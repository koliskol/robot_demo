# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two independent Isaac Sim demos sharing the same Galbot G1 robot rig:
- A WebRTC live-streaming pipeline: `stream_demo.py` drives the robot around a small scene and
  pushes its chassis camera (RGB + depth) and chassis lidar (point cloud) to a browser via
  `streaming_server.py`. The browser side lives in `static/` (plain HTML/JS, three.js via CDN
  import map for the point cloud, canvas 2D for the top-down map).
- A LeRobot imitation-learning data pipeline: `collect_pickplace_demo.py` builds a
  table+pushcart+box scene and records keyboard-teleoperated pick-and-place demonstrations to
  disk; `convert_to_lerobot.py` (run in a separate environment) converts those into a
  LeRobotDataset for training an ACT-style policy. See "LeRobot pick-and-place pipeline" below.

## Commands

Run the full sim + streaming demo (requires an Isaac Sim conda env; there is no venv/requirements
setup for this half — it depends on `isaacsim`, `carb`, `omni.*`, `pxr`, etc. from that
environment):

```
conda run -n isaac_sim python stream_demo.py
conda run -n isaac_sim python stream_demo.py --port 8080 --drive-speed 1.0 --turn-speed 0.3 --arm-speed 0.4 --torso-speed 0.4
```

Then open `http://<host>:<port>/` in a browser and click Connect.

Run just the streaming server standalone, with a synthetic moving pattern instead of real sim
data, to check the server/viewer independent of Isaac Sim (only needs `requirements.txt`, no
Isaac Sim install):

```
pip install -r requirements.txt
python streaming_server.py
```

Collect pick-and-place demonstrations (same conda env, no server/streaming involved):

```
conda run -n isaac_sim python collect_pickplace_demo.py
conda run -n isaac_sim python collect_pickplace_demo.py --out ./raw_episodes --deck-riser 0.5
```

Convert recorded episodes into a LeRobotDataset (separate environment — see the architecture note
below for why):

```
conda create -n lerobot python=3.10 -y && conda activate lerobot && pip install lerobot
python convert_to_lerobot.py --raw-dir ./raw_episodes --repo-id local/pick_box_table_to_cart --root ./lerobot_dataset
```

There is no test suite, linter, or build step configured in this repo.

## Architecture

**Two independent runtimes glued by one shared object.** `stream_demo.py` (Isaac Sim, synchronous,
owns the main thread — Kit/PhysX require that) and `streaming_server.py` (asyncio aiohttp/aiortc
server) run on separate threads. `run_in_background()` starts the server's own asyncio event loop
in a daemon thread and returns immediately. The only thing crossing the thread boundary is a
`FrameStore` instance (lock-protected latest-value holder, no queue): the sim loop calls
`update_rgb`/`update_depth`/`update_points`/`update_world_state` every physics step, and the
server's tracks/senders read whatever's currently there whenever a client asks — like a live
camera, not a buffered stream. A slow client just sees fewer/staler frames, never a growing
backlog. `streaming_server.py` has no `import isaacsim` anywhere, which is what makes it
standalone-runnable for testing the viewer without Isaac Sim.

**`build_app`/`run_in_background` serve a different static page per consuming script**, via
`static_index`/`static_viewer_js` params (default `"index.html"`/`"viewer.js"`, so `stream_demo.py`'s
own call site is unchanged). `stream_demo.py` gets the original page (RGB, depth, point cloud, top-
down map); `collect_pickplace_demo.py` gets its own `static/collect_index.html` +
`static/collect_viewer.js` (RGB + recorder controls only — no map/point-cloud/three.js, since that
task has no lidar or world-state data). The URL path is always `/` and `/viewer.js` either way —
only which file on disk answers those routes changes. Video track negotiation itself is
unconditional regardless of which page connects (`offer()` always calls `pc.addTrack(RGBTrack(...))`
then `pc.addTrack(DepthTrack(...))`), so `collect_viewer.js` still declares two recvonly
transceivers even though it only renders the first track — skipping a *render* is free; skipping a
*transceiver* would mean the server's second `addTrack()` has no matching offer m-section, hitting
the same JSEP violation the data-channel comment below describes.

**Six data paths total, split across the two pages, each shaped differently:**
- RGB and depth are separate WebRTC video tracks (`RGBTrack`, `DepthTrack`), on both pages. Depth
  is false-colored (`_depth_to_rgb`: blue near → green mid → grey far, fixed 0–20m scale so a
  color always means the same distance across frames) — exact float depth is never sent anywhere.
- Point cloud (`stream_demo.py`'s page only) goes out over a `"pointcloud"` WebRTC data channel as
  raw `float32` xyz bytes, paced at 5 Hz by the server (`_send_point_cloud`), decoded client-side
  straight into a `Float32Array` for a three.js `Points` cloud.
- World state (`stream_demo.py`'s page only: room outline, static object footprints, robot pose)
  goes out over a `"worldmap"` data channel as JSON at 10 Hz (`_send_world_state`), rendered as a
  top-down view on a plain `<canvas>`.
- Status (`collect_pickplace_demo.py`'s page only) goes out over a `"status"` data channel as
  JSON at 5 Hz (`_send_status`, `FrameStore.update_status`/`get_status`) — deliberately generic
  (an arbitrary dict; this module has no concept of what's in it). `collect_pickplace_demo.py`
  uses it for the recorder's state/episode/frame-count.
- **The one path in the other direction, also `collect_pickplace_demo.py`-only**: a `"control"`
  data channel carries browser→server messages (button clicks), relayed via
  `FrameStore.push_command`/`pop_commands` into whatever the consuming script's main loop wants to
  do with them — again generic, this module doesn't interpret command contents.
  `collect_pickplace_demo.py` maps `{"action": "toggle_record"}` / `{"action": "label", "value":
  "success"|"fail"}` / `{"action": "discard"}` onto the exact same flags its keyboard handler
  sets, so a browser button and a keypress are interchangeable inputs into one state machine, not
  two parallel ones.

**Data channels must be created client-side, including the reverse-direction "control" one.**
Per WebRTC/JSEP, an answer can't introduce an SCTP "application" section that wasn't in the
offer, so the browser (`static/viewer.js` / `static/collect_viewer.js`) calls `createDataChannel`
for every channel it needs — even `"control"`, which the browser sends on and the server only
listens to — before generating its offer, and the server only ever listens via
`pc.on("datachannel")` — a server-side `createDataChannel()` call after receiving the offer
cannot negotiate (confirmed live: `readyState` stuck at `"connecting"` forever). Same reasoning
shapes the video side: both tracks land in one remote `MediaStream` on the client (server never
assigns them to distinct streams), so `viewer.js` wraps each `ontrack` event's own track in a
fresh `MediaStream` rather than using `event.streams[0]`, or both `<video>` elements end up
showing the RGB feed.

**Robot control in `stream_demo.py`** is a per-frame position/velocity command loop keyed off
`held_keys` (see the module docstring for the full key map). Two recurring patterns worth knowing
before touching joint control:
- Every jogged joint target is passed through `clamp_to_actual()` before being applied, capping
  how far a position command can run ahead of the joint's actual physical position
  (`MAX_JOINT_LEAD_RAD`, tighter `ARM_CONTACT_MAX_LEAD_RAD`/`GRIPPER_MAX_LEAD_RAD` for the
  arm/gripper). This exists because an unclamped lead against something rigid was live-observed to
  cause joint velocity spikes and fling the robot out of the scene — don't relax these without a
  reason.
- `ROBOT_FORWARD_OFFSET_RAD` (−π/2 correction in `robot_heading_yaw` usage) exists because the
  Galbot G1 asset's root +X axis is not the direction it actually drives, confirmed empirically.
  Any new code deriving "which way is the robot facing" from orientation needs this offset too.

**This project intentionally mirrors, but does not import, a sibling project**
(`../Robot_project/capture_cube_rgbd.py`): the same Galbot G1 asset, camera/lidar mounting, and
arm/hand/torso jog controls/joint constants are ported as-is from there (see that file's docstring
for the full derivation — FK sweeps, live-tuned instability fixes). This project deliberately
leaves out that sibling's pushcart/cube/capture-to-disk features to stay focused on the streaming
pipeline. When changing shared constants (arm poses, lead clamps, wheel/holonomic setup), check
whether the sibling file should change too rather than assuming this repo is the sole source of
truth for that robot rig.

**Lidar is a real sensor model, not a simplification**: `OS1_REV6_32ch10hz512res` is a 32-channel
3D lidar (32 stacked scan rings), so the point cloud has real vertical structure — this is a
deliberate config choice, not incidental.

## LeRobot pick-and-place pipeline

`collect_pickplace_demo.py` is a third sibling to `stream_demo.py`/`capture_cube_rgbd.py`, built
the same way (copied and adapted, not imported — see the mirroring note above, which applies
here too). It drops lidar entirely (not needed for offline data collection) but captures and
records both RGB and depth (raw float32 meters per frame, `frames/NNNNNN_depth.npy`, "just in
case" a future policy wants it — `convert_to_lerobot.py` doesn't use it yet, see that script's own
docstring for why) and reuses `streaming_server.py`'s WebRTC server — same module `stream_demo.py`
uses — but with its own dedicated page (`static/collect_index.html`/`collect_viewer.js`, RGB +
depth + recorder controls, no map/point-cloud) rather than the shared one, so nothing here touches
`stream_demo.py`'s page. This is unrelated to what actually gets recorded (`EpisodeRecorder`
samples the same camera independently, at `--record-fps`, regardless of whether anyone's watching
the live stream). It
also pushes recorder state through the `"status"` channel and reads browser button clicks back
through `"control"` (see the streaming architecture section above) — both drive the *same*
`record_requested`/`label_success_requested`/etc. flags the keyboard handler sets, so
`B`/`Y`/`F`/`Backspace` and the browser's
Start/Stop/Success/Fail/Discard buttons are interchangeable inputs into one state machine. It adds
a pushcart (`build_pushcart`, ported from
`capture_cube_rgbd.py`) placed **adjacent** to the
table rather than across the room, so the task is pure fixed-base arm/gripper/torso manipulation
— no driving during an episode, no base pose in the recorded state/action space (21 dims: both
7-DOF arms, 5-DOF torso, both grippers). `--cube-start {table,cart}` controls where the box
spawns on reset — run one recording session per value to collect both directions (table→cart and
cart→table); the task name recorded in each episode's manifest defaults accordingly unless
`--task` overrides it. On table-start sessions, two extra bigger boxes spawn by default
(`--cube2-scale`/`--cube3-scale`, `--no-extra-boxes` to disable) as real pick-up targets for size
variety — not on cart-start sessions, since the pushcart deck (`PUSHCART_DECK_HALF_EXTENT`) is
too small to fit 3 boxes side by side. None of this is tracked in the recorded state/action
(robot-only, 21 dims) — box choice only affects what the camera sees, the same way varying
`--cube-scale` across sessions would.

**Boxes are real warehouse cardboard-box assets, not procedural cubes.** `spawn_real_box()`
references three distinct real box props from Isaac's warehouse/logistics environment set
(`BOX_ASSET_MAIN`/`CUBE2`/`CUBE3` — generic shipping boxes, sized ~0.38m/0.50m/0.70m). Unlike an
earlier iteration using Isaac's YCB grocery-item assets (which already carried
`RigidBodyAPI`/`MassAPI`), these warehouse props ship as **static, collision-only meshes** —
confirmed live: the mesh's collision approximation defaults to `"none"` (an exact triangle mesh),
which PhysX accepts for a static collider but rejects for a *dynamic* rigid body. `make_box_dynamic()`
explicitly authors `RigidBodyAPI`+`MassAPI` on the root and overrides the mesh's collision
approximation to `convexHull` (tested live: all three settle cleanly under gravity with negligible
drift, no instability) — any new real-asset prop added to this scene needs the same treatment
unless it's independently confirmed to already ship with dynamic-body-compatible physics.
`place_on_surface`/`scaled_footprint` extend `place_on_ground`'s scale-then-measure trick to rest
something on an arbitrary surface height (a tabletop or cart deck) and to measure a scaled
footprint without moving the prim — both must be called only once per prim, at its
just-referenced identity transform, or the measurement is wrong (see their docstrings). Friction
is left at each asset's own baked-in default for now — not overridden with a custom
`PhysicsMaterial`, since that would mix the `isaacsim.core.api` (legacy) and
`isaacsim.core.experimental` physics-material APIs without live verification; that's the next
lever if the hug hold proves unreliable, not a kinematic attach. `BOX_ASSET_MAIN` (the smallest,
CardBoxD) was deliberately chosen small enough to also fit the pushcart deck; `CUBE2`/`CUBE3` are
table-only and `CUBE3`'s 0.7m width leaves only ~0.05m margin against the table's 0.8m x-extent
(confirmed live, not just estimated) — reduce `--cube3-scale` if the table asset ever changes.

**The central open risk is holding the object itself.** Neither this project nor
`capture_cube_rgbd.py` has ever demonstrated actually lifting and carrying a loose object — only
pinching a fixed obstacle or a heavy pushcart handle. The chosen approach is a bimanual **hug**
(both arms swinging forward via `U` to compress the box between the forearms) rather than a
single gripper's fingertip pinch — hence real box assets sized well beyond a gripper's grasp
margin; gripper open/close (`M`/`N`) is optional extra contact, not the primary hold. This is
still friction-only contact, so the same constraint applies as would have applied to a gripper
pinch: `capture_cube_rgbd.py`'s own comments record that every *kinematic* grasp assist tried on
this hand link (a hand-authored `FixedJoint`, Isaac Sim's `IsaacSurfaceGripper`) reproducibly
destabilized the whole robot, because the hand is an actively-driven articulation link, not a
simple jointed body. Never grow a joint-based/kinematic attach mechanism to stabilize the hug —
if it proves unreliable, the fix space is box mass/scale/friction and arm swing-in distance, not
a new attach primitive. Run through one full pick/place/pick/place cycle by hand (see the
script's module docstring, "Stage 0") before trusting any recorded data — in particular, confirm
both arms can actually converge around the box from a single parked robot pose without
repositioning.

**Cart deck height is a tunable, not a constant**: the pushcart's stock deck sits ~0.15m off the
floor (designed for "push by the handle," not "place a box here"), almost certainly well below
table height. `pushcart_deck_top_z()`/`--deck-riser` raise the deck without touching the
already-stability-tuned caster joints — tune this empirically per the module docstring rather than
assuming a value. Likewise `ROBOT_APPROACH_GAP_M`/`CART_TABLE_GAP_M` (parking distance and
cart-table clearance) are first guesses, not verified reach envelopes.

**`--drive-speed`/`--turn-speed` default lower here (0.4/0.15) than `stream_demo.py`'s (1.0/0.3)**
— live-observed the robot tipping over at the higher defaults, especially on diagonal drive+strafe
key combos (command magnitude adds across axes, so holding two drive keys can exceed 1.0 even at
`--drive-speed 1.0`). Only affects parking/repositioning between episodes, never anything
recorded — raise them back via the flags if driving feels too sluggish and tipping isn't an issue
for your setup.

**Lighting is authored explicitly in `main()`** (a `DomeLight` + `DistantLight`, matching
`stream_demo.py`'s `build_scene`) — `build_room` only builds walls, it was never responsible for
lighting, and an earlier version of this script omitted lights entirely (scene was too dark to
see anything). If a future refactor moves scene-building around, keep the light authoring
somewhere that always runs.

**The camera is head-mounted (`HEAD_CAMERA_MOUNT`), not chassis-mounted like `stream_demo.py`'s
`front_camera`.** This asset has a real 2-DOF head (`Head_Golf`) with its own purpose-built
sensor mount point (`head_end_effector_mount_link`), found by walking the robot's full prim tree
live, not guessed — nothing drives the head joints, so it sits at rest, but a head-mounted camera
still moves with torso crouch (`I`/`K`), unlike a fixed chassis mount. Two earlier iterations got
this wrong before landing here, both confirmed live rather than assumed: (1) a chassis-mounted,
flat/level mount (`stream_demo.py`'s exact mount) put the box completely out of frame at the
robot's normal table approach distance, because the table's own edge geometry blocks line of
sight to anything on top of it at a grazing near-horizontal angle — fixed by tilting down, not by
repositioning; (2) the head mount's own local frame doesn't face the robot's forward direction at
all (confirmed by rendering at identity orientation — showed a sideways, rolled view) and needs a
90° roll correction (`CAMERA_ROLL_DEG`) before any additional downward pitch makes sense; that
correction is composed via `quat_multiply`/`camera_head_mount_quat`, verified against
`scipy.spatial.transform.Rotation`'s composition before trusting it. The needed downward pitch
(`CAMERA_TILT_DEG`, 15°) is much smaller here than the old chassis mount needed (45°), because
the head sits much higher and further forward, so the look-down angle to a table-height box is
shallow — confirmed by computing the actual head-to-box world vector live, not guessed by feel.
FOV widened to `CAMERA_FOV_DEG` (90°) to help keep a nearby box in frame. Confirmed clear (and
actually improved vs. the original 10°) for the box-on-table view at the robot's normal ~0.9m
parked distance. Pushcart framing is still not reliable, but that's the existing robot/cart
y-alignment issue above, not something this mount
causes.

**Isaac Sim's `Camera` defaults to a 1.0m near clipping plane** (confirmed live via
`camera.get_clipping_range()` — not a documented default anyone would guess), meaning anything
closer than 1m to the lens is silently not rendered at all. This was the actual cause of the box
(and the robot's own hand) "disappearing" once brought close during the hug — not a framing/angle
problem like the box-on-table case above, an outright render-time clip. Confirmed directly: with
the default clip, an object ~0.4m from the lens rendered as nothing; with `CAMERA_NEAR_CLIP_M`
applied via `camera.set_clipping_range()`, the same object is visible.

**`CAMERA_NEAR_CLIP_M` is 0.1, not something smaller — a first attempt at 0.02 broke rendering
entirely** (confirmed live via a sweep, not assumed): mean frame brightness collapsed from ~195
to ~0.15 with `near=0.02` — independent of the far value (both 50m and the 1,000,000m default
were equally broken at that near value). 0.03 and 0.05 were also broken/badly dark; 0.08 partially
recovered; 0.1 and above exactly matched normal baseline brightness. This isn't the usual
near/far-ratio depth-precision story (far value provably didn't matter) — more likely something
specific to how the RTX renderer's auto-exposure or a similar pass reacts to a near-zero near
plane. 0.1 is still a real improvement over the 1.0m default (confirmed a box at ~0.35m renders
clearly) without triggering whatever breaks at smaller values — **do not lower this without
re-testing actual rendered brightness**, not just whether `set_clipping_range()` succeeds without
erroring, since the failure mode here is silent (no exception, just a near-black frame).
`CAMERA_FAR_CLIP_M` (50m) is just tightened from the 1,000,000m default to match this scene's
actual scale — confirmed not implicated in the near-value breakage above.

**Two distinct kinds of self-occlusion were found and confirmed, not guessed — don't conflate
them:**
- **A dark curved shape intruding into the frame even at rest pose (no crouch, no arm swing) is
  the camera seeing part of its own head.** Root-caused via a physics raycast (`raycast_closest`
  from the camera through the exact screen region the shape occupied), not visual guessing: every
  ray in that region hit `head_link2`'s own collision mesh — the camera mount point sits close
  enough to the head's own physical shell that its lower edge pokes into the camera's field of
  view. Fixed by `CAMERA_MOUNT_FORWARD_OFFSET_M` (0.1m along the mount's local +X), confirmed both
  directions: 0.0 shows the obstruction, +0.1 fully clears it, -0.1 makes it fill most of the
  frame instead.
- **A dark shape filling most of the frame during a deep torso crouch + forward arm swing is a
  separate issue: the robot's own torso/shoulder**, not the head. Confirmed via a controlled test
  (driving torso/arms to their crouched/forward poses through the same `clamp_to_actual` scheme
  `main()`'s loop uses, not letting them free-fall-settle, then rendering): once the torso is
  crouched most of the way down while the arms are also swung forward, the torso/shoulder ends up
  directly in front of the head-mounted camera. Tilting down further makes this **worse**, not
  better — it was tested, and it just puts the torso even more squarely in frame, since the torso
  is now the closest thing to the lens. If the hand still isn't visible, the fix is less torso
  crouch (partial `I`/`K`) relying more on elbow lift (`J`/`L`) to keep the torso out of the
  camera's line of sight — a teleoperation-technique fix, not something tunable via these
  constants, unlike the head-housing case above.

**The head can visibly wobble/tilt in unintended directions during ordinary torso+arm motion —
not a bug in `HEAD_CAMERA_MOUNT`'s roll/tilt math.** `head_joint1`/`head_joint2` ship with very
weak PhysX drive stiffness/damping (~2.8/0.001 and ~0.99/0.0004, confirmed live) and, unlike
every other controlled joint group in this file, nothing commanded them at all until
`stiffen_head_joints()`/`hold_head_joints()` were added. Live-tested under an aggressive 1s
full-range torso-crouch + arm-swing stress cycle: uncommanded, the head can swing up to ~36° —
easily enough to look like the camera itself is tilted sideways rather than down, since the drift
direction isn't controlled. `stiffen_head_joints()` (called once, before `world.reset()`, raising
stiffness/damping to 200/20) plus `hold_head_joints()` (called every physics step, same as every
other joint group, no lead clamp — tested and confirmed one makes no measurable difference here,
unlike arm/gripper) together cut worst-case drift to ~12° under that same aggressive test; going
stiffer still (2000/100) barely helped further (~11°), so it isn't pushed beyond 200/20. Real
teleoperation (not a full-range flip every single second) should see less than this worst case,
but some residual wobble during fast motion is expected, not eliminated.

**Recording is decoupled from LeRobot on purpose.** `collect_pickplace_demo.py` writes raw
per-episode data (`manifest.json` + `data.npz` + `frames/*.png`) with zero `lerobot` dependency,
because `lerobot` isn't installed anywhere on this machine and installing it into the `isaac_sim`
conda env risks conflicting with Isaac Sim's own pinned deps (opencv/av/gymnasium etc.).
`convert_to_lerobot.py` has zero Isaac Sim imports and is meant to run in a separate, disposable
`lerobot`-pip-installed env. Its exact `LeRobotDataset` API calls are unverified against a real
install (the module docstring says what to check) — don't assume they're correct without testing
against whatever lerobot version actually gets installed.
