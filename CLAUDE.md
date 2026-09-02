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
a pick/place partner placed **adjacent** to the
table rather than across the room, so the task is pure fixed-base arm/gripper/torso manipulation
— no driving during an episode, no base pose in the recorded state/action space (21 dims: both
7-DOF arms, 5-DOF torso, both grippers). `--place-target {cart,table2}` chooses which partner the
scene builds — a pushcart (`build_pushcart`, ported from `capture_cube_rgbd.py`) or a second
table (same asset/height as the main one, `TABLE2_GAP_M`/`TABLE2_EDGE_INSET_M`) — and only one is
ever built per session; they're alternative task variants (table↔cart vs. table↔table2), not
simultaneous targets, since a single parked robot pose can't reach both a cart and a full-size
table2 at once. Unlike the small pushcart deck, table2 uses the same full-size table asset as the
main table (0.8m × 2.8m) — far too long to reach across from one parked pose — so its actual
pick/place point is `TABLE2_EDGE_INSET_M` onto its surface from the near edge, not its centroid;
most of table2's surface sits out of reach, which is fine, real tables are bigger than their
contact patch too. `--cube-start {table,cart,table2}` controls where the box spawns on reset (must
be `table` or match `--place-target`) — run one recording session per value to collect both
directions; the task name recorded in each episode's manifest defaults accordingly (e.g.
`pick_box_table_to_table2`) unless `--task` overrides it. On table-start sessions, two extra
bigger boxes spawn by default (`--cube2-scale`/`--cube3-scale`, `--no-extra-boxes` to disable) as
real pick-up targets for size variety — only on `table`-start sessions (checked via
`args.cube_start == "table"` alone, not `--place-target`), never on `cart`- or `table2`-start
sessions, since the pushcart deck (`PUSHCART_DECK_HALF_EXTENT`) is too small to fit 3 boxes side
by side and table2's extra-box placement was never added (it spawns extras on table1's surface
specifically, not wherever the box starts). None of this is tracked in the recorded state/action
(robot-only, 21 dims originally, now 22 with `chassis_forward` — see below) — box choice only
affects what the camera sees, the same way varying `--cube-scale` across sessions would.

**Table2 was moved from ahead of table1 to its side, with a real (not token) gap, once
`chassis_forward` existed to make driving there worth recording.** Originally table2 sat directly
north of table1 (same X-center, offset a bare `TABLE2_GAP_M`=0.15m along Y) specifically so a
single parked pose could reach it by arm swing alone, without any driving - the recorder used to
have no way to capture chassis motion at all, so driving was something to design around, not use.
Now that it can, per the user's own request ("move table2 farther left so the robot could move to
the side of the table") table2 sits offset along X instead (table1's *short*, 0.8m axis) with
`TABLE2_GAP_M` raised to 0.6m, so the robot approaches table2's *long* 2.8m edge rather than its
narrow 0.8m end, and covering the gap is a deliberate, recordable drive rather than something to
avoid. `TABLE2_SIDE_SIGN` (+1 = table1's +X/xmax side, the default; -1 = the other side, same side
the robot parks on) picks which side - flip it if the layout reads backward once viewed live,
nothing else depends on which sign is used. Unverified live, same as the box-jitter and rollout
features above - re-run Step 0 to confirm the new reach distance/direction actually works before
trusting it for a real `place_policy` session.

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

**The main pick box's spawn/reset pose is randomized per episode, not fixed** — and so is table2's,
as of a later addition (see below). Originally the box spawned at the exact same (x, y, yaw) every
episode (including across `R`-triggered resets within one session), which risks training a policy
that only ever saw one pixel-perfect box pose and doesn't generalize to any real-world placement
error. `sample_pose_jitter()` (renamed from `sample_box_jitter()` once table2 started reusing it -
it was already fully generic, box-specific in name only) draws a fresh `(dx, dy, yaw)` each time -
dx/dy uniform over a disk (not a square) of radius `--box-jitter-m` (default 0.03m) around the
tuned anchor position, yaw uniform in `±--box-yaw-jitter-deg` (default 10°) — applied both at
initial spawn and, via `box_xform.set_world_pose()` right after `world.reset()`, on every
subsequent reset. `--seed` makes the jitter sequence reproducible; default is a fresh sequence each
run. **This is unverified live** (written without an Isaac Sim install available) — the specific
risk is that `ARM_FORWARD_POSE`'s hug convergence was tuned against one exact box position, and
hasn't been confirmed to still converge from the jittered extremes; the *default* 3cm/10° jitter is
also small enough that it may not be visually obvious it's happening at all when watching the
viewport - check the printed `[box] episode NNNN spawn offset dx=... dy=... yaw=...` console line
each reset to confirm it's actually sampling different values, rather than assuming from a glance
that nothing moved. Re-run the Stage 0 manual reach cycle after enabling this (it's on by default)
and watch several resets play out before trusting it for a real collection session; lower
`--box-jitter-m`/`--box-yaw-jitter-deg` (or pass `0` to disable either) if the hug stops reliably
converging.
Only the main box (`/World/Cube`) is randomized — the two decorative extra boxes (`Cube2`/`Cube3`,
table-start only) stay at their fixed offsets from it, since they're untracked distractors, not
the pick target.

**Table2 itself is also jittered now** (`--table2-jitter-m`/`--table2-yaw-jitter-deg`, defaults
0.05m/5° — a bit bigger than the box's default since imprecise destination placement matters less
than imprecise grasp positioning, and a bit smaller on yaw since a full table rotating meaningfully
changes the whole reach geometry far more than a small box does), same disk/reset pattern as the
box, using the same now-generic `sample_pose_jitter()`. `rng` had to move earlier in `main()` (from
just above the box's setup to just above the place-target build) since table2 is built before the
box and now also needs it. `TABLE2_EDGE_INSET_M`'s reach point (`target_x`/`target_y`) shifts with
table2's *initial* jitter sample so the two stay consistent at spawn, but does not keep re-tracking
table2 on every subsequent `R`-reset (nothing in the main loop reads `target_x`/`target_y` again
after scene setup, so this was never wired up) - **a known, currently-unaddressed edge case**: for
`--cube-start table2` sessions specifically (box starts already on table2, not table1), the box's
spawn anchor stays pinned to table2's *first* jittered position even as table2 keeps moving on
later resets, so the box and table2 can drift out of alignment after a few resets in that
configuration. Not an issue for the current `pickup_policy`/`place_policy` workflow (`pickup_policy`
always uses `--cube-start table`; `place_policy` starts the box already held via manual hug, not
auto-spawned on table2), but worth fixing properly before ever using `--cube-start table2` for real
collection. Not yet implemented for `--place-target cart` (`table2_xform` stays `None` in that
branch) - the cart is a 9-body dynamic assembly (see below), and jittering a dynamic multi-body
prim the way the box was carries the same live-verification risk described above, doubled.

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

**Live camera pan/tilt calibration exists to find a better mount angle interactively, not just by
editing constants and relaunching.** Arrow keys (Left/Right pan, Up/Down tilt) or the browser's
rotate buttons adjust `camera_pan_deg`/`camera_tilt_deg` at runtime via `camera_pan_tilt_quat()`,
printing the current values (on key release, or after each browser click) to paste back into
`CAMERA_PAN_DEG`/`CAMERA_TILT_DEG` once satisfied. Adding pan required more than just plugging a
third rotation into `camera_head_mount_quat`'s existing roll+tilt composition — naively rotating
about a raw local axis (X, Y, or Z, tried in every composition order relative to the existing
roll/tilt) reproducibly showed up as an unwanted **extra tilt** instead of a clean left/right pan,
confirmed via many live render comparisons, not just one. Root cause, found by reading Isaac Sim's
own source rather than guessing further: `Camera.set_local_pose(camera_axes="world")` (the
default) silently right-multiplies the given orientation by a fixed correction matrix
(`isaacsim.sensors.camera.camera.W_U_TRANSFORM`) before authoring it, and that correction doesn't
commute cleanly with a naive extra local-axis rotation. `camera_pan_tilt_quat()` instead applies
pan as a genuine **world-space yaw** (rotation about the global up axis) on top of the existing
roll+tilt orientation, explicitly undoing and redoing that same correction matrix (`CAMERA_WU_QUAT`,
the matrix converted to a quaternion) plus the mount's own live world rotation (read fresh each
call, since it moves with the torso/arm chain) - confirmed live: the box shifts purely
horizontally, with the horizon/amount-of-floor-visible unchanged, unlike every raw-axis attempt.
At `pan_deg=0` this is mathematically guaranteed (and confirmed live) to reduce to exactly
`camera_head_mount_quat(roll_deg, tilt_deg)`, so existing tuned defaults are unaffected.

**Recording is decoupled from LeRobot on purpose.** `collect_pickplace_demo.py` writes raw
per-episode data (`manifest.json` + `data.npz` + `frames/*.png`) with zero `lerobot` dependency,
because installing it into the `isaac_sim` conda env risks conflicting with Isaac Sim's own pinned
deps (opencv/av/gymnasium etc.). `convert_to_lerobot.py` has zero Isaac Sim imports and is meant to
run in a separate, disposable `lerobot`-pip-installed env (`conda create -n lerobot python=3.10 -y
&& conda activate lerobot && pip install lerobot`).

**Its `LeRobotDataset` API calls have now been confirmed against a real install (lerobot v0.6,
`codebase_version: "v3.0"`)** — `create()`'s signature, `add_frame(dict)`, and `save_episode()`
all matched the script's existing usage as originally written, with one exception: `create()`'s
`fps` parameter is typed `int`, but `manifest.json` stores `fps` as a JSON float (`15.0`).
Passing the float through reached PyAV's `add_stream()` during video encoding and crashed in
`to_avrational` with `'float' object has no attribute 'numerator'` — fixed by casting
`fps = int(manifest["fps"])` right where it's read from the manifest, before it reaches
`create()`. Ran a full conversion (41 success-labeled episodes, 27,154 frames) and read every
sample back via `LeRobotDataset(repo_id, root=...)` + indexing to confirm decoded video shape
`[3, H, W]` and state/action shape `[21]` — this isn't just "the script exited 0", the output was
actually loaded and inspected.

**Video-frame decoding needs ffmpeg's shared libs present, and a system ffmpeg isn't enough by
itself.** `lerobot`'s video backend is `torchcodec`, which ships prebuilt binaries pinned to a
specific ffmpeg ABI/so-version. On a machine with no `ffmpeg` binary at all and only a mismatched
system `libavutil.so.58` (torchcodec wanted `.so.56`/`.so.4`), reading back any sample (`ds[0]`)
failed with `OSError: Could not load this library: .../libtorchcodec_core*.so`, even though
dataset *creation*/encoding had already succeeded — the crash only surfaces on read, so a
conversion run finishing without error is not proof the dataset is actually loadable. Fixed by
`conda install -n lerobot -c conda-forge ffmpeg -y`, which drops matching-ABI shared libs inside
the env itself rather than depending on whatever ffmpeg (if any) the system happens to have.

**Evaluating a trained checkpoint has two tiers, open-loop then closed-loop, because closed-loop
needs a second process.** `evaluate_act_checkpoint.py` (runs in the `lerobot` env, zero Isaac Sim
imports) replays recorded episodes frame-by-frame through the policy's real inference path
(`policy.select_action()`) and compares predicted vs. recorded actions - fast and needs nothing
but the `lerobot` env, but it's an in-sample, open-loop tracking check: the policy is always fed
the *true* recorded observation, never what it would have seen after acting on its own prediction,
so it says nothing about whether the policy can actually control the robot end-to-end. A first ACT
checkpoint (table-to-table2, 35 episodes/23,307 frames, 30k steps) scored mean MAE 0.0068 rad
across 3 episodes this way - a useful sanity check that training converged, not evidence the hug
would succeed.

Actually finding that out needs closed-loop rollout inside the real Isaac Sim scene, which can't
be one script: policy inference needs `torch`/`lerobot`, which must not be installed into the
`isaac_sim` conda env (same conflict-risk reasoning as the recording/conversion split above), so
`collect_pickplace_demo.py --rollout` (isaac_sim env) and `policy_server.py` (lerobot env, loads a
checkpoint via `lerobot_policy_utils.load_policy()` - the same loading path
`evaluate_act_checkpoint.py` uses, factored out once it was needed by both) talk over a
`127.0.0.1`-only TCP socket. Wire protocol lives in `policy_wire.py` (stdlib + numpy only, no
torch/lerobot - safe to import from either env): 4-byte length prefix + JSON, numpy arrays as
base64 with explicit shape/dtype - **not pickle**, even though this never leaves localhost, since
avoiding an arbitrary-code-exec deserializer here costs nothing. `policy_client.py` (also stdlib +
numpy only) is what `collect_pickplace_demo.py` imports. Confirmed live end-to-end from this
session (no Isaac Sim involved, just the two processes talking): connect → reset → 5×predict all
succeeded, returned `(21,)` finite `float32` vectors, ~8-9ms latency per predict after a ~280ms
first-call warmup - comfortably inside the 67ms budget a 15Hz control loop allows.

**Real finding, not assumed**: `predict_action()`'s own docstring claims it strips the batch
dimension before returning; the installed `lerobot` (0.4.4) doesn't - it returns shape `(1, N)`
(kept for vectorized-env compatibility upstream, where a real env step wants one action per parallel
env). `policy_server.py` squeezes this explicitly (`action.squeeze(0)`) before sending it over the
wire, rather than relying on numpy's broadcasting rules to silently paper over the extra dimension
downstream (which is what let `evaluate_act_checkpoint.py`'s original in-process version pass
without erroring - broadcasting a `(1, 21)` array into a `(21,)` slot works, so nothing flagged it
until this was made explicit for the wire protocol). Don't trust that docstring for this lerobot
version.

`collect_pickplace_demo.py --rollout` reuses the *exact* same `clamp_to_actual()` calls (same
per-group `max_lead` constants: `ARM_CONTACT_MAX_LEAD_RAD`/default `MAX_JOINT_LEAD_RAD`/
`GRIPPER_MAX_LEAD_RAD`) teleop already uses - only the pre-clamp target's source changes (a policy
prediction instead of a held-key-driven fraction). This is deliberate: those clamps are the guard
against the joint-velocity-spike/fling failure mode described above, and they must apply
identically no matter where the target came from. The policy is queried once per `record_fps` tick
(matching the 15Hz rate training data was sampled at); between queries the same last-received
target keeps getting re-applied every physics step, same as teleop's fraction state does between
key-presses. `B`/`Y`/`F`/`Backspace`/`R` are unchanged - `B`-start now also calls
`policy_client.reset()` (clears ACT's internal action-chunk queue for a fresh attempt) and `R`
still re-triggers the box-jitter re-randomization already wired in, so every rollout attempt gets a
fresh box pose. Rollout attempts record through the same generic `EpisodeRecorder`, defaulting to
`./rollout_episodes` (not `raw_episodes/`) so policy predictions never silently mix into training
data.

**The Isaac Sim side of `--rollout` has now been watched live, once, and it surfaced a real bug
that's since been fixed**: the idle-pose fallback (`policy_action_vec is None`, i.e. before the
first prediction of an attempt arrives) originally used the fully-open arm pose, which turned out
to be close enough to the robot's raw spawn pose that the robot looked completely frozen at launch
- there was no version of the visible ~5s settle-into-position motion teleop mode shows. Fixed by
making the fallback match teleop's actual initial condition (`STARTING_LEFT_ARM_SWING_FRACTION`=
0.815/`STARTING_RIGHT_ARM_SWING_FRACTION`=0.663/`STARTING_HAND_UPDOWN_RAD`=1.173-derived pose, not
the open pose). Whether a full pick/place attempt actually succeeds end-to-end is still not
confirmed - that first watched session ended with the policy connected and the robot correctly
parked, but no attempt had been graded yet (see the chassis_forward addition below, which changes
what "attempt" now means anyway).

**The recorded state/action space grew from 21 to 22 dims: `chassis_forward` was added so
forward/backward chassis motion is finally part of what gets learned, not silently dropped.**
Motivated by a live-observed rollout failure pattern: the trained (21-dim, fixed-base) policy could
reach and hug the box only once a human manually drove the chassis into range first, and manual
driving mid-attempt then made the later place/release phase near table2 fail most of the time -
because the recorder was *already* silently discarding wheel motion (`wheel_dof_indices` drives the
wheels live every step via `compute_drive_command`/`held_keys`, but was never part of
`state_dof_indices` or `action_vec`), so any chassis motion during a recording showed up in the
video but not in the labels the policy trained on - a real, confirmed train/inference mismatch, not
a modeling issue.

The fix does *not* add full SLAM/navigation into the policy - per the user's own target
architecture, SLAM (a separate, not-yet-built component) is responsible for getting the robot from
one location to the general vicinity of the next; the policy only needs to handle the short final
approach/retreat around the pick or place point, which is exactly what `chassis_forward` covers.
The 22nd dim is asymmetric between state and action, unlike every other dim (where action = target
position for the same joint state reads):
- **State** (`forward_displacement_m`): cumulative *signed* distance (meters) the chassis has moved
  along its own forward axis since the current attempt started (0.0 at the first recorded frame,
  reset on every `B`-press-start and `R`-reset) - not an absolute world position, so it stays
  meaningful regardless of where the robot happens to be parked. Computed via
  `robot_forward_reference()`/`forward_displacement()` (new), which needed `robot_heading_yaw()` +
  `ROBOT_FORWARD_OFFSET_RAD` ported from `stream_demo.py` (same Galbot G1 asset/root prim, same
  -π/2 heading-vs-actual-drive-direction offset documented there) - this file never needed heading
  before.
- **Action** (`chassis_forward`): the forward/back drive *command* that tick (`command[0]` from
  `compute_drive_command`, roughly `[-args.drive_speed, +args.drive_speed]`) - a velocity command,
  not a position target, because the chassis is velocity-controlled, unlike every joint group.

In `--rollout`, forward/back is now policy-controlled (`command[0]` is overridden from
`policy_action_vec[21]` once a prediction exists) while strafe/rotate (`A`/`D`/`Q`/`E`) stay
manual-only - they were never in the recorded action space either way, so leaving them manual
doesn't introduce a new train/inference mismatch.

**This is a breaking schema change, not an additive one**: `state_dim`/`action_dim` are now 22
everywhere new data gets recorded, but every existing episode (`raw_episodes/`,
`raw_episodes_cart/`) and the already-trained checkpoint
(`act_training/table_to_table2/checkpoints/*`) are 21-dim. They cannot be mixed with new
recordings, and the existing checkpoint cannot be queried with a 22-dim state (the normalizer's
input layer is shape-locked to 21 - expect a loud shape-mismatch crash, not silent misbehavior, if
you try). New data must be collected from scratch and a new checkpoint trained before `--rollout`
works again.

**Two-policy plan, not one continuous pick-to-place trajectory**: per the user's target inference
architecture (SLAM between locations, one short fixed-base policy at each end), `pickup` and
`place` should be recorded as **separate, independently-labeled episodes**, not as one long session
like the original 21-dim data was. No new scene-setup code was needed for this - `--task` already
accepts any free-form string (e.g. `--task pickup_policy` for one session, `--task place_policy`
for another), and a "place" episode's "already holding the box" starting condition is achieved the
same way every episode's starting condition always has been: the operator manually jogs the robot
into the hug pose *before* pressing `B`, exactly like the existing Stage 0 workflow's "jog into
position, then start recording." Pickup episodes should end once the box is lifted and the robot
has backed away (not continue into a full carry-to-destination, which is what all existing episodes
did) - end each episode at the point where SLAM would take over, per the plan above.

## GR00T conversion (for training on a bigger GPU, e.g. the user's H200)

ACT is what trains and evaluates on this machine (12GB laptop GPU); NVIDIA's Isaac GR00T N1.7
(`/home/kholis/Isaac-GR00T-main`, a sibling repo, not part of this one) is a ~3B-parameter VLA that
needs 40GB+ VRAM even for its lightweight default fine-tune mode - out of reach locally, but well
within a single H200. This section is about *preparing the data* for that, done entirely on this
machine (no GPU needed for conversion) - actually fine-tuning GR00T still has to happen on the H200.

**GR00T needs LeRobot v2, not v3** - our datasets (`lerobot_dataset_pickup`/`lerobot_dataset_place`,
built by `convert_to_lerobot.py`) are v3.0. GR00T ships `scripts/lerobot_conversion/convert_v3_to_v2.py`
for this, but it needs a *specific pinned* `lerobot` git commit in its own isolated env (not our
`lerobot` conda env's pip-installed 0.4.x, which would conflict) - installed into a new `gr00t_convert`
conda env (python 3.10; the subproject requires `<3.12`, and system Python here is 3.12) via
`cd scripts/lerobot_conversion && pip install -e .` per that directory's own README. Also needed
`conda install -c conda-forge ffmpeg` in that env too - same missing-ffmpeg-binary issue as the
`lerobot` env earlier, except this one shells out to the `ffmpeg` binary directly via `subprocess.run`
rather than through a Python video-decode library, so the fix is the same but the failure mode looks
different (a `FileNotFoundError: ffmpeg`, not a decode crash).

**One conversion gotcha, confirmed live**: `convert_v3_to_v2.py --root <path> --repo-id <repo_id>`
does *not* use `<path>` as the dataset location directly - it resolves to `Path(root) / repo_id`
internally. Our `convert_to_lerobot.py` writes flat (`--root ./lerobot_dataset_pickup` *is* the
dataset root), so passing that straight through made the script look for a nonexistent
`lerobot_dataset_pickup/local/pickup_policy` and silently fall through to attempting a Hub download
(which then 401'd, since `local/pickup_policy` isn't a real Hub repo). Fixed by copying (not
symlinking - the script does in-place move/rename, risky with symlinks) both datasets into the
structure the flag actually expects: `gr00t_datasets/local/{pickup_policy,place_policy}/`, then
`--root ./gr00t_datasets`. Converted in place: the v3.0 original gets renamed to a `_v3.0` suffix
(kept, not deleted) and the new v2.1 version takes the original path.

**`meta/modality.json` (GR00T's one real schema addition over plain LeRobot v2)** was hand-authored
once (`gr00t_config/modality.json`, copied identically into both converted datasets - same robot,
same 22-dim layout, only the recorded behavior differs) since the converter doesn't generate one.
Maps directly onto our existing `state_names` breakdown: `left_arm` [0:7], `right_arm` [7:14],
`torso` [14:19], `left_gripper` [19:20], `right_gripper` [20:21], `chassis_forward` [21:22], video
`head_camera` -> `observation.images.head_camera`, and language sourced straight from the
`task_index` column `convert_to_lerobot.py` already writes (no extra annotation work needed).

**`gr00t_config/galbot_g1_config.py`** (modeled on their `examples/SO100/so100_config.py`) is the
Python modality config GR00T's fine-tuning script actually reads - registers under
`EmbodimentTag.NEW_EMBODIMENT` (checked: GR00T does have a pretrained `REAL_G1` tag, but that's
almost certainly Unitree's G1 humanoid, not our Galbot G1 - a naming coincidence, not a shortcut;
verify before ever assuming otherwise). Arm/torso/gripper actions are marked
`ActionRepresentation.ABSOLUTE` - not a stylistic choice like SO-100's `RELATIVE` pick for its arm,
but because that's literally what we recorded: the post-`clamp_to_actual` target position sent to
the joint controller each tick, not a delta. `chassis_forward` doesn't fit this taxonomy cleanly at
all - it's a recorded drive *velocity* command, not a position target in any representation - marked
`ABSOLUTE`/`NON_EEF` as the closest approximation; **unverified** whether GR00T's normalization/
diffusion head handles a velocity-typed channel labeled `ABSOLUTE` sensibly, worth specifically
checking that dimension in `open_loop_eval.py`'s per-dimension plots once fine-tuned.

**Verified structurally (parquet columns, video paths, task labels), not yet end-to-end** - loading
through GR00T's own dataset class needs the full `gr00t` package (`uv sync` in the repo root, heavy:
torch/diffusers/etc.), which wasn't installed here since actual fine-tuning happens on the H200
machine, not this one. Confirmed directly instead: both `data/chunk-000/episode_*.parquet` files
have 22-element `observation.state`/`action` arrays, `videos/chunk-000/observation.images.head_camera/
episode_*.mp4` files exist matching `modality.json`'s `original_key`, and `meta/tasks.jsonl` has the
right task string per dataset. Run GR00T's own loader on the H200 as the real first test before
trusting this further.

**To move to the H200**: copy `gr00t_datasets/local/{pickup_policy,place_policy}/` (327MB total,
`_v3.0` backups included) and `gr00t_config/galbot_g1_config.py` over, `uv sync --all-extras` in
the GR00T repo there, then per policy:
```
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path <path>/pickup_policy \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path <path>/galbot_g1_config.py \
    --num-gpus 1 --output-dir <out>/pickup_policy \
    --save-steps 2000 --max-steps 2000 --global-batch-size 32
```
(swap `pickup_policy` for `place_policy` for the other one - two separate fine-tunes, same as ACT).
