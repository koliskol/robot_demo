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

**A third task variant is being developed: grabbing/releasing the pushcart handle itself** (as
opposed to picking a box up off it), motivated by wanting the robot able to move the cart, not just
place things on it. This surfaced a real, still-only-partially-solved instability: gripping the
handle and then rotating the chassis (`Q`/`E`) can visibly fling the arm apart, confirmed live to
be a genuine physics divergence, not a rendering glitch - the console showed real `[watchdog]`
joint-velocity spikes (`left_arm_joint5` recorded at -34.7 rad/s in one case) followed by an
`Invalid PhysX transform` error across the *entire* robot body, meaning the physics solver's
position/rotation math went to NaN, not just "a big but valid" value. Straight pushing (`W`) with
the same grip is fine - only rotation triggers it.

**Root cause, confirmed through live testing rather than assumed**: rotating the whole chassis
while rigidly gripping a fixed external point forces the wrist to sweep through more range than it
physically has to keep the hand roughly stationary - `left_arm_joint6` only has about `-42/+47deg`
of travel (confirmed from the live asset's own authored PhysX joint limits, not the offline
reference URDF, via a `dump_arm_joint_limits()` diagnostic added specifically to check this - the
live asset's limits matched the reference URDF exactly once converted from degrees to radians, so
"missing/unenforced limits" was ruled out, not confirmed). Once the required wrist travel exceeds
what's available, the joint gets wrenched hard enough to overpower even the arm's very strong
native PhysX drive (600000 stiffness / 60000 damping, confirmed live - an earlier attempt to fix
this by further *raising* wrist joint stiffness was based on a wrong assumption that these joints
were weakly driven, like the head joints were; it wasn't, and that change made a later test
noticeably worse, consistent with over-stiffening feeding a growing oscillation rather than
damping one - reverted, function kept but disabled). Every mitigation tried so far reduces
frequency but does not guarantee prevention, since this is a genuine kinematic mismatch between
the task and the wrist's range, not a tuning error:

- **`JOINT_EFFORT_LIMIT_NM`** (15.0, a reasoned guess bracketing observed clean-vs-danger effort
  values, not measured) drives a new compliance mechanism: once a joint's measured effort (read
  via `get_measured_joint_efforts()`, same API the watchdog already used) exceeds this, further
  motion *into* the load is blocked for that joint - forward-only for arm swing (`joint2`, retreat
  via `O` always still works), both directions for elbow/wrist-rotate (no assumed safe direction).
  Confirmed live to let several smaller overload events recover cleanly that would previously have
  needed to; did not prevent the one large single-spike escalation in the same test.
- **`--turn-speed` default lowered 0.15 -> 0.08** - slower chassis rotation gives the wrist more
  time to track the changing geometry per unit of motion.
- **Gripper stability fixes, unrelated to the wrist issue above but found along the way**: fixed a
  real bug where releasing `M`/`N` didn't stop the gripper from continuing to close - the software
  target (`gripper_rad`) accumulated at `GRIPPER_SPEED_RAD_S` (2.5 rad/s) while held, far faster
  than the lead-clamped joint could physically follow (~0.5 rad/s), so it could race far ahead of
  the real position; release now re-syncs it to the actual position every step instead of letting
  it drift. Also found `GRIPPER_MAX_LEAD_RAD` at a temporarily-raised 0.02 (sped up on request)
  reproduced finger-juddering (repeated exact `+-0.5 rad/s` readings, a stick-slip pattern against
  handle contact) that fed into the same class of arm fling - reverted to the original 0.008.
  Separately, `stiffen_gripper_friction()` (added to help the pinch hold better without needing
  more closing force, which is unsafe) had silently never worked - the keyword match found zero
  collision prims (confirmed live). A new `dump_gripper_hierarchy()` diagnostic found the gripper's
  actual internal link names (`gripper_{l,r}_finger_link`, `gripper_{l,r}_knuckle_link`,
  `gripper_{l,r}_inner_knuckle_link`, each with their own `visuals`/`collisions` children) by
  reading the drive joint's own `body0`/`body1` targets rather than guessing - `HasAPI(UsdPhysics.
  CollisionAPI)` reads `False` on literally every prim in the hand assembly for reasons not
  understood, so the fix targets these real names directly instead of filtering on that check.

**A promising alternative grip approach was found live, not yet implemented as a policy**: rotating
both wrists (see the wrist-rotate controls below) can orient the hands so the handle is squeezed
between both open palms from either side, instead of pinched by the fingers closing - the same
bimanual-compression principle as the box hug, applied to the handle. This avoids the gripper's
closing mechanism (the actual fragile part in every test above) entirely. Plan is to record this as
two more independent policies (`grab_cart_policy`/`release_cart_policy`), same pattern as
`pickup_policy`/`place_policy` - not yet attempted live for whether it's actually more stable under
chassis rotation, which is the real test given the root cause above.

**Manual wrist-rotate controls** (`C`/`V`, `,`/`.`, `[`/`]`) jog `joint5`/`joint6`/`joint7`
independently - added specifically to fix the gripper's resting angle (it sat diagonal, not flat,
at the asset's authored zero position) and now also used to explore the open-palm grip above. Each
joint's mapping to a local axis of the gripper end link (`left`/`right_arm_link7`) was derived by
composing the URDF's origin rotations, not guessed - `joint7`'s own `<axis>` **is** link7's local Z
by definition (its child prim is link7 itself); `joint6`'s and `joint5`'s axes were derived by
composing joint7's and joint6's fixed mounting rotations respectively, landing on link7's local Y
and negative X. `C`/`V` (X/joint5) is confirmed live as the correct rotation to fix the diagonal
resting angle, calibrated to `-0.7400 rad` (`-42.4deg`) against a user-supplied reference photo and
set as the default (`STARTING_WRIST_X_RAD`) so no key press is needed for that fix specifically;
`Y`/`Z` have no live-calibrated default yet (both `0`). Two earlier guesses on `joint7` alone
(`+45deg` then `-45deg`, before this per-axis, per-key setup existed) were tried and confirmed
wrong first - worth remembering before assuming a similar rotation fix will work on the first try
elsewhere in this file.

**Pushcart size/mass/friction were re-tuned multiple times this session, ending in a deliberately
split design**: originally sized up together (chassis mass `4.4kg -> 25kg`, meant to resist an
accidental bump) but that conflated two different things - mass also determines how much reaction
force a deliberate push/grip transmits back into the wrist, and 25kg turned out to be enough to
contribute to the fling issue above (confirmed via the watchdog before the chassis-rotation root
cause was isolated). Split apart instead: chassis mass brought back down to `6kg` (close to
original, keeps the wrist's reaction-force budget small), while `CASTER_ROLLING_FRICTION_NM` was
raised steeply instead (`0.05 -> 2.0`) to resist a light accidental bump via wheel friction,
independent of mass - a real sustained push should still overcome it. Also this session: the
handle was moved to the deck's opposite side (`PUSHCART_HANDLE_SIDE_SIGN`, a sign flip exploiting
the caster/deck layout's symmetry rather than an authored rotation), the deck/wheels sized up
(`(0.45, 0.225)/0.05 wheel radius -> (0.55, 0.30)/0.08`), and the wheels changed from flat
`Cylinder` prims to `Sphere` prims (a cylinder only looks right spinning about its own axis; the
casters' swivel+spin joints were already unlimited/360deg-free, the cylinder shape was just what
looked wrong from an arbitrary fork heading). None of this is live-verified beyond what's noted
above - re-run the Stage 0 reach/hug cycle against the new geometry before trusting it fully.

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

**A `pickup_policy` checkpoint came back from the H200 (`gr00t_model_trained/`, safetensors + a
`gr00t_model.zip` of config/processor files) and local GR00T inference has now been confirmed to
actually work on this machine's RTX 4080 Laptop (12GB)** - not just theorized about, unlike
everything else in this section up to now. Checkpoint's `config.json` declares
`"architectures": ["Gr00tN1d6"]` (GR00T N1.6, not N1.7) and `dataset_statistics.json` confirms the
22-dim `chassis_forward`-inclusive schema, so this is current-format data, not a stale 21-dim run.

**Getting a loadable environment took real, non-obvious work - each step here was a genuine
blocker, not a formality**:
- `/home/kholis/Isaac-GR00T-main` referenced elsewhere in this doc didn't exist on this machine at
  all (training only ever ran on the H200) - re-cloned from `github.com/NVIDIA/Isaac-GR00T`.
- **The repo's `main` branch has moved on to N1.7-only and can't load this checkpoint** -
  `gr00t/model/` only contains `gr00t_n1d7/`, and `MODEL_REGISTRY` has nothing registered for
  `"Gr00tN1d6"`, confirmed by reading `gr00t/model/registry.py` directly rather than guessing from
  an import error. Fixed by checking out the `n1.6-release` git tag (found via `git fetch
  --unshallow`, which also revealed `n1.5-release`/`n1.7-release` tags) - that tag's `gr00t/model/`
  still has `gr00t_n1d6/`, matching the checkpoint exactly. Any future GR00T checkpoint made against
  a different model version needs its own matching tag, not whatever `main` happens to be.
- New `gr00t_infer` conda env, Python 3.10 (matches the `n1.6-release` `pyproject.toml`'s
  `>=3.10,<3.13` and its prebuilt `flash-attn` wheel's `cp310` tag) - kept separate from
  `isaac_sim`/`lerobot` for the same dependency-conflict reasoning already established for those
  two, just with a heavier stack (a ~2B-param Eagle VLM backbone, flash-attn).
- Installed `torch==2.7.1`/`torchvision==0.22.1` from the `cu128` PyTorch index and the rest of
  that tag's pinned runtime deps via plain `pip`, then `pip install -e . --no-deps` for the `gr00t`
  package itself - skipping the full `pyproject.toml` dependency list on purpose
  (`deepspeed`/`tensorrt`/`onnx` are training/export-only weight, not needed to just call
  `get_action()`, and risk being slow or failing to build in a plain pip install with no matching
  system CUDA toolkit).
- **`flash_attn` turned out to be a hard requirement despite looking optional** - the Eagle
  backbone's own `modeling_siglip2.py` only imports it behind `is_flash_attn_2_available()`, but
  loading the checkpoint still raised `ImportError: FlashAttention2 has been toggled on` from deep
  inside `transformers`' own `_autoset_attn_implementation` - confirmed live, not assumed from
  reading the code alone. Fixed by installing the exact prebuilt wheel `n1.6-release`'s
  `pyproject.toml` names for `cp310`/`torch2.7`/`cu12`/`x86_64`
  (`flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` from the
  `Dao-AILab/flash-attention` GitHub releases) rather than `pip install flash-attn`, which compiles
  from source and can take a very long time (or fail) without a matching CUDA toolkit installed.
- **The checkpoint directory `gr00t_model.zip` unpacks to isn't directly loadable** -
  `Gr00tN1d6Processor.from_pretrained` (confirmed by reading it directly) looks for
  `processor_config.json`/`statistics.json`/`embodiment_id.json` at the top level of the model
  directory, but the zip nests them under a `processor/` subfolder. Fixed by assembling
  `gr00t_model_trained/pickup_policy_ckpt/` with those three files copied up to the top level
  alongside `config.json`/`model.safetensors.index.json` (from the zip) and the two
  `model-*-of-00002.safetensors` shards symlinked in from the parent directory (avoids duplicating
  9.8GB) - `AutoModel.from_pretrained`/`AutoProcessor.from_pretrained` both load cleanly from that
  assembled directory. Any other checkpoint pulled from the H200 the same way will need the same
  flattening.

**Confirmed live, with real numbers, not estimated**: loading the checkpoint onto `cuda:0` in
bf16 and running one `get_action()` call used **6.6-6.9GB of the 12GB card** - comfortably inside
budget, and notably less than the 9.81GB raw weight size implied by `model.safetensors.index.json`
(the fp32-upcast top-4 LLM layers `tune_top_llm_layers` keeps trainable don't change this enough to
matter here since this is inference, not training). This is a materially better result than the
"barely fits, unverified" concern this doc raised before an actual `gr00t_infer` env existed to
test it. Per-call latency, measured over several calls after the first (warmup) one: **~85-95ms**,
versus the ACT checkpoint's confirmed ~8-9ms and the 67ms/15Hz budget `--rollout`'s control loop
targets - a real, confirmed gap (roughly 11Hz achievable, not 15Hz), though not necessarily fatal
since the rollout loop already re-applies the last received target between predicts rather than
blocking on one every physics step (see the ACT `--rollout` section above) - untested whether
tracking quality actually holds up at that effective rate, only that it runs.

**`gr00t_policy_server.py`** is a drop-in alternative to `policy_server.py` for
`collect_pickplace_demo.py --rollout` - it speaks the identical `policy_wire.py` protocol
(`PolicyClient.reset()`/`predict()` don't know or care which model answers), so no changes to
`collect_pickplace_demo.py` were needed; point `--policy-host`/`--policy-port` at it instead of an
ACT server. It builds the `video`/`state`/`language` observation dict `Gr00tPolicy.get_action()`
expects (confirmed against the checkpoint's own `processor_config.json`, which stores the exact
per-key modality breakdown from training) and returns only the predicted chunk's first timestep
(`action_horizon=16` per call, unlike ACT's own internal one-step dequeue) - it does not attempt to
consume or cache the remaining 15 steps. **Verified live end-to-end**: `policy_client.py`, imported
from the `isaac_sim` env exactly as `collect_pickplace_demo.py` would, connected, reset, and ran
several `predict()` calls against a running `gr00t_policy_server.py`, returning finite `(22,)`
`float32` vectors at the latency above - with a synthetic random image and zeroed state, not a real
Isaac Sim scene. **Not yet attempted**: an actual closed-loop rollout inside Isaac Sim itself (the
`--rollout` flag pointed at this server) - that's the next real test, same caveat this doc already
applies to the ACT rollout path.

## SmolVLA - fine-tuning (not just inference) confirmed live on this machine

Separately from GR00T, `lerobot[smolvla]==0.4.4` (installed into the existing `lerobot` conda env
via `pip install "lerobot[smolvla]==0.4.4"`, which pulls in `transformers` and the other smolvla
extras that plain `lerobot` doesn't include) was used to actually run `lerobot-train
--policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base` against
`lerobot_dataset_pickup` (the same v3.0 dataset `convert_to_lerobot.py` already produces for ACT) -
a real fine-tuning run, not inference. Two CLI gotchas hit before it ran: the output dir must not
already exist unless `--resume=true`, and `--policy.repo_id`/`--policy.push_to_hub=false` are
required even for a fully local run (the validator errors without them, despite nothing actually
being pushed to the Hub).

**Confirmed live: loss actually decreases with real gradient updates** (10-step run: 2.29 -> 0.83,
noisy but trending down, batch size 4) - this is a genuine fine-tuning step, not just a forward
pass. `num_learnable_params=99.88M` out of `num_total_params=450M` printed at start, confirming
SmolVLA's own defaults (`freeze_vision_encoder=True`, `train_expert_only=True`) are what's actually
running - only the ~100M-param action expert trains, the SmolVLM2-500M-Video-Instruct backbone
stays frozen.

**Peak VRAM measured at batch_size=8 over a real 300-step run (1-second `nvidia-smi` polling for
the full ~70s duration, not a single snapshot): ~3.0GB** - dramatically better than the
~10-16GB LeRobot's own `hardware_guide.mdx` states for this exact tier/batch size. Not fully
reconciled why the gap is this large (possibly that guide's number assumes something this
default-settings run doesn't - a different effective batch via gradient accumulation, an unfrozen
backbone, or eval/logging overhead) - flagging the discrepancy rather than picking one explanation,
since both numbers came from a stated source (the doc) vs. a real local measurement (this run), and
only the second was actually observed on this machine. Either way, **this is comfortably inside the
12GB budget with a lot of room to spare** - unlike GR00T, where two 3B+-class checkpoints don't fit
together, batch_size=8 SmolVLA training leaves enough headroom that raising the batch size further,
or eventually running training alongside something else, is plausible (untested how far).

The one-time `lerobot/smolvla_base` pretrained checkpoint download (907MB, cached under
`~/.cache/huggingface/hub/models--lerobot--smolvla_base`) only needs to happen once - reused across
runs. Test checkpoints from these verification runs were written to a scratch directory and deleted
afterward, not kept - a real fine-tuning session should pick a real `--output_dir` under this repo
(gitignored, same as `act_training/`) and a realistic `--steps` count, not the 10/300-step smoke
tests used here to confirm the mechanism works.

**A real 20,000-step fine-tune (matching the GR00T H200 run's step count, for comparability) has
now been run end-to-end**: `smolvla_training/pickup_policy/`, same `lerobot_dataset_pickup` data,
batch_size=8, checkpoints every 4000 steps. Took **77 minutes wall-clock** on this GPU (~4.3
steps/s sustained, matching the earlier smoke test's throughput) - VRAM stayed flat at the same
~3GB the shorter run showed, no growth over the full run. Loss trend across the run: `step 200:
1.156 -> step 5K: 0.076 -> step 10K: 0.046 -> step 15K: 0.033 -> step 20K: 0.038` - converges hard
by ~5-10K steps (roughly epoch 2-4 over this 71-episode dataset) and mostly plateaus/noises around
0.03-0.04 after, suggesting 20K steps is already past the point of obviously-still-improving returns
for a dataset this size, not that more steps are clearly needed. Final checkpoint:
`smolvla_training/pickup_policy/checkpoints/020000/pretrained_model` (also symlinked as
`checkpoints/last`). Five checkpoints (4K/8K/12K/16K/20K) total ~7.5GB on disk - **not yet trimmed
down to just the final one**, worth deleting the intermediate four if disk space matters (this
machine was at ~91% full/23GB free right after this run). **Not yet done**: actually evaluating this
checkpoint's real pick/place behavior - either open-loop against held-out frames
(`evaluate_act_checkpoint.py`'s pattern, not yet ported to SmolVLA) or closed-loop via a
`smolvla`-serving equivalent of `gr00t_policy_server.py`/`policy_server.py` through
`collect_pickplace_demo.py --rollout`. A low final loss here says the model fits its own training
distribution, not that it can actually control the robot - same caveat this doc already applies to
every other checkpoint before its first real rollout.

**Closed-loop rollout now attempted live, and it actually works.** No new server was needed -
`policy_server.py`/`lerobot_policy_utils.py` (originally built for ACT) turned out to already be
policy-agnostic: `PreTrainedConfig.from_pretrained()` reads the policy type straight from the
checkpoint's own `config.json` and `make_policy()`/`make_pre_post_processors()` dispatch on it, so
pointing `policy_server.py --checkpoint-dir` at the SmolVLA checkpoint just worked, unmodified.
Confirmed live: ~1.5GB VRAM for inference (lighter than training's ~3GB, no gradients/optimizer
state) and ~10-13ms per `predict()` call after warmup - dramatically faster than GR00T's ~85-95ms,
comfortably inside the 67ms/15Hz `--rollout` budget with room to spare.

**User-confirmed result from the first live watched session**: pressing `M` in the Isaac Sim
window activates the SmolVLA `pickup_policy` checkpoint, and **it can actually grab both the
medium and big boxes** - the central open risk this whole pipeline has carried since
`capture_cube_rgbd.py` (actually lifting a loose object, not just pinching a fixed obstacle) is
resolved for this checkpoint, at least for the cube-scale-cycle boxes tested. Not yet confirmed:
whether it holds up across the box-jitter range, whether it succeeds reliably vs. sometimes, or
whether the small box also works - only stated as "it works" from watching it grab two of the
three sizes.

**Follow-up user-run success-rate check, 10 attempts per box size (each presumably a fresh
`R`-reset drawing a new box-jitter sample, so this does also partially answer the jitter-range
question above): medium 9/10, big 8/10, small 6/10.** Confirms the checkpoint is reliable, not
just "worked once," on medium/big, and that it does generalize across at least the default
jitter range rather than only the one exact pose it was watched succeed on earlier. The small box
is the clear weak point - notably worse than the other two rather than uniformly good, worth
keeping in mind before trusting `pickup_policy` on a small object in any downstream use (e.g. a
future closed-loop eval script or a place_policy handoff). Root cause not yet investigated - could
be training-data imbalance (if small-box episodes were under-represented in the 71-episode
dataset), the smaller visual/contact margin making the hug's compression window narrower, or
something else; not distinguished yet.

**Follow-up (2026-09-10): the training-data-imbalance hypothesis above was acted on, not just
noted.** User-counted split of the original 71-episode `pickup_policy` set was roughly big=31/
medium=25/small=20 - real imbalance, and it lines up with small being the weak point above.
Collected 30 more episodes directly targeting the underrepresented sizes (16 small, 14 medium,
fixed `--cube-scale` instead of `--cube-scale-cycle` so every new episode is the intended size, not
a 1-in-3 chance of it) - `raw_pickup/` is now 104 episodes (101 success/3 fail), roughly
big=31/medium=39/small=36, so small and medium are no longer trailing big. One real mistake caught
before converting: all 30 new episodes were recorded without `--task pickup_policy`, so they landed
with the auto-derived default task string (`pick_box_table_to_cart`) instead - would have split the
dataset's language-conditioning across two different task labels for what should be one task. Fixed
by patching all 30 manifests' `task` field directly (metadata-only, no frames/actions touched) before
reconverting. `lerobot_dataset_pickup` has been rebuilt from this corrected 104-episode set.

**A retrain on this rebalanced dataset (`pickup_policy_v2`, same 20k-step/batch-8 SmolVLA recipe,
output `smolvla_training/pickup_policy_v2/`) was started 2026-09-10 and deliberately paused
partway through for the day, not run to completion** - stopped cleanly (SIGTERM to the main
`lerobot-train` process, not a kill -9) at step ~8000-9000/20000, loss already down from 1.14 to
~0.12, tracking the same fast-converge curve the original `pickup_policy`/`place_policy` runs
showed. A checkpoint exists at `smolvla_training/pickup_policy_v2/checkpoints/008000` (`last`
symlink points to it), so this resumes rather than restarts:

    conda run -n lerobot lerobot-train \
        --config_path=./smolvla_training/pickup_policy_v2/checkpoints/last/train_config.json \
        --resume=true

The original `pickup_policy` checkpoint (`smolvla_training/pickup_policy/checkpoints/020000`,
trained on the old 74-episode/imbalanced data) is deliberately left in place, not overwritten, so
once `pickup_policy_v2` finishes it can be compared against - specifically, re-run the same
10-attempts-per-box-size check from above and see whether small's 6/10 actually improved, not just
whether loss went down.

**`place_policy` (SmolVLA) has now been fine-tuned the same way, on the already-recorded/already-
converted `raw_place`/`lerobot_dataset_place` data (50/50 success-labeled episodes, 11,775 frames,
22-dim schema - this data predates this session and just hadn't been trained on yet).** Same
command shape as `pickup_policy`'s run (`--policy.type=smolvla
--policy.pretrained_path=lerobot/smolvla_base`, `batch_size=8`, `steps=20000`,
`save_freq=4000`), output to `smolvla_training/place_policy/`. Took **~82 minutes wall-clock**
(15:43-17:05), ~4.0-4.1 steps/s sustained, matching `pickup_policy`'s throughput on this GPU.
Loss trend across the run (single-batch instantaneous value at each checkpoint, not an averaged
epoch loss - noisy by nature): `step 4K: 0.107 -> step 8K: 0.066 -> step 12K: 0.038 -> step 16K:
0.031 -> step 20K: 0.040`. Note this is not monotonic at the tail (16K's single-sample reading is
lower than 20K's) - consistent with the same "converges hard early, then plateaus/noises in the
0.03-0.04 band" pattern `pickup_policy` showed, not evidence that 16K is actually the better
checkpoint. Kept the **final step-20000 checkpoint** as the one to carry forward (`checkpoints/020000`,
symlinked as `checkpoints/last`) rather than picking whichever single noisy reading happened to be
lowest - it's the only checkpoint saved after the cosine learning-rate schedule had annealed
nearly to zero (`lr:2.5e-06` at 20K vs `9.1e-05` at 4K), which is the principled reason to prefer
it over an earlier one absent a real held-out-eval signal to rank them by. Trimmed the four
intermediate checkpoints (4K/8K/12K/16K, ~6GB) immediately after, same as `pickup_policy` was
eventually trimmed to just its final checkpoint - disk was at 92%/21GB free right after the run
finished, tight enough that keeping every checkpoint from a future training run isn't viable
without trimming as you go.

**Closed-loop rollout now attempted live for `place_policy`, and it mostly succeeds** - the first
real evaluation this checkpoint has had of any kind (no open-loop check was done either). Run via
`run_policy_inference.py` (not `collect_pickplace_demo.py --rollout`) against `policy_server.py
--checkpoint-dir smolvla_training/place_policy/checkpoints/020000/pretrained_model` on port 8766
alongside `pickup_policy`'s server on 8765, starting from a manually-jogged "already holding the
box" pose per how `place_policy` episodes were always recorded. User-observed result: **place
succeeded on most attempts** - not yet a formal counted success rate like `pickup_policy`'s 10-per-
box-size check above, so treat "mostly" as a qualitative first read, not a number to cite. A
proper per-box-size or per-jitter-range count is the natural next step if this checkpoint is going
to be relied on, same as was done for `pickup_policy`.

`run_policy_inference.py` (not `collect_pickplace_demo.py --rollout`) was the right tool for this
test, for a reason worth remembering: it was forked from `collect_pickplace_demo.py` on 2026-09-04,
*before* the wrist-flattening calibration (`STARTING_WRIST_X_RAD`, added 2026-09-08) existed - its
`idle_arm_pose()` takes no wrist parameter at all, so it starts from wrist joint5=0.0, which is
exactly the pose distribution `pickup_policy`/`place_policy` were both recorded under (see that
flag's own help text). Running these two checkpoints through `collect_pickplace_demo.py --rollout`
instead requires remembering to pass `--starting-wrist-x-rad 0.0` explicitly, or the gripper starts
from the wrong pose relative to training data - confirmed live as a real failure mode (reported as
"the gripper position is wrong again, not similar with the dataset") before switching to
`run_policy_inference.py` sidestepped it entirely. `run_policy_inference.py` is otherwise stale
relative to `collect_pickplace_demo.py` (no `JOINT_EFFORT_LIMIT_NM` wrist compliance, e.g.) - fine
for pick/place, since that mechanism was added for the unrelated pushcart-grip instability.

**A related `collect_pickplace_demo.py --rollout`-only bug was found and fixed along the way**:
pressing `M` then `M` again (stop) left the recorder in `AWAITING_LABEL`; pressing `N` from there
silently set `active_policy_client` to place without restarting recording (the auto-start check
only fired from `IDLE`), since `predict()`/`recorder.append()` are both gated on `RECORDING` - so
the key press appeared to do nothing, with a misleading "active policy -> PLACE" print alongside.
Fixed (`0d56839`) by having `M`/`N` auto-discard the pending unlabeled attempt when pressed from
`AWAITING_LABEL`, instead of requiring an explicit `Y`/`F`/`Backspace` first - `Y`/`F`/`Backspace`
still work exactly as before when you do want to label explicitly. Doesn't affect
`run_policy_inference.py`, which has no labeling state machine to get stuck in.

**Real usability gap found and fixed**: none of these policies (SmolVLA, ACT, GR00T) predict any
kind of "done"/termination signal - they're pure behavior-cloning, trained only on "given this
observation, what's the next action," with episode boundaries decided entirely by whoever was
pressing keys during data collection. So a running `--rollout` attempt predicts forever with no
natural stopping point; the *only* way it stops is a human pressing `B`/`Y`/`F`/`Backspace`, which
is easy to not realize since `M`/`N` are the buttons actually used to start it. Fixed by making
`M`/`N` toggle: pressing the same key again while that policy is the one currently running now
stops the attempt (the same transition `B` triggers) so `Y`/`F`/`Backspace` can label it; pressing
the *other* key while one is running still just switches the active policy without stopping,
unchanged. This applies to all three policy types equally, since the toggle lives in
`collect_pickplace_demo.py`'s key handler, not in anything policy-specific.
