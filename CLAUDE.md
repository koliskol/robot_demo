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

**Four data paths to the browser, each shaped differently:**
- RGB and depth are separate WebRTC video tracks (`RGBTrack`, `DepthTrack`). Depth is
  false-colored (`_depth_to_rgb`: blue near → green mid → grey far, fixed 0–20m scale so a color
  always means the same distance across frames) — exact float depth is never sent anywhere.
- Point cloud goes out over a `"pointcloud"` WebRTC data channel as raw `float32` xyz bytes,
  paced at 5 Hz by the server (`_send_point_cloud`), decoded client-side straight into a
  `Float32Array` for a three.js `Points` cloud.
- World state (room outline, static object footprints, robot pose) goes out over a `"worldmap"`
  data channel as JSON at 10 Hz (`_send_world_state`), rendered as a top-down view on a plain
  `<canvas>` (simpler than three.js for flat rectangles/labels).

**Data channels must be created client-side.** Per WebRTC/JSEP, an answer can't introduce an SCTP
"application" section that wasn't in the offer, so the browser (`static/viewer.js`) calls
`createDataChannel` for both channels before generating its offer, and the server only ever
listens via `pc.on("datachannel")` — a server-side `createDataChannel()` call after receiving the
offer cannot negotiate (confirmed live: `readyState` stuck at `"connecting"` forever). Same
reasoning shapes the video side: both tracks land in one remote `MediaStream` on the client
(server never assigns them to distinct streams), so `viewer.js` wraps each `ontrack` event's own
track in a fresh `MediaStream` rather than using `event.streams[0]`, or both `<video>` elements
end up showing the RGB feed.

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
here too). It drops lidar/depth entirely (not needed for offline data collection) but does reuse
`streaming_server.py`'s WebRTC RGB track — same server `stream_demo.py` uses — purely so you can
watch the camera feed in a browser while teleoperating; this is unrelated to what actually gets
recorded (`EpisodeRecorder` samples the same camera independently, at `--record-fps`, regardless
of whether anyone's watching the live stream). It adds a pushcart (`build_pushcart`, ported from
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

**Recording is decoupled from LeRobot on purpose.** `collect_pickplace_demo.py` writes raw
per-episode data (`manifest.json` + `data.npz` + `frames/*.png`) with zero `lerobot` dependency,
because `lerobot` isn't installed anywhere on this machine and installing it into the `isaac_sim`
conda env risks conflicting with Isaac Sim's own pinned deps (opencv/av/gymnasium etc.).
`convert_to_lerobot.py` has zero Isaac Sim imports and is meant to run in a separate, disposable
`lerobot`-pip-installed env. Its exact `LeRobotDataset` API calls are unverified against a real
install (the module docstring says what to check) — don't assume they're correct without testing
against whatever lerobot version actually gets installed.
