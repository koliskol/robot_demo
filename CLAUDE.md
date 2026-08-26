# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A WebRTC live-streaming pipeline for an Isaac Sim robot demo: `stream_demo.py` drives a Galbot
G1 mobile robot around a small scene in Isaac Sim and pushes its chassis camera (RGB + depth)
and chassis lidar (point cloud) to a browser via `streaming_server.py`. The browser side lives
in `static/` (plain HTML/JS, three.js via CDN import map for the point cloud, canvas 2D for the
top-down map).

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
