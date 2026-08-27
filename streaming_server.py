"""WebRTC media server for streaming RGB, depth, and lidar point cloud data out of a running
Isaac Sim process to a browser, via aiortc + aiohttp.

Runs its own asyncio event loop in a background thread (see run_in_background) so it can sit
alongside a synchronous Isaac Sim simulation loop, which owns the main thread (Kit/PhysX
require that). The two sides only share a FrameStore instance: the sim loop calls
FrameStore.update_*() every physics step, and this module's video tracks / point-cloud
sender read whatever's currently there each time they're asked - the same way a live camera
doesn't queue frames for a slow client. If the browser or network falls behind, it just sees
fewer, more current frames, not a growing backlog.

Standalone-testable: nothing in this module imports isaacsim. Run it directly
(`python streaming_server.py`) to stream a synthetic moving pattern instead of real sim data,
to check the server/viewer independent of Isaac Sim.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

STATIC_DIR = Path(__file__).parent / "static"


class FrameStore:
    """Thread-safe latest-frame holder - see module docstring."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rgb: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._points: np.ndarray | None = None
        self._world_state: dict | None = None
        self._status: dict | None = None
        self._commands: list[dict] = []

    def update_rgb(self, rgb: np.ndarray) -> None:
        with self._lock:
            self._rgb = rgb

    def update_depth(self, depth: np.ndarray) -> None:
        with self._lock:
            self._depth = depth

    def update_points(self, points: np.ndarray) -> None:
        with self._lock:
            self._points = points

    def update_world_state(self, world_state: dict) -> None:
        """Room outline, static object footprints, and robot pose (all world xy, meters, plus
        a heading in radians for the robot) - see stream_demo.py's ROOM_MIN/MAX and
        build_scene for what actually populates this. Sent to the browser as JSON over the
        "worldmap" data channel for the 2D top-down map view (see static/viewer.js).
        """
        with self._lock:
            self._world_state = world_state

    def get_rgb(self) -> np.ndarray | None:
        with self._lock:
            return self._rgb

    def get_depth(self) -> np.ndarray | None:
        with self._lock:
            return self._depth

    def get_points(self) -> np.ndarray | None:
        with self._lock:
            return self._points

    def get_world_state(self) -> dict | None:
        with self._lock:
            return self._world_state

    def update_status(self, status: dict) -> None:
        """Arbitrary small JSON-able status dict (e.g. a recorder's state/episode/frame-count in
        collect_pickplace_demo.py) sent to the browser as JSON over the "status" data channel,
        for a UI element like a status badge. Generic on purpose - this module has no concept of
        "recording"; the consuming script decides what goes in the dict, same as
        update_world_state's room/objects/robot shape is stream_demo.py's own concept, not this
        module's.
        """
        with self._lock:
            self._status = status

    def get_status(self) -> dict | None:
        with self._lock:
            return self._status

    def push_command(self, command: dict) -> None:
        """Queue a command received from the browser's "control" data channel (see build_app's
        on_datachannel) - e.g. a button click - for the sim loop to pick up via pop_commands().
        Generic on purpose, same rationale as update_status: this module doesn't interpret
        command contents, just relays them.
        """
        with self._lock:
            self._commands.append(command)

    def pop_commands(self) -> list[dict]:
        """Drain and return all commands queued since the last call - call this once per sim loop
        iteration (see collect_pickplace_demo.py) rather than polling get_status-style, so a
        command sent between two slow frames isn't silently dropped or double-counted.
        """
        with self._lock:
            commands, self._commands = self._commands, []
            return commands


DEPTH_MIN_M = 0.0
DEPTH_MAX_M = 20.0


_DEPTH_NEAR_COLOR = np.array([0, 60, 255], dtype=np.float32)  # blue
_DEPTH_MID_COLOR = np.array([0, 200, 0], dtype=np.float32)  # green
_DEPTH_FAR_COLOR = np.array([160, 160, 160], dtype=np.float32)  # grey


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Map a float depth buffer (meters, possibly with inf for no-hit pixels) to a
    blue (near) -> green (mid) -> grey (far) colormap for video encoding: two linear RGB
    lerps, blue->green over the first half of the range and green->grey over the second half.
    Fixed scale (DEPTH_MIN_M to DEPTH_MAX_M) rather than per-frame min/max, so a color always
    means the same distance across frames - 0m is pure blue, DEPTH_MAX_M+ (and inf/no-hit
    pixels) clip to pure grey. Lossy/for-viewing only - the exact float values are never sent
    over the depth video track, only this preview; nothing in this project currently streams
    raw depth floats (see _send_point_cloud for the one place exact numeric data - xyz points
    - goes out unmodified).
    """
    # 0 = nearest (0m), 1 = farthest (DEPTH_MAX_M+) - inf (no-hit) pixels clip to 1.0, i.e. grey.
    t = np.clip(depth, DEPTH_MIN_M, DEPTH_MAX_M) / DEPTH_MAX_M
    t = np.nan_to_num(t, nan=1.0, posinf=1.0)

    first_half = np.clip(t / 0.5, 0.0, 1.0)[..., None]
    second_half = np.clip((t - 0.5) / 0.5, 0.0, 1.0)[..., None]
    color = _DEPTH_NEAR_COLOR + first_half * (_DEPTH_MID_COLOR - _DEPTH_NEAR_COLOR)
    color = np.where(t[..., None] > 0.5, _DEPTH_MID_COLOR + second_half * (_DEPTH_FAR_COLOR - _DEPTH_MID_COLOR), color)

    return np.clip(color, 0, 255).astype(np.uint8)


class RGBTrack(VideoStreamTrack):
    def __init__(self, frame_store: FrameStore) -> None:
        super().__init__()
        self._frame_store = frame_store

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        rgba = self._frame_store.get_rgb()
        image = np.zeros((480, 640, 3), dtype=np.uint8) if rgba is None else np.ascontiguousarray(rgba[:, :, :3])
        frame = VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


class DepthTrack(VideoStreamTrack):
    def __init__(self, frame_store: FrameStore) -> None:
        super().__init__()
        self._frame_store = frame_store

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        depth = self._frame_store.get_depth()
        image = np.zeros((480, 640, 3), dtype=np.uint8) if depth is None else _depth_to_rgb(depth)
        frame = VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


async def _send_point_cloud(channel, frame_store: FrameStore, hz: float = 5.0) -> None:
    """Point clouds go out over a data channel, not a video track, since they're structured
    numeric data (xyz per point) rather than an image - a browser-side renderer (see
    static/viewer.js) turns each binary message straight back into a Float32Array. Paced much
    slower than video (5Hz default): lidar scans don't need to update at video framerate, and
    unlike the video tracks (paced automatically by VideoStreamTrack.next_timestamp), a data
    channel sender has to pace itself.
    """
    interval = 1.0 / hz
    # The datachannel event (see build_app) can fire slightly before the channel finishes
    # opening - wait it out rather than exiting immediately on a readyState check.
    while channel.readyState == "connecting":
        await asyncio.sleep(0.05)
    while channel.readyState == "open":
        points = frame_store.get_points()
        if points is not None and len(points):
            channel.send(np.ascontiguousarray(points, dtype=np.float32).tobytes())
        await asyncio.sleep(interval)


async def _send_world_state(channel, frame_store: FrameStore, hz: float = 10.0) -> None:
    """Room outline / object footprints / robot pose, as JSON text (not binary like the point
    cloud - this is small, human-readable, and heterogeneous enough - a room, a list of
    objects, a robot pose - that a fixed binary layout isn't worth it). See
    FrameStore.update_world_state and static/viewer.js's map renderer.
    """
    interval = 1.0 / hz
    while channel.readyState == "connecting":
        await asyncio.sleep(0.05)
    while channel.readyState == "open":
        world_state = frame_store.get_world_state()
        if world_state is not None:
            channel.send(json.dumps(world_state))
        await asyncio.sleep(interval)


async def _send_status(channel, frame_store: FrameStore, hz: float = 5.0) -> None:
    """Generic small JSON status blob (see FrameStore.update_status) - low rate, since a status
    badge doesn't need to update at anywhere near video framerate. Only ever sends something if
    the consuming script (e.g. collect_pickplace_demo.py) actually calls update_status(); a
    script that never does (e.g. stream_demo.py) leaves this channel silent, which is what keeps
    static/index.html's status UI hidden for it (see viewer.js).
    """
    interval = 1.0 / hz
    while channel.readyState == "connecting":
        await asyncio.sleep(0.05)
    while channel.readyState == "open":
        status = frame_store.get_status()
        if status is not None:
            channel.send(json.dumps(status))
        await asyncio.sleep(interval)


def build_app(frame_store: FrameStore) -> web.Application:
    pcs: set[RTCPeerConnection] = set()

    async def index(request: web.Request) -> web.Response:
        return web.FileResponse(STATIC_DIR / "index.html")

    async def viewer_js(request: web.Request) -> web.Response:
        return web.FileResponse(STATIC_DIR / "viewer.js")

    async def offer(request: web.Request) -> web.Response:
        params = await request.json()
        rtc_offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_state_change() -> None:
            if pc.connectionState in ("failed", "closed"):
                pcs.discard(pc)
                await pc.close()

        pc.addTrack(RGBTrack(frame_store))
        pc.addTrack(DepthTrack(frame_store))

        # The client creates this channel (before generating its offer - see static/viewer.js)
        # and we just listen for it, rather than the server calling createDataChannel() here.
        # That's not a style choice: per WebRTC/JSEP, an answer can't introduce a new m-section
        # (the SCTP "application" section a data channel needs) that wasn't already in the
        # offer - a server-side createDataChannel() call made only after receiving the offer
        # can never actually negotiate, confirmed live (readyState stuck at "connecting"
        # forever, no "open" event, even though the video tracks - negotiated via transceivers
        # the client did include in its offer - worked fine over the same connection).
        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            if channel.label == "pointcloud":
                asyncio.ensure_future(_send_point_cloud(channel, frame_store))
            elif channel.label == "worldmap":
                asyncio.ensure_future(_send_world_state(channel, frame_store))
            elif channel.label == "status":
                asyncio.ensure_future(_send_status(channel, frame_store))
            elif channel.label == "control":
                # Reverse direction from the others - the browser sends, we just relay each
                # message into frame_store for the sim loop to pick up via pop_commands().
                # Malformed JSON is dropped rather than raised, since this callback runs inside
                # aiortc's own event loop - an uncaught exception here would surface as an
                # unhelpful asyncio traceback, not a clean error back to the browser.
                @channel.on("message")
                def on_control_message(message) -> None:
                    try:
                        frame_store.push_command(json.loads(message))
                    except (json.JSONDecodeError, TypeError):
                        pass

        await pc.setRemoteDescription(rtc_offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    async def on_shutdown(app: web.Application) -> None:
        await asyncio.gather(*(pc.close() for pc in list(pcs)))
        pcs.clear()

    app = web.Application()
    app["frame_store"] = frame_store
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/viewer.js", viewer_js)
    app.router.add_post("/offer", offer)
    return app


def run_in_background(frame_store: FrameStore, host: str = "0.0.0.0", port: int = 8080) -> threading.Thread:
    """Start the WebRTC signaling/media server in a daemon background thread with its own
    asyncio event loop, and return immediately - the caller's own (Isaac Sim) loop keeps
    running on the main thread untouched. Daemon thread: exits automatically when the main
    process does, no explicit shutdown call needed from the sim side.
    """

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _serve() -> None:
            app = build_app(frame_store)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            print(f"Streaming server listening on http://{host}:{port}/ (open in a browser)")
            await asyncio.Event().wait()  # run forever

        loop.run_until_complete(_serve())

    thread = threading.Thread(target=_run, name="webrtc-streaming-server", daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    import time

    store = FrameStore()

    def _synthetic_feed() -> None:
        t = 0
        while True:
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb[:, :, 0] = t * 4 % 255
            rgb[:, :, 1] = t * 2 % 255
            store.update_rgb(rgb)

            store.update_depth(np.random.uniform(0.5, 5.0, size=(480, 640)).astype(np.float32))

            angle = np.linspace(0, 2 * np.pi, 500, dtype=np.float32)
            radius = 2.0 + 0.5 * np.sin(angle * 5 + t * 0.1)
            points = np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)], axis=1)
            store.update_points(points)

            store.update_world_state(
                {
                    "room": {"min": [-3.0, -3.0], "max": [3.0, 3.0]},
                    "objects": [
                        {"label": "table", "x": 1.0, "y": 1.0, "hw": 0.6, "hh": 0.4, "color": "#8a8a8a"},
                        {"label": "box0", "x": -1.0, "y": 0.5, "hw": 0.3, "hh": 0.3, "color": "#cc3333"},
                    ],
                    "robot": {"x": float(2.0 * np.cos(t * 0.05)), "y": float(2.0 * np.sin(t * 0.05)), "heading": t * 0.05},
                }
            )

            t += 1
            time.sleep(1 / 20)

    threading.Thread(target=_synthetic_feed, daemon=True).start()
    server_thread = run_in_background(store)
    server_thread.join()
