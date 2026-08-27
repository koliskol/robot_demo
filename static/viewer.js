import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const statusEl = document.getElementById("status");
const connectBtn = document.getElementById("connect");
const rgbVideo = document.getElementById("rgb");
const depthVideo = document.getElementById("depth");

const recorderBar = document.getElementById("recorder-bar");
const recorderBadge = document.getElementById("recorder-badge");
const recordToggleBtn = document.getElementById("record-toggle");
const labelSuccessBtn = document.getElementById("label-success");
const labelFailBtn = document.getElementById("label-fail");
const labelDiscardBtn = document.getElementById("label-discard");

// Set by connect() once the peer connection exists - the button handlers below are wired once,
// up front, and just no-op (via the readyState guard in sendControl) until a connection with an
// open "control" channel actually exists.
let controlChannel = null;

function sendControl(message) {
  if (controlChannel && controlChannel.readyState === "open") {
    controlChannel.send(JSON.stringify(message));
  }
}

recordToggleBtn.addEventListener("click", () => sendControl({ action: "toggle_record" }));
labelSuccessBtn.addEventListener("click", () => sendControl({ action: "label", value: "success" }));
labelFailBtn.addEventListener("click", () => sendControl({ action: "label", value: "fail" }));
labelDiscardBtn.addEventListener("click", () => sendControl({ action: "discard" }));

// Only ever called if the server actually sends a "status" message (see build_app in
// streaming_server.py) - a page whose script never calls FrameStore.update_status (e.g.
// stream_demo.py) leaves recorder-bar hidden forever, which is the point.
function updateRecorderStatus(status) {
  recorderBar.style.display = "flex";
  const state = status.state; // "IDLE" | "RECORDING" | "AWAITING_LABEL"
  recorderBadge.className = state === "RECORDING" ? "recording" : state === "AWAITING_LABEL" ? "awaiting-label" : "";
  recorderBadge.textContent =
    state === "RECORDING"
      ? `recording (ep ${status.episode_index}, ${status.num_frames} frames)`
      : state === "AWAITING_LABEL"
      ? `awaiting label (ep ${status.episode_index}, ${status.num_frames} frames)`
      : "idle";
  recordToggleBtn.textContent = state === "RECORDING" ? "Stop Recording" : "Start Recording";
  recordToggleBtn.disabled = state === "AWAITING_LABEL";
  const awaitingLabel = state === "AWAITING_LABEL";
  labelSuccessBtn.disabled = !awaitingLabel;
  labelFailBtn.disabled = !awaitingLabel;
  labelDiscardBtn.disabled = !awaitingLabel;
}

connectBtn.addEventListener("click", () => {
  connectBtn.disabled = true;
  connect().catch((err) => {
    statusEl.textContent = `error: ${err}`;
    connectBtn.disabled = false;
  });
});

async function connect() {
  statusEl.textContent = "connecting...";
  const pc = new RTCPeerConnection();

  // recvonly: we only ever receive the server's two video tracks (RGB, then depth, in the
  // same order the server calls addTrack), never send our own.
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("video", { direction: "recvonly" });

  // We create this channel (even though the server is the one sending on it, via its own
  // ondatachannel handler) so our offer's SDP includes the SCTP section a data channel needs.
  // A server-side createDataChannel() call made only after seeing the offer can't negotiate -
  // an answer can't introduce a new media section that wasn't already in the offer.
  const pointChannel = pc.createDataChannel("pointcloud");
  pointChannel.binaryType = "arraybuffer";
  pointChannel.onmessage = (msg) => updatePointCloud(new Float32Array(msg.data));

  const mapChannel = pc.createDataChannel("worldmap");
  mapChannel.onmessage = (msg) => updateMap(JSON.parse(msg.data));

  const statusChannel = pc.createDataChannel("status");
  statusChannel.onmessage = (msg) => updateRecorderStatus(JSON.parse(msg.data));

  // Same "client creates it" reasoning as the others, but this one's the reverse direction: we
  // send on it (see sendControl above), the server just listens (see build_app's on_datachannel
  // in streaming_server.py).
  controlChannel = pc.createDataChannel("control");

  const videoEls = [rgbVideo, depthVideo];
  let nextVideoIndex = 0;
  pc.ontrack = (event) => {
    const el = videoEls[nextVideoIndex++];
    // Deliberately not event.streams[0]: the server adds both tracks without assigning them
    // to distinct streams, so both tracks land in the same remote MediaStream and
    // event.streams[0] would be identical for both ontrack calls - confirmed live, both video
    // elements showed the same (RGB) feed. Wrapping just this event's own track in a fresh
    // MediaStream guarantees each <video> only ever gets the track it was assigned.
    if (el) el.srcObject = new MediaStream([event.track]);
  };

  pc.onconnectionstatechange = () => {
    statusEl.textContent = pc.connectionState;
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const response = await fetch("/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
  });
  const answer = await response.json();
  await pc.setRemoteDescription(answer);
}

// --- Point cloud viewer (three.js) ---

const container = document.getElementById("pointcloud");
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.05, 500);
camera.position.set(4, 4, 4);
camera.up.set(0, 0, 1); // match Isaac Sim's Z-up convention

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AxesHelper(1));
// GridHelper is built flat in the XZ plane (Y up) by default, but this scene is Z-up (see
// camera.up above, matching Isaac Sim) - without this rotation the grid stands upright like a
// wall instead of lying flat as a floor. Rotating -90deg about X maps its XZ plane onto XY.
const gridHelper = new THREE.GridHelper(10, 10);
gridHelper.rotation.x = Math.PI / 2;
scene.add(gridHelper);

const pointsGeometry = new THREE.BufferGeometry();
const pointsMaterial = new THREE.PointsMaterial({ color: 0x00ffaa, size: 0.03 });
const pointCloud = new THREE.Points(pointsGeometry, pointsMaterial);
scene.add(pointCloud);

let loggedPointStats = false;
function updatePointCloud(flatXYZ) {
  pointsGeometry.setAttribute("position", new THREE.BufferAttribute(flatXYZ, 3));
  pointsGeometry.computeBoundingSphere();
  if (!loggedPointStats && flatXYZ.length > 0) {
    loggedPointStats = true;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < flatXYZ.length; i += 3) {
      minX = Math.min(minX, flatXYZ[i]); maxX = Math.max(maxX, flatXYZ[i]);
      minY = Math.min(minY, flatXYZ[i + 1]); maxY = Math.max(maxY, flatXYZ[i + 1]);
      minZ = Math.min(minZ, flatXYZ[i + 2]); maxZ = Math.max(maxZ, flatXYZ[i + 2]);
    }
    console.log(
      `[point cloud diagnostics, one-time] count=${flatXYZ.length / 3} ` +
      `x=[${minX.toFixed(3)}, ${maxX.toFixed(3)}] y=[${minY.toFixed(3)}, ${maxY.toFixed(3)}] ` +
      `z=[${minZ.toFixed(3)}, ${maxZ.toFixed(3)}] first_point=[${flatXYZ[0]}, ${flatXYZ[1]}, ${flatXYZ[2]}]`
    );
  }
}

window.addEventListener("resize", () => {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();

// --- 2D top-down map (plain canvas - simpler than three.js for flat rectangles/labels) ---

const mapCanvas = document.getElementById("map");
const mapCtx = mapCanvas.getContext("2d");
let latestWorldState = null;

function resizeMapCanvas() {
  mapCanvas.width = mapCanvas.clientWidth;
  mapCanvas.height = mapCanvas.clientHeight;
}
resizeMapCanvas();
window.addEventListener("resize", resizeMapCanvas);

function updateMap(worldState) {
  latestWorldState = worldState;
}

function drawMap() {
  requestAnimationFrame(drawMap);
  if (!latestWorldState) return;
  const { room, objects, robot } = latestWorldState;
  const w = mapCanvas.width, h = mapCanvas.height;
  mapCtx.fillStyle = "#000";
  mapCtx.fillRect(0, 0, w, h);
  if (!room) return;

  const [x0, y0] = room.min;
  const [x1, y1] = room.max;
  const padding = 24;
  const scale = Math.min((w - 2 * padding) / (x1 - x0), (h - 2 * padding) / (y1 - y0));

  // World (x, y) meters -> canvas pixels, flipping Y so +Y in world reads as "up" on screen -
  // the usual top-down-map convention.
  const toScreen = (x, y) => [padding + (x - x0) * scale, h - padding - (y - y0) * scale];

  const [rx0, ry0] = toScreen(x0, y0);
  const [rx1, ry1] = toScreen(x1, y1);
  mapCtx.strokeStyle = "#888";
  mapCtx.lineWidth = 2;
  mapCtx.strokeRect(Math.min(rx0, rx1), Math.min(ry0, ry1), Math.abs(rx1 - rx0), Math.abs(ry1 - ry0));

  mapCtx.textAlign = "center";
  mapCtx.font = "12px sans-serif";
  for (const obj of objects || []) {
    const [cx, cy] = toScreen(obj.x, obj.y);
    const halfW = obj.hw * scale, halfH = obj.hh * scale;
    mapCtx.fillStyle = obj.color || "#888";
    mapCtx.fillRect(cx - halfW, cy - halfH, halfW * 2, halfH * 2);
    mapCtx.fillStyle = "#fff";
    mapCtx.fillText(obj.label, cx, cy - halfH - 4);
  }

  if (robot) {
    const [rx, ry] = toScreen(robot.x, robot.y);
    // Elongated along the heading axis (nose far forward, narrow base) rather than a
    // stubby equilateral triangle - reads as a ">"-style direction arrow at a glance.
    const noseLength = 18;
    const backLength = 8;
    const halfWidth = 6;
    mapCtx.save();
    mapCtx.translate(rx, ry);
    // Screen Y is flipped relative to world Y (see toScreen), so negate heading to keep the
    // triangle's rotation visually consistent with the robot actually turning in the world.
    mapCtx.rotate(-robot.heading);
    mapCtx.fillStyle = "#00aaff";
    mapCtx.beginPath();
    mapCtx.moveTo(noseLength, 0);
    mapCtx.lineTo(-backLength, halfWidth);
    mapCtx.lineTo(-backLength, -halfWidth);
    mapCtx.closePath();
    mapCtx.fill();
    mapCtx.restore();
    mapCtx.fillStyle = "#fff";
    mapCtx.fillText("robot", rx, ry - noseLength - 6);
  }
}
requestAnimationFrame(drawMap);
