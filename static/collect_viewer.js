// Dedicated, minimal viewer for collect_pickplace_demo.py - RGB + depth video and recorder
// controls only. No map/point-cloud/three.js: this task has no lidar or world-state data to show.

const statusEl = document.getElementById("status");
const connectBtn = document.getElementById("connect");
const rgbVideo = document.getElementById("rgb");
const depthVideo = document.getElementById("depth");

const recorderBadge = document.getElementById("recorder-badge");
const recordToggleBtn = document.getElementById("record-toggle");
const labelSuccessBtn = document.getElementById("label-success");
const labelFailBtn = document.getElementById("label-fail");
const labelDiscardBtn = document.getElementById("label-discard");

const camUpBtn = document.getElementById("cam-up");
const camDownBtn = document.getElementById("cam-down");
const camLeftBtn = document.getElementById("cam-left");
const camRightBtn = document.getElementById("cam-right");

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

// Signs match the server's CAMERA_ROTATE_KEYS_PAN/TILT (LEFT=+1, RIGHT=-1, DOWN=+1, UP=-1) -
// confirmed live server-side: increasing pan rotates the view left, increasing tilt pitches down.
camLeftBtn.addEventListener("click", () => sendControl({ action: "camera_rotate", axis: "pan", delta: 1 }));
camRightBtn.addEventListener("click", () => sendControl({ action: "camera_rotate", axis: "pan", delta: -1 }));
camDownBtn.addEventListener("click", () => sendControl({ action: "camera_rotate", axis: "tilt", delta: 1 }));
camUpBtn.addEventListener("click", () => sendControl({ action: "camera_rotate", axis: "tilt", delta: -1 }));

function updateRecorderStatus(status) {
  const state = status.state; // "IDLE" | "RECORDING" | "AWAITING_LABEL"
  recorderBadge.className = state === "RECORDING" ? "recording" : state === "AWAITING_LABEL" ? "awaiting-label" : "";
  recorderBadge.textContent =
    state === "RECORDING"
      ? `recording (ep ${status.episode_index}, ${status.num_frames} frames)`
      : state === "AWAITING_LABEL"
      ? `awaiting label (ep ${status.episode_index}, ${status.num_frames} frames)`
      : "idle";
  recordToggleBtn.textContent = state === "RECORDING" ? "Stop Recording" : "Start Recording";
  const awaitingLabel = state === "AWAITING_LABEL";
  recordToggleBtn.disabled = awaitingLabel;
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

  // Server always sends RGB then depth, in that order (see build_app's offer()).
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("video", { direction: "recvonly" });

  const statusChannel = pc.createDataChannel("status");
  statusChannel.onmessage = (msg) => updateRecorderStatus(JSON.parse(msg.data));

  controlChannel = pc.createDataChannel("control");

  const videoEls = [rgbVideo, depthVideo];
  let nextVideoIndex = 0;
  pc.ontrack = (event) => {
    // Deliberately not event.streams[0] - see the shared viewer.js's ontrack for why (both
    // tracks land in the same remote MediaStream, wrapping just this event's own track in a
    // fresh MediaStream is what keeps each <video> showing only the track it was assigned).
    const el = videoEls[nextVideoIndex++];
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
