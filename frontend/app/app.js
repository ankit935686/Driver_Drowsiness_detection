const statusElements = {
  apiHealth: document.getElementById("api-health"),
  serviceRunning: document.getElementById("service-running"),
  driverState: document.getElementById("driver-state"),
  ear: document.getElementById("ear"),
  earSmoothed: document.getElementById("ear-smoothed"),
  fps: document.getElementById("fps"),
  closedFrames: document.getElementById("closed-frames"),
  raw: document.getElementById("raw-json"),
  message: document.getElementById("message"),
};

const controls = {
  mode: document.getElementById("mode"),
  threshold: document.getElementById("ear-threshold"),
  consecutive: document.getElementById("consecutive-frames"),
  warning: document.getElementById("warning-frames"),
  recovery: document.getElementById("recovery-frames"),
  camera: document.getElementById("camera-index"),
  stage: document.getElementById("stage"),
};

const cameraPreview = document.getElementById("camera-preview");
const frameCanvas = document.getElementById("frame-canvas");
const frameCtx = frameCanvas.getContext("2d");

const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const refreshBtn = document.getElementById("refresh-btn");
const installBtn = document.getElementById("install-btn");

let pollTimer = null;
let frameTimer = null;
let mediaStream = null;
let deferredInstallPrompt = null;

function setMessage(text) {
  statusElements.message.textContent = text;
}

function readConfig() {
  return {
    mode: controls.mode.value,
    ear_threshold: Number(controls.threshold.value),
    consecutive_frames: Number(controls.consecutive.value),
    warning_frames: Number(controls.warning.value),
    recovery_frames: Number(controls.recovery.value),
    camera_index: Number(controls.camera.value),
    stage: Number(controls.stage.value),
  };
}

async function ensureCameraStream() {
  if (mediaStream) return;
  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 360 },
      facingMode: "user",
    },
    audio: false,
  });
  cameraPreview.srcObject = mediaStream;
  await cameraPreview.play();
}

function stopCameraStream() {
  if (!mediaStream) return;
  mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;
  cameraPreview.srcObject = null;
}

async function sendBrowserFrame() {
  if (!mediaStream || cameraPreview.readyState < 2) return;
  frameCtx.drawImage(cameraPreview, 0, 0, frameCanvas.width, frameCanvas.height);
  const image = frameCanvas.toDataURL("image/jpeg", 0.7);
  const response = await apiPost("/detect/frame", { image });
  renderStatus(response);
}

function startFrameLoop() {
  stopFrameLoop();
  frameTimer = setInterval(async () => {
    try {
      await sendBrowserFrame();
    } catch (error) {
      setMessage(error.message);
    }
  }, 250);
}

function stopFrameLoop() {
  if (frameTimer) {
    clearInterval(frameTimer);
    frameTimer = null;
  }
}

async function apiGet(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`GET ${path} failed: ${response.status}`);
  }
  return response.json();
}

async function apiPost(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.message || `POST ${path} failed: ${response.status}`);
  }
  return data;
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toFixed(digits);
}

function renderStatus(payload) {
  const status = payload?.status || {};
  const runtime = status.runtime || {};
  const state = runtime.state || "IDLE";

  statusElements.serviceRunning.textContent = `Service: ${status.running ? "Running" : "Stopped"}`;
  statusElements.driverState.textContent = `State: ${state}`;
  statusElements.driverState.classList.remove("state-idle", "state-awake", "state-warning", "state-drowsy");
  if (state === "AWAKE") {
    statusElements.driverState.classList.add("state-awake");
  } else if (state === "WARNING") {
    statusElements.driverState.classList.add("state-warning");
  } else if (state === "DROWSY") {
    statusElements.driverState.classList.add("state-drowsy");
  } else {
    statusElements.driverState.classList.add("state-idle");
  }

  statusElements.ear.textContent = fmt(runtime.ear);
  statusElements.earSmoothed.textContent = fmt(runtime.ear_smoothed);
  statusElements.fps.textContent = fmt(runtime.fps, 1);

  if (runtime.closed_frames !== undefined && runtime.consecutive_frames !== undefined) {
    statusElements.closedFrames.textContent = `${runtime.closed_frames}/${runtime.consecutive_frames}`;
  } else {
    statusElements.closedFrames.textContent = "-";
  }

  statusElements.raw.textContent = JSON.stringify(payload, null, 2);
}

async function refreshStatus() {
  try {
    const payload = await apiGet("/detect/status");
    renderStatus(payload);
  } catch (error) {
    setMessage(error.message);
  }
}

async function checkHealth() {
  try {
    const health = await apiGet("/health");
    statusElements.apiHealth.textContent = `API: ${health.status}`;
  } catch {
    statusElements.apiHealth.textContent = "API: Down";
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(refreshStatus, 1000);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

startBtn.addEventListener("click", async () => {
  try {
    setMessage("Starting detection...");
    const config = readConfig();
    if (config.mode === "browser") {
      await ensureCameraStream();
    }

    await apiPost("/detect/start", config);
    await refreshStatus();
    startPolling();

    if (config.mode === "browser") {
      startFrameLoop();
    }

    setMessage("Detection started.");
  } catch (error) {
    setMessage(error.message);
  }
});

stopBtn.addEventListener("click", async () => {
  try {
    setMessage("Stopping detection...");
    await apiPost("/detect/stop");
    await refreshStatus();
    stopFrameLoop();
    stopCameraStream();
    stopPolling();
    setMessage("Detection stopped.");
  } catch (error) {
    setMessage(error.message);
  }
});

refreshBtn.addEventListener("click", async () => {
  await refreshStatus();
  setMessage("Status refreshed.");
});

installBtn.addEventListener("click", async () => {
  if (!deferredInstallPrompt) return;
  deferredInstallPrompt.prompt();
  const choice = await deferredInstallPrompt.userChoice;
  deferredInstallPrompt = null;
  installBtn.hidden = true;
  if (choice.outcome === "accepted") {
    setMessage("App installation accepted.");
  } else {
    setMessage("App installation dismissed.");
  }
});

(async function init() {
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("/service-worker.js").catch(() => {
        setMessage("Service worker registration failed.");
      });
    });
  }

  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    deferredInstallPrompt = event;
    installBtn.hidden = false;
  });

  window.addEventListener("appinstalled", () => {
    deferredInstallPrompt = null;
    installBtn.hidden = true;
    setMessage("App installed.");
  });

  await checkHealth();
  await refreshStatus();
})();
