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
const detectionView = document.getElementById("detection-view");

const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const refreshBtn = document.getElementById("refresh-btn");
const installBtn = document.getElementById("install-btn");

let pollTimer = null;
let frameTimer = null;
let mediaStream = null;
let deferredInstallPrompt = null;
let frameSeq = 0;
let frameInFlight = false;
let activeConfig = null;

function setMessage(text) {
  statusElements.message.textContent = text;
}

function readConfig() {
  const warningFrames = Number(controls.warning.value);
  const consecutiveFrames = Number(controls.consecutive.value);
  const recoveryFrames = Number(controls.recovery.value);

  return {
    mode: controls.mode.value,
    ear_threshold: Number(controls.threshold.value),
    consecutive_frames: consecutiveFrames,
    warning_frames: warningFrames,
    recovery_frames: recoveryFrames,
    drowsy_seconds: consecutiveFrames / 10.0,
    warning_seconds: warningFrames / 10.0,
    recovery_seconds: recoveryFrames / 10.0,
    camera_index: Number(controls.camera.value),
    stage: Number(controls.stage.value),
  };
}

async function ensureCameraStream() {
  if (mediaStream) return;
  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 480 },
      height: { ideal: 270 },
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
  if (!mediaStream || cameraPreview.readyState < 2 || frameInFlight) return;
  frameInFlight = true;
  frameCtx.drawImage(cameraPreview, 0, 0, frameCanvas.width, frameCanvas.height);
  const image = frameCanvas.toDataURL("image/jpeg", 0.6);
  frameSeq += 1;
  const includeProcessedFrame = frameSeq % 8 === 0;

  try {
    let response;
    try {
      response = await apiPost("/detect/frame", {
        image,
        include_processed_frame: includeProcessedFrame,
        include_status: false,
        auto_start: true,
        ...(activeConfig || readConfig()),
      });
    } catch (error) {
      if (!error.message.includes("Detection is not running")) {
        throw error;
      }

      const fallback = { ...(activeConfig || readConfig()), mode: "browser" };
      try {
        await apiPost("/detect/start", fallback);
      } catch (startError) {
        if (!startError.message.includes("already running")) {
          throw startError;
        }
      }
      response = await apiPost("/detect/frame", {
        image,
        include_processed_frame: includeProcessedFrame,
        include_status: false,
        auto_start: true,
        ...fallback,
      });
    }

    if (response.processed_image) {
      detectionView.src = response.processed_image;
    }
    renderStatus(response);
  } finally {
    frameInFlight = false;
  }
}

function startFrameLoop() {
  stopFrameLoop();
  frameTimer = setInterval(async () => {
    try {
      await sendBrowserFrame();
    } catch (error) {
      setMessage(error.message);
    }
  }, 75);
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
  const status = payload?.status || null;
  const runtime = payload?.runtime || status?.runtime || {};
  const state = runtime.state || "IDLE";

  if (status) {
    statusElements.serviceRunning.textContent = `Service: ${status.running ? "Running" : "Stopped"}`;
  }
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

  const safePayload = { ...(payload || {}) };
  if (typeof safePayload.processed_image === "string") {
    safePayload.processed_image = `[omitted base64 image: ${safePayload.processed_image.length} chars]`;
  }
  statusElements.raw.textContent = JSON.stringify(safePayload, null, 2);
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
    frameSeq = 0;
    frameInFlight = false;
    const config = readConfig();
    activeConfig = { ...config };
    if (config.mode === "browser") {
      await ensureCameraStream();
    }

    try {
      await apiPost("/detect/start", config);
    } catch (error) {
      if (!error.message.includes("already running")) {
        throw error;
      }
    }
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
    setMessage("Detection stopped.");
  } catch (error) {
    if (error.message.includes("Detection is not running")) {
      setMessage("Detection already stopped.");
    } else {
      setMessage(error.message);
    }
  } finally {
    stopFrameLoop();
    stopCameraStream();
    frameSeq = 0;
    frameInFlight = false;
    activeConfig = null;
    detectionView.removeAttribute("src");
    stopPolling();
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
