import os
import time
import threading
import subprocess
import webbrowser
from datetime import datetime

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
    send_from_directory
)


# ============================================================
# Camera configuration
# ============================================================

DEVICE = os.environ.get("DEVICE", "0")

# 256x192 @ 25 FPS is the sensor's only real image-capture mode. The
# camera's avfoundation device also reports other pixel sizes, but those
# are radiometric/metadata packings, not higher-resolution images -
# decoding them as ordinary video produces corrupted color data.
CAPTURE_WIDTH = int(os.environ.get("WIDTH", 256))
CAPTURE_HEIGHT = int(os.environ.get("HEIGHT", 192))
CAPTURE_FPS = int(os.environ.get("FPS", 25))

# "Resolution" options are display/recording output sizes, produced by
# scaling up the single native capture - not different camera modes.
OUTPUT_SCALES = {
    "1x": {"label": f"{CAPTURE_WIDTH}x{CAPTURE_HEIGHT} (native)", "scale": 1},
    "2x": {"label": f"{CAPTURE_WIDTH * 2}x{CAPTURE_HEIGHT * 2} (2x)", "scale": 2},
    "3x": {"label": f"{CAPTURE_WIDTH * 3}x{CAPTURE_HEIGHT * 3} (3x)", "scale": 3},
    "4x": {"label": f"{CAPTURE_WIDTH * 4}x{CAPTURE_HEIGHT * 4} (4x)", "scale": 4},
}

DEFAULT_OUTPUT_SCALE_KEY = os.environ.get("OUTPUT_SCALE", "3x")

if DEFAULT_OUTPUT_SCALE_KEY not in OUTPUT_SCALES:
    DEFAULT_OUTPUT_SCALE_KEY = "3x"

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "recordings")

os.makedirs(RECORDINGS_DIR, exist_ok=True)


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)


# ============================================================
# Global state
# ============================================================

lock = threading.Lock()

latest_frame = None

running = True

ffmpeg_process = None

current_output_scale_key = DEFAULT_OUTPUT_SCALE_KEY

_default_scale = OUTPUT_SCALES[DEFAULT_OUTPUT_SCALE_KEY]["scale"]
stream_width = CAPTURE_WIDTH * _default_scale
stream_height = CAPTURE_HEIGHT * _default_scale

palette_name = "inferno"

flip_horizontal = True
flip_vertical = False

recording = False
recording_paused = False
video_writer = None
recording_filename = None
recording_start_time = None
recording_paused_seconds = 0.0
pause_started_at = None


# ============================================================
# Palette mapping
# ============================================================

PALETTES = {
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "turbo": cv2.COLORMAP_TURBO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "bone": cv2.COLORMAP_BONE,
    "ocean": cv2.COLORMAP_OCEAN,
    "rainbow": cv2.COLORMAP_RAINBOW,
}


# ============================================================
# Camera capture thread
# ============================================================

def build_ffmpeg_command():
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",

        "-f", "avfoundation",

        "-framerate", str(CAPTURE_FPS),
        "-video_size", f"{CAPTURE_WIDTH}x{CAPTURE_HEIGHT}",

        "-i", f"{DEVICE}:none",

        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",

        "-"
    ]


def camera_loop():
    global latest_frame
    global running
    global video_writer
    global ffmpeg_process

    command = build_ffmpeg_command()

    print("Starting FFmpeg:")
    print(" ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=10**8
    )

    ffmpeg_process = process

    frame_size = CAPTURE_WIDTH * CAPTURE_HEIGHT * 3

    time.sleep(0.5)

    if process.poll() is not None:
        print("FFmpeg failed to start.")
        running = False
        return

    while running:
        raw = process.stdout.read(frame_size)

        if not raw or len(raw) != frame_size:
            print("Camera stream ended.")
            break

        frame = np.frombuffer(
            raw,
            dtype=np.uint8
        ).reshape(
            CAPTURE_HEIGHT,
            CAPTURE_WIDTH,
            3
        )

        # ----------------------------------------------------
        # Flip
        # ----------------------------------------------------

        with lock:
            do_flip_h = flip_horizontal
            do_flip_v = flip_vertical

        if do_flip_h and do_flip_v:
            frame = cv2.flip(frame, -1)
        elif do_flip_h:
            frame = cv2.flip(frame, 1)
        elif do_flip_v:
            frame = cv2.flip(frame, 0)

        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Normalize contrast
        # ----------------------------------------------------

        gray_normalized = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        # ----------------------------------------------------
        # Get current palette safely
        # ----------------------------------------------------

        with lock:
            current_palette = palette_name

        palette = PALETTES.get(
            current_palette,
            cv2.COLORMAP_INFERNO
        )

        colored = cv2.applyColorMap(
            gray_normalized,
            palette
        )

        # ----------------------------------------------------
        # Resize stream to the current output scale
        # ----------------------------------------------------

        with lock:
            target_width = stream_width
            target_height = stream_height

        display = cv2.resize(
            colored,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------------------------
        # Record processed frame (clean, no overlay burned in)
        # ----------------------------------------------------

        with lock:
            is_recording = recording
            is_paused = recording_paused

            if (
                recording
                and not recording_paused
                and video_writer is not None
            ):
                video_writer.write(display)

        # ----------------------------------------------------
        # Overlay status (live preview only)
        # ----------------------------------------------------

        live_frame = display

        if is_recording:
            live_frame = display.copy()

            status = "RECORDING PAUSED" if is_paused else "REC"

            cv2.putText(
                live_frame,
                status,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Store latest frame
        # ----------------------------------------------------

        with lock:
            latest_frame = live_frame.copy()

    if process.poll() is None:
        process.terminate()

    if process.stdout is not None:
        process.stdout.close()

    print("Camera thread stopped.")


# ============================================================
# MJPEG generator
# ============================================================

def generate_mjpeg():
    while running:
        with lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        ok, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )

        time.sleep(1 / CAPTURE_FPS)


# ============================================================
# HTML
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>HIKMICRO Thermal Camera</title>

    <style>
        :root {
            --bg: #0c0e11;
            --panel-bg: #16191d;
            --panel-bg-raised: #1c2025;
            --border: #262b31;
            --text: #e8eaed;
            --text-dim: #8b93a1;
            --accent: #4da6ff;
            --danger: #ef4444;
            --warning: #f59e0b;
            --success: #22c55e;
            --neutral: #3a3f47;
            --radius: 12px;
            --radius-sm: 8px;
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 0;
        }

        #layout {
            display: flex;
            align-items: flex-start;
            min-height: 100vh;
        }

        #sidebar {
            width: 340px;
            flex-shrink: 0;
            padding: 24px;
            border-right: 1px solid var(--border);
            text-align: left;
        }

        #main {
            flex: 1;
            padding: 32px;
            text-align: center;
        }

        #recordingsCard {
            margin-top: 24px;
            width: 768px;
            max-width: 88vw;
            margin-left: auto;
            margin-right: auto;
            text-align: left;
        }

        .recordings-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .recordings-header h3 {
            margin: 0;
        }

        #recordingsScroll {
            max-height: 220px;
            overflow-y: auto;
            margin-top: 10px;
        }

        #recordingsScroll table {
            margin-top: 0;
        }

        #recordingsScroll thead th {
            position: sticky;
            top: 0;
            background: var(--panel-bg);
        }

        .brand {
            margin-bottom: 24px;
        }

        .brand h1 {
            margin: 0;
            font-size: 19px;
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        .brand .subtitle {
            margin-top: 4px;
            font-size: 12px;
            color: var(--text-dim);
            font-family: "SF Mono", Menlo, monospace;
        }

        .card {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px;
            margin-bottom: 16px;
        }

        .card h3 {
            margin: 0 0 12px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-dim);
        }

        .card.danger-zone {
            border-color: rgba(239, 68, 68, 0.35);
            background: rgba(239, 68, 68, 0.06);
        }

        .row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .field-label {
            display: block;
            font-size: 12px;
            color: var(--text-dim);
            margin-bottom: 6px;
        }

        #videoContainer {
            position: relative;
            display: inline-block;
            border-radius: var(--radius);
            overflow: hidden;
            border: 1px solid var(--border);
            background: #000;
        }

        #video {
            max-width: min(768px, 88vw);
            max-height: 70vh;
            display: block;
        }

        #videoContainer:fullscreen {
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100vw;
            height: 100vh;
            border-radius: 0;
        }

        #videoContainer:fullscreen #video {
            width: auto;
            height: 100vh;
            max-width: 100vw;
            max-height: 100vh;
        }

        #fullscreenBtn {
            position: absolute;
            top: 12px;
            right: 12px;
            background: rgba(12, 14, 17, 0.65);
            backdrop-filter: blur(4px);
            margin: 0;
            padding: 8px 14px;
            font-size: 13px;
        }

        #cameraInfo {
            position: absolute;
            top: 12px;
            left: 12px;
            padding: 5px 12px;
            border-radius: 999px;
            background: rgba(12, 14, 17, 0.65);
            backdrop-filter: blur(4px);
            border: 1px solid var(--border);
            font-family: "SF Mono", Menlo, monospace;
            font-size: 12px;
            color: var(--text-dim);
        }

        button,
        select {
            font-family: inherit;
            font-size: 14px;
            padding: 10px 14px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border);
            cursor: pointer;
            transition: background 0.15s ease, border-color 0.15s ease;
        }

        select {
            width: 100%;
            background: var(--panel-bg-raised);
            color: var(--text);
            margin-bottom: 8px;
        }

        button {
            background: var(--neutral);
            color: var(--text);
        }

        button:hover:not(:disabled) {
            background: #4a505a;
        }

        button.primary {
            background: var(--accent);
            border-color: var(--accent);
            color: #06121f;
            font-weight: 600;
            width: 100%;
        }

        button.primary:hover:not(:disabled) {
            background: #6bb6ff;
        }

        button.record {
            background: var(--danger);
            border-color: var(--danger);
        }

        button.pause {
            background: var(--warning);
            border-color: var(--warning);
            color: #1a1200;
        }

        button.resume {
            background: var(--success);
            border-color: var(--success);
            color: #05170b;
        }

        button.stop {
            background: var(--neutral);
        }

        button.danger {
            background: transparent;
            border-color: var(--danger);
            color: var(--danger);
            width: 100%;
        }

        button.danger:hover:not(:disabled) {
            background: rgba(239, 68, 68, 0.12);
        }

        button:disabled {
            opacity: 0.4;
            cursor: default;
        }

        .toggle {
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            padding: 6px 0;
            user-select: none;
        }

        .toggle input {
            display: none;
        }

        .toggle-track {
            position: relative;
            width: 38px;
            height: 22px;
            flex-shrink: 0;
            background: var(--neutral);
            border-radius: 999px;
            transition: background 0.15s ease;
        }

        .toggle-thumb {
            position: absolute;
            top: 2px;
            left: 2px;
            width: 18px;
            height: 18px;
            background: #fff;
            border-radius: 50%;
            transition: transform 0.15s ease;
        }

        .toggle input:checked + .toggle-track {
            background: var(--accent);
        }

        .toggle input:checked + .toggle-track .toggle-thumb {
            transform: translateX(16px);
        }

        .toggle-label {
            font-size: 14px;
            color: var(--text);
        }

        #status {
            margin-top: 4px;
            font-family: "SF Mono", Menlo, monospace;
            font-size: 12px;
            color: var(--text-dim);
            word-break: break-word;
        }

        #recordTimer {
            display: none;
            margin-top: 10px;
            font-family: "SF Mono", Menlo, monospace;
            font-size: 22px;
            font-weight: 600;
            color: var(--danger);
        }

        table {
            border-collapse: collapse;
            margin-top: 8px;
            width: 100%;
        }

        th,
        td {
            padding: 6px 8px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            font-family: "SF Mono", Menlo, monospace;
            font-size: 12px;
        }

        th {
            color: var(--text-dim);
            font-weight: 500;
        }

        td a {
            color: var(--accent);
            text-decoration: none;
        }

        td a:hover {
            text-decoration: underline;
        }
    </style>
</head>

<body>

    <div id="layout">

        <div id="sidebar">

            <div class="brand">
                <h1>HIKMICRO Mini2 V2</h1>
                <div class="subtitle">Thermal camera control</div>
            </div>

            <div class="card">
                <h3>Output Size</h3>

                <select id="outputScale">
                    {% for key, label in output_scales %}
                    <option value="{{ key }}" {% if key == selected_output_scale %}selected{% endif %}>
                        {{ label }}
                    </option>
                    {% endfor %}
                </select>

                <button id="outputScaleApplyBtn" class="primary" onclick="changeOutputScale()">
                    Apply
                </button>
            </div>

            <div class="card">
                <h3>Palette</h3>

                <select id="palette">
                    <option value="inferno">Inferno</option>
                    <option value="jet">Jet</option>
                    <option value="hot">Hot</option>
                    <option value="turbo">Turbo</option>
                    <option value="magma">Magma</option>
                    <option value="plasma">Plasma</option>
                    <option value="viridis">Viridis</option>
                    <option value="bone">Bone</option>
                    <option value="ocean">Ocean</option>
                    <option value="rainbow">Rainbow</option>
                </select>

                <button class="primary" onclick="changePalette()">
                    Apply Palette
                </button>
            </div>

            <div class="card">
                <h3>Orientation</h3>

                <label class="toggle">
                    <input type="checkbox" id="flipHToggle" onchange="toggleFlip('horizontal')">
                    <span class="toggle-track"><span class="toggle-thumb"></span></span>
                    <span class="toggle-label">Flip Horizontal</span>
                </label>

                <label class="toggle">
                    <input type="checkbox" id="flipVToggle" onchange="toggleFlip('vertical')">
                    <span class="toggle-track"><span class="toggle-thumb"></span></span>
                    <span class="toggle-label">Flip Vertical</span>
                </label>
            </div>

            <div class="card">
                <h3>Capture</h3>

                <div class="row">
                    <button
                        id="recordBtn"
                        class="record"
                        onclick="toggleRecord()"
                    >
                        Start Recording
                    </button>

                    <button
                        id="pauseBtn"
                        class="pause"
                        onclick="togglePause()"
                        disabled
                    >
                        Pause
                    </button>

                    <button
                        id="photoBtn"
                        onclick="takePhoto()"
                    >
                        📷 Take Photo
                    </button>
                </div>

                <div id="recordTimer"></div>
                <div id="status">Ready</div>
            </div>

            <div class="card danger-zone">
                <h3>Danger Zone</h3>

                <button class="danger" onclick="quitApp()">
                    Quit Application
                </button>
            </div>

        </div>

        <div id="main">

            <div id="videoContainer">

                <div id="cameraInfo">
                    {{ width }}x{{ height }} @ {{ fps }} FPS
                </div>

                <img
                    id="video"
                    src="/video_feed"
                >

                <button id="fullscreenBtn" onclick="toggleFullscreen()">
                    ⛶ Fullscreen
                </button>

            </div>

            <div class="card" id="recordingsCard">
                <div class="recordings-header">
                    <h3>Recordings</h3>

                    <div class="row">
                        <button onclick="openRecordingsFolder()">
                            📂 Open Folder
                        </button>

                        <button onclick="loadRecordings()">
                            Refresh
                        </button>
                    </div>
                </div>

                <div id="recordingsScroll">
                    <table>
                        <thead>
                            <tr>
                                <th>File</th>
                                <th>Size</th>
                                <th>Resolution</th>
                                <th>Length</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody id="recordingsBody">
                        </tbody>
                    </table>
                </div>
            </div>

        </div>

    </div>

<script>

let recording = false;
let recordingPaused = false;

function setStatus(message) {
    document.getElementById("status").innerText = message;
}


function renderRecordingButtons() {

    const recordBtn = document.getElementById("recordBtn");
    const pauseBtn = document.getElementById("pauseBtn");
    const outputScaleSelect = document.getElementById("outputScale");
    const outputScaleApplyBtn = document.getElementById("outputScaleApplyBtn");

    if (recording) {
        recordBtn.innerText = "Stop Recording";
        recordBtn.className = "stop";
    } else {
        recordBtn.innerText = "Start Recording";
        recordBtn.className = "record";
    }

    pauseBtn.disabled = !recording;
    pauseBtn.innerText = recordingPaused ? "Resume" : "Pause";
    pauseBtn.className = recordingPaused ? "resume" : "pause";

    outputScaleSelect.disabled = recording;
    outputScaleApplyBtn.disabled = recording;
}


function formatElapsed(seconds) {

    if (seconds === null || seconds === undefined) {
        return "";
    }

    const total = Math.max(0, Math.round(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;

    const mm = String(minutes).padStart(2, "0");
    const ss = String(secs).padStart(2, "0");

    return hours > 0 ? (hours + ":" + mm + ":" + ss) : (mm + ":" + ss);
}


function renderTimer(data) {

    const timerEl = document.getElementById("recordTimer");

    if (!data.recording) {
        timerEl.style.display = "none";
        return;
    }

    const label = data.recording_paused ? "PAUSED" : "REC";

    timerEl.innerText = label + " " + formatElapsed(data.recording_elapsed_seconds);
    timerEl.style.display = "block";
}


async function refreshStatus(syncSelects) {

    const response = await fetch("/status");
    const data = await response.json();

    recording = data.recording;
    recordingPaused = data.recording_paused;

    // Only force the palette/output-size dropdowns to match the server on
    // load or right after we apply a change - not on every periodic poll,
    // otherwise a poll landing between "pick a new value" and "click Apply"
    // silently reverts the user's in-progress selection.
    if (syncSelects) {
        document.getElementById("palette").value = data.palette;
        document.getElementById("outputScale").value = data.output_scale;
    }

    document.getElementById("flipHToggle").checked = data.flip_horizontal;
    document.getElementById("flipVToggle").checked = data.flip_vertical;

    renderRecordingButtons();
    renderTimer(data);
}


async function changePalette() {

    const palette =
        document.getElementById("palette").value;

    const response = await fetch(
        "/palette/" + palette,
        {
            method: "POST"
        }
    );

    const data = await response.json();

    setStatus(
        "Palette: " + data.palette
    );
}


async function changeOutputScale() {

    const key = document.getElementById("outputScale").value;

    const response = await fetch(
        "/output-scale/" + key,
        {
            method: "POST"
        }
    );

    const data = await response.json();

    setStatus(data.ok ? ("Output size: " + key) : (data.error || "Could not change output size"));

    await refreshStatus(true);
}


async function takePhoto() {

    const response = await fetch(
        "/photo",
        {
            method: "POST"
        }
    );

    const data = await response.json();

    setStatus(data.message);

    if (data.ok) {
        loadRecordings();
    }
}


async function quitApp() {

    if (!confirm("Quit the application? This stops the server and camera capture.")) {
        return;
    }

    setStatus("Shutting down...");

    try {
        await fetch("/quit", { method: "POST" });
    } catch (e) {
        // The connection drops as the process exits - that's expected.
    }

    setStatus("Application stopped. You can close this tab.");
}


function toggleFullscreen() {

    const container = document.getElementById("videoContainer");

    if (!document.fullscreenElement) {
        if (container.requestFullscreen) {
            container.requestFullscreen();
        } else if (container.webkitRequestFullscreen) {
            container.webkitRequestFullscreen();
        }
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        }
    }
}


async function toggleFlip(axis) {

    const response = await fetch(
        "/flip/" + axis,
        {
            method: "POST"
        }
    );

    const data = await response.json();

    document.getElementById("flipHToggle").checked = data.flip_horizontal;
    document.getElementById("flipVToggle").checked = data.flip_vertical;
}


async function toggleRecord() {

    const endpoint = recording ? "/record/stop" : "/record/start";

    const response = await fetch(
        endpoint,
        {
            method: "POST"
        }
    );

    const data = await response.json();

    setStatus(data.message);

    await refreshStatus(true);

    if (!recording) {
        loadRecordings();
    }
}


async function togglePause() {

    const endpoint = recordingPaused ? "/record/resume" : "/record/pause";

    const response = await fetch(
        endpoint,
        {
            method: "POST"
        }
    );

    const data = await response.json();

    setStatus(data.message);

    await refreshStatus(true);
}


function formatSize(bytes) {

    const mb = bytes / (1024 * 1024);

    return mb.toFixed(1) + " MB";
}


function formatDuration(seconds) {

    if (seconds === null || seconds === undefined) {
        return "?";
    }

    const totalSeconds = Math.round(seconds);
    const minutes = Math.floor(totalSeconds / 60);
    const remainder = totalSeconds % 60;

    return minutes + ":" + String(remainder).padStart(2, "0");
}


async function loadRecordings() {

    const response = await fetch("/recordings/list");
    const data = await response.json();

    const body = document.getElementById("recordingsBody");
    body.innerHTML = "";

    for (const file of data.files) {

        const row = document.createElement("tr");

        const nameCell = document.createElement("td");
        const link = document.createElement("a");
        link.href = "/recordings/file/" + encodeURIComponent(file.name);
        link.target = "_blank";
        link.innerText = file.name;
        nameCell.appendChild(link);

        const sizeCell = document.createElement("td");
        sizeCell.innerText = formatSize(file.size_bytes);

        const resolutionCell = document.createElement("td");
        resolutionCell.innerText = file.resolution || "?";

        const lengthCell = document.createElement("td");
        lengthCell.innerText = file.name.toLowerCase().endsWith(".jpg")
            ? "Photo"
            : formatDuration(file.duration_seconds);

        const actionCell = document.createElement("td");
        const renameBtn = document.createElement("button");
        renameBtn.innerText = "Rename";
        renameBtn.onclick = () => renameRecording(file.name);
        actionCell.appendChild(renameBtn);

        row.appendChild(nameCell);
        row.appendChild(sizeCell);
        row.appendChild(resolutionCell);
        row.appendChild(lengthCell);
        row.appendChild(actionCell);

        body.appendChild(row);
    }
}


async function renameRecording(oldName) {

    const dotIndex = oldName.lastIndexOf(".");
    const base = dotIndex >= 0 ? oldName.slice(0, dotIndex) : oldName;
    const ext = dotIndex >= 0 ? oldName.slice(dotIndex) : "";

    const newBase = prompt("Rename file to:", base);

    if (!newBase || newBase === base) {
        return;
    }

    const response = await fetch(
        "/recordings/rename",
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                old_name: oldName,
                new_name: newBase + ext
            })
        }
    );

    const data = await response.json();

    setStatus(data.message);

    if (data.ok) {
        loadRecordings();
    }
}


async function openRecordingsFolder() {

    const response = await fetch(
        "/recordings/open-folder",
        {
            method: "POST"
        }
    );

    const data = await response.json();

    setStatus(data.message);
}


refreshStatus(true);
loadRecordings();

setInterval(() => refreshStatus(false), 1000);

</script>

</body>
</html>
"""


# ============================================================
# Web routes
# ============================================================

@app.route("/")
def index():
    with lock:
        selected_output_scale = current_output_scale_key

    return render_template_string(
        HTML,
        width=CAPTURE_WIDTH,
        height=CAPTURE_HEIGHT,
        fps=CAPTURE_FPS,
        output_scales=[
            (key, value["label"])
            for key, value in OUTPUT_SCALES.items()
        ],
        selected_output_scale=selected_output_scale
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# Palette control
# ============================================================

@app.route(
    "/palette/<name>",
    methods=["POST"]
)
def set_palette(name):
    global palette_name

    name = name.lower()

    if name not in PALETTES:
        return jsonify({
            "ok": False,
            "error": "Unknown palette"
        }), 400

    with lock:
        palette_name = name

    return jsonify({
        "ok": True,
        "palette": palette_name
    })


# ============================================================
# Flip control
# ============================================================

@app.route(
    "/flip/<axis>",
    methods=["POST"]
)
def toggle_flip(axis):
    global flip_horizontal
    global flip_vertical

    if axis not in ("horizontal", "vertical"):
        return jsonify({
            "ok": False,
            "error": "Unknown axis"
        }), 400

    with lock:
        if axis == "horizontal":
            flip_horizontal = not flip_horizontal
        else:
            flip_vertical = not flip_vertical

        result = {
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical
        }

    return jsonify({
        "ok": True,
        **result
    })


# ============================================================
# Output size control
# ============================================================

@app.route(
    "/output-scale/<key>",
    methods=["POST"]
)
def set_output_scale(key):
    global current_output_scale_key
    global stream_width
    global stream_height

    if key not in OUTPUT_SCALES:
        return jsonify({
            "ok": False,
            "error": "Unknown output size"
        }), 400

    with lock:
        if recording:
            return jsonify({
                "ok": False,
                "error": "Stop recording before changing output size"
            }), 400

        scale = OUTPUT_SCALES[key]["scale"]
        current_output_scale_key = key
        stream_width = CAPTURE_WIDTH * scale
        stream_height = CAPTURE_HEIGHT * scale

    return jsonify({
        "ok": True,
        "output_scale": key
    })


# ============================================================
# Status
# ============================================================

@app.route("/status")
def status():
    with lock:
        elapsed_seconds = None

        if recording and recording_start_time is not None:
            if recording_paused and pause_started_at is not None:
                reference_time = pause_started_at
            else:
                reference_time = time.monotonic()

            elapsed_seconds = (
                reference_time
                - recording_start_time
                - recording_paused_seconds
            )

        data = {
            "recording": recording,
            "recording_paused": recording_paused,
            "recording_elapsed_seconds": elapsed_seconds,
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
            "palette": palette_name,
            "output_scale": current_output_scale_key
        }

    return jsonify(data)


# ============================================================
# Recording controls
# ============================================================

@app.route(
    "/record/start",
    methods=["POST"]
)
def record_start():
    global recording
    global recording_paused
    global video_writer
    global recording_filename
    global recording_start_time
    global recording_paused_seconds
    global pause_started_at

    with lock:

        if recording:
            return jsonify({
                "ok": False,
                "message": "Already recording"
            })

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        recording_filename = os.path.join(
            RECORDINGS_DIR,
            f"thermal_{timestamp}.mp4"
        )

        # H.264 (avc1) plays natively in every browser; mp4v (MPEG-4 Part 2)
        # does not, so only fall back to it if this system's OpenCV/ffmpeg
        # build has no H.264 encoder available.
        video_writer = cv2.VideoWriter(
            recording_filename,
            cv2.VideoWriter_fourcc(*"avc1"),
            CAPTURE_FPS,
            (
                stream_width,
                stream_height
            )
        )

        if not video_writer.isOpened():
            video_writer = cv2.VideoWriter(
                recording_filename,
                cv2.VideoWriter_fourcc(*"mp4v"),
                CAPTURE_FPS,
                (
                    stream_width,
                    stream_height
                )
            )

        if not video_writer.isOpened():
            video_writer = None

            return jsonify({
                "ok": False,
                "message": "Could not create recording file"
            }), 500

        recording = True
        recording_paused = False
        recording_start_time = time.monotonic()
        recording_paused_seconds = 0.0
        pause_started_at = None

    return jsonify({
        "ok": True,
        "message": (
            "Recording started: "
            + recording_filename
        )
    })


@app.route(
    "/record/pause",
    methods=["POST"]
)
def record_pause():
    global recording_paused
    global pause_started_at

    with lock:

        if not recording:
            return jsonify({
                "ok": False,
                "message": "Not recording"
            })

        recording_paused = True
        pause_started_at = time.monotonic()

    return jsonify({
        "ok": True,
        "message": "Recording paused"
    })


@app.route(
    "/record/resume",
    methods=["POST"]
)
def record_resume():
    global recording_paused
    global recording_paused_seconds
    global pause_started_at

    with lock:

        if not recording:
            return jsonify({
                "ok": False,
                "message": "Not recording"
            })

        if pause_started_at is not None:
            recording_paused_seconds += time.monotonic() - pause_started_at
            pause_started_at = None

        recording_paused = False

    return jsonify({
        "ok": True,
        "message": "Recording resumed"
    })


@app.route(
    "/record/stop",
    methods=["POST"]
)
def record_stop():
    global recording
    global recording_paused
    global video_writer
    global recording_filename
    global recording_start_time
    global recording_paused_seconds
    global pause_started_at

    with lock:

        if not recording:
            return jsonify({
                "ok": False,
                "message": "Not recording"
            })

        recording = False
        recording_paused = False
        recording_start_time = None
        recording_paused_seconds = 0.0
        pause_started_at = None

        if video_writer is not None:
            video_writer.release()
            video_writer = None

        filename = recording_filename

        recording_filename = None

    return jsonify({
        "ok": True,
        "message": (
            "Recording saved: "
            + str(filename)
        )
    })


# ============================================================
# Recordings browser
# ============================================================

def probe_duration_seconds(path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        return float(result.stdout.strip())

    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def probe_resolution(path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                path
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        output = result.stdout.strip()

        return output if output else None

    except (OSError, subprocess.TimeoutExpired):
        return None


@app.route("/recordings/list")
def recordings_list():
    files = []

    for name in os.listdir(RECORDINGS_DIR):
        lower_name = name.lower()

        is_video = lower_name.endswith(".mp4")
        is_photo = lower_name.endswith(".jpg")

        if not (is_video or is_photo):
            continue

        path = os.path.join(RECORDINGS_DIR, name)

        files.append({
            "name": name,
            "size_bytes": os.path.getsize(path),
            "duration_seconds": probe_duration_seconds(path) if is_video else None,
            "resolution": probe_resolution(path),
            "modified_at": os.path.getmtime(path)
        })

    files.sort(key=lambda f: f["modified_at"], reverse=True)

    for file_info in files:
        del file_info["modified_at"]

    return jsonify({
        "ok": True,
        "files": files
    })


@app.route(
    "/photo",
    methods=["POST"]
)
def take_photo():
    with lock:
        if latest_frame is None:
            return jsonify({
                "ok": False,
                "message": "No frame available yet"
            }), 503

        frame = latest_frame.copy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = os.path.join(
        RECORDINGS_DIR,
        f"photo_{timestamp}.jpg"
    )

    if not cv2.imwrite(filename, frame):
        return jsonify({
            "ok": False,
            "message": "Could not save photo"
        }), 500

    return jsonify({
        "ok": True,
        "message": "Photo saved: " + filename
    })


@app.route("/recordings/file/<path:filename>")
def recordings_file(filename):
    return send_from_directory(
        RECORDINGS_DIR,
        filename,
        as_attachment=False
    )


@app.route(
    "/recordings/open-folder",
    methods=["POST"]
)
def open_recordings_folder():
    folder_path = os.path.abspath(RECORDINGS_DIR)

    try:
        subprocess.run(["open", folder_path], check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        return jsonify({
            "ok": False,
            "message": f"Could not open folder: {error}"
        }), 500

    return jsonify({
        "ok": True,
        "message": "Opened: " + folder_path
    })


@app.route(
    "/recordings/rename",
    methods=["POST"]
)
def rename_recording():
    data = request.get_json(silent=True) or {}

    old_name = data.get("old_name", "")
    new_name = data.get("new_name", "")

    if not old_name or not new_name:
        return jsonify({
            "ok": False,
            "message": "Missing filename"
        }), 400

    if (
        os.path.basename(old_name) != old_name
        or os.path.basename(new_name) != new_name
    ):
        return jsonify({
            "ok": False,
            "message": "Invalid filename"
        }), 400

    old_ext = os.path.splitext(old_name)[1].lower()
    new_ext = os.path.splitext(new_name)[1].lower()

    if new_ext != old_ext:
        return jsonify({
            "ok": False,
            "message": f"Keep the {old_ext} extension"
        }), 400

    old_path = os.path.join(RECORDINGS_DIR, old_name)
    new_path = os.path.join(RECORDINGS_DIR, new_name)

    if not os.path.isfile(old_path):
        return jsonify({
            "ok": False,
            "message": "File not found"
        }), 404

    if os.path.exists(new_path):
        return jsonify({
            "ok": False,
            "message": "A file with that name already exists"
        }), 409

    os.rename(old_path, new_path)

    return jsonify({
        "ok": True,
        "message": "Renamed to " + new_name
    })


# ============================================================
# Quit
# ============================================================

@app.route(
    "/quit",
    methods=["POST"]
)
def quit_app():
    global running

    running = False

    def shutdown():
        global video_writer

        time.sleep(0.3)

        with lock:
            if video_writer is not None:
                video_writer.release()
                video_writer = None

        if ffmpeg_process is not None and ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()

        os._exit(0)

    threading.Thread(target=shutdown, daemon=True).start()

    return jsonify({
        "ok": True,
        "message": "Shutting down"
    })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5999))

    camera_thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )

    camera_thread.start()

    threading.Timer(
        1.0,
        lambda: webbrowser.open(f"http://localhost:{port}")
    ).start()

    try:

        app.run(
            host="0.0.0.0",
            port=port,
            threaded=True,
            debug=False
        )

    finally:

        running = False

        with lock:
            if video_writer is not None:
                video_writer.release()

        print("Shutting down.")
