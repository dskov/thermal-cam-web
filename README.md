# Thermal camera viewer for HIKMICRO Mini series

A small tool to run a HIKMICRO Mini series thermal camera (256×192) on a Mac.
It shows the live thermal image in your browser, and lets you switch color
palettes, record video and take photos.

Tested with the HIKMICRO Mini2 V2.

## Features

- Live view in the browser, with fullscreen
- 10 color palettes: Inferno, Jet, Hot, Turbo, Magma, Plasma, Viridis, Bone, Ocean, Rainbow
- Horizontal and vertical flip (horizontal flip is on by default)
- Output size from native 256×192 up to 4× (1024×768)
- Video recording to MP4, with pause and resume
- Photos as JPG
- Recordings list in the page: play, rename, or open the folder in Finder

## How it works

ffmpeg reads the camera through macOS AVFoundation at 256×192, 25 fps. Each
frame is converted to grayscale, contrast-stretched, colored with an OpenCV
colormap, and streamed to the browser as MJPEG by a Flask server.

- The larger output sizes are the same image scaled up. 256×192 is the
  sensor's only real image mode.
- Contrast is stretched on every frame, so colors show what's hotter or colder
  in the current view, not absolute temperatures. The app doesn't read
  temperature values.

## Requirements

- macOS
- Python 3
- ffmpeg: `brew install ffmpeg`
- A HIKMICRO Mini series camera plugged in over USB-C

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```sh
make run
```

This starts the server on port 5999 and opens http://localhost:5999 in your
browser.

If the picture is black or shows your webcam instead, the thermal camera is on
a different device index. List the devices, test them, and pick the right one:

```sh
make test
make run DEVICE=1
```

The first time, macOS asks to give your terminal app camera access. If you
denied it, `make clear` resets camera permissions (for all apps) so macOS asks
again.

## Configuration

Set these as environment variables:

| Variable         | Default      | What it does                              |
|------------------|--------------|-------------------------------------------|
| `DEVICE`         | `0`          | AVFoundation video device index           |
| `PORT`           | `5999`       | Web server port                           |
| `OUTPUT_SCALE`   | `3x`         | Starting output size: `1x`, `2x`, `3x`, `4x` |
| `RECORDINGS_DIR` | `recordings` | Where videos and photos are saved         |

Example:

```sh
DEVICE=1 OUTPUT_SCALE=2x .venv/bin/python thermal_web.py
```

## Recordings

Files go to `recordings/`:

- Videos: `thermal_YYYYMMDD_HHMMSS.mp4`, H.264 at the selected output size,
  without the REC overlay
- Photos: `photo_YYYYMMDD_HHMMSS.jpg`

The output size can't be changed while recording, so set it before you start.

## Network access

The server listens on all network interfaces, so a phone or another computer
on the same network can open `http://<your-mac-ip>:5999`. There is no login:
anyone who can reach that address can also record, rename files or quit the
app. Only run it on networks you trust.

## Files

- `thermal_web.py`: the app (server and web page)
- `Makefile`: `make run`, `make test`, `make clear`
- `reset.sh`: resets macOS camera permissions
- `requirements.txt`: Python dependencies

## Disclaimer

This is an independent project, not affiliated with or endorsed by HIKMICRO.
