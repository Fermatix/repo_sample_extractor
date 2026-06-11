#!/usr/bin/env python
"""Capture frames from a macOS camera (default: the iPhone via Continuity Camera).

Uses ffmpeg + AVFoundation (already installed) rather than OpenCV, because
OpenCV + Continuity Camera has flaky device-index / permission behaviour on
macOS. The iPhone is auto-located by name so it keeps working even if the
AVFoundation index shifts when other virtual cameras (Camo, Desk View, ...)
come and go.

This is the drop-in image source for a real-robot `observe()` (the analogue of
the sim's rendered screenshots).

CLI:
    # list the available video devices and their indices
    python scripts/real_camera.py --probe

    # grab one warmed-up frame from the iPhone and save it
    python scripts/real_camera.py --capture agent_runs/camera-test/frame.png

    # override the device or output width
    python scripts/real_camera.py --capture out.png --index 2 --width 1280
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

FFMPEG = "ffmpeg"
DEFAULT_NAME_MATCH = "iphone"
DEFAULT_NAME_EXCLUDE = "desk view"  # the "Desk View" companion cam, not what we want
DEFAULT_WIDTH = 1280
DEFAULT_WARMUP = 24  # frames to let auto-exposure/white-balance settle (~0.8s @30fps)
_DEVICE_LINE = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")


class CameraError(RuntimeError):
    pass


def list_video_devices() -> list[tuple[int, str]]:
    """Return [(index, name), ...] for AVFoundation *video* devices."""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    # ffmpeg prints the device list to stderr and exits non-zero; that's expected.
    devices: list[tuple[int, str]] = []
    in_video = False
    for line in proc.stderr.splitlines():
        low = line.lower()
        if "avfoundation video devices" in low:
            in_video = True
            continue
        if "avfoundation audio devices" in low:
            in_video = False
            continue
        if not in_video:
            continue
        match = _DEVICE_LINE.search(line)
        if match:
            devices.append((int(match.group(1)), match.group(2).strip()))
    return devices


def find_device_index(
    name_match: str = DEFAULT_NAME_MATCH,
    name_exclude: str = DEFAULT_NAME_EXCLUDE,
) -> int:
    """Find the AVFoundation video index whose name contains `name_match`."""
    devices = list_video_devices()
    if not devices:
        raise CameraError(
            "no AVFoundation video devices found (is the camera connected / "
            "Terminal granted Camera permission in System Settings > Privacy?)"
        )
    name_match = name_match.lower()
    name_exclude = (name_exclude or "").lower()
    for index, name in devices:
        low = name.lower()
        if name_match in low and (not name_exclude or name_exclude not in low):
            return index
    available = ", ".join(f"[{i}] {n}" for i, n in devices)
    raise CameraError(
        f"no video device matching '{name_match}' found. Available: {available}"
    )


class RealCamera:
    """On-demand still capture from a macOS AVFoundation camera via ffmpeg.

    Each `capture()` spawns a short ffmpeg run that grabs `warmup` frames and
    keeps the last one (so auto-exposure has settled), optionally downscaled to
    `width`. Returns PNG bytes. There is no persistent capture handle, which
    keeps Continuity Camera happy and avoids holding the device between steps.
    """

    def __init__(
        self,
        index: int | None = None,
        *,
        name_match: str = DEFAULT_NAME_MATCH,
        name_exclude: str = DEFAULT_NAME_EXCLUDE,
        width: int | None = DEFAULT_WIDTH,
        warmup: int = DEFAULT_WARMUP,
        framerate: int = 30,
    ) -> None:
        self.index = index if index is not None else find_device_index(name_match, name_exclude)
        self.width = width
        self.warmup = max(1, warmup)
        self.framerate = framerate

    def capture(self, out_path: str | Path | None = None) -> bytes:
        """Grab one frame; return PNG bytes (and also write to out_path if given)."""
        target = Path(out_path) if out_path is not None else None
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_png = Path(tmp) / "frame.png"
            cmd = [
                FFMPEG, "-hide_banner", "-loglevel", "error",
                "-f", "avfoundation",
                "-framerate", str(self.framerate),
                "-i", str(self.index),
                "-frames:v", str(self.warmup),
                "-update", "1",
            ]
            if self.width:
                # scale to width, keep aspect, force even height (-2)
                cmd += ["-vf", f"scale={int(self.width)}:-2"]
            cmd += ["-y", str(tmp_png)]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if not tmp_png.exists() or tmp_png.stat().st_size == 0:
                raise CameraError(
                    f"ffmpeg capture failed (device index {self.index}).\n"
                    f"exit={proc.returncode}\nstderr:\n{proc.stderr.strip()[-800:]}"
                )
            data = tmp_png.read_bytes()

        if target is not None:
            target.write_bytes(data)
        return data


def _probe() -> int:
    devices = list_video_devices()
    if not devices:
        print("No AVFoundation video devices found.", file=sys.stderr)
        return 1
    print("AVFoundation video devices:")
    for index, name in devices:
        print(f"  [{index}] {name}")
    try:
        idx = find_device_index()
        print(f"\n-> iPhone auto-selected: index {idx}")
    except CameraError as exc:
        print(f"\n(iPhone auto-select: {exc})")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture a frame from the iPhone (or another) camera.")
    p.add_argument("--probe", action="store_true", help="list video devices and exit")
    p.add_argument("--capture", metavar="PATH", help="grab one frame and save it to PATH")
    p.add_argument("--index", type=int, default=None, help="force AVFoundation device index")
    p.add_argument("--name", default=DEFAULT_NAME_MATCH, help="substring to match the device name")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="output width in px (0 = native)")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="warmup frames to discard")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.probe or not args.capture:
        return _probe()
    cam = RealCamera(
        index=args.index,
        name_match=args.name,
        width=(args.width or None),
        warmup=args.warmup,
    )
    data = cam.capture(args.capture)
    print(f"captured {len(data)} bytes from device index {cam.index} -> {args.capture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
