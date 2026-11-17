"""Guided recording: BASELINE, IDLE, STAND, WALK.

    python3 -m scripts.record_hop --room <name> --baseline-min 3 --activity-min 5 --auto IDLE
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "acquisition"))

from csi_serial_logger import run_session as run_csi   # noqa: E402
from record_video import record_session as run_video   # noqa: E402

STAGES = [
    ("BASELINE", "EMPTY ROOM — leave the room and close the door.",
     "Nobody inside. This becomes the reference for what the room looks like "
     "with no one in it."),
    ("IDLE", "EMPTY ROOM AGAIN — stay outside.",
     "Same as the baseline, recorded separately so the two share no packets."),
    ("STAND", "STAND STILL inside the room, in the signal path.",
     "Stand between the sensor and where the signal travels. Keep still."),
    ("WALK", "WALK around the room, continuously.",
     "Normal pace, back and forth. Do not leave the room."),
]


def say(msg=""):
    print(msg, flush=True)


def rule(ch="="):
    say(ch * 70)


def preflight(port, camera_index, use_camera):
    """Check the serial port, CSI and camera before recording."""
    rule()
    say("PRE-FLIGHT")
    rule()

    import serial
    say(f"  serial port {port} ...")
    try:
        ser = serial.Serial(port, 921600, timeout=0.5)
    except Exception as exc:
        sys.exit(f"  FAILED: {exc}\n  Is the ESP32 plugged in? Is another program "
                 f"holding the port?\n  Only ONE process may read it at a time.")
    say("    opened — the board has just reset, waiting for CSI to resume ...")

    seen, deadline = 0, time.time() + 20
    while time.time() < deadline and seen < 20:
        try:
            line = ser.readline().decode("utf-8", "ignore")
        except Exception:
            break
        if line.startswith("CSI_DATA,"):
            seen += 1
    ser.close()
    if seen < 20:
        sys.exit(f"  FAILED: only {seen} CSI lines in 20 s.\n"
                 f"  The board may still be joining the network, or no client is\n"
                 f"  connected to it. Check the serial monitor before retrying.")
    say(f"    CSI flowing — {seen} packets read")

    if use_camera:
        import cv2
        say(f"  camera index {camera_index} ...")
        cap = cv2.VideoCapture(camera_index)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if not ok:
            sys.exit("  FAILED: camera did not open.\n"
                     "  On macOS: System Settings -> Privacy & Security -> Camera,\n"
                     "  and allow the terminal app you are running this from.\n"
                     "  Run with --no-camera to record CSI only.")
        say("    camera OK")
    else:
        say("  camera: SKIPPED (--no-camera)")
    say("  pre-flight passed.")
    say()


def record_stage(label, duration, csi_dir, video_dir, port, camera_index, use_camera):
    csi_meta, video_meta = {}, {}
    barrier = threading.Barrier(2 if use_camera else 1)

    def csi_worker():
        barrier.wait()
        csi_meta.update(run_csi(label=label, duration=duration,
                                output_dir=str(csi_dir), port=port, live=True) or {})

    def video_worker():
        barrier.wait()
        video_meta.update(run_video(label=label, duration=duration,
                                    output_dir=str(video_dir),
                                    camera_index=camera_index) or {})

    threads = [threading.Thread(target=csi_worker)]
    if use_camera:
        threads.append(threading.Thread(target=video_worker))
    for t in threads:
        t.start()
    last = -10
    end = time.time() + duration
    while any(t.is_alive() for t in threads):
        left = end - time.time()
        if time.time() - last >= 15 and left > 0:
            say(f"    ... {left/60:4.1f} min remaining")
            last = time.time()
        time.sleep(0.5)
    for t in threads:
        t.join()
    return csi_meta, video_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbserial-0001",
                    help="RX board serial port (default /dev/cu.usbserial-0001)")
    ap.add_argument("--room", required=True, help="a tag for this room, e.g. hostel_room")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--baseline-min", type=float, default=3.0)
    ap.add_argument("--activity-min", type=float, default=10.0)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--countdown", type=int, default=10,
                    help="seconds between pressing ENTER and recording starting, "
                         "so you can leave the room and close the door (default 10)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="record only these stages, e.g. --only WALK")
    ap.add_argument("--auto", nargs="*", default=[],
                    help="stages to start automatically with no ENTER prompt, "
                         "e.g. --auto IDLE lets BASELINE flow straight into IDLE "
                         "while you stay out of the room")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    use_camera = not args.no_camera
    preflight(args.port, args.camera_index, use_camera)
    if args.preflight_only:
        say("Pre-flight only — nothing recorded.")
        return

    stages = [s for s in STAGES if args.only is None or s[0] in args.only]
    total = sum(args.baseline_min if s[0] == "BASELINE" else args.activity_min
                for s in stages)

    session = f"{args.room}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = ROOT / "data" / "raw" / session
    csi_dir, video_dir = out / "csi", out / "video"
    for d in (csi_dir, video_dir):
        d.mkdir(parents=True, exist_ok=True)

    rule()
    say(f"SESSION {session}")
    rule()
    say(f"  room          : {args.room}")
    say(f"  stages        : {', '.join(s[0] for s in stages)}")
    say(f"  camera        : {'on' if use_camera else 'OFF'}")
    say(f"  total time    : about {total:.0f} minutes of recording")
    say(f"  saving to     : {out}")
    say()
    say("  Keep the door closed and the room otherwise undisturbed for the")
    say("  whole session. Do not move the ESP32 or the camera between stages —")
    say("  the baseline is only valid while the geometry stays put.")
    say()

    manifest = {"session": session, "room": args.room, "stages": [],
                "camera": use_camera, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    for name, headline, detail in stages:
        minutes = args.baseline_min if name == "BASELINE" else args.activity_min
        rule()
        say(f"STAGE: {name}   ({minutes:.0f} minutes)")
        rule()
        say(f"  {headline}")
        say(f"  {detail}")
        say()
        if name in args.auto:
            say(f"  [AUTO] {name} starts on its own — no ENTER needed. Stay put.")
        else:
            input(f"  Press ENTER, then get into position for {name} ... ")
        for i in range(args.countdown, 0, -1):
            if i <= 5 or i % 5 == 0:
                say(f"    starting in {i:2d}")
            time.sleep(1)
        say(f"  >>> RECORDING {name} <<<")

        csi_meta, video_meta = record_stage(
            name, minutes * 60.0, csi_dir, video_dir, args.port,
            args.camera_index, use_camera)

        packets = csi_meta.get("csi_packets", 0)
        frames = video_meta.get("frame_count", 0)
        rate = csi_meta.get("csi_rate_host_span_hz", 0)
        say(f"  done: {packets} packets ({rate} Hz)"
            + (f", {frames} video frames" if use_camera else ""))
        if packets == 0:
            say("  !!! ZERO CSI PACKETS — stop and check the ESP32 before continuing.")
        elif packets < minutes * 60 * 50:
            say("  !!! Packet rate looks low. Expected roughly 110 per second.")
        if use_camera and frames == 0:
            say("  !!! ZERO VIDEO FRAMES — check camera permission.")
        manifest["stages"].append({"label": name, "minutes": minutes,
                                   "csi": csi_meta, "video": video_meta})
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        say()

    rule()
    say("SESSION COMPLETE")
    rule()
    say(f"  {out}")
    say()
    say("  Next:")
    say(f"    python3 -m scripts.prepare --room {args.room} --empty ... --idle ... --stand ... --walk ...")
    say(f"    python3 -m scripts.label --features artifacts/features_{args.room}.csv --session {session}")
    say(f"    python3 -m scripts.train --features artifacts/features_{args.room}.csv --save artifacts/model_{args.room}.joblib")


if __name__ == "__main__":
    main()
