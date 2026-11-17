"""Webcam recorder timestamped with the same clock as the CSI logger."""

import argparse
import csv
import time
from pathlib import Path

import cv2


def record_session(label, duration, output_dir, camera_index=0, target_fps=30, stop_event=None):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = label.upper()
    session_id = f"{safe_label}_{stamp}"
    video_path = out_dir / f"{session_id}.mp4"
    timestamps_path = out_dir / f"{session_id}.frame_times.csv"

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera. On macOS, check System Settings -> Privacy & "
            "Security -> Camera and allow access for the app running this script."
        )

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Camera opened but failed to read a frame.")
    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, target_fps, (width, height))

    frame_count = 0
    start = time.time()
    with timestamps_path.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["frame_index", "host_time_ms", "host_time_ns", "label", "session_id"])
        while time.time() - start < duration:
            if stop_event is not None and stop_event.is_set():
                break
            ok, frame = cap.read()
            if not ok:
                continue
            now_ns = time.time_ns()
            writer.write(frame)
            csv_writer.writerow([frame_count, now_ns // 1_000_000, now_ns, safe_label, session_id])
            frame_count += 1

    cap.release()
    writer.release()

    elapsed = time.time() - start
    meta = {
        "session_id": session_id,
        "label": safe_label,
        "frame_count": frame_count,
        "elapsed_sec": round(elapsed, 3),
        "avg_fps": round(frame_count / elapsed, 2) if elapsed > 0 else 0.0,
        "video_path": str(video_path),
        "timestamps_path": str(timestamps_path),
    }
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="UNKNOWN")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", default="data/video")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    meta = record_session(args.label, args.duration, args.output_dir, args.camera_index)
    print(meta)


if __name__ == "__main__":
    main()
