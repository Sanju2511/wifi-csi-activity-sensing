"""Check window labels against the video with YOLO (the camera is never a model input).

    python3 -m scripts.label --features artifacts/features_<name>.csv --session <session>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from core import ARTIFACTS, ROOT

FRAMES_PER_WINDOW = 10
SEEN = 0.4          # "a person is visible in this window"
MOVING = 0.05       # box_cx_std above this = the person is travelling


def yolo_window(model, cap, frame_indices):
    """Pool YOLO over the given frames into 9 numbers."""
    cx, cy, area, conf, seen = [], [], [], [], []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = model.predict(frame, classes=[0], verbose=False)[0]   # 0 = person
        if len(res.boxes) == 0:
            seen.append(0)
            continue
        best = max(res.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = [float(v) for v in best.xyxy[0]]
        cx.append((x1 + x2) / 2 / w)
        cy.append((y1 + y2) / 2 / h)
        area.append((x2 - x1) * (y2 - y1) / (w * h))
        conf.append(float(best.conf[0]))
        seen.append(1)

    out = {"person_detected_frac": float(np.mean(seen)) if seen else 0.0}
    for name, vals in (("box_cx", cx), ("box_cy", cy), ("box_area", area), ("conf", conf)):
        out[f"{name}_mean"] = float(np.mean(vals)) if vals else 0.0
        out[f"{name}_std"] = float(np.std(vals)) if vals else 0.0
    return out


def verdict(label, f):
    """Does the camera agree with the label? Returns (keep, reason)."""
    frac, moved = f["person_detected_frac"], f["box_cx_std"]
    if label in ("BASELINE", "IDLE"):
        if frac >= SEEN:
            return False, f"room should be empty, person seen in {frac:.0%} of frames"
    elif label == "STAND":
        if frac <= SEEN:
            return False, f"should be standing here, but only seen in {frac:.0%} of frames"
        if moved > MOVING:
            return False, f"labelled STAND but the person travelled (box_cx_std {moved:.3f})"
    elif label == "WALK":
        if frac >= 0.5 and moved <= MOVING:
            return False, f"labelled WALK but the person stood still (box_cx_std {moved:.3f})"
    return True, ""


def find_video(video_dir, label):
    vids = sorted(Path(video_dir).glob(f"{label}_*.mp4"))
    if not vids:
        return None, None
    v = vids[0]
    ft = v.with_suffix("").with_suffix("")
    ft = v.parent / (v.stem + ".frame_times.csv")
    return (v, ft) if ft.exists() else (v, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None, help="features csv with t_start_ns/t_end_ns")
    ap.add_argument("--session", default=None, help="folder under data/raw/")
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--frames", type=int, default=FRAMES_PER_WINDOW)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="check only the first N windows per label (quick test)")
    args = ap.parse_args()

    if args.session:
        video_dir = ROOT / "data" / "raw" / args.session / "video"
    elif args.video_dir:
        video_dir = Path(args.video_dir)
    else:
        sys.exit("Pass --session or --video-dir.")
    if not video_dir.exists():
        sys.exit(f"No such video directory: {video_dir}")

    fpath = Path(args.features) if args.features else None
    if fpath is None:
        found = [f for f in sorted(ARTIFACTS.glob("features_*.csv"))
                 if "nosub" not in f.name and "_agg" not in f.name]
        if not found:
            sys.exit("No features found — run scripts.prepare first.")
        fpath = found[0]
    df = pd.read_csv(fpath)
    if "t_start_ns" not in df.columns:
        sys.exit("This features file has no timestamps — re-run scripts.prepare.")

    from ultralytics import YOLO
    model = YOLO(str(ROOT.parent / "models" / "yolov8n.pt"))

    print(f"features {fpath.name}   video {video_dir}")
    rows = []
    for label, g in df.groupby("label"):
        video, ftimes = find_video(video_dir, label)
        if video is None or ftimes is None:
            print(f"  {label:9s} no video found — skipped")
            continue
        ft = pd.read_csv(ftimes)
        t = ft["host_time_ns"].to_numpy(dtype=np.float64)
        cap = cv2.VideoCapture(str(video))
        print(f"  {label:9s} {video.name}  ({len(ft)} frames)")

        kept = dropped = novideo = 0
        gg = g.sort_values("window_index")
        if args.max_windows:
            gg = gg.head(args.max_windows)
        for _, w in gg.iterrows():
            inside = np.where((t >= w.t_start_ns) & (t <= w.t_end_ns))[0]
            if len(inside) < args.frames:
                novideo += 1
                rows.append({"label": label, "window_index": int(w.window_index),
                             "keep": False, "reason": f"only {len(inside)} frames in span",
                             "n_frames_available": len(inside)})
                continue
            pick = inside[np.linspace(0, len(inside) - 1, args.frames).astype(int)]
            f = yolo_window(model, cap, pick)
            ok, why = verdict(label, f)
            kept += ok
            dropped += (not ok)
            rows.append({"label": label, "window_index": int(w.window_index),
                         "keep": ok, "reason": why,
                         "n_frames_available": len(inside), **f})
        cap.release()
        print(f"            kept {kept}, flagged {dropped}, no video {novideo}")

    out = Path(args.out) if args.out else ARTIFACTS / f"audit_{fpath.stem}.csv"
    a = pd.DataFrame(rows)
    a.to_csv(out, index=False)

    print()
    print("=" * 74)
    print("AUDIT")
    print("=" * 74)
    print(f"  windows checked : {len(a)}")
    print(f"  kept            : {int(a.keep.sum())}")
    print(f"  flagged         : {int((~a.keep).sum())}")
    bad = a[~a.keep]
    if len(bad):
        print("\n  flagged windows:")
        for _, r in bad.head(25).iterrows():
            print(f"    {r.label:9s} #{int(r.window_index):<4d} {r.reason}")
        if len(bad) > 25:
            print(f"    ... and {len(bad) - 25} more")
    if "box_cx_std" in a.columns:
        print("\n  box_cx_std by label (how much the person moved sideways):")
        for k, v in a.groupby("label")["box_cx_std"].mean().items():
            print(f"    {k:9s} {v:.4f}")
    print(f"\n  saved -> {out}")
    print("  pass it to scripts.prepare with --audit to drop the flagged windows.")


if __name__ == "__main__":
    main()
