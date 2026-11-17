"""Build the empty-room baseline and the feature table.

    python3 -m scripts.prepare --room <name> --empty B.csv --idle I.csv --stand S.csv --walk W.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core import ARTIFACTS
from core import baseline as bl
from core.features import (CROP_PACKETS, META_COLS, WINDOW_SECONDS, extract)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True, help="tag identifying the physical room")
    ap.add_argument("--empty", required=True, help="recording used to build the baseline")
    ap.add_argument("--idle", required=True)
    ap.add_argument("--stand", required=True)
    ap.add_argument("--walk", required=True)
    ap.add_argument("--mode", choices=["physics", "per_subcarrier", "aggregate"],
                    default="physics")
    ap.add_argument("--no-subtract", action="store_true",
                    help="A/B control: identical windows, baseline removal OFF")
    ap.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS,
                    help="window DURATION in seconds (default 2.0). Fixed time, "
                         "so FFT bin spacing is 1/window-seconds in every window")
    ap.add_argument("--stride-seconds", type=float, default=None,
                    help="hop between windows (default: half the window)")
    ap.add_argument("--window-packets", type=int, default=None,
                    help="CONTROL: use the legacy fixed-packet-count windows "
                         "instead of fixed duration, for comparison only")
    ap.add_argument("--crop", type=int, default=CROP_PACKETS,
                    help="packets discarded from each end (walk-in / walk-out). "
                         "Default 3300 assumes ~110 packets/s. At a lower passive "
                         "rate, scale it down: crop = rate * 30")
    ap.add_argument("--baseline-packets", type=int, default=None,
                    help="packets used to build the baseline (default 13000)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--audit", default=None,
                    help="audit csv from scripts.label — flagged windows are dropped")
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sub = not args.no_subtract

    print("=" * 74)
    print("1 · EMPTY-ROOM BASELINE")
    print("=" * 74)
    kw = {"crop": args.crop}
    if args.baseline_packets:
        kw["n_packets"] = args.baseline_packets
    base = bl.build(args.empty, room=args.room, **kw)
    bl.save(base, ARTIFACTS / f"baseline_{args.room}.npz")
    print("  " + bl.describe(base))

    same_file = Path(args.empty).resolve() == Path(args.idle).resolve()
    idle_skip = base["first_unused_index"] if same_file else args.crop
    if same_file:
        print(f"  baseline and idle share a file -> idle class starts at "
              f"packet {idle_skip}")

    print()
    print("=" * 74)
    win = (f"{args.window_packets} packets/window — LEGACY CONTROL"
           if args.window_packets else f"{args.window_seconds:g}s windows")
    print(f"2 · FEATURES   ({args.mode}, {win}, background subtraction "
          f"{'ON' if sub else 'OFF — A/B control'})")
    print("=" * 74)

    blocks, meta_all, cols = [], [], None
    for path, label, skip in ((args.idle, "IDLE", idle_skip),
                              (args.stand, "STAND", args.crop),
                              (args.walk, "WALK", args.crop)):
        X, cols, meta = extract(path, label, base, args.room, skip=skip,
                                drop_last=args.crop,
                                window_seconds=args.window_seconds,
                                stride_seconds=args.stride_seconds,
                                window_packets=args.window_packets,
                                mode=args.mode, subtract=sub)
        blocks.append(X)
        meta_all += meta

    X = np.vstack(blocks)
    df = pd.DataFrame(X, columns=cols)
    for k in META_COLS:
        df[k] = [m[k] for m in meta_all]

    if args.audit:
        a = pd.read_csv(args.audit)
        bad = {(r.label, int(r.window_index)) for r in a.itertuples() if not r.keep}
        before = len(df)
        df = df[[(l, int(w)) not in bad
                 for l, w in zip(df.label, df.window_index)]].reset_index(drop=True)
        print(f"\n  camera audit: {before} -> {len(df)} windows "
              f"({before - len(df)} dropped)")

    suffix = {"physics": "", "per_subcarrier": "_persc", "aggregate": "_agg"}[args.mode]
    name = args.out or (f"features_{args.room}"
                        f"{'' if sub else '_nosub'}{suffix}.csv")
    out = ARTIFACTS / name
    df.to_csv(out, index=False)

    n_nan = int(df[cols].isna().sum().sum())
    n_inf = int(np.isinf(df[cols].to_numpy()).sum())
    print(f"\n  {len(df)} windows x {len(cols)} features -> {out}")
    print(f"  per class: " + " · ".join(
        f"{k} {v}" for k, v in df.label.value_counts().sort_index().items()))
    print(f"  NaN {n_nan}   Inf {n_inf}")
    if n_nan or n_inf:
        sys.exit("!!! non-finite values present — do not train on this !!!")


if __name__ == "__main__":
    main()
