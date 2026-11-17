"""Train the agent and evaluate on the last 25% of each recording.

    python3 -m scripts.train --features artifacts/features_<name>.csv --save <model.joblib>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from core import ARTIFACTS
from core.brl import ACTIONS, ContinuousBayesianRL, calibration
from core.features import META_COLS, block_split, room_split


def report(model, te, feature_cols, title):
    r = model.evaluate(te)
    # decide() reports an action index, or None when it will not commit
    pred_lab = np.array(["UNCERTAIN" if p is None else ACTIONS[p] for p in r["pred"]])
    answered = pred_lab != "UNCERTAIN"
    truth_lab = np.array([ACTIONS[i] for i in r["truth"]])
    n = len(truth_lab)

    acc_answered = float((pred_lab[answered] == truth_lab[answered]).mean()) if answered.any() else float("nan")
    acc_allin = float((pred_lab == truth_lab).mean())
    acc_forced = float((np.array([ACTIONS[g] for g in r["guess"]]) == truth_lab).mean())

    print()
    print("=" * 74)
    print(title)
    print("=" * 74)
    print(f"windows                        : {n}")
    print(f"answered / abstained           : {int(answered.sum())} / {int((~answered).sum())}")
    print(f"accuracy when it answered      : {acc_answered:.3f}")
    print(f"accuracy, abstain = wrong      : {acc_allin:.3f}   <- headline figure")
    print(f"accuracy if forced to answer   : {acc_forced:.3f}")

    idx = {a: i for i, a in enumerate(ACTIONS)}
    cm = np.zeros((len(ACTIONS), len(ACTIONS) + 1), dtype=int)
    for t, p in zip(truth_lab, pred_lab):
        cm[idx[t], len(ACTIONS) if p == "UNCERTAIN" else idx[p]] += 1
    print(f"\nconfusion (rows true, cols predicted + abstain) {ACTIONS}")
    print(cm)
    with np.errstate(invalid="ignore"):
        rec = np.array([cm[i, i] / cm[i].sum() if cm[i].sum() else np.nan
                        for i in range(len(ACTIONS))])
    print("recall: " + "  ".join(f"{a}={v:.2f}" for a, v in zip(ACTIONS, rec)))

    ok = np.array([ACTIONS[g] for g in r["guess"]]) == truth_lab
    bands, ece = calibration(r["prob"], ok)
    print(f"\nexpected calibration error : {ece:.4f}  (0 = perfect)")
    print(f"mean credible interval width: {np.mean(r['hi'] - r['lo']):.3f}")
    return dict(answered=acc_answered, all_in=acc_allin, forced=acc_forced,
                n_abstain=int((~answered).sum()), ece=float(ece))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None)
    ap.add_argument("--pca", type=int, default=12, help="0 disables the projection")
    ap.add_argument("--passes", type=int, default=40)
    ap.add_argument("--no-memory", action="store_true",
                    help="remove the previous-decision slots from the state vector")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--test-room", default=None,
                    help="hold out an entire room instead of the tail of each recording")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    path = Path(args.features) if args.features else None
    if path is None:
        found = sorted(ARTIFACTS.glob("features_*.csv"))
        found = [f for f in found if "nosub" not in f.name and "_agg" not in f.name]
        if not found:
            sys.exit("No features found — run scripts.prepare first.")
        path = found[0]
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in META_COLS]

    if args.test_room:
        tr, te = room_split(df, args.test_room)
        title = f"HELD-OUT ROOM: {args.test_room}  (trained on every other room)"
        if len(tr) == 0 or len(te) == 0:
            sys.exit(f"room split gave {len(tr)} train / {len(te)} test — check --test-room")
    else:
        tr, te = block_split(df, args.test_frac)
        title = "HELD-OUT BLOCK (last 25% of each recording, never trained on)"

    print(f"features {path.name}")
    print(f"train {len(tr)} / held-out {len(te)} windows · {len(feature_cols)} features")
    print(f"PCA {args.pca or 'off'} · "
          f"{args.passes} passes")
    print()

    model = ContinuousBayesianRL(memory=not args.no_memory,
                                   n_pca=args.pca or None,
                                   n_passes=args.passes)
    model.fit(tr, feature_cols)
    stats = report(model, te, feature_cols, title)

    out = Path(args.save) if args.save else ARTIFACTS / "model.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols,
                 "features_file": str(path), "stats": stats}, out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
