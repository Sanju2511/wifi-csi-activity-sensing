"""Empty-room baseline (per-subcarrier mean and std) and z-score calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .csi import load_recording

CROP_PACKETS = 3300         # ~30 s at each end, when I walk in/out to start and stop
BASELINE_PACKETS = 13000    # ~2 min of empty room
NULL_STD_THRESHOLD = 1e-6


def build(csi_path, room, crop=CROP_PACKETS, n_packets=BASELINE_PACKETS):
    """Per-subcarrier ruler and wobble from an empty-room recording."""
    amps, _pd, rssis, _t, n_corrupt = load_recording(csi_path, skip=crop, limit=n_packets)
    if len(amps) < 200:
        raise SystemExit(f"Only {len(amps)} usable packets — far too few for a baseline.")

    wob = amps.std(axis=0)
    return {
        "avg": amps.mean(axis=0),
        "wob": wob,
        "null_mask": wob < NULL_STD_THRESHOLD,
        "rssi_mean": float(rssis.mean()),
        "rssi_std": float(rssis.std()),
        "room": str(room),
        "n_packets": int(len(amps)),
        "n_corrupt": int(n_corrupt),
        "crop": int(crop),
        "source": str(csi_path),
        # IDLE data must not overlap the packets used for the baseline
        "first_unused_index": int(crop + len(amps)),
    }


def calibrate(amps, base):
    """(amp - avg) / wob, per subcarrier, nulls forced to 0."""
    denom = base["wob"].copy()
    denom[denom < NULL_STD_THRESHOLD] = 1.0
    out = (amps - base["avg"]) / denom
    out[:, base["null_mask"]] = 0.0
    return out


def active(amps, base):
    """Drop the null subcarriers, leaving only slots that carry energy."""
    return amps[:, ~base["null_mask"]]


def n_active(base):
    return int((~base["null_mask"]).sum())


def save(base, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **base)
    return p


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def describe(base):
    m = ~base["null_mask"]
    return (f"room {base['room']} · {base['n_packets']} packets · "
            f"{int(base['null_mask'].sum())}/64 null, {int(m.sum())} active · "
            f"wobble {base['wob'][m].mean():.2f} · rssi {base['rssi_mean']:.2f} dBm · "
            f"class data may start at packet {base['first_unused_index']}")
