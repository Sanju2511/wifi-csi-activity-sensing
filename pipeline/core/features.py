"""Windowing (2 s windows, 1 s stride) and the 37 physics features."""

from __future__ import annotations

import numpy as np
from scipy import fft as _fft
from scipy import stats as _st

from .baseline import CROP_PACKETS, active, calibrate
from .csi import load_recording

WINDOW_SECONDS = 2.0                   # 0.5 Hz FFT bin spacing
STRIDE_SECONDS = WINDOW_SECONDS / 2    # 50% overlap

MIN_WINDOW_FILL = 0.5

WINDOW_PACKETS = 200                   # packet-count windows, only used for comparison
STRIDE_PACKETS = WINDOW_PACKETS // 2

# window metadata, not model inputs
META_COLS = ["label", "room", "window_index", "t_start_ns", "t_end_ns"]

PER_SC_STATS = ["std", "mad", "range", "mean"]


TEMPORAL_STATS = ["mean", "var", "skew", "kurt", "range", "iqr",
                  "dom_freq", "motion_pw", "n_change", "max_step"]
SHAPE_STATS = ["mean", "sd", "max", "p90", "p50", "p10"]
SHAPE_SOURCES = ["std", "mad", "range"]
SERIES = ["energy", "spread"]

EXCLUDED_LEVEL_COLS = {"energy_mean"}

PHYSICS_COLS = [
    c for c in (
        [f"{sr}_{st}" for sr in SERIES for st in TEMPORAL_STATS]
        + [f"{sc}_{st}" for sc in SHAPE_SOURCES for st in SHAPE_STATS]
    ) if c not in EXCLUDED_LEVEL_COLS
]

_ALL_TEMPORAL_COLS = [f"{sr}_{st}" for sr in SERIES for st in TEMPORAL_STATS]
_KEEP_MASK = np.array([c not in EXCLUDED_LEVEL_COLS for c in
                       _ALL_TEMPORAL_COLS
                       + [f"{sc}_{st}" for sc in SHAPE_SOURCES for st in SHAPE_STATS]])

assert not any("rssi" in c for c in PHYSICS_COLS), "signal-strength feature leaked in"
assert len(PHYSICS_COLS) == int(_KEEP_MASK.sum())


def temporal_features(x, fs):
    """Ten time- and frequency-domain statistics of one time series sampled at fs Hz."""
    x = np.asarray(x, dtype=float)
    spec = np.abs(_fft.rfft(x - x.mean())) ** 2
    freq = _fft.rfftfreq(len(x), d=1.0 / fs)

    def band(lo, hi):
        return float(spec[(freq >= lo) & (freq <= hi)].sum())

    dom = float(freq[1:][np.argmax(spec[1:])]) if len(spec) > 1 else 0.0

    # CUSUM change points: how many times the level shifted decisively
    thr = 3.0 * x.std()
    mu = x.mean()
    pos = neg = 0.0
    changes = 0
    for i in range(1, len(x)):
        pos = max(0.0, pos + x[i] - mu)
        neg = min(0.0, neg + x[i] - mu)
        if pos > thr or neg < -thr:
            changes += 1
            pos = neg = 0.0

    return [
        float(x.mean()), float(x.var()), float(_st.skew(x)), float(_st.kurtosis(x)),
        float(np.ptp(x)), float(np.subtract(*np.percentile(x, [75, 25]))),
        dom, band(0.5, 3.0),
        float(changes), float(np.abs(np.diff(x)).max()),
    ]


def shape_features(v):
    """Order statistics of a profile across subcarriers."""
    v = np.sort(np.asarray(v, dtype=float))
    n = len(v)
    return [float(v.mean()), float(v.std()), float(v[-1]),
            float(v[int(0.9 * n)]), float(v[int(0.5 * n)]), float(v[int(0.1 * n)])]


def physics_features(cal_amp, fs):
    """One window -> the 37 default features. `fs` is that window's own rate."""
    energy = cal_amp.mean(axis=1)     # average level across subcarriers, per packet
    spread = cal_amp.std(axis=1)      # how uneven the spectrum is, per packet

    sd = cal_amp.std(axis=0)
    mad = (np.abs(np.diff(cal_amp, axis=0)).mean(axis=0)
           if cal_amp.shape[0] > 1 else np.zeros(cal_amp.shape[1]))
    rng = cal_amp.max(axis=0) - cal_amp.min(axis=0)

    v = np.array(
        temporal_features(energy, fs) + temporal_features(spread, fs)
        + shape_features(sd) + shape_features(mad) + shape_features(rng),
        dtype=float,
    )
    # energy_mean is dropped (it is just a signal-level reading)
    return v[_KEEP_MASK]


def per_subcarrier_names(n_sc):
    return [f"{s}_sc{i:02d}" for s in PER_SC_STATS for i in range(n_sc)]


def per_subcarrier_features(cal_amp):
    """(W, n_active) calibrated window -> (4 * n_active,) feature vector."""
    if cal_amp.shape[0] > 1:
        mad = np.abs(np.diff(cal_amp, axis=0)).mean(axis=0)
    else:
        mad = np.zeros(cal_amp.shape[1])
    return np.concatenate([
        cal_amp.std(axis=0),
        mad,
        cal_amp.max(axis=0) - cal_amp.min(axis=0),
        cal_amp.mean(axis=0),
    ])


AGGREGATE_COLS = [
    "amp_mean", "amp_std", "amp_peak", "amp_range", "subcarrier_spread",
    "mad_mean", "mad_std", "mad_max", "jerk_mean",
    "phase_diff_std", "rssi_delta", "rssi_std",
]


def aggregate_features(cal_amp, phase_diff, rssi, base):
    """Pooled statistics, used only for comparison."""
    flat = cal_amp.ravel()
    if cal_amp.shape[0] > 2:
        d1 = np.abs(np.diff(cal_amp, axis=0))
        mad = d1.mean(axis=0)
        jerk = float(np.abs(np.diff(cal_amp, n=2, axis=0)).mean())
        mad_mean, mad_std, mad_max = float(mad.mean()), float(mad.std()), float(mad.max())
    else:
        mad_mean = mad_std = mad_max = jerk = 0.0
    return np.array([
        flat.mean(), flat.std(), flat.max(), flat.max() - flat.min(),
        cal_amp.mean(axis=0).std(),
        mad_mean, mad_std, mad_max, jerk,
        phase_diff.std(),
        rssi.mean() - base["rssi_mean"], rssi.std(),
    ], dtype=float)


def time_windows(times, window_seconds, stride_seconds, min_fill=MIN_WINDOW_FILL):
    """Yield (i0, i1, t0_ns, t1_ns) index spans of fixed DURATION."""
    t = np.asarray(times, dtype=float)
    if len(t) < 2:
        return

    span_ns = window_seconds * 1e9
    step_ns = stride_seconds * 1e9

    gap_ns = float(np.median(np.diff(t)))
    expected = span_ns / gap_ns if gap_ns > 0 else 0.0
    floor = max(2, int(min_fill * expected))

    start_ns, end_ns = t[0], t[-1]
    n_steps = int((end_ns - start_ns - span_ns) // step_ns) + 1
    for k in range(max(0, n_steps)):
        w0 = start_ns + k * step_ns
        i0, i1 = np.searchsorted(t, [w0, w0 + span_ns])
        if i1 - i0 < floor:
            continue
        yield int(i0), int(i1), float(t[i0]), float(t[i1 - 1])


def extract(csi_path, label, base, room, skip=0, drop_last=CROP_PACKETS,
            window_seconds=WINDOW_SECONDS, stride_seconds=None,
            window_packets=None, mode="physics",
            subtract=True, verbose=True, stride=None):
    """One recording -> (feature matrix, column names, metadata rows)."""
    amps, pds, rssis, times, n_corrupt = load_recording(csi_path, skip=skip,
                                                        drop_last=drop_last)

    cal = calibrate(amps, base) if subtract else amps
    cal = active(cal, base)

    if window_packets:
        if len(amps) < window_packets:
            raise SystemExit(f"{csi_path}: only {len(amps)} packets after cropping.")
        stride = stride or max(1, window_packets // 2)
        spans = [(w0, w0 + window_packets,
                  float(times[w0]), float(times[w0 + window_packets - 1]))
                 for w0 in range(0, len(cal) - window_packets + 1, stride)]
    else:
        stride_seconds = stride_seconds or window_seconds / 2
        spans = list(time_windows(times, window_seconds, stride_seconds))
        if not spans:
            raise SystemExit(
                f"{csi_path}: no usable {window_seconds}s windows after cropping "
                f"({len(amps)} packets, spanning "
                f"{(float(times[-1]) - float(times[0])) / 1e9:.1f}s).")

    rows, meta = [], []
    for w, (i0, i1, t0, t1) in enumerate(spans):
        s = slice(i0, i1)
        if mode == "physics":
            span_s = (window_seconds if not window_packets
                      else max((t1 - t0) / 1e9, 1e-9))
            fs = (i1 - i0) / span_s
            rows.append(physics_features(cal[s], fs))
        elif mode == "per_subcarrier":
            rows.append(per_subcarrier_features(cal[s]))
        else:
            rows.append(aggregate_features(cal[s], pds[s], rssis[s], base))
        meta.append({"label": label, "room": room, "window_index": w,
                     "t_start_ns": t0, "t_end_ns": t1})

    cols = (PHYSICS_COLS if mode == "physics"
            else per_subcarrier_names(cal.shape[1]) if mode == "per_subcarrier"
            else AGGREGATE_COLS)
    if verbose:
        note = f", {n_corrupt} corrupt skipped" if n_corrupt else ""
        if window_packets:
            fill = f"{window_packets} pkt/window (fixed)"
        else:
            n = np.array([i1 - i0 for i0, i1, _, _ in spans])
            fill = (f"{window_seconds:g}s windows, "
                    f"{int(np.median(n))} pkt median ({n.min()}–{n.max()})")
        print(f"  {label:8s} {len(amps):6d} packets -> {len(rows):4d} windows"
              f"  [{fill}]{note}")
    return np.asarray(rows), cols, meta


def block_split(df, test_frac=0.25, by="label"):
    """Hold out the last test_frac of each class (no shuffling)."""
    import pandas as pd
    tr, te = [], []
    for _key, g in df.groupby(by):
        g = g.sort_values("window_index")
        n_test = max(1, int(len(g) * test_frac))
        tr.append(g.iloc[:-n_test])
        te.append(g.iloc[-n_test:])
    return pd.concat(tr), pd.concat(te)


def room_split(df, test_room):
    """Train on every other room and test on test_room."""
    return df[df.room != test_room], df[df.room == test_room]
