"""Parsing of ESP32 CSI_DATA lines and recorded CSV rows into amplitudes."""

from __future__ import annotations

import ast
import csv
import io
import json
from dataclasses import dataclass

import numpy as np

N_SUBCARRIERS = 64
N_RAW_VALUES = 2 * N_SUBCARRIERS      # 128: interleaved I/Q
BAUD_RATE = 921600

# field names emitted by the ESP32 firmware, in order, after the CSI_DATA tag
CSV_FIELDS = [
    "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing",
    "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor",
    "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant",
    "sig_len", "rx_format", "len", "first_word", "data",
]


class CorruptPacket(Exception):
    """Raised for anything that is not a well-formed 128-value CSI packet."""


@dataclass
class Packet:
    rssi: float
    amplitude: np.ndarray      # (64,)
    phase_diff: np.ndarray     # (63,)
    raw: list                  # the 128 interleaved values, kept for logging


def parse_line(line: str) -> dict | None:
    """One raw serial line -> dict, or None if it is not a valid CSI record."""
    if not line.startswith("CSI_DATA,"):
        return None
    try:
        fields = next(csv.reader(io.StringIO(line)))
    except Exception:
        return None
    if len(fields) < 20:
        return None
    values = fields[1:]                       # drop the "CSI_DATA" tag

    data_field = None
    for f in reversed(values):
        fs = f.strip()
        if fs.startswith("["):
            data_field = fs
            break
    if data_field is None:
        return None
    try:
        parsed = ast.literal_eval(data_field)
    except Exception:
        return None
    # a cut-off serial line can still parse, so check the length
    if not isinstance(parsed, list) or len(parsed) != N_RAW_VALUES:
        return None

    record = {k: "0" for k in CSV_FIELDS}
    for i, k in enumerate(CSV_FIELDS[:19]):
        if i < len(values):
            record[k] = values[i]
    record["data"] = parsed
    return record


def iq_to_physics(raw) -> tuple[np.ndarray, np.ndarray]:
    """128 interleaved I/Q values -> (amplitude[64], adjacent phase diff[63])."""
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != N_RAW_VALUES:
        raise CorruptPacket(f"expected {N_RAW_VALUES} values, got {len(raw) if hasattr(raw,'__len__') else type(raw)}")
    arr = np.asarray(raw, dtype=np.float64)
    i_vals, q_vals = arr[0::2], arr[1::2]
    amplitude = np.sqrt(i_vals ** 2 + q_vals ** 2)
    phase = np.arctan2(q_vals, i_vals)
    return amplitude, phase[1:] - phase[:-1]


def packet_from_line(line: str) -> Packet | None:
    rec = parse_line(line)
    if rec is None:
        return None
    amp, pd_ = iq_to_physics(rec["data"])
    return Packet(rssi=float(rec["rssi"]), amplitude=amp, phase_diff=pd_, raw=rec["data"])


def packet_from_csv_row(row: dict) -> Packet:
    """Row from a recorded CSV, where `data` is a JSON string."""
    try:
        raw = json.loads(row["data"])
    except (json.JSONDecodeError, TypeError) as exc:
        raise CorruptPacket(str(exc)) from exc
    amp, pd_ = iq_to_physics(raw)
    return Packet(rssi=float(row["rssi"]), amplitude=amp, phase_diff=pd_, raw=raw)


def load_recording(path, skip=0, limit=None, drop_last=0):
    """Read a recorded CSV -> (amps[n,64], phase_diffs[n,63], rssi[n], times[n], n_corrupt)."""
    amps, pds, rssis, times = [], [], [], []
    n_corrupt = 0
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < skip:
                continue
            if limit is not None and len(amps) >= limit:
                break
            try:
                p = packet_from_csv_row(row)
            except (CorruptPacket, ValueError, KeyError):
                n_corrupt += 1
                continue
            amps.append(p.amplitude)
            pds.append(p.phase_diff)
            rssis.append(p.rssi)
            times.append(float(row.get("host_time_ns") or 0))

    amps = np.asarray(amps)
    pds = np.asarray(pds)
    rssis = np.asarray(rssis)
    times = np.asarray(times, dtype=np.float64)
    if drop_last and len(amps) > drop_last:
        amps, pds, rssis, times = (amps[:-drop_last], pds[:-drop_last],
                                   rssis[:-drop_last], times[:-drop_last])
    return amps, pds, rssis, times, n_corrupt
