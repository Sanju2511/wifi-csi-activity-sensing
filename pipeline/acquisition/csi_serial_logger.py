"""Serial CSI logger: reads CSI_DATA lines from the RX board and writes them to CSV."""

import argparse
import ast
import collections
import csv
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path

import serial

BAUD_RATE = 921600


def _frame_amplitude(data):
    """Mean subcarrier amplitude of one 128-value CSI array."""
    mags = []
    for i in range(0, len(data) - 1, 2):
        q, ii = data[i], data[i + 1]
        if q == 0 and ii == 0:
            continue
        mags.append(math.hypot(ii, q))
    return sum(mags) / len(mags) if mags else 0.0


def _live_bar(value, lo, hi, width=34):
    if hi <= lo:
        hi = lo + 1
    n = int(max(0.0, min(1.0, (value - lo) / (hi - lo))) * width)
    return "#" * n + "." * (width - n)

CSV_FIELDS = [
    "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing",
    "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor",
    "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant",
    "sig_len", "rx_format", "len", "first_word", "data",
]


def parse_csi_line(line):
    """Parse one CSI_DATA line into a CSV record."""
    if not line.startswith("CSI_DATA,"):
        return None
    try:
        fields = list(csv.reader(io.StringIO(line)))[0]
    except Exception:
        return None
    if len(fields) < 20:
        return None
    values = fields[1:]                      # drop the "CSI_DATA" tag

    # the data array is the last field that looks like "[...]"
    data_field = None
    for f in reversed(values):
        fs = f.strip()
        if fs.startswith("["):
            data_field = fs
            break
    if data_field is None:
        return None
    try:
        parsed_data = ast.literal_eval(data_field)
    except Exception:
        return None
    if not isinstance(parsed_data, list) or len(parsed_data) != 128:
        return None

    record = {k: "0" for k in CSV_FIELDS}
    for i, k in enumerate(CSV_FIELDS[:19]):
        if i < len(values):
            record[k] = values[i]
    record["data"] = parsed_data
    return record


def run_session(label, duration, output_dir, port, target_packets=None, max_duration=None,
                 progress_cb=None, live=False):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = label.upper()
    session_id = f"{safe_label}_{stamp}"
    csi_path = out_dir / f"{session_id}.csi.csv"
    meta_path = out_dir / f"{session_id}.meta.json"

    ser = serial.Serial(port, BAUD_RATE, timeout=0.25)
    ser.reset_input_buffer()

    csv_header = ["host_time_ms", "host_time_ns", "label", "session_id"] + CSV_FIELDS
    packets = 0
    bytes_total = 0
    host_times_ns = []
    first_local_ts = None
    last_local_ts = None
    hit_target = False

    hard_cap = max_duration if max_duration is not None else duration
    start = time.time()
    last_progress = 0.0

    # live CSI meter state (only used when live=True)
    live_recent = collections.deque(maxlen=60)    # ~0.6 s for motion
    live_seen = collections.deque(maxlen=400)      # for bar auto-scale
    live_calib = None
    last_live = 0.0

    with csi_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        while True:
            elapsed = time.time() - start
            if target_packets is not None:
                if packets >= target_packets:
                    hit_target = True
                    break
                if elapsed >= hard_cap:
                    break
            else:
                if elapsed >= duration:
                    break

            raw = ser.readline()
            if not raw:
                if progress_cb and time.time() - last_progress >= 1.0:
                    last_progress = time.time()
                    progress_cb(elapsed, packets, target_packets)
                continue
            bytes_total += len(raw)
            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            record = parse_csi_line(line)
            if record is None:
                continue

            now_ns = time.time_ns()
            writer.writerow([
                now_ns // 1_000_000, now_ns, safe_label, session_id,
                *[record[k] if k != "data" else json.dumps(record["data"]) for k in CSV_FIELDS],
            ])
            packets += 1
            host_times_ns.append(now_ns)
            local_ts = int(record["local_timestamp"])
            if first_local_ts is None:
                first_local_ts = local_ts
            last_local_ts = local_ts

            if live:
                amp = _frame_amplitude(record["data"])
                live_recent.append(amp)
                live_seen.append(amp)
                now_t = time.time()
                if now_t - last_live >= 0.1:
                    last_live = now_t
                    motion = statistics.pstdev(live_recent) if len(live_recent) > 5 else 0.0
                    if live_calib is None and len(live_seen) >= 120:
                        live_calib = max(0.5, statistics.pstdev(list(live_seen)[:120]))
                    lo, hi = min(live_seen), max(live_seen)
                    flag = ""
                    if live_calib:
                        if motion > 4 * live_calib:
                            flag = "  <<< MOTION >>>"
                        elif motion > 2 * live_calib:
                            flag = "  < motion >"
                    left = max(0.0, duration - (now_t - start))
                    sys.stdout.write(
                        f"\r    [{label:5s}] live |{_live_bar(amp, lo, hi)}| "
                        f"amp {amp:5.1f}  motion {motion:4.1f}{flag}   "
                        f"{left:4.0f}s left   ")
                    sys.stdout.flush()

            if progress_cb and time.time() - last_progress >= 1.0:
                last_progress = time.time()
                progress_cb(elapsed, packets, target_packets)

    ser.close()
    if live:
        sys.stdout.write("\n")
        sys.stdout.flush()

    duration_sec = max(
        (host_times_ns[-1] - host_times_ns[0]) / 1e9 if len(host_times_ns) >= 2 else duration,
        0.001,
    )
    esp_span_sec = max(
        (last_local_ts - first_local_ts) / 1e6 if first_local_ts is not None and last_local_ts is not None else duration_sec,
        0.001,
    )

    n_buckets = max(1, int(duration_sec // 5.0) + 1)
    bucket_counts = [0] * n_buckets
    if host_times_ns:
        t0 = host_times_ns[0]
        for t in host_times_ns:
            idx = int((t - t0) / 1e9 // 5.0)
            if 0 <= idx < n_buckets:
                bucket_counts[idx] += 1
    bucket_rates_hz = [round(c / 5.0, 2) for c in bucket_counts]

    meta = {
        "architecture": "esp32_espnow_pair_serial",
        "session_id": session_id,
        "label": safe_label,
        "duration_sec": round(time.time() - start, 3),
        "csi_packets": packets,
        "csi_rate_host_span_hz": round(packets / duration_sec, 3),
        "csi_rate_esp_span_hz": round(packets / esp_span_sec, 3),
        "bucket_sec": 5.0,
        "bucket_rates_hz": bucket_rates_hz,
        "bytes_total": bytes_total,
        "target_packets": target_packets,
        "hit_target_packets": hit_target if target_packets is not None else None,
        "files": {"csi": str(csi_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Collect one CSI session over USB serial (ESP-NOW RX board).")
    parser.add_argument("--label", default="UNKNOWN")
    parser.add_argument("--port", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", default="data/validation")
    args = parser.parse_args()

    def progress(elapsed, packets, target):
        rate = packets / elapsed if elapsed > 0 else 0.0
        print(f"  t={elapsed:5.1f}s  packets={packets:5d}  running_rate={rate:6.1f} Hz", flush=True)

    meta = run_session(args.label, args.duration, args.output_dir, args.port, progress_cb=progress)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
