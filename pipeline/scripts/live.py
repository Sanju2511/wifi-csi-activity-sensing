"""Real-time recognition from the RX serial port, or replay of a recording.

    python3 -m scripts.live --port <RX_PORT> --room <name> --model <model.joblib>
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from core import ARTIFACTS
from core import baseline as bl
from core.brl import ACTIONS, START_TOKEN
from core.csi import BAUD_RATE, packet_from_line
from core.features import (MIN_WINDOW_FILL, STRIDE_SECONDS, WINDOW_SECONDS,
                           physics_features)

BAR = "=" * 74


def serial_source(ser):
    """Yield (packet|None, wall_clock_time) from the serial port, forever."""
    while True:
        raw = ser.readline().decode("utf-8", "ignore").strip()
        if not raw:
            continue
        yield packet_from_line(raw), time.time()


def replay_source(path, speed=0.0):
    """Same as serial_source, but reads a recorded CSV."""
    import csv as _csv
    from core.csi import packet_from_csv_row, CorruptPacket
    t0 = None
    started = time.time()
    with open(path, newline="") as fh:
        for row in _csv.DictReader(fh):
            try:
                pkt = packet_from_csv_row(row)
            except (CorruptPacket, KeyError, TypeError):
                yield None, (t0 or 0.0)
                continue
            t = float(row.get("host_time_ns") or 0) / 1e9
            if t0 is None:
                t0 = t
            if speed > 0:
                behind = (t - t0) / speed - (time.time() - started)
                if behind > 0:
                    time.sleep(min(behind, 1.0))
            yield pkt, t


def open_port(port, settle):
    import serial
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.5)
    except Exception as exc:
        sys.exit(f"Could not open {port}: {exc}\n"
                 f"  Is the board plugged in?\n"
                 f"  Is idf.py monitor still running? Only one reader is allowed.")
    print(f"  opened {port} — the board has just reset, waiting {settle}s ...")
    time.sleep(settle)
    ser.reset_input_buffer()
    return ser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--replay", default=None,
                    help="a recorded .csi.csv to feed through the SAME code path "
                         "instead of the serial port — for testing and demos")
    ap.add_argument("--replay-speed", type=float, default=0.0,
                    help="0 = as fast as possible, 1 = real time")
    ap.add_argument("--room", required=True,
                    help="which baseline to use, e.g. hostel_room")
    ap.add_argument("--model", default=None,
                    help="default: artifacts/model_<room>.joblib")
    ap.add_argument("--window", type=float, default=WINDOW_SECONDS)
    ap.add_argument("--stride", type=float, default=STRIDE_SECONDS)
    ap.add_argument("--settle", type=float, default=10.0,
                    help="seconds to wait after opening the port for the board "
                         "to reboot and resume CSI")
    ap.add_argument("--no-gate", action="store_true", default=False,
                    help="disable the out-of-distribution gate")
    ap.add_argument("--gate", action="store_true", default=False,
                    help="enable the out-of-distribution gate (prints OUT-OF-ROOM)")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after this many seconds (default: run until Ctrl-C)")
    ap.add_argument("--show-all", action="store_true",
                    help="also print the interval and evidence for every window")
    args = ap.parse_args()

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = ARTIFACTS / f"model_{args.room}.joblib"
        if not model_path.exists():
            model_path = ARTIFACTS / "model.joblib"
    if not model_path.exists():
        sys.exit(f"No model at {model_path}\n"
                 f"  Train one first:  python3 -m scripts.train --save {model_path}")
    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    base_path = ARTIFACTS / f"baseline_{args.room}.npz"
    if not base_path.exists():
        sys.exit(f"No baseline at {base_path}\n"
                 f"  It is written by scripts.prepare for the room you trained on.")
    base = bl.load(base_path)

    # Plausibility (out-of-distribution) gate: off by default, enabled with --gate.
    use_gate = args.gate and not args.no_gate

    print(BAR)
    print("LIVE PREDICTION")
    print(BAR)
    is_rf = bundle.get("model_type") == "rf" or (hasattr(model, "predict_proba") and not hasattr(model, "decide"))
    m_type_name = "Random Forest" if is_rf else "Bayesian RL"
    print(f"  model type: {m_type_name} ({model_path.name})")
    print(f"  baseline  : {base_path.name}")
    print(f"  window    : {args.window:g} s, stride {args.stride:g} s")
    print(f"  actions   : {', '.join(ACTIONS)}" + ("" if is_rf else "  (+ UNCERTAIN)"))
    if not is_rf:
        if not use_gate:
            print(f"  OOD gate  : off (enable with --gate)")
        else:
            print(f"  OOD gate  : on (floor = {getattr(model, 'll_floor', float('nan')):.1f})")
    if "stats" in bundle and bundle["stats"]:
        s = bundle["stats"]
        if isinstance(s, dict):
            if "acc_allin" in s:
                print(f"  held-out  : {s['acc_allin']:.3f} abstain-as-wrong")
            elif "acc" in s:
                print(f"  held-out  : {s['acc']:.3f} accuracy")
    print()
    print("  The baseline is only valid while the geometry has not changed.")
    print("  Ctrl-C to stop.")
    print()

    if not args.port and not args.replay:
        sys.exit("Pass --port (live) or --replay <file.csi.csv> (offline test).")

    if args.replay:
        source = replay_source(args.replay, args.replay_speed)
        print(f"  REPLAY from {Path(args.replay).name} — not a live sensor\n")
        ser = None
    else:
        ser = open_port(args.port, args.settle)
        source = serial_source(ser)

    amps: deque = deque()
    times: deque = deque()
    prev = START_TOKEN
    next_emit = None
    started = time.time()
    n_pkt = n_bad = n_odd = 0
    last_line = ""
    t_first = None          # clock origin: wall time live, recording time on replay

    print(f"  {'time':>8s}  {'rate':>7s}  {'answer':<12s}  {'p':>6s}")
    print("  " + "-" * 42)

    try:
        while True:
            if args.duration and time.time() - started > args.duration:
                break
            got = next(source, None)
            if got is None:
                break
            pkt, now = got
            if pkt is None:
                n_bad += 1
                continue
            if t_first is None:
                t_first = now
            amps.append(pkt.amplitude)
            times.append(now)
            n_pkt += 1

            if next_emit is None:
                next_emit = now + args.window
                continue

            # drop anything older than the window
            cutoff = now - args.window
            while times and times[0] < cutoff:
                times.popleft()
                amps.popleft()

            if now < next_emit:
                continue
            next_emit = now + args.stride

            n = len(amps)
            rate = n / args.window
            if n < max(2, int(MIN_WINDOW_FILL * args.window * 100)):
                print(f"  {now-t_first:7.1f}s  {rate:6.1f}/s  "
                      f"{'-- LINK STALLED --':<10s}   too few packets ({n})")
                continue

            block = np.asarray(amps, dtype=float)
            cal = bl.calibrate(block, base)
            cal = bl.active(cal, base)
            fs = n / args.window          # fixed duration -> constant bin spacing
            vec = physics_features(cal, fs)

            if is_rf:
                probs = model.predict_proba([vec])[0]
                classes = list(model.classes_)
                top_idx = int(np.argmax(probs))
                label = classes[top_idx]
                prob = float(probs[top_idx])
                line = f"  {now-t_first:7.1f}s  {rate:6.1f}/s  {label:<12s}  {prob:6.3f}"
            else:
                resp = model.responsibilities_for(vec)
                reported, guess, prob, lo, hi, evidence = model.decide(
                    resp, model.effective_prev(prev))
                prev = guess                  # the chain advances on the guess, not the report

                ll, plausible = model.plausibility(vec)
                if use_gate and not plausible:
                    reported = None
                    n_odd += 1

                label = ACTIONS[reported] if reported is not None else "UNCERTAIN"
                if use_gate and not plausible:
                    label = "OUT-OF-ROOM"
                line = f"  {now-t_first:7.1f}s  {rate:6.1f}/s  {label:<12s}  {prob:6.3f}"
            if args.show_all or line[:35] != last_line[:35]:
                print(line)
            last_line = line

    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        if ser is not None:
            ser.close()
        el = time.time() - started
        print()
        print(BAR)
        print(f"  ran {el:.0f}s · {n_pkt} packets ({n_pkt/max(el,1):.1f}/s) · "
              f"{n_bad} non-CSI lines skipped")
        if n_odd and use_gate:
            floor = getattr(model, "ll_floor", None)
            print()
            print(f"  {n_odd} windows were OUT-OF-ROOM — below the training floor "
                  f"({floor:.1f} log-likelihood)." if floor is not None else "")
            print("  The channel does not match the baseline this model was trained")
            print("  against. Either an ESP32 board has moved since the")
            print("  recording, or something large in the room has. Predictions in")
            print("  that state are meaningless — record a new baseline.")
        print(BAR)


if __name__ == "__main__":
    main()
