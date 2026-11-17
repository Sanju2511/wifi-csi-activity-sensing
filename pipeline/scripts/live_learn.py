"""Real-time recognition with online updates from operator feedback.

    python3 -m scripts.live_learn --port <RX_PORT> --room <name>

Keys: i = IDLE, l = STAND, p = WALK, s = save, q = save and quit.
"""
from __future__ import annotations
import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from core import ARTIFACTS
from core import baseline as bl
from core.brl import ACTIONS, START_TOKEN
from core.features import MIN_WINDOW_FILL, physics_features
from scripts.live import serial_source, open_port

BAR = "=" * 70
KEYMAP = {"i": "IDLE", "1": "IDLE", "l": "STAND", "2": "STAND",
          "v": "STAND", "p": "WALK", "3": "WALK"}

truth_label = None          # activity confirmed with the keyboard
running = True
save_now = False


def key_listener():
    global truth_label, running, save_now
    while running:
        ch = sys.stdin.read(1)
        if not ch:
            continue
        if ch in ("q", "\x03"):
            running = False
            break
        if ch == "s":
            save_now = True
            continue
        if ch == " ":
            i = ACTIONS.index(truth_label) if truth_label in ACTIONS else -1
            truth_label = ACTIONS[(i + 1) % len(ACTIONS)]
            continue
        low = ch.lower()
        if low in KEYMAP:
            truth_label = KEYMAP[low]
            continue
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            truth_label = {"[D": "IDLE", "[B": "STAND", "[C": "WALK",
                           "[A": "IDLE"}.get(seq, truth_label)


def save_model(bundle, model, room, src_name):
    out = ARTIFACTS / f"model_{room}_online.joblib"
    bundle = dict(bundle)
    bundle["model"] = model
    bundle["online_learned_from"] = src_name
    joblib.dump(bundle, out)
    return out


def main():
    global running, save_now
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbserial-0001")
    ap.add_argument("--room", required=True, help="which baseline/model to use, e.g. hostel_room")
    ap.add_argument("--model", default=None, help="default: artifacts/model.joblib")
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--settle", type=float, default=10.0)
    args = ap.parse_args()

    model_path = Path(args.model) if args.model else ARTIFACTS / "model.joblib"
    if not model_path.exists():
        sys.exit(f"No model at {model_path} — train one first.")
    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    if not hasattr(model, "observe"):
        sys.exit("This model has no observe() — re-run scripts.train so the BRL supports online learning.")

    base_path = ARTIFACTS / f"baseline_{args.room}.npz"
    if not base_path.exists():
        sys.exit(f"No baseline at {base_path}")
    base = bl.load(base_path)

    print(BAR)
    print("ONLINE RL  —  the agent learns from your live feedback")
    print(BAR)
    print(f"  model     : {model_path.name}  (updated copy saved separately)")
    print(f"  baseline  : {base_path.name}")
    print(f"  keys      : i=IDLE  l=STAND  p=WALK  space=cycle  s=save  q=quit")
    print(f"  tip       : set the key to what you are ACTUALLY doing — wrong keys teach it wrong")
    print(BAR)
    print()

    ser = open_port(args.port, args.settle)
    source = serial_source(ser)

    amps: deque = deque()
    times: deque = deque()
    prev = START_TOKEN
    next_emit = None
    t_first = None
    n_seen = n_correct = n_learned = 0

    old = None; fd = None
    if sys.stdin.isatty():
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] = new[3] & ~termios.ECHO & ~termios.ICANON
        termios.tcsetattr(fd, termios.TCSANOW, new)
        threading.Thread(target=key_listener, daemon=True).start()

    print(f"  {'time':>7s}  {'rate':>6s}  {'predicted':<11s}  {'you say':<8s}  {'learn':<6s}  acc")
    print("  " + "-" * 60)
    try:
        while running:
            got = next(source, None)
            if got is None:
                break
            pkt, now = got
            if pkt is None:
                continue
            if t_first is None:
                t_first = now
            amps.append(pkt.amplitude)
            times.append(now)
            if next_emit is None:
                next_emit = now + args.window
                continue
            cutoff = now - args.window
            while times and times[0] < cutoff:
                times.popleft(); amps.popleft()
            if now < next_emit:
                continue
            next_emit = now + args.stride

            n = len(amps)
            rate = n / args.window
            if n < max(2, int(MIN_WINDOW_FILL * args.window * 100)):
                continue

            block = np.asarray(amps, dtype=float)
            cal = bl.active(bl.calibrate(block, base), base)
            vec = physics_features(cal, n / args.window)

            resp = model.project_features(vec)
            reported, guess, prob, lo, hi, ev = model.decide(resp, model.effective_prev(prev))
            pred_label = ACTIONS[reported] if reported is not None else "UNCERTAIN"

            # only learn when the true activity has been set
            learn_flag = ""
            if truth_label in ACTIONS:
                true_a = ACTIONS.index(truth_label)
                n_seen += 1
                if guess == true_a:
                    n_correct += 1
                model.observe(resp, true_a, model.effective_prev(prev))
                n_learned += 1
                learn_flag = "LEARN"
            acc = (n_correct / n_seen) if n_seen else 0.0

            prev = guess
            you = truth_label if truth_label in ACTIONS else "-"
            print(f"  {now-t_first:6.1f}s  {rate:5.1f}/s  {pred_label:<11s}  {you:<8s}  "
                  f"{learn_flag:<6s}  {acc:4.0%} ({n_seen})")

            if save_now:
                out = save_model(bundle, model, args.room, model_path.name)
                print(f"  >>> saved {n_learned} updates -> {out.name}")
                save_now = False
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        if old and fd is not None:
            import termios
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        out = save_model(bundle, model, args.room, model_path.name)
        print(f"\n  Session done: {n_learned} learning updates, "
              f"{n_correct}/{n_seen} correct before learning.")
        print(f"  Saved learned model -> {out}")
        print(f"  Use it live:  python3 -m scripts.live --port {args.port} "
              f"--room {args.room} --model {out}")


if __name__ == "__main__":
    main()
