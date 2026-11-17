# Privacy-Aware Wi-Fi Activity Sensing

**Bayesian reinforcement learning on ESP32 Channel State Information, with key-driven frequency hopping**

B.Tech Project · Sanjay · Indian Institute of Technology Patna · 2026

## Overview

In this project I recognise human activity (IDLE, STAND, WALK) from Wi-Fi Channel State
Information (CSI) measured between two ESP32 boards, without any camera or wearable device.
Since CSI can be measured by any receiver tuned to the same channel, I made the link hop
across all 13 channels of the 2.4 GHz band in a secret, key-driven order. A fixed-channel
eavesdropper then receives only fragments of the signal, while my receiver keeps sensing.

- **Link:** connectionless ESP-NOW broadcast between two ESP32 boards, about 100 packets/s,
  forced to 802.11g OFDM at 6 Mbps so that every packet yields CSI.
- **Hopping:** channel = 1 + (H(step, K) mod 13), step = seq / 200 (2 s per channel). The
  receiver derives the step from the sequence number in each packet and applies the same key;
  the key and the channel order are never transmitted.
- **Learning:** a contextual Bayesian bandit (linear-Gaussian beliefs, Thompson sampling) that
  abstains with UNCERTAIN when its credible intervals do not separate the best action from the
  runner-up, and can continue to learn online from operator feedback.

## Results

I evaluate on the last 25% (in time) of each activity recording, which is never used in training.

| | Phase A: fixed channel | Phase B: hopping (13 channels) |
|---|---|---|
| Train / held-out windows | 518 / 171 | 241 / 78 |
| Accuracy when answered | 1.000 (167 of 171) | 1.000 (64 of 78) |
| Accuracy, abstention counted as error | **0.977** | **0.821** |
| Accuracy if forced to answer | 1.000 | 0.923 |
| Misclassifications | 0 | 0 |
| Expected calibration error | 0.117 | 0.136 |

Hopping performance: 92% of air time usable while hopping over 13 channels (about 0.12 s
outage per hop), compared with 73% (3.9 s re-association per hop) for an access-point-based
link. Phase A labels were verified against video with a YOLOv8n person detector (689 of 689
windows kept).

## Repository structure

```
firmware/                     ESP-IDF project, one source for both boards
  main/app_main.c             TX/RX roles (#define ROLE), hopping (#define HOP_ENABLE)
  sdkconfig.defaults          CSI enabled, power save off, console at 921600 baud
pipeline/
  core/                       library
    csi.py                    parsing of CSI_DATA lines and recorded CSV rows
    baseline.py               empty-room baseline and per-subcarrier calibration
    features.py               2 s windows, 37 features, time-block split
    brl.py                    contextual Bayesian RL agent
  acquisition/                serial CSI logger and synchronised webcam recorder
  scripts/
    record_hop.py             guided recording: baseline, IDLE, STAND, WALK
    prepare.py                baseline construction and feature extraction
    label.py                  YOLO audit of window labels against video
    train.py                  training and held-out evaluation
    live.py                   real-time recognition (or replay of a recording)
    live_learn.py             real-time recognition with online Bayesian updates
  data/raw/                   recorded CSI sessions (Phase A and Phase B)
  artifacts/                  baselines, feature tables and trained models
docs/Project_Presentation.pptx
```

## Requirements

- Two ESP32 boards (tested on ESP32-D0WD-V3) and ESP-IDF v5.3
- Python 3.9 or later: `pip install -r requirements.txt`

## Usage

### 1. Flash the boards

From `firmware/`, with ESP-IDF loaded (`source $HOME/esp/esp-idf/export.sh`). Set
`HOP_ENABLE` to `0` (fixed channel) or `1` (hopping) in `main/app_main.c`, identically for both
boards, then flash each role:

```bash
sed -i '' -E 's/^#define ROLE +ROLE_(RX|TX)/#define ROLE      ROLE_TX/' main/app_main.c
idf.py -p <TX_PORT> flash
sed -i '' -E 's/^#define ROLE +ROLE_(RX|TX)/#define ROLE      ROLE_RX/' main/app_main.c
idf.py -p <RX_PORT> flash
```

### 2. Record, prepare, train and run (from `pipeline/`)

```bash
python3 -m scripts.record_hop --port <RX_PORT> --room <name> --baseline-min 3 --activity-min 5 --auto IDLE

D=$(ls -d data/raw/<name>_*/csi | tail -1)
python3 -m scripts.prepare --room <name> --empty $D/BASELINE_*.csi.csv \
  --idle $D/IDLE_*.csi.csv --stand $D/STAND_*.csi.csv --walk $D/WALK_*.csi.csv

python3 -m scripts.train --features artifacts/features_<name>.csv --save artifacts/model_<name>.joblib
python3 -m scripts.live --port <RX_PORT> --room <name> --model artifacts/model_<name>.joblib
```

### 3. Reproduce the reported results with the included data

```bash
python3 -m scripts.train --features artifacts/features_hostel_room.csv --save /tmp/phase_a.joblib
python3 -m scripts.train --features artifacts/features_hostel_hop.csv  --save /tmp/phase_b.joblib
python3 -m scripts.live --replay data/raw/hostel_hop_20261116_203445/csi/WALK_*.csi.csv \
  --room hostel_hop --model artifacts/model_hostel_hop.joblib --replay-speed 1
```

## Data

I recorded two sessions on 16 November 2026 in one room, with the boards unmoved:

| Session | Firmware | Stages | Packets (baseline / IDLE / STAND / WALK) |
|---|---|---|---|
| `hostel_room_20261116_191429` | fixed channel 6 | 3 + 3 × 5 min | 17,222 / 28,708 / 29,122 / 28,882 |
| `hostel_hop_20261116_203445` | hopping, channels 1–13 | 2 + 3 × 3 min | 11,170 / 16,726 / 16,237 / 16,665 |

Each `.csi.csv` has one row per packet, including `rssi`, `channel`, `host_time_ns` and `data`
(128 raw I/Q values). Video recordings (used only for the label check) are not included.

## Limitations

- One room and one session per phase; cross-session and cross-room accuracy are not yet measured.
- Three coarse activities and a single subject; IDLE denotes an empty room.
- Phase B has less than half the training data of Phase A and uses one blended baseline for all
  channels; channels 11–13 show 1.5–1.9× higher empty-room fluctuation than channels 1–10.
- The privacy benefit has not yet been measured with a separate eavesdropping receiver. The
  sequence number is sent in plain text and the hopping hash is not a cryptographic construction.
- The out-of-distribution gate is available in `live.py` with `--gate` and is off by default.

## Future work

Next I plan to add per-channel normalisation for Phase B, a third-board eavesdropper experiment, cross-session and
cross-room evaluation, a keyed cryptographic hopping sequence, and more activities and
subjects.
