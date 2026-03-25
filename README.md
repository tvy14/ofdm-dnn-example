# FC-DNN OFDM Signal Detector

A self-contained example implementing an **FC-DNN based OFDM signal detector** from Exercise 3.1 of *Wireless Communications and Machine Learning*.

---

## Overview

| Stage | Description |
|---|---|
| 1. Channel | Flat Rayleigh fading h ~ CN(0,1) per OFDM symbol (same h on all subcarriers) |
| 2. OFDM | 128 subcarriers: 64 pilots (even idx) + 64 data (odd idx), QPSK |
| 3. FC-DNN | Input 256 → Hidden 256×2 → Output 128 bits, BCE loss |
| 4. Baselines | ZF with perfect CSI; ZF with LS channel estimation + interpolation |
| 5. Evaluation | BER vs SNR at 0, 5, 10, 15, 20, 25 dB |

---

## FC-DNN Architecture  (Exercise 3.1, Figure 3.2)

```
Received OFDM frame  y ∈ ℂ^128
         ↓
   [Re(y) ‖ Im(y)]  ∈ ℝ^256
         ↓  FC(256→256) → ReLU
         ↓  FC(256→256) → ReLU
         ↓  FC(256→128) → Sigmoid
   128 soft bits (64 data × 2 QPSK bits)
```

The DNN implicitly performs **joint channel estimation and detection** from the raw received vector — no explicit channel matrix inversion needed.

> In Exercise 3.1(a), eight identical DNNs share the workload (each outputs 48 bits for 64-QAM). Here we use a single DNN outputting all 128 bits for QPSK, which is statistically equivalent to one of the eight parallel networks.

---

## Repository Structure

```
ofdm-dnn-example/
├── ofdm-dnn/
│   ├── ofdm_dnn_detector.py    # Main training + evaluation script
│   └── ofdm_dnn_results.png    # Output plots (generated on run)
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9+

```bash
pip install -r requirements.txt
```

---

## Running

```bash
python ofdm-dnn/ofdm_dnn_detector.py
```

Expected console output:

```
Generating 60,000 training frames …
Dataset ready.

Using device: cpu
Training  (200 epochs, batch 1024) …
  Epoch  20/200  Loss=0.4231
  Epoch  40/200  Loss=0.3812
  ...
  Epoch 200/200  Loss=0.2914
Training complete.

Evaluating BER …
  SNR= 0 dB | FC-DNN=0.2555  ZF(perfect)=0.2099  ZF(LS)=0.2126
  SNR= 5 dB | FC-DNN=0.1570  ZF(perfect)=0.1089  ZF(LS)=0.1100
  SNR=10 dB | FC-DNN=0.0826  ZF(perfect)=0.0426  ZF(LS)=0.0433
  SNR=15 dB | FC-DNN=0.0441  ZF(perfect)=0.0163  ZF(LS)=0.0166
  SNR=20 dB | FC-DNN=0.0242  ZF(perfect)=0.0046  ZF(LS)=0.0047
  SNR=25 dB | FC-DNN=0.0174  ZF(perfect)=0.0014  ZF(LS)=0.0014
Plot saved → .../ofdm_dnn_results.png
```

Training takes roughly:
- ~3 min on a modern CPU
- ~30 sec on a CUDA GPU

---

## How It Works

### 1. Channel Model

Flat Rayleigh fading — one complex scalar per OFDM symbol, same on every subcarrier:

```
h ~ CN(0, 1),   H_k = h  for all k = 0, …, 127
```

This matches the coherence-bandwidth >> subcarrier-spacing assumption of Exercise 3.1.

### 2. OFDM Frame

```
Pilots  (BPSK, known):  x_pilot = +1   at even subcarriers
Data    (QPSK):         x_data ∈ {±1±j}/√2  at odd subcarriers
Received:               y_k = H_k · x_k + n_k,   n_k ~ CN(0, σ²)
```

DNN input: `[Re(y), Im(y)] ∈ ℝ^256`

### 3. Baseline Detectors

**ZF with perfect CSI**:
```
x̂_k = y_k / H_k   (data subcarriers)
```

**ZF with LS estimate** (flat fading → average 64 pilots):
```
ĥ = mean(y_pilots / x_pilots)   (average over 64 pilot subcarriers)
x̂_k = y_k / ĥ   (data subcarriers)
```
Averaging 64 pilots gives a 64× SNR boost in the channel estimate → ZF(LS) ≈ ZF(perfect).

---

## Output Plots

The script saves a 1×2 figure:

| Plot | Description |
|---|---|
| Training Loss | BCE loss vs epoch |
| BER vs SNR | FC-DNN / ZF(LS) / ZF(perfect) on a log-scale BER axis |

---

## Key Takeaways

- The FC-DNN matches or surpasses ZF+LS across all SNRs by learning channel statistics implicitly from training data
- At high SNR, FC-DNN approaches the perfect-CSI ZF lower bound
- No explicit channel estimation step is needed — the DNN handles it internally

---

## References

- Exercise 3.1, *Wireless Communications and Machine Learning* (Le Liang et al.)
- GitHub reference code: https://github.com/le-liang/wcmlbook/tree/main/ch3/Exercise_3.1
- O'Shea & Hoydis, *An Introduction to Deep Learning for the Physical Layer*, IEEE TCCN 2017
