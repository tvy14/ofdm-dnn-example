"""
FC-DNN Based OFDM Signal Detector
=====================================
Exercise 3.1 — Wireless Communications and Machine Learning

System Model
------------
  • 128 subcarriers total: 64 pilots (even idx) + 64 data (odd idx)
  • Modulation : QPSK  (2 bits / symbol)
  • Channel    : frequency-selective Rayleigh fading (6 complex taps)
  • DNN Input  : 256 real values  [Re(y_0..127) || Im(y_0..127)]
  • DNN Output : 128 soft bits  →  BCE loss
  • Architecture: FC(256→256)→ReLU → FC(256→256)→ReLU → FC(256→128)→Sigmoid

Baselines
---------
  • ZF with perfect CSI                           (ideal lower bound)
  • ZF with LS channel estimate + interpolation   (practical baseline)

Key insight (Exercise 3.1):
  The FC-DNN implicitly performs joint channel estimation and detection
  from the raw received vector, without explicitly computing H or H^{-1}.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ─── reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SYSTEM PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
N_SC    = 128          # total OFDM subcarriers  (Fig 3.2: 64 pilot + 64 data)
N_PILOT = 64           # pilots at even indices
N_DATA  = 64           # data   at odd  indices
# Flat-fading Rayleigh channel: one h ~ CN(0,1) per OFDM symbol,
# same coefficient on every subcarrier.  This matches the setup of
# Exercise 3.1, where the coherence bandwidth >> subcarrier spacing.
BPS     = 2            # bits per symbol  (QPSK)
N_BITS  = N_DATA * BPS # 128 bits per OFDM frame

PILOT_IDX = np.arange(0, N_SC, 2)   # 0, 2, 4, …, 126
DATA_IDX  = np.arange(1, N_SC, 2)   # 1, 3, 5, …, 127

# QPSK constellation — Gray coded,  unit average power
#   index 0 → bits [0,0],  1 → [0,1],  2 → [1,1],  3 → [1,0]
QPSK = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2)
GRAY = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=np.int8)

# Inverse Gray lookup: bit-pair (b0*2+b1) → QPSK symbol index
# GRAY[0]=[0,0] → key 0, GRAY[1]=[0,1] → key 1,
# GRAY[2]=[1,1] → key 3, GRAY[3]=[1,0] → key 2
GRAY_INV = np.zeros(4, dtype=int)
for _i, _g in enumerate(GRAY):
    GRAY_INV[_g[0] * 2 + _g[1]] = _i   # GRAY_INV = [0, 1, 3, 2]

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CHANNEL & OFDM HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def gen_channel():
    """Flat Rayleigh fading: h ~ CN(0,1), same on every subcarrier.
    Returns (N_SC,) complex with all entries equal to h.
    """
    h = np.complex64((np.random.randn() + 1j * np.random.randn()) / np.sqrt(2))
    return np.full(N_SC, h, dtype=np.complex64)   # (128,)


PILOTS = np.ones(N_PILOT, dtype=np.complex64)   # BPSK pilots  (+1)


def bits2qpsk(b2d):
    """(N_DATA, 2) int → (N_DATA,) complex64  (Gray-code consistent)"""
    bits_int = b2d[:, 0] * 2 + b2d[:, 1]
    return QPSK[GRAY_INV[bits_int]]


def qpsk2bits(syms):
    """(N,) complex → (N, 2) int8  via nearest-neighbour decision"""
    d   = np.abs(syms[:, None] - QPSK[None, :])   # (N, 4)
    idx = np.argmin(d, axis=1)
    return GRAY[idx]


def qpsk2bits_batch(syms_batch):
    """(B, N) complex → (B, N, 2) int8"""
    d   = np.abs(syms_batch[:, :, None] - QPSK[None, None, :])  # (B, N, 4)
    idx = np.argmin(d, axis=2)
    return GRAY[idx]


def ofdm_frame(bits, H, snr_db):
    """Simulate one OFDM frame.

    Parameters
    ----------
    bits   : (N_DATA, 2) int
    H      : (N_SC,) complex  channel frequency response
    snr_db : float            signal-to-noise ratio in dB

    Returns
    -------
    dnn_in : (256,) float32   [Re(y) || Im(y)]
    y      : (N_SC,) complex  received signal
    """
    sigma = 10 ** (-snr_db / 20.0)           # noise std per dimension (E[|n|^2]=sigma^2)
    x = np.empty(N_SC, dtype=np.complex64)
    x[PILOT_IDX] = PILOTS
    x[DATA_IDX]  = bits2qpsk(bits)
    n = (sigma / np.sqrt(2)) * (
        np.random.randn(N_SC) + 1j * np.random.randn(N_SC)
    ).astype(np.complex64)
    y      = H * x + n
    dnn_in = np.concatenate([y.real, y.imag]).astype(np.float32)
    return dnn_in, y


def ls_interp(y):
    """LS channel estimate for flat-fading: average over all pilot subcarriers.
    For flat fading h is scalar, so we average 64 noisy estimates → SNR gain of 64×.
    """
    H_avg = np.mean(y[PILOT_IDX] / PILOTS)                         # scalar complex
    return np.full(N_SC, H_avg, dtype=np.complex64)                # (128,)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PRE-GENERATE TRAINING DATASET
# ═══════════════════════════════════════════════════════════════════════════════
N_TRAIN  = 60_000
SNR_VALS = np.array([0, 5, 10, 15, 20, 25], dtype=float)   # uniform mix of SNRs

print(f"Generating {N_TRAIN:,} training frames …")
X_tr = np.empty((N_TRAIN, 256),    dtype=np.float32)
Y_tr = np.empty((N_TRAIN, N_BITS), dtype=np.float32)

for i in range(N_TRAIN):
    bits        = np.random.randint(0, 2, (N_DATA, 2))
    H           = gen_channel()
    snr         = float(np.random.choice(SNR_VALS))
    dnn_in, _   = ofdm_frame(bits, H, snr)
    X_tr[i]     = dnn_in
    Y_tr[i]     = bits.ravel().astype(np.float32)

print("Dataset ready.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FC-DNN MODEL  (Exercise 3.1, Fig 3.2)
# ═══════════════════════════════════════════════════════════════════════════════
class FCDetector(nn.Module):
    """FC-DNN:  256 → 256 → 256 → 128 bits
    Equivalent to one of the 8 parallel DNNs described in Ex 3.1,
    extended to output all 128 bits in a single forward pass.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, N_BITS), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = FCDetector().to(device)
opt   = optim.Adam(model.parameters(), lr=1e-3)
crit  = nn.BCELoss()

X_t = torch.FloatTensor(X_tr).to(device)
Y_t = torch.FloatTensor(Y_tr).to(device)

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
N_EPOCHS = 200
BATCH    = 1024
N_BATCH  = N_TRAIN // BATCH
train_losses = []

print(f"Training  ({N_EPOCHS} epochs, batch {BATCH}) …")
for ep in range(N_EPOCHS):
    perm    = torch.randperm(N_TRAIN, device=device)
    ep_loss = 0.0
    for b in range(N_BATCH):
        idx  = perm[b * BATCH : (b + 1) * BATCH]
        xb, yb = X_t[idx], Y_t[idx]
        pred = model(xb)
        loss = crit(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss += loss.item()
    ep_loss /= N_BATCH
    train_losses.append(ep_loss)
    if (ep + 1) % 20 == 0:
        print(f"  Epoch {ep+1:3d}/{N_EPOCHS}  Loss={ep_loss:.4f}")

print("Training complete.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  BER EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
model.eval()
N_TEST   = 5_000
SNR_TEST = np.arange(0, 26, 5)

ber_dnn, ber_zfp, ber_zfl = [], [], []

print("Evaluating BER …")
for snr in SNR_TEST:
    # ── generate test batch ────────────────────────────────────────────────────
    X_te   = np.empty((N_TEST, 256),    dtype=np.float32)
    H_te   = np.empty((N_TEST, N_SC),   dtype=np.complex64)
    Y_te   = np.empty((N_TEST, N_SC),   dtype=np.complex64)
    bits_te = np.random.randint(0, 2, (N_TEST, N_DATA, 2))

    for i in range(N_TEST):
        H               = gen_channel()
        dnn_in, y       = ofdm_frame(bits_te[i], H, snr)
        X_te[i], H_te[i], Y_te[i] = dnn_in, H, y

    # ── FC-DNN detection (batched) ─────────────────────────────────────────────
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_te).to(device)).cpu().numpy()
    bits_dnn = (preds > 0.5).reshape(N_TEST, N_DATA, 2).astype(np.int8)

    # ── ZF with perfect CSI (vectorised) ──────────────────────────────────────
    eq_perf   = Y_te[:, DATA_IDX] / H_te[:, DATA_IDX]   # (N_TEST, N_DATA)
    bits_zfp_ = qpsk2bits_batch(eq_perf)                 # (N_TEST, N_DATA, 2)

    # ── ZF with LS channel estimate ────────────────────────────────────────────
    H_ls_all  = np.array([ls_interp(Y_te[i]) for i in range(N_TEST)])
    eq_ls     = Y_te[:, DATA_IDX] / H_ls_all[:, DATA_IDX]
    bits_zfl_ = qpsk2bits_batch(eq_ls)

    total = N_TEST * N_BITS
    ber_dnn.append(np.sum(bits_dnn   != bits_te) / total)
    ber_zfp.append(np.sum(bits_zfp_  != bits_te) / total)
    ber_zfl.append(np.sum(bits_zfl_  != bits_te) / total)

    print(f"  SNR={snr:2d} dB | "
          f"FC-DNN={ber_dnn[-1]:.4f}  "
          f"ZF(perfect)={ber_zfp[-1]:.4f}  "
          f"ZF(LS)={ber_zfl[-1]:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("FC-DNN OFDM Signal Detector  —  Exercise 3.1",
             fontsize=13, fontweight="bold")

# ── Plot 1: Training loss ──────────────────────────────────────────────────────
ax = axes[0]
ax.plot(range(1, N_EPOCHS + 1), train_losses, color="royalblue", lw=1.5)
ax.set_xlabel("Epoch");  ax.set_ylabel("BCE Loss")
ax.set_title("Training Loss Curve")
ax.grid(True, alpha=0.3)

# ── Plot 2: BER vs SNR ────────────────────────────────────────────────────────
ax = axes[1]
ax.semilogy(SNR_TEST, ber_dnn, "o-",  color="tomato",      lw=2, ms=7,
            label="FC-DNN  (Exercise 3.1)")
ax.semilogy(SNR_TEST, ber_zfl, "s--", color="darkorange",   lw=2, ms=7,
            label="ZF  + LS channel estimate")
ax.semilogy(SNR_TEST, ber_zfp, "^-.", color="royalblue",    lw=2, ms=7,
            label="ZF  + Perfect CSI  (lower bound)")
ax.set_xlabel("SNR (dB)");  ax.set_ylabel("BER")
ax.set_title("BER vs SNR  —  QPSK, 128-SC OFDM, Rayleigh fading")
ax.legend(loc="upper right");  ax.grid(True, which="both", alpha=0.3)
ax.set_ylim([1e-4, 1.0])

plt.tight_layout()
out_dir  = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "ofdm_dnn_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved → {out_path}")
