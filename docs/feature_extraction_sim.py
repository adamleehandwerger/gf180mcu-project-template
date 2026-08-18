"""
confusion_comparison_m6.py — ECE410 Milestone 6
================================================
Side-by-side confusion matrix: sklearn binary OVR float vs ASIC Q6.10.

  Col 1: sklearn binary OVR (float64) — same model architecture as ASIC.
  Col 2: ASIC Q6.10 — same 5 binary OVR SVMs, pure-Python Q6.10 hardware
                       model (identical algorithm to svm_compute_core RTL).

SV allocation: [120,120,120,120,120] = 600 total (uniform — confirmed optimal at 600 SV).
Target: 98.67% float and Q6.10, 0 quantization flips.

Real data only — raises RuntimeError if wfdb / PhysioNet unavailable.

Outputs:
  confusion_comparison_m6.png  in the same directory as this script
"""

import os, sys, math, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FRAC_BITS     = 10
SCALE         = 1 << FRAC_BITS       # 1024
FEATURE_DIM   = 256
FEAT_SINGLE   = 128
FEAT_10BEAT   = 64
FEAT_100RR    = 64
NUM_CLASSES   = 5
CLASS_NAMES   = ["Normal", "PVC", "AFib", "VT", "SVT"]
DEFAULT_GAMMA = 0.25
NORMAL_RR     = 308

SV_ALLOC = [120, 120, 120, 120, 120]   # m6 uniform optimal allocation, total = 600

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (256-dim multi-scale: 128 beat + 64 mean + 64 RR)
# ─────────────────────────────────────────────────────────────────────────────
HALF_SINGLE = FEAT_SINGLE // 2
HALF_10BEAT = FEAT_10BEAT // 2
N_BEATS_10  = 10
N_BEATS_100 = 100
_BEAT_SYMS  = set("NLReEjJAaSVF/fQ")

def extract_multiscale(sig, all_beat_samples, beat_idx):
    s = int(all_beat_samples[beat_idx]); n = len(sig)
    if s < HALF_SINGLE or s + HALF_SINGLE > n:
        return None
    seg1 = sig[s - HALF_SINGLE : s + HALF_SINGLE].copy()
    pk1  = np.max(np.abs(seg1))
    if pk1 < 1e-6: return None
    seg1 = (seg1 / pk1).astype(np.float32)

    half10 = N_BEATS_10 // 2
    i0 = max(0, beat_idx - half10); i1 = min(len(all_beat_samples), beat_idx + half10)
    segs2 = []
    for bi in range(i0, i1):
        bs = int(all_beat_samples[bi])
        if bs >= HALF_10BEAT and bs + HALF_10BEAT <= n:
            seg = sig[bs - HALF_10BEAT : bs + HALF_10BEAT].astype(np.float32)
            pk2 = np.max(np.abs(seg))
            if pk2 > 1e-6: segs2.append(seg / pk2)
    if not segs2: return None
    seg2 = np.mean(segs2, axis=0).astype(np.float32)

    j0 = max(0, beat_idx - N_BEATS_100)
    rr_raw = np.diff(all_beat_samples[j0 : beat_idx + 1]).astype(np.float32)
    if len(rr_raw) < 2: return None
    rr_norm = np.clip(rr_raw / NORMAL_RR, 0.0, 2.0)
    seg3 = np.interp(np.linspace(0, 1, FEAT_100RR),
                     np.linspace(0, 1, len(rr_norm)), rr_norm).astype(np.float32)
    return np.concatenate([seg1, seg2, seg3])


def load_mitbih_beats(max_per_class=300):
    try:
        import wfdb
    except ImportError:
        raise RuntimeError("wfdb not installed. Run: pip install wfdb")
    BMAP = {"N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
            "V": 1, "E": 1, "F": 3,
            "A": 4, "a": 4, "J": 4, "S": 4, "/": 4, "f": 4, "Q": 4}
    sources = [
        (['100','101','102','103','104','105','106','107','108','109',
          '111','112','113','114','115','116','117','118','119','121',
          '122','123','124','200','201','202','203','205','207','208',
          '209','210','212','213','214','215','217','219','220','221',
          '222','223','228','230','231','232','233','234'], 'mitdb'),
        ([f'e{i:04d}' for i in range(1, 79)], 'svdb'),
        ([f'I{i:02d}' for i in range(1, 76)], 'incartdb'),
    ]
    beats = {i: [] for i in range(NUM_CLASSES)}
    for rec_list, pn_dir in sources:
        if all(len(v) >= max_per_class for v in beats.values()): break
        for rec in rec_list:
            if all(len(v) >= max_per_class for v in beats.values()): break
            try:
                r   = wfdb.rdrecord(rec, pn_dir=pn_dir)
                ann = wfdb.rdann(rec, 'atr', pn_dir=pn_dir)
            except Exception: continue
            sig = r.p_signal[:, 0].astype(np.float32)
            beat_samp_list, beat_sym_list = [], []
            for s, sym in zip(ann.sample, ann.symbol):
                if sym in _BEAT_SYMS:
                    beat_samp_list.append(s); beat_sym_list.append(sym)
            if len(beat_samp_list) < 3: continue
            all_beat_samples = np.array(beat_samp_list, dtype=np.int32)
            afib_regions = []
            if hasattr(ann, 'aux_note'):
                in_afib = False; afib_start = None
                for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
                    if sym == '+' and aux:
                        if '(AFIB' in aux: in_afib = True; afib_start = s
                        elif in_afib and '(' in aux:
                            afib_regions.append((afib_start, s)); in_afib = False
                if in_afib and afib_start is not None:
                    afib_regions.append((afib_start, len(sig)))
            for beat_idx, (s_idx, sym) in enumerate(
                    zip(all_beat_samples.tolist(), beat_sym_list)):
                in_afib_flag = any(a0 <= s_idx <= a1 for a0, a1 in afib_regions)
                if in_afib_flag and sym in ('N','L','R','e','j','V','A','a','J','S'):
                    cls = 2
                elif sym in BMAP: cls = BMAP[sym]
                else: continue
                if len(beats[cls]) >= max_per_class: continue
                feat = extract_multiscale(sig, all_beat_samples, beat_idx)
                if feat is not None: beats[cls].append(feat)
    return beats


def build_dataset(n_per_class=300):
    print("\n=== Loading ECG databases: MIT-BIH + SVDB + INCART (real data only) ===")
    real = load_mitbih_beats(max_per_class=n_per_class)
    X, y = [], []
    for cls in range(NUM_CLASSES):
        for b in real.get(cls, []):
            X.append(b); y.append(cls)
    if not X:
        raise RuntimeError("No real beats found — install wfdb and check network access.")
    for cls in range(NUM_CLASSES):
        print(f"  Class {cls} ({CLASS_NAMES[cls]:7s}): {len(real.get(cls, []))} real beats")
    return np.array(X, np.float32), np.array(y, np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Train 5 binary OVR SVMs with per-class SV budget
# ─────────────────────────────────────────────────────────────────────────────
def train_binary_ovr(X_tr, y_tr, sv_budget):
    sv_vecs, sv_alphas, biases, sv_counts = [], [], [], []
    for c in range(NUM_CLASSES):
        y_bin = np.where(y_tr == c, 1, -1)
        svm_c = SVC(kernel="rbf", gamma=DEFAULT_GAMMA, C=1.0, random_state=42)
        svm_c.fit(X_tr, y_bin)
        alphas = svm_c.dual_coef_[0]
        svs    = svm_c.support_vectors_
        n      = min(len(alphas), sv_budget[c])
        idx    = (np.argsort(-np.abs(alphas))[:sv_budget[c]]
                  if len(alphas) > sv_budget[c] else np.arange(len(alphas)))
        sv_vecs.append(svs[idx])
        sv_alphas.append(alphas[idx])
        biases.append(svm_c.intercept_[0])
        sv_counts.append(n)
        print(f"  Class {c} ({CLASS_NAMES[c]:7s}): {len(alphas):3d} natural SVs, "
              f"budget={sv_budget[c]}, using {n}")
    return sv_vecs, sv_alphas, np.array(biases), np.array(sv_counts, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Float OVR inference
# ─────────────────────────────────────────────────────────────────────────────
def ovr_predict_float(X_te, sv_vecs, sv_alphas, biases, gamma=DEFAULT_GAMMA):
    n_test = len(X_te)
    scores = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)
    for c in range(NUM_CLASSES):
        sv  = sv_vecs[c]; alp = sv_alphas[c]
        sq_x  = np.sum(X_te**2, axis=1, keepdims=True)
        sq_sv = np.sum(sv**2,   axis=1, keepdims=True).T
        dist2 = sq_x - 2 * (X_te @ sv.T) + sq_sv
        scores[:, c] = np.exp(-gamma * dist2) @ alp + biases[c]
    return np.argmax(scores, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Q6.10 OVR inference — pure Python, models svm_compute_core RTL exactly
# ─────────────────────────────────────────────────────────────────────────────
HORNER_COEFFS = [1024, 1024, 512, 170, 42, 8, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
EXP_INT_LUT   = [round(math.exp(-i) * SCALE) for i in range(16)]


def _to_i16(v):
    v = int(v) & 0xFFFF
    return v - 65536 if v >= 32768 else v


def hw_kernel(gamma_q, x_q, sv_q):
    """RBF kernel: distance accumulator + range-reduction exp LUT, all Q6.10."""
    acc = 0; prev_d = 0; prev_dsq = 0
    for k in range(FEATURE_DIM):
        xi = _to_i16(x_q[k]); si = _to_i16(sv_q[k])
        diff = _to_i16(xi - si)
        acc     += prev_dsq >> FRAC_BITS
        prev_dsq = prev_d * prev_d
        prev_d   = diff
    acc += prev_dsq >> FRAC_BITS      # drain 1
    prev_dsq = prev_d * prev_d
    acc += prev_dsq >> FRAC_BITS      # drain 2
    acc = min(acc, 0xFFFFF)
    P = _to_i16(gamma_q) * acc
    I = P >> 20; F_q = (P >> 10) & 1023
    if I >= 16: return 0
    lut_val = EXP_INT_LUT[I]
    x = _to_i16(-F_q)
    result = _to_i16(HORNER_COEFFS[15])
    for n in range(14, -1, -1):
        result = _to_i16(HORNER_COEFFS[n] + (_to_i16(x) * _to_i16(result) >> FRAC_BITS))
    horner_val = max(0, min(1024, result))
    return max(0, min(1024, (lut_val * horner_val) >> FRAC_BITS))


def vecs_to_q10(X):
    return np.clip(np.round(X * SCALE), -32768, 32767).astype(np.int32)


def ovr_predict_q10(X_te, sv_vecs, sv_alphas, biases, gamma=DEFAULT_GAMMA):
    gamma_q = int(round(gamma * SCALE)) & 0xFFFF
    n_test  = len(X_te)
    scores  = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)
    X_q     = vecs_to_q10(X_te)
    for c in range(NUM_CLASSES):
        SV_q = vecs_to_q10(sv_vecs[c])
        alp  = sv_alphas[c]
        for i in range(n_test):
            k_sum = 0.0
            for j in range(len(alp)):
                k = hw_kernel(gamma_q, X_q[i], SV_q[j])
                k_sum += alp[j] * (k / 1024.0)
            scores[i, c] = k_sum + biases[c]
    return np.argmax(scores, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_cm(ax, cm, title, acc, n_flips, fig):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True")
    flip_str = "" if n_flips is None else f"\nQuantization flips vs float: {n_flips}"
    ax.set_title(f"{title}\nAccuracy: {acc:.2%}{flip_str}", fontsize=10, fontweight="bold")
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh else "black")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    alloc_str = str(SV_ALLOC)
    print(f"\n{'='*60}")
    print(f"confusion_comparison_m6.py — sklearn OVR float vs ASIC Q6.10")
    print(f"SV allocation: {alloc_str}  total={sum(SV_ALLOC)}")
    print(f"{'='*60}")

    X, y = build_dataset(n_per_class=300)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    N_eval = len(X_te)
    print(f"\n  Train: {len(X_tr)}  Test: {N_eval}")

    # Train binary OVR SVMs
    print(f"\n=== Training 5 binary OVR SVMs — {alloc_str} ===")
    sv_vecs, sv_alphas, biases, sv_counts = train_binary_ovr(
        X_tr, y_tr, SV_ALLOC)
    print(f"  SV counts: {sv_counts}  total: {sv_counts.sum()}")

    # Col 1 — sklearn binary OVR float64
    print(f"\n=== Col 1: sklearn binary OVR (float64) ===")
    y_float = ovr_predict_float(X_te, sv_vecs, sv_alphas, biases)
    acc_float = accuracy_score(y_te, y_float)
    print(f"  Accuracy: {acc_float:.4f}")

    # Col 2 — ASIC Q6.10
    print(f"\n=== Col 2: ASIC Q6.10 hardware model ===")
    t0 = time.perf_counter()
    y_q10 = ovr_predict_q10(X_te, sv_vecs, sv_alphas, biases)
    acc_q10   = accuracy_score(y_te, y_q10)
    n_flips   = int(np.sum(y_float != y_q10))
    print(f"  Q6.10 done in {time.perf_counter()-t0:.1f}s  accuracy: {acc_q10:.4f}")
    print(f"  Quantization flips vs float: {n_flips}")
    if n_flips > 0:
        for i in np.where(y_float != y_q10)[0]:
            print(f"    Sample {i:3d}: true={CLASS_NAMES[y_te[i]]:<7} "
                  f"float={CLASS_NAMES[y_float[i]]:<7} "
                  f"Q6.10={CLASS_NAMES[y_q10[i]]}")

    cm_float = confusion_matrix(y_te, y_float, labels=list(range(NUM_CLASSES)))
    cm_q10   = confusion_matrix(y_te, y_q10,   labels=list(range(NUM_CLASSES)))

    # Summary line
    print(f"\n{'='*55}")
    print(f"{'':30} {'Float':>10} {'Q6.10':>10} {'Flips':>6}")
    print(f"{'-'*55}")
    print(f"{'Binary OVR '+alloc_str:<30} {acc_float:>10.4f} {acc_q10:>10.4f} {n_flips:>6}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_cm(axes[0], cm_float,
            f"sklearn binary OVR (float64)\n"
            f"SV allocation {alloc_str}",
            acc_float, None, fig)
    plot_cm(axes[1], cm_q10,
            f"ASIC (binary OVR, Q6.10)\n"
            f"SV allocation {alloc_str}",
            acc_q10, n_flips, fig)

    fig.suptitle(
        "RBF-SVM — sklearn binary OVR (float) vs ASIC Q6.10\n"
        f"SV allocation: {alloc_str}  (total={sum(SV_ALLOC)}, uniform optimal)\n"
        "gamma=0.25  ·  256-dim features (128+64+64)  ·  "
        "MIT-BIH + SVDB + INCART  ·  ECE410 PSU  ·  m6 IHP SG13G2",
        fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(SCRIPT_DIR, "confusion_comparison_m6.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
