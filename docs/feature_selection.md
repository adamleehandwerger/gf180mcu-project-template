# Feature Selection — Multi-Scale ECG Features for the GF180MCU SVM Classifier

The classifier consumes a **256-dimensional feature vector** per heartbeat. It is deliberately
**multi-scale**: three concatenated blocks that each capture the ECG at a different time horizon.
This is what lets one RBF-SVM separate all five classes (Normal, PVC, AFib, VT, SVT) — some
arrhythmias are defined by a *single beat's shape*, others only by *rhythm over many beats*.

| Block | Dims | Time scale | What it captures |
|-------|-----:|-----------|------------------|
| **Single beat** | 128 | ~0.36 s (one QRS) | beat morphology (shape) |
| **10-beat average** | 64 | ~7–9 s | denoised "typical" morphology |
| **100-beat RR** | 64 | ~60–90 s | rhythm / rate / variability |
| **Total** | **256** | — | shape **and** timing, multi-scale |

Signals are MIT-BIH-family recordings sampled at **360 Hz**; beats are located at their annotated
R-peaks. All amplitude blocks are **peak-normalized** (divided by their max |amplitude|) so the
Q6.10 fixed-point range is used well and inter-patient gain differences are removed.

Data sources (real only): **MIT-BIH Arrhythmia** (`mitdb`), **MIT-BIH Supraventricular** (`svdb`),
and **St. Petersburg INCART** (`incartdb`).

---

## 1. Single-beat characteristics — 128 dims (`seg1`)

The 128 raw signal samples centered on the R-peak (**±64 samples ≈ ±0.18 s**), peak-normalized:

```
seg1 = sig[R-64 : R+64] / max(|sig[R-64 : R+64]|)
```

- **Purpose:** the instantaneous **morphology** of the beat — QRS width, R-wave amplitude/polarity,
  ST shape. This is the primary discriminator for beats defined by their shape.
- **Why it matters per class:** a **PVC** is a wide, bizarre QRS with no preceding P-wave — visible
  directly in a single beat. **VT** (ventricular, fusion-type here) also has an abnormal single-beat
  QRS. Normal beats have a narrow, consistent QRS.
- **Limitation it exposes:** a single noisy beat can be misleading, and rhythm information (rate,
  regularity) is entirely absent — which is why the next two blocks exist.

## 2. Ten-beat characteristics — 64 dims (`seg2`)

The **mean** of 10 consecutive beats (the current beat ±5), each taken as 64 samples (**±32 ≈
±0.09 s**) around its R-peak and peak-normalized before averaging:

```
seg2 = mean_over_10_beats( sig[Rk-32 : Rk+32] / max|·| )
```

- **Purpose:** a **noise-averaged, stable template** of the patient's beat shape over a short window
  (~7–9 s). Averaging suppresses baseline wander, muscle noise, and one-off artifacts that would
  distort a single beat.
- **Why it matters per class:** it establishes the **local "normal" morphology** so the classifier
  can judge how anomalous the current single beat is *relative to the patient's own recent beats*
  (inter-patient morphology varies a lot; this normalizes for it). Sustained ventricular rhythms
  keep an abnormal averaged shape, while isolated ectopy averages back toward normal.
- **Scale choice:** 10 beats is long enough to denoise but short enough to stay within one rhythm
  episode (it won't blur across a rhythm change the way a very long average would).

## 3. Hundred-beat characteristics — 64 dims (`seg3`)

The **RR-interval sequence** over the last ~100 beats — the timing between successive R-peaks —
normalized and resampled to 64 points:

```
rr      = diff(R_positions over last 100 beats)          # samples between beats
rr_norm = clip(rr / NORMAL_RR, 0, 2)                      # NORMAL_RR = 308 samples ≈ 0.855 s (~70 bpm)
seg3    = interp(rr_norm -> 64 points)
```

- **Purpose:** **rhythm** — heart rate, regularity, and beat-to-beat variability over a long window
  (~60–90 s). This is *timing*, not shape, and it's the only block that sees the long horizon.
- **Why it matters per class:** several arrhythmias are **defined by rhythm, not morphology**:
  - **AFib** — irregularly irregular RR intervals (high variability, no pattern). Individual beats
    can look near-normal; only the RR sequence reveals it. In this dataset AFib is labeled from the
    recording's rhythm annotations, so RR features are essential to learn it.
  - **SVT / VT** — sustained *fast* rhythms show a run of short, regular RR intervals.
  - **Normal** — regular RR near 1.0 (after normalization by `NORMAL_RR`).
- **Design choices:** normalizing by `NORMAL_RR` makes ~70 bpm map to ~1.0 and clipping to `[0,2]`
  bounds pathological extremes; resampling to a fixed 64 points makes the block a constant length
  regardless of how many beats are available.

---

## Why multi-scale (design rationale)

No single scale separates all five classes:

| Class | Distinguished mainly by |
|-------|-------------------------|
| Normal | narrow QRS + regular RR |
| PVC | abnormal **single-beat** morphology |
| VT | abnormal **sustained** morphology (single + 10-beat) |
| SVT | **fast, regular RR** (100-beat) + morphology |
| AFib | **irregular RR** (100-beat), near-normal beats |

Concatenating morphology (128 + 64) with rhythm (64) gives the RBF kernel enough to place all five
classes. The full-dataset RTL cosim confirms the choice: **98.67% (296/300)**, with the only errors
being VT/SVT→PVC (the hardest morphology overlap). See `../../Indently/project/m7/cosim/COSIM_RESULTS.md`.

## Fixed-point / hardware notes

- The feature vector is quantized to **Q6.10** (`value × 1024`, clamped to signed 16-bit) before it
  enters the chip — the same format as the SVs and alphas. Peak-normalization keeps morphology
  blocks in `[-1, 1]` and the RR block in `[0, 2]`, well inside Q6.10 range.
- `FEATURE_DIM = 256` is fixed in silicon (the distance accumulator and the off-chip RAM row width),
  so the **256-dim layout above is part of the hardware contract** — retrained models must keep the
  128 / 64 / 64 block structure and the same extraction to stay compatible.

## Reference implementation

The exact feature extraction used throughout this project (and in the accuracy simulation that
realizes 98.67%) is `extract_multiscale()` in **`feature_extraction_sim.py`** (this directory) — the
same script that loads the MIT-BIH/SVDB/INCART data, trains the 600-SV [120×5] OVR RBF-SVM, and runs
the Q6.10 hardware-model confusion matrix. Constants: `FEAT_SINGLE=128`, `FEAT_10BEAT=64`,
`FEAT_100RR=64`, `HALF_SINGLE=64`, `HALF_10BEAT=32`, `N_BEATS_10=10`, `N_BEATS_100=100`,
`NORMAL_RR=308`.
