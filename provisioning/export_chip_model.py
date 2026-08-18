#!/usr/bin/env python3
"""
export_chip_model.py — emit the deployable model artifact `svm_model_gf180.json`.

This is the file the field loader (`load_chip.py`) programs into the delivered chip.
Everything is stored as the exact Q6.10 16-bit integers the silicon consumes, so no
math is needed at bring-up time.

Source of the model:
  * default: reuse /tmp/svm_cosim_model.npz produced by ../../Indently/project/m7/cosim/
    export_model.py (the trained 600-SV [120x5] model that cosims at 98.67%).
  * pass --npz PATH to point elsewhere.

Layout of svm_model_gf180.json:
  meta        : format/version, dims, class names, Q-format, register map (reference)
  gamma, c    : Q6.10 ints (kernel width, SVM C)
  bias        : [5] Q6.10 ints (per-class OVR intercepts)
  sv_counts   : [5] support-vector counts (sum <= 600)
  alpha       : [600] Q6.10 ints (dual coefficients, class-ordered)
  sv_matrix   : [600][256] Q6.10 ints (support vectors, class-ordered; -> off-chip RAM)
"""
import argparse, json, os, sys
import numpy as np

CLASS_NAMES = ["Normal", "PVC", "AFib", "VT", "SVT"]
FRAC_BITS   = 10                       # Q6.10
C_DEFAULT_Q = 1 << FRAC_BITS           # C = 1.0

def i16(v):                            # store signed Q6.10 as a signed int (fits int16)
    v = int(v) & 0xFFFF
    return v - 65536 if v >= 32768 else v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/svm_cosim_model.npz")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "svm_model_gf180.json"))
    a = ap.parse_args()

    if not os.path.exists(a.npz):
        sys.exit(f"model source {a.npz} not found — first run "
                 f"Indently/project/m7/cosim/export_model.py to build it.")
    M = np.load(a.npz)
    SV, ALPHA, BIAS = M["SV"], M["alpha"], M["bias"]
    GAMMA, COUNTS   = int(M["gamma"]), [int(c) for c in M["counts"]]
    num_sv, feat_dim = SV.shape

    model = {
        "meta": {
            "format": "gf180mcu-svm-model", "version": 1,
            "device": "SVM cardiac arrhythmia classifier (GF180MCU / wafer.space, slot 0p5x1)",
            "num_sv": num_sv, "feature_dim": feat_dim, "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES, "q_frac_bits": FRAC_BITS, "q_note": "value = int/1024",
            "sv_matrix_target": "off-chip RAM, address = sv_row*256 + feature_col (rows 0..num_sv-1)",
            "spi_register_map": {
                "CTRL": "0x00", "NSAMP": "0x01", "NSVPC": "0x02", "PARAM": "0x03", "ALPHA": "0x04",
                "STATUS": "0x40", "GAMMA": "0x41", "C": "0x42", "KERNEL": "0x43", "SCORES": "0x44"},
            "param_subaddr": {"gamma": 0, "C": 1, "bias0": 2, "bias1": 3, "bias2": 4, "bias3": 5, "bias4": 6},
        },
        "gamma":     i16(GAMMA),
        "c":         C_DEFAULT_Q,
        "bias":      [i16(v) for v in BIAS],
        "sv_counts": COUNTS,
        "alpha":     [i16(v) for v in ALPHA],
        "sv_matrix": [[i16(v) for v in row] for row in SV],
    }
    with open(a.out, "w") as f:
        json.dump(model, f)
    sz = os.path.getsize(a.out)
    print(f"wrote {a.out}  ({sz/1e6:.2f} MB)")
    print(f"  num_sv={num_sv}  feat_dim={feat_dim}  sv_counts={COUNTS}  gamma={i16(GAMMA)}  "
          f"bias={[i16(v) for v in BIAS]}")

if __name__ == "__main__":
    main()
