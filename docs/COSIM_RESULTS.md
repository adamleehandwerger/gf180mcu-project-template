# Full-Dataset RTL Cosimulation — GF180MCU 600-SV SVM Classifier

**Design under test:** the signed-off `svm_compute_core` RTL (GF180MCU / wafer.space,
slot 0p5x1) — 600 support vectors [120×5], feature bank + alpha table in on-chip SRAM,
pipelined α×kernel MAC, Q6.10 fixed-point.

**Method:** the entire 300-sample held-out PhysioNet (MIT-BIH) test set is streamed through
the **actual RTL** in a native Verilog testbench (`cosim_tb.sv`). The unified off-chip RAM is
loaded with the 600-SV matrix (rows 0–599) and the 300 test-feature vectors (rows 600–899);
the Q6.10 model (gamma, 5 OVR biases, 600 alphas) is loaded over the real param/alpha
interface; `class_out` is captured at every `sample_rdy`. Predictions are scored against the
ground-truth labels and the pure-Python Q6.10 reference model.

Data + model come from `export_model.py`, which reuses the m6 pipeline that realizes 98.67%
(`random_state=42`, so the split and result reproduce exactly).

## Result

| Metric | Value |
|--------|-------|
| **Hardware (RTL) accuracy** | **98.67 % (296 / 300)** |
| Python Q6.10 reference | 98.67 % (296 / 300) |
| **RTL ≡ Q6.10 agreement** | **300 / 300 (100 %)** |

The RTL matches the reference model on **every** sample — the fixed-point hardware is
bit-for-bit faithful, with zero quantization flips.

## Confusion matrix (`confusion_matrix_cosim.png`)

| True \ Pred | Normal | PVC | AFib | VT | SVT | Recall |
|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Normal** | 60 | 0 | 0 | 0 | 0 | 100 % |
| **PVC**    | 0 | 60 | 0 | 0 | 0 | 100 % |
| **AFib**   | 0 | 0 | 60 | 0 | 0 | 100 % |
| **VT**     | 0 | 3 | 0 | 57 | 0 | 95.0 % |
| **SVT**    | 0 | 1 | 0 | 0 | 59 | 98.3 % |

All 4 errors are VT/SVT → PVC confusions — the clinically hardest pair to separate (all are
ventricular-morphology arrhythmias). Normal / PVC / AFib are classified perfectly.

## Reproduce

```
python3 export_model.py        # -> /tmp/svm_cosim_model.npz  (arm_env: sklearn+wfdb+net)
python3 export_hex.py          # -> /tmp/cosim_{ram,alpha}.hex  ($readmemh images)
iverilog -g2012 -DSIMULATION -DALPHA_SRAM_BEHAV_CELL -o /tmp/cosim_tb.out \
    cosim_tb.sv ../rt1/compute_core.sv ../rt1/alpha_sram.sv
vvp /tmp/cosim_tb.out           # -> /tmp/cosim_preds.txt  (~45 min, 300 samples)
python3 score_cosim.py          # -> confusion_matrix_cosim.png + accuracy
```

(A cocotb equivalent, `test_full_dataset_cosim.py`, produces the identical result but is ~30×
slower on long runs — the native TB is preferred for the full set.)
