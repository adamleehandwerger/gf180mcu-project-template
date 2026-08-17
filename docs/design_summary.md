# SVM Compute Core — Full-Chip Design Summary (GF180MCU / wafer.space Re-target)

**Project:** Multi-Class Cardiac Arrhythmia Detection — GlobalFoundries GF180MCU Open Shuttle
**Technology:** GF180MCU 180 nm (variant **gf180mcuD**, 5 metal layers, 1.1 µm top metal)
**Flow:** LibreLane (nix / `leo/gf180mcu` branch) via the wafer.space `gf180mcu-project-template`, hardened on Orca HPC (SLURM + Apptainer)
**Repo:** [adamleehandwerger/gf180mcu-project-template](https://github.com/adamleehandwerger/gf180mcu-project-template) (fork of `wafer-space/gf180mcu-project-template`)
**Architecture:** SPI slave + unified SVM core, wrapped in the wafer.space fixed pad frame (`chip_top` → `chip_core`)
**RTL:** `src/compute_core.sv` (unified core) + `src/chip_core.sv` (SPI bridge) — NUM_SV=600, SV_ALLOC=[120,120,120,120,120], **alpha store = on-chip SRAM macro** (was register file)
**Status:** RTL integrated + elaborates (yosys); harden pending on Orca

---

## Motivation for the GF180MCU Re-target

The m6 design targeted IHP SG13G2 (130 nm) as a standalone SPI die. During hardening the
IHP die hit **local metal-density (fill) DRC violations** that could not be cleared without
a multi-day top-level re-integration, and a foundry waiver was not available. The register-based
`alpha_table[NUM_SV]` (NUM_SV × 16 flip-flops) was a major area/density contributor and forced
the SV count down from the accuracy-optimal **600** to **500** ([95,95,95,120,95]).

wafer.space offers a GF180MCU open shuttle with two properties that directly fix both problems:

| Feature | IHP SG13G2 (m6) | GF180MCU / wafer.space (this build) |
|---------|-----------------|--------------------------------------|
| Process | 130 nm BiCMOS | 180 nm (gf180mcuD, 5-metal, 1.1 µm TM) |
| Metal layers | Metal1–Metal5 | Metal1–Metal5 (**+ thick top metal**) |
| On-chip SRAM macros | not used (alpha in FF) | **available** → alpha table in SRAM |
| Die | standalone 2400×2400 µm (self-drawn pad ring) | **fixed template pad frame**, slot-selectable |
| Interface | SPI slave (direct GPIO) | SPI slave (same) |
| Flow | OpenLane 2 | LibreLane (nix), template-driven |
| Submission | IHP quarterly MPW | wafer.space GF180MCU shuttle |

The GF180 SRAM macros let the alpha coefficients live in a compact macro instead of ~9,600
flip-flops, **removing the NUM_SV↔area coupling** and enabling the intended 600-SV optimum
on a modest slot.

---

## Slot & Die (wafer.space `0p5x1`)

The template die and pad ring are fixed by the chosen `SLOT`; the user design occupies the core.

| Parameter | Value |
|-----------|-------|
| Slot | `0p5x1` (`SLOT=0p5x1`) |
| Die area | 1936 × 5122 µm |
| Core area (usable) | 1052 × 4238 µm ≈ **4.46 mm²** |
| Pads | 4 input-only + 44 bidir + 6 analog (+ power/ground) |
| PL_TARGET_DENSITY_PCT | 35 (template default) |
| Clock | `clk_PAD`, 25 MHz (40 ns) template default |

*(The roomier `1x1` slot — core 3048×4238 ≈ 12.9 mm² — is the fallback if 600 SVs + the SRAM
macro do not close on `0p5x1`.)*

---

## Component Summary

### svm_compute_core (unified core, GF180MCU — harden pending)

| Metric | Value |
|--------|-------|
| NUM_SV | **600** ([120,120,120,120,120], runtime `num_sv_per_class`) |
| FEATURE_DIM | 256 (128 single-beat + 64 10-beat mean + 64 RR) |
| Precision | Q6.10 (16-bit signed) |
| Alpha store | **on-chip single-port SRAM (`alpha_mem`, macro-ready ~1024×16)** — was `ram_style="registers"` |
| alpha_addr width | 10-bit |
| Off-chip RAM | unified read bus `ram_addr[18:0]`/`ram_rdata[15:0]`/`ram_ren` (SV matrix rows 0..599, input rows 600..) |
| Result | `class_out[2:0]` + `sample_rdy` per beat, `done` at batch end |
| RAM_LATENCY | 3 |
| Timing / power / DRC | TBD (Orca harden) |

### chip_core (SPI bridge + pad map — `src/chip_core.sv`)

The wafer.space pad budget (~48 digital pads) cannot expose the core's wide config bus, so
`chip_core` is an **SPI-slave register bridge** (Wishbone was tried on m5 and ran out of pins):

| Interface | Pads |
|-----------|------|
| `clk`, `rst_n` | dedicated pads |
| SPI in (`sclk`, `cs_n`, `mosi`) | 3 input-only pads |
| Off-chip RAM bus (`ram_addr[18:0]`, `ram_ren`, `ram_rdata[15:0]`) | 36 bidir |
| `spi_miso`, `done`, `error`, `sample_rdy`, `class_out[2:0]`, `kernel_valid` | remaining bidir |

SPI register file expands to `param_*` / `alpha_*` / `num_sv_per_class` / `num_samples` / `start`
and shifts results back (`STATUS`/`GAMMA`/`C`/`KERNEL`/`SCORES`). See `src/chip_core.sv`.

---

## Results Architecture — Classifications over Time

The clinically meaningful output is **the arrhythmia class per heartbeat over the recording**
(the physician reviews the class time series over days/weeks). The chip emits one
`class_out[2:0]` per beat (`sample_rdy` pulse); at ~1 beat/s the sizing is:

| Scope | Size | Lives in |
|-------|------|----------|
| One batch (1000 beats) | ~1 KB | small on-chip result buffer (drained per batch) |
| Weeks of batches | several MB | **MCU external storage** (flash/SD) |

The chip only holds one batch; the MCU appends each drained batch to the long-term record.

---

## Functional Results (model-level, carried from m6)

Training/testing on pooled PhysioNet ECG (MIT-BIH + SVDB + INCART), 80/20 stratified,
`random_state=42`, 240 train + 60 test beats/class (1,200 train / 300 test), 256-dim features,
Q6.10.

| Implementation | Accuracy | SVs |
|---|---|---|
| sklearn binary OVR (float) | **98.67%** (296/300) | 600 [120×5] |
| ASIC binary OVR (Q6.10, cosim) | **98.67%** (296/300, 0 quant flips) | 600 [120×5] |

Per-class recall (final RTL cosim): Normal 100% · PVC 100% · AFib 100% · VT 95.0% (57/60) ·
SVT 98.3% (59/60); all 4 errors are VT/SVT→PVC. 600-SV uniform `[120×5]` is the accuracy optimum.

**Full-dataset RTL cosim (DONE 2026-08-16):** the entire 300-sample PhysioNet test set was
streamed through the actual signed-off `compute_core` RTL (native Verilog TB, `m7/cosim/`):
**98.67% (296/300)**, and the RTL matched the Q6.10 reference model on **300/300 samples**
(bit-exact, 0 flips). 24/24 RTL testbenches also pass. See `COSIM_RESULTS.md` +
`confusion_matrix_cosim.png`.

**Physical sign-off (job 127101):** DRC 0, LVS 0, antenna 0, hold +0.26 ns. Setup closes 25 MHz
at typical (`tt_025C_5v00` +14 ns) and fast (`ff` +20 ns) corners; the worst-case slow corner
(`ss_125C_4v50`) is −5.07 ns — a documented limitation of the slot's 1:4 aspect ratio (long
wire-dominated nets; density can't rise without routing congestion). GDS built from committed RTL.

---

## m6 (IHP) → GF180MCU Delta

| Parameter | m6 (IHP SG13G2) | GF180MCU (this build) |
|-----------|------------------|------------------------|
| Process | IHP 130 nm | GF180MCU 180 nm (gf180mcuD) |
| Flow | OpenLane 2 | LibreLane (nix) / wafer.space template |
| Die | standalone 2400×2400 µm | template slot `0p5x1` (core 4.46 mm²) |
| NUM_SV (achieved) | 500 [95,95,95,120,95] (density-forced) | **600 [120,120,120,120,120]** |
| Alpha store | `alpha_table` = registers (~9.6k FF) | **on-chip SRAM macro (~1024×16)** |
| Density/fill | 5 unresolved M2/M3 violations | avoided (SRAM macro removes FF bulk) |
| Interface | SPI | SPI (same) |
| Accuracy (Q6.10 target) | 98.67% (design) / 500-SV silicon | **98.67% (600-SV optimum)** |

---

## Runtime Model Reload / Field Update

The chip is fully field-reprogrammable — a retrained model is loaded without re-synthesis:

| Component | Location | Update path |
|-----------|----------|-------------|
| SV matrix (600×256×16b, ~300 KB) | external SV SRAM | MCU writes SRAM |
| Alpha coeffs (600×16b, ~1.2 KB) | on-chip `alpha_mem` SRAM | SPI `ALPHA_WR` ×600 |
| Gamma / C / bias[0-4] | on-chip param regs | SPI `PARAM_WR` |
| SV counts per class | runtime regs | SPI `NUM_SV[0-4]` |

**Constraints (fixed silicon):** ≤ 600 SVs (alpha SRAM depth; 10-bit addr ceiling 1024),
exactly 256 features, 5 classes, Q6.10 range. **Behavior change vs m6:** the alpha SRAM has
**no reset-to-1.0 default** — the host must load all 600 coefficients before the first classify.

---

## Orca Harden (pending)

```bash
# In the fork, on Orca (nix + wafer.space gf180 PDK)
make clone-pdk                 # wafer.space gf180mcu PDK fork
nix-shell                      # pinned LibreLane (leo/gf180mcu)
SLOT=0p5x1 make librelane      # synth → PnR → GDS for chip_top
```

Remaining work before submission:
1. Instantiate the concrete GF180 SRAM macro for `alpha_mem` (currently macro-ready RTL).
2. Re-simulate the unified core (alpha-SRAM read latency, NUM_SV=600) — cocotb + iverilog.
3. Close timing/DRC/density on `0p5x1`; fall back to `1x1` slot if needed.
4. Post-layout power (GF180 Liberty `nom_tt` corner).

---

*Document version: GF180MCU re-target · derived from ECE410 `main` `project/m6/design_summary.md`.
Key changes: GF180MCU/wafer.space flow + fixed pad frame; NUM_SV=600 [120×5] achieved via
on-chip alpha SRAM macro (removes the IHP density limiter); SPI interface retained.*
