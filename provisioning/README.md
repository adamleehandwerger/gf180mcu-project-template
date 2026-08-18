# Chip Provisioning Kit — GF180MCU SVM Cardiac Arrhythmia Classifier

Everything needed to load the trained model into the **delivered chip** at bring-up.

| File | Purpose |
|------|---------|
| **`svm_model_gf180.json`** | the deployable model — all weights as the exact Q6.10 integers the silicon consumes (gamma, C, 5 OVR biases, per-class SV counts, 600 alphas, the 600×256 SV matrix). Common JSON, language-agnostic. |
| **`load_chip.py`** | the loader/driver — programs the model over SPI + the off-chip RAM bus. Pure-Python, no math at bring-up. |
| **`export_chip_model.py`** | regenerates `svm_model_gf180.json` from the trained model (only needed to update the model). |

## What gets loaded where

The chip stores the model in two places (see `../docs/dataflow_block_diagram.png`):

- **Over SPI** (`chip_core` bridge — mode 0, MSB-first): `gamma`, `C`, the 5 per-class OVR
  **biases**, the per-class **SV counts**, and the **600 alpha** coefficients.
- **Over the off-chip RAM bus**: the **600×256 support-vector matrix** (rows 0…599, word
  address = `sv_row*256 + col`). The chip reads SVs from external SRAM at classification time.

`load_chip.py` implements both. `svm_model_gf180.json` carries both.

## Hardware hookup (two backends to wire)

The loader is board-agnostic — implement the two abstract backends for your setup:

1. **`SpiPort.xfer(tx) -> rx`** — a full-duplex SPI master to the chip's SPI pins
   (`input[0]=sclk`, `input[1]=cs_n`, `input[2]=mosi`, `bidir[36]=miso`), **mode 0, MSB-first**.
   A ready-made `SpidevPort` (Linux `/dev/spidev`, e.g. Raspberry Pi) is included.
2. **`RamWriter.write(addr, word)`** — writes one 16-bit word to the external SRAM on the
   chip's RAM bus. Board-specific (host MCU, or a memory-mapped interface).

Also required: assert **`rst_n`** low then high before loading (power-up reset).

## Run

```bash
# 1. Dry run — prints the exact byte stream, no hardware needed (sanity check):
python3 load_chip.py svm_model_gf180.json

# 2. Real hardware over Linux spidev (implement your RamWriter first, see load_chip.py):
python3 load_chip.py svm_model_gf180.json --spidev 0 0
```

The loader verifies by reading `GAMMA` back (expects `0x0100`) and `STATUS` (idle).

## After loading — run a classification

The model is now resident. To classify heartbeats:

1. Write the beat's 256 features to off-chip RAM rows ≥ `num_sv` (row `600+i` for beat `i`).
2. `NSAMP` = number of beats; pulse `CTRL.start`.
3. Poll `STATUS` (or watch pins) for `sample_rdy`/`done`; read **`class_out[2:0]`**:
   `0=Normal 1=PVC 2=AFib 3=VT 4=SVT`.

See `../docs/startup_instructions.pdf` §5–6 for the full run protocol.

## Notes

- The alpha SRAM is **volatile** — re-run `load_chip.py` after every power-up (keep the JSON
  in the host's non-volatile storage).
- The model is field-updatable: retrain, regenerate the JSON with `export_chip_model.py`, reload.
  The exp() kernel LUT is fixed in silicon (model-independent) and never changes.
- This exact model cosims at **98.67% (296/300)** on the PhysioNet test set through the signed-off
  RTL (`../../Indently/project/m7/cosim/`), matching the reference model bit-for-bit.
