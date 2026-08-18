#!/usr/bin/env python3
"""
load_chip.py — field provisioning for the GF180MCU SVM cardiac-arrhythmia chip.

Programs a delivered chip with `svm_model_gf180.json`:
  * over SPI  (chip_core bridge): gamma, C, 5 OVR biases, per-class SV counts, 600 alphas
  * over the off-chip RAM bus     : the 600x256 support-vector matrix (rows 0..599)

The SPI byte protocol below matches the chip_core RTL exactly (mode 0, MSB-first,
header = {rd(1)/wr(0), addr[6:0]}). Wire the two abstract backends to your board:
  - SpiPort.xfer(tx) -> rx : full-duplex SPI (e.g. spidev on a Raspberry Pi, an FTDI
                             MPSSE adapter, or the host MCU's SPI master)
  - RamWriter.write(addr,word): write one 16-bit word to the external SRAM on the
                             chip's RAM bus (board-specific: MCU, or a mem-mapped port)

Usage:
  python3 load_chip.py svm_model_gf180.json            # dry-run: prints the byte stream
  python3 load_chip.py svm_model_gf180.json --spidev 0 0   # program via /dev/spidev0.0
"""
import argparse, json, sys
from abc import ABC, abstractmethod

# ---- register + param map (must match chip_core.sv) ----
ADDR_CTRL, ADDR_NSAMP, ADDR_NSVPC, ADDR_PARAM, ADDR_ALPHA = 0x00, 0x01, 0x02, 0x03, 0x04
ADDR_STATUS, ADDR_GAMMA, ADDR_C = 0x40, 0x41, 0x42
PARAM_GAMMA, PARAM_C, PARAM_BIAS0 = 0, 1, 2          # bias c -> PARAM_BIAS0 + c

def _hdr(addr, read=False):
    return (0x80 if read else 0x00) | (addr & 0x7F)

def _u16(v):
    v &= 0xFFFF
    return [(v >> 8) & 0xFF, v & 0xFF]              # hi, lo (MSB-byte first)


# ---------------------------------------------------------------- backends
class SpiPort(ABC):
    """Full-duplex SPI master. xfer(tx: bytes) -> rx: bytes of equal length. Mode 0, MSB-first."""
    @abstractmethod
    def xfer(self, tx: bytes) -> bytes: ...

class RamWriter(ABC):
    """Writes one 16-bit word (Q6.10) to the external SRAM on the chip's RAM bus."""
    @abstractmethod
    def write(self, addr: int, word: int) -> None: ...

class DryRunSpi(SpiPort):
    def __init__(self): self.n = 0
    def xfer(self, tx):
        self.n += len(tx)
        if len(tx) <= 6:
            print("  SPI ->", " ".join(f"{b:02x}" for b in tx))
        return bytes(len(tx))

class DryRunRam(RamWriter):
    def __init__(self): self.n = 0
    def write(self, addr, word): self.n += 1

class SpidevPort(SpiPort):
    """Linux spidev backend (e.g. Raspberry Pi). pip install spidev."""
    def __init__(self, bus, dev, hz=1_000_000):
        import spidev
        self.s = spidev.SpiDev(); self.s.open(bus, dev)
        self.s.mode = 0; self.s.max_speed_hz = hz
    def xfer(self, tx):
        return bytes(self.s.xfer2(list(tx)))


# ---------------------------------------------------------------- SPI ops
def spi_write(spi, addr, payload=()):
    spi.xfer(bytes([_hdr(addr, False)]) + bytes(payload))

def spi_read(spi, addr, n):
    rx = spi.xfer(bytes([_hdr(addr, True)] + [0x00] * n))
    return rx[1:1 + n]                              # first byte overlaps the header

def write_param(spi, sub, q):     spi_write(spi, ADDR_PARAM, [sub & 0x07] + _u16(q))
def write_alpha(spi, addr, q):    spi_write(spi, ADDR_ALPHA, [(addr >> 8) & 0x03, addr & 0xFF] + _u16(q))
def write_nsvpc(spi, counts):     spi_write(spi, ADDR_NSVPC, [counts[4], counts[3], counts[2], counts[1], counts[0]])
def write_nsamp(spi, n):          spi_write(spi, ADDR_NSAMP, [(n >> 8) & 0x03, n & 0xFF])
def write_ctrl(spi, start=0, vbatt_warn=0, vbatt_ok=1, kernel_ready=1):
    spi_write(spi, ADDR_CTRL, [(start & 1) | ((vbatt_warn & 1) << 1) |
                               ((vbatt_ok & 1) << 2) | ((kernel_ready & 1) << 3)])


# ---------------------------------------------------------------- load sequence
def load_model(spi, ram, model, verify=True):
    m = model
    ns = m["meta"]["num_sv"]
    print(f"[1/6] operating flags (vbatt_ok, kernel_ready)")
    write_ctrl(spi, vbatt_ok=1, kernel_ready=1)
    print(f"[2/6] gamma={m['gamma']}  C={m['c']}")
    write_param(spi, PARAM_GAMMA, m["gamma"])
    write_param(spi, PARAM_C,     m["c"])
    print(f"[3/6] biases {m['bias']}")
    for c, b in enumerate(m["bias"]):
        write_param(spi, PARAM_BIAS0 + c, b)
    print(f"[4/6] SV counts {m['sv_counts']}  (sum={sum(m['sv_counts'])})")
    write_nsvpc(spi, m["sv_counts"])
    print(f"[5/6] {ns} alpha coefficients over SPI")
    for a, q in enumerate(m["alpha"]):
        write_alpha(spi, a, q)
    print(f"[6/6] {ns}x{m['meta']['feature_dim']} SV matrix -> off-chip RAM (rows 0..{ns-1})")
    for r, row in enumerate(m["sv_matrix"]):
        base = r * 256
        for c, v in enumerate(row):
            ram.write(base + c, v & 0xFFFF)

    if verify:
        g = spi_read(spi, ADDR_GAMMA, 2)
        got = (g[0] << 8) | g[1] if len(g) == 2 else None
        exp = m["gamma"] & 0xFFFF
        ok = (got == exp)
        print(f"[verify] GAMMA readback = 0x{(got or 0):04x} (expected 0x{exp:04x}) -> "
              f"{'OK' if ok else 'MISMATCH — check SPI wiring/mode'}")
        st = spi_read(spi, ADDR_STATUS, 2)
        if len(st) == 2:
            status = (st[0] << 8) | st[1]
            print(f"[verify] STATUS = 0x{status:04x} (done={ (status>>14)&1 } error={ (status>>13)&1 })")
        return ok
    return True


def main():
    ap = argparse.ArgumentParser(description="Program the GF180 SVM chip from a model JSON.")
    ap.add_argument("model", help="svm_model_gf180.json")
    ap.add_argument("--spidev", nargs=2, type=int, metavar=("BUS", "DEV"),
                    help="use /dev/spidevBUS.DEV (real hardware)")
    ap.add_argument("--hz", type=int, default=1_000_000)
    args = ap.parse_args()

    with open(args.model) as f:
        model = json.load(f)
    assert model["meta"]["format"] == "gf180mcu-svm-model", "unrecognized model file"

    if args.spidev:
        spi = SpidevPort(args.spidev[0], args.spidev[1], args.hz)
        ram = _require_ram_backend()           # user must supply their board's RAM writer
    else:
        print("=== DRY RUN (no hardware) — showing the programming sequence ===")
        spi, ram = DryRunSpi(), DryRunRam()

    ok = load_model(spi, ram, model)
    print(f"\nprovisioning {'complete' if ok else 'FAILED verify'}: "
          f"{getattr(spi,'n',0)} SPI bytes, {getattr(ram,'n',0)} RAM words written.")
    print("Chip is ready — set NSAMP, stream input beats to RAM rows >= num_sv, pulse CTRL.start.")
    sys.exit(0 if ok else 1)

def _require_ram_backend():
    raise SystemExit(
        "No off-chip RAM backend configured. The SV matrix must be written to the external\n"
        "SRAM on the chip's RAM bus (addr = sv_row*256 + col). Implement RamWriter for your\n"
        "board (host MCU / memory-mapped port) and pass it to load_model().")

if __name__ == "__main__":
    main()
