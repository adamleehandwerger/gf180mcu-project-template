// SPDX-FileCopyrightText: © 2026 Adam Handwerger
// SPDX-License-Identifier: Apache-2.0
//
// alpha_sram_1024x16 — on-chip alpha coefficient store for svm_compute_core.
//
// Replaces the old NUM_SV x 16 flip-flop register file (which made SV count an
// on-chip area cost). NUM_SV = 600 (the accuracy optimum) needs a 1024-deep store,
// built from 4x gf180mcu_fd_ip_sram__sram512x8m8wm1 foundry macros:
//   2 banks deep (addr[9] selects) x 2 byte lanes ([7:0], [15:8]) = 1024 x 16.
//
// Single-port, synchronous, 1-cycle read latency. Controls are ACTIVE-LOW
// (CEN/GWEN/WEN) per the GF180 SRAM convention. Power is delivered via
// PDN_MACRO_CONNECTIONS in the LibreLane config (and via USE_POWER_PINS for
// power-aware gate-level sim).
//
// For tools without the GF180 PDK (local iverilog/yosys elaboration, functional
// sim) define ALPHA_SRAM_BEHAV_CELL to compile the behavioral macro model below;
// on Orca the PDK supplies the real blackbox (synth) + model (sim), so leave it
// undefined there.

`default_nettype none

module alpha_sram_1024x16 (
`ifdef USE_POWER_PINS
    inout  wire        VDD,
    inout  wire        VSS,
`endif
    input  wire        clk,
    input  wire        ce,        // access enable (active high)
    input  wire        we,        // write enable (active high)
    input  wire [9:0]  addr,
    input  wire [15:0] wdata,
    output wire [15:0] rdata
);
    wire        bank = addr[9];
    wire [8:0]  a    = addr[8:0];
    wire        cen0 = ~(ce & ~bank);        // active-low chip enable, bank 0
    wire        cen1 = ~(ce &  bank);        // active-low chip enable, bank 1
    wire        gwen = ~we;                  // active-low global write enable
    wire [7:0]  wen  = we ? 8'h00 : 8'hFF;   // active-low per-bit write (0=write)
    wire [7:0]  q0lo, q0hi, q1lo, q1hi;

    gf180mcu_fd_ip_sram__sram512x8m8wm1 u0lo (
    `ifdef USE_POWER_PINS .VDD(VDD), .VSS(VSS), `endif
        .CLK(clk), .CEN(cen0), .GWEN(gwen), .WEN(wen), .A(a), .D(wdata[7:0]),  .Q(q0lo));
    gf180mcu_fd_ip_sram__sram512x8m8wm1 u0hi (
    `ifdef USE_POWER_PINS .VDD(VDD), .VSS(VSS), `endif
        .CLK(clk), .CEN(cen0), .GWEN(gwen), .WEN(wen), .A(a), .D(wdata[15:8]), .Q(q0hi));
    gf180mcu_fd_ip_sram__sram512x8m8wm1 u1lo (
    `ifdef USE_POWER_PINS .VDD(VDD), .VSS(VSS), `endif
        .CLK(clk), .CEN(cen1), .GWEN(gwen), .WEN(wen), .A(a), .D(wdata[7:0]),  .Q(q1lo));
    gf180mcu_fd_ip_sram__sram512x8m8wm1 u1hi (
    `ifdef USE_POWER_PINS .VDD(VDD), .VSS(VSS), `endif
        .CLK(clk), .CEN(cen1), .GWEN(gwen), .WEN(wen), .A(a), .D(wdata[15:8]), .Q(q1hi));

    // Q is registered inside the macro (1-cycle); delay bank select to match.
    reg bank_r;
    always @(posedge clk) if (ce) bank_r <= bank;
    assign rdata = bank_r ? {q1hi, q1lo} : {q0hi, q0lo};
endmodule


// feature_sram_512x16 — on-chip feature bank (FEATURE_DIM=256 used of 512 depth).
// Restores the intended SRAM implementation of feature_bank (the IHP/m5 core had
// it forced to flip-flops via ram_style="registers", which created the 256:1 read
// mux + ~1030-fanout feat_rd_addr broadcast + 4096 loads on the clock that jammed
// routing). Single bank, 2 byte lanes = 2x gf180mcu_fd_ip_sram__sram512x8m8wm1.
// Single-port, 1-cycle registered read (matches the existing feat_rd_data timing).
module feature_sram_512x16 (
`ifdef USE_POWER_PINS
    inout  wire        VDD,
    inout  wire        VSS,
`endif
    input  wire        clk,
    input  wire        ce,        // access enable (active high)
    input  wire        we,        // write enable (active high)
    input  wire [8:0]  addr,
    input  wire [15:0] wdata,
    output wire [15:0] rdata
);
    wire       cen  = ~ce;                  // active-low chip enable
    wire       gwen = ~we;                  // active-low global write enable
    wire [7:0] wen  = we ? 8'h00 : 8'hFF;   // active-low per-bit write
    wire [7:0] qlo, qhi;

    gf180mcu_fd_ip_sram__sram512x8m8wm1 ulo (
    `ifdef USE_POWER_PINS .VDD(VDD), .VSS(VSS), `endif
        .CLK(clk), .CEN(cen), .GWEN(gwen), .WEN(wen), .A(addr), .D(wdata[7:0]),  .Q(qlo));
    gf180mcu_fd_ip_sram__sram512x8m8wm1 uhi (
    `ifdef USE_POWER_PINS .VDD(VDD), .VSS(VSS), `endif
        .CLK(clk), .CEN(cen), .GWEN(gwen), .WEN(wen), .A(addr), .D(wdata[15:8]), .Q(qhi));

    assign rdata = {qhi, qlo};   // Q is registered inside the macro (1-cycle)
endmodule

`ifdef ALPHA_SRAM_BEHAV_CELL
// Behavioral model of the foundry 512x8 single-port SRAM — functional sim / local
// elaboration ONLY. The GF180 PDK provides the real macro on Orca; do not define
// ALPHA_SRAM_BEHAV_CELL there.
module gf180mcu_fd_ip_sram__sram512x8m8wm1 (
`ifdef USE_POWER_PINS
    inout  wire        VDD,
    inout  wire        VSS,
`endif
    input  wire        CLK,
    input  wire        CEN,   // active-low chip enable
    input  wire        GWEN,  // active-low global write enable
    input  wire [7:0]  WEN,   // active-low per-bit write enable
    input  wire [8:0]  A,
    input  wire [7:0]  D,
    output reg  [7:0]  Q
);
    reg [7:0] mem [0:511];
    integer b;
    always @(posedge CLK) if (!CEN) begin
        if (!GWEN) begin
            for (b = 0; b < 8; b = b + 1)
                if (!WEN[b]) mem[A][b] <= D[b];
        end else begin
            Q <= mem[A];
        end
    end
endmodule
`endif

`default_nettype wire
