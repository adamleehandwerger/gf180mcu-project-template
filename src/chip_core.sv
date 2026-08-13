// SPDX-FileCopyrightText: © 2026 Adam Handwerger
// SPDX-License-Identifier: Apache-2.0
//
// chip_core — GF180MCU wafer.space integration of svm_compute_core (SVM5740).
//
// Pin budget (slot 0p5x1): 4 input-only pads + 44 bidir pads (clk/rst_n are
// dedicated pads). svm_compute_core exposes ~217 signals assuming an external
// MCU + off-chip unified RAM, so a serial (SPI-like) host bridge is used for
// all config/results (Wishbone was tried on m5 and ran out of pins), while the
// wide, high-bandwidth off-chip RAM bus is mapped straight to bidir pads.
//
//   input pads : [0]=spi_sclk  [1]=spi_cs_n  [2]=spi_mosi  [3]=unused
//   bidir pads : [18:0]=ram_addr(out) [19]=ram_ren(out) [35:20]=ram_rdata(in)
//                [36]=spi_miso(out) [37]=done [38]=error [39]=sample_rdy
//                [42:40]=class_out[2:0] [43]=kernel_valid
//
// NOTE: the SPI bridge is new RTL; validate with the template's cocotb flow
// before trusting silicon behaviour. Synthesis is the first ORCA checkpoint.

`default_nettype none

module chip_core #(
    parameter NUM_INPUT_PADS,
    parameter NUM_BIDIR_PADS,
    parameter NUM_ANALOG_PADS
    )(
    `ifdef USE_POWER_PINS
    inout  wire VDD,
    inout  wire VSS,
    `endif

    input  wire clk,
    input  wire rst_n,

    input  wire [NUM_INPUT_PADS-1:0] input_in,
    output wire [NUM_INPUT_PADS-1:0] input_pu,
    output wire [NUM_INPUT_PADS-1:0] input_pd,

    input  wire [NUM_BIDIR_PADS-1:0] bidir_in,
    output wire [NUM_BIDIR_PADS-1:0] bidir_out,
    output wire [NUM_BIDIR_PADS-1:0] bidir_oe,
    output wire [NUM_BIDIR_PADS-1:0] bidir_cs,
    output wire [NUM_BIDIR_PADS-1:0] bidir_sl,
    output wire [NUM_BIDIR_PADS-1:0] bidir_ie,
    output wire [NUM_BIDIR_PADS-1:0] bidir_pu,
    output wire [NUM_BIDIR_PADS-1:0] bidir_pd,

    inout  wire [NUM_ANALOG_PADS-1:0] analog
);

    localparam int DW = 16;

    // ------------------------------------------------------------------
    // Pad static controls: no pulls; per-bit output-enable set below.
    // ------------------------------------------------------------------
    assign input_pu = '0;
    assign input_pd = '0;
    assign bidir_cs = '0;      // CMOS buffers
    assign bidir_sl = '1;      // slow slew (safe default for bring-up)
    assign bidir_pu = '0;
    assign bidir_pd = '0;

    // ------------------------------------------------------------------
    // SPI inputs (host is slow relative to clk): 2-FF synchronize + edge detect
    // ------------------------------------------------------------------
    logic [2:0] sclk_sync, cs_sync, mosi_sync;
    always_ff @(posedge clk) begin
        sclk_sync <= {sclk_sync[1:0], input_in[0]};
        cs_sync   <= {cs_sync[1:0],   input_in[1]};
        mosi_sync <= {mosi_sync[1:0], input_in[2]};
    end
    wire sclk_q   = sclk_sync[1];
    wire sclk_rise = (sclk_sync[1] & ~sclk_sync[2]);
    wire sclk_fall = (~sclk_sync[1] & sclk_sync[2]);
    wire cs_active = ~cs_sync[1];              // CS active low
    wire cs_start  = (~cs_sync[1] & cs_sync[2]); // falling edge = frame start

    // ------------------------------------------------------------------
    // SPI shift engine (mode 0, MSB first, byte-oriented)
    //   byte 0 = header: bit7 = 1:read / 0:write, bits[6:0] = address
    //   subsequent bytes = payload (MSB byte first)
    // ------------------------------------------------------------------
    logic [7:0]  rx_shift;
    logic [2:0]  bit_cnt;
    logic [3:0]  byte_cnt;
    logic        have_hdr;
    logic        rd_nwr;         // 1 = read transaction
    logic [6:0]  addr;
    logic [7:0]  rx_byte;
    logic        rx_byte_valid;  // 1-cycle pulse when a byte completes

    logic [7:0]  tx_shift;       // outgoing byte, shifted out MSB first on miso
    logic        miso_q;

    // Config holding registers (drive svm_compute_core inputs)
    logic [9:0]  cfg_num_samples;
    logic [39:0] cfg_num_sv_per_class;
    logic        cfg_vbatt_warn, cfg_vbatt_ok, cfg_kernel_ready;
    logic [2:0]  cfg_param_addr;
    logic [15:0] cfg_param_data;
    logic [9:0]  cfg_alpha_addr;
    logic [15:0] cfg_alpha_data;
    logic        start_pulse, param_we_pulse, alpha_we_pulse;

    // Byte-assembly staging for multi-byte writes
    logic [39:0] wr_accum;

    // Result sources (from core, latched for stable read-out)
    logic [15:0] gamma_reg, c_reg, kernel_out;
    logic [2:0]  class_out;
    logic        done_r, error_r, sample_rdy_r, kernel_valid_r;
    logic [3:0]  error_code;
    logic [127:0] class_scores_la;

    // Read address map
    localparam ADDR_CTRL=7'h00, ADDR_NSAMP=7'h01, ADDR_NSVPC=7'h02,
               ADDR_PARAM=7'h03, ADDR_ALPHA=7'h04,
               ADDR_STATUS=7'h40, ADDR_GAMMA=7'h41, ADDR_C=7'h42,
               ADDR_KERNEL=7'h43, ADDR_SCORES=7'h44;

    // Pack a status word for read-back
    wire [15:0] status_word =
        {sample_rdy_r, done_r, error_r, kernel_valid_r, error_code, 5'b0, class_out};

    // Select the byte to shift out for reads, indexed by byte_cnt (0 = first payload byte)
    function automatic [7:0] read_byte(input [6:0] a, input [3:0] idx);
        case (a)
            ADDR_STATUS: read_byte = idx[0] ? status_word[7:0]  : status_word[15:8];
            ADDR_GAMMA : read_byte = idx[0] ? gamma_reg[7:0]    : gamma_reg[15:8];
            ADDR_C     : read_byte = idx[0] ? c_reg[7:0]        : c_reg[15:8];
            ADDR_KERNEL: read_byte = idx[0] ? kernel_out[7:0]   : kernel_out[15:8];
            ADDR_SCORES: read_byte = class_scores_la[8*(15-idx) +: 8]; // MSB byte first
            default    : read_byte = 8'h00;
        endcase
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            bit_cnt <= '0; byte_cnt <= '0; have_hdr <= 1'b0;
            rd_nwr <= 1'b0; addr <= '0; rx_byte_valid <= 1'b0;
            start_pulse <= 1'b0; param_we_pulse <= 1'b0; alpha_we_pulse <= 1'b0;
            cfg_num_samples <= '0; cfg_num_sv_per_class <= '0;
            cfg_vbatt_warn <= 1'b0; cfg_vbatt_ok <= 1'b1; cfg_kernel_ready <= 1'b1;
            cfg_param_addr <= '0; cfg_param_data <= '0;
            cfg_alpha_addr <= '0; cfg_alpha_data <= '0;
            wr_accum <= '0; tx_shift <= '0; miso_q <= 1'b0;
        end else begin
            // default pulse clears
            rx_byte_valid  <= 1'b0;
            start_pulse    <= 1'b0;
            param_we_pulse <= 1'b0;
            alpha_we_pulse <= 1'b0;

            if (cs_start) begin
                bit_cnt <= '0; byte_cnt <= '0; have_hdr <= 1'b0;
            end

            if (cs_active && sclk_rise) begin
                // sample MOSI (MSB first)
                rx_shift <= {rx_shift[6:0], mosi_sync[1]};
                if (bit_cnt == 3'd7) begin
                    bit_cnt       <= '0;
                    rx_byte       <= {rx_shift[6:0], mosi_sync[1]};
                    rx_byte_valid <= 1'b1;
                end else begin
                    bit_cnt <= bit_cnt + 3'd1;
                end
            end

            // Prepare outgoing byte for reads at frame/byte boundaries
            if (cs_active && sclk_fall) begin
                miso_q  <= tx_shift[7];
                tx_shift <= {tx_shift[6:0], 1'b0};
            end

            // Byte-level processing
            if (rx_byte_valid) begin
                if (!have_hdr) begin
                    have_hdr <= 1'b1;
                    rd_nwr   <= rx_byte[7];
                    addr     <= rx_byte[6:0];
                    byte_cnt <= '0;
                    // preload first read byte
                    if (rx_byte[7]) tx_shift <= read_byte(rx_byte[6:0], 4'd0);
                end else begin
                    byte_cnt <= byte_cnt + 4'd1;
                    if (rd_nwr) begin
                        tx_shift <= read_byte(addr, byte_cnt + 4'd1);
                    end else begin
                        // WRITE payload handling (MSB byte first)
                        wr_accum <= {wr_accum[31:0], rx_byte};
                        case (addr)
                            ADDR_CTRL: begin
                                cfg_vbatt_warn   <= rx_byte[1];
                                cfg_vbatt_ok     <= rx_byte[2];
                                cfg_kernel_ready <= rx_byte[3];
                                if (rx_byte[0]) start_pulse <= 1'b1;
                            end
                            // multi-byte fields sent MSB-byte first; partial
                            // high byte carries the surplus bits in its LSBs
                            ADDR_NSAMP: if (byte_cnt==4'd1)
                                cfg_num_samples <= {wr_accum[1:0], rx_byte};
                            ADDR_NSVPC: if (byte_cnt==4'd4)
                                cfg_num_sv_per_class <= {wr_accum[31:0], rx_byte};
                            ADDR_PARAM: begin
                                if (byte_cnt==4'd0) cfg_param_addr <= rx_byte[2:0];
                                if (byte_cnt==4'd2) begin
                                    cfg_param_data <= {wr_accum[7:0], rx_byte};
                                    param_we_pulse <= 1'b1;
                                end
                            end
                            ADDR_ALPHA: begin
                                if (byte_cnt==4'd1) cfg_alpha_addr <= {wr_accum[1:0], rx_byte};
                                if (byte_cnt==4'd3) begin
                                    cfg_alpha_data <= {wr_accum[7:0], rx_byte};
                                    alpha_we_pulse <= 1'b1;
                                end
                            end
                            default: ;
                        endcase
                    end
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // Off-chip unified RAM bus <-> bidir pads
    // ------------------------------------------------------------------
    wire [18:0] ram_addr;
    wire [15:0] ram_rdata = bidir_in[35:20];
    wire        ram_ren;

    // ------------------------------------------------------------------
    // SVM compute core (full configuration)
    // ------------------------------------------------------------------
    svm_compute_core #(
        .DATA_WIDTH   (DW),
        .FEATURE_DIM  (256),
        .NUM_SV       (600)
    ) u_core (
    `ifdef USE_POWER_PINS
        .VDD                  (VDD),
        .VSS                  (VSS),
    `endif
        .clk                  (clk),
        .rst_n                (rst_n),
        .param_write_en       (param_we_pulse),
        .param_addr           (cfg_param_addr),
        .param_data           (cfg_param_data),
        .gamma_reg            (gamma_reg),
        .c_reg                (c_reg),
        .num_sv_per_class_flat(cfg_num_sv_per_class),
        .ram_addr             (ram_addr),
        .ram_rdata            (ram_rdata),
        .ram_ren              (ram_ren),
        .vbatt_warn           (cfg_vbatt_warn),
        .vbatt_ok             (cfg_vbatt_ok),
        .start                (start_pulse),
        .num_samples          (cfg_num_samples),
        .sample_rdy           (sample_rdy_r),
        .class_out            (class_out),
        .done                 (done_r),
        .error                (error_r),
        .error_code           (error_code),
        .kernel_out           (kernel_out),
        .kernel_valid         (kernel_valid_r),
        .kernel_ready         (cfg_kernel_ready),
        .class_scores_la      (class_scores_la),
        .alpha_write_en       (alpha_we_pulse),
        .alpha_addr           (cfg_alpha_addr),
        .alpha_data           (cfg_alpha_data)
    );

    // ------------------------------------------------------------------
    // Bidir pad output drive + per-bit output-enable
    //   outputs: ram_addr[18:0], ram_ren[19], miso[36], status bits[37..43]
    //   inputs : ram_rdata[35:20]
    // ------------------------------------------------------------------
    logic [NUM_BIDIR_PADS-1:0] o, oe;
    always_comb begin
        o  = '0;
        oe = '0;
        o[18:0]  = ram_addr;   oe[18:0]  = '1;
        o[19]    = ram_ren;    oe[19]    = 1'b1;
        // [35:20] ram_rdata are inputs -> oe stays 0
        o[36]    = miso_q;     oe[36]    = 1'b1;
        o[37]    = done_r;     oe[37]    = 1'b1;
        o[38]    = error_r;    oe[38]    = 1'b1;
        o[39]    = sample_rdy_r; oe[39]  = 1'b1;
        o[42:40] = class_out;  oe[42:40] = '1;
        o[43]    = kernel_valid_r; oe[43]= 1'b1;
    end

    assign bidir_out = o;
    assign bidir_oe  = oe;
    assign bidir_ie  = ~oe;   // enable input buffer only where not driving

    // input pad [3] and analog pads unused for now
    logic _unused;
    assign _unused = &{input_in[3], bidir_in[19:0], bidir_in[NUM_BIDIR_PADS-1:36],
                       sclk_q, cs_active, 1'b0};

endmodule

`default_nettype wire
