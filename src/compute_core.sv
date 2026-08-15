// SPDX-FileCopyrightText: 2026 Adam Handwerger
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Multi-Class Cardiac Arrhythmia Detection — SVM Compute Core  v12  (batch)
// ECE 410 Project  |  Milestone: m6
//
// Batch architecture:
//   Host collects 1000 heartbeats at low power, extracts 256-dim features,
//   pre-loads both the input matrix and the SV matrix into off-chip SRAM,
//   then fires start.  The ASIC drives the outer sample loop autonomously.
//
// v12 changes vs v11:
//   - FEATURE_DIM unchanged at 256; feature split back to original 128-64-64
//     (128 single-beat morphology + 64 10-beat avg + 64 RR intervals)
//   - NUM_SV reduced 600 -> 500; allocation [95,95,95,120,95] (VT-boosted)
//   - Q6.10 accuracy: 97.67% (146/150)
//
// Off-chip address map  (row x FEATURE_DIM layout, FEATURE_DIM = 256 = 2^8):
//   Rows  0 .. NUM_SV-1              SV matrix      (500 x 256 = 128 000 words)
//   Rows  NUM_SV .. NUM_SV+batch-1   input matrix   (1000 x 256 = 256 000 words)
//   Maximum address: (500 + 1000) x 256 - 1 = 384 000  ->  19-bit address bus
//
// Per-sample output:
//   sample_rdy pulses one cycle per WRITE_CLASS.
//   class_out[2:0] is stable when sample_rdy fires.
//   Host captures class_out on every sample_rdy IRQ.
//   done pulses once at the end of the batch (last WRITE_CLASS).
//
// FIXED-POINT FORMAT
//   Q6.10  (16-bit, 10 fractional bits)
//   real_value = raw / 1024.0
//   γ = 0.25  →  raw = 256 = 0x0100  (exact)
// ============================================================================

`default_nettype none

module svm_compute_core #(
    parameter int  DATA_WIDTH     = 16,
    parameter int  FRAC_BITS      = 10,
    parameter int  DIST_WIDTH     = 20,
    parameter int  FEATURE_DIM    = 256,
    parameter int  NUM_SV         = 600,
    parameter int  MAX_BATCH_SIZE = 1000,
    parameter int  RAM_LATENCY    = 3,     // cycles from ram_ren assert to ram_rdata valid
    parameter real DEFAULT_GAMMA  = 0.25,
    parameter real DEFAULT_C      = 1.0,
    parameter real DEFAULT_BIAS_0 = 0.0,
    parameter real DEFAULT_BIAS_1 = 0.0,
    parameter real DEFAULT_BIAS_2 = 0.0,
    parameter real DEFAULT_BIAS_3 = 0.0,
    parameter real DEFAULT_BIAS_4 = 0.0
) (
`ifdef USE_POWER_PINS
    inout wire                      VDD,
    inout wire                      VSS,
`endif
    input  logic                    clk,
    input  logic                    rst_n,

    // Parameter write port (Wishbone-driven, config only)
    input  logic                    param_write_en,
    input  logic [2:0]              param_addr,
    input  logic [DATA_WIDTH-1:0]   param_data,
    output logic [DATA_WIDTH-1:0]   gamma_reg,
    output logic [DATA_WIDTH-1:0]   c_reg,

    // SV counts (Wishbone registers, latched at start)
    input  logic [39:0]             num_sv_per_class_flat,

    // Unified off-chip RAM  (input matrix + SV matrix, host serves from SRAM)
    output logic [18:0]             ram_addr,
    input  logic [DATA_WIDTH-1:0]   ram_rdata,
    output logic                    ram_ren,

    // Battery monitors (async; 2-FF synchronized internally)
    input  logic                    vbatt_warn,
    input  logic                    vbatt_ok,

    // Batch control
    input  logic                    start,
    input  logic [9:0]              num_samples,

    // Per-sample result — fires every heartbeat
    output logic                    sample_rdy,
    output logic [2:0]              class_out,

    // Batch completion
    output logic                    done,
    output logic                    error,
    output logic [3:0]              error_code,

    // Test / debug visibility
    output logic [DATA_WIDTH-1:0]   kernel_out,
    output logic                    kernel_valid,
    input  logic                    kernel_ready,
    output logic [127:0]            class_scores_la,

    // Alpha dual-coefficient write port (Wishbone-driven)
    input  logic                    alpha_write_en,
    input  logic [9:0]              alpha_addr,
    input  logic [DATA_WIDTH-1:0]   alpha_data
);

    // =========================================================================
    // Error codes
    // =========================================================================
    localparam logic [3:0] ERR_NONE             = 4'h0;
    localparam logic [3:0] ERR_SV_ZERO          = 4'h1; // Σsv_count = 0
    localparam logic [3:0] ERR_SV_OVERFLOW      = 4'h2; // Σsv_count > NUM_SV
    localparam logic [3:0] ERR_ILLEGAL_STATE    = 4'h3; // FSM default branch
    localparam logic [3:0] ERR_GAMMA_SAT        = 4'h4; // gamma > 8.0 (all kernels → 0)
    // 4'h5 reserved (was ERR_FIFO_OVERFLOW)
    localparam logic [3:0] ERR_GAMMA_ZERO       = 4'h6; // gamma = 0 (all kernels = 1)
    localparam logic [3:0] ERR_NUM_SAMPLES_ZERO = 4'h7; // num_samples = 0
    localparam logic [3:0] ERR_WARMING_UP       = 4'h8; // < 100 heartbeats classified (advisory)
    localparam logic [3:0] ERR_INTERRUPTED      = 4'h9; // reset mid-warmup (advisory)
    localparam logic [3:0] ERR_LOW_BATTERY      = 4'hA; // vbatt_warn asserted (advisory)
    localparam logic [3:0] ERR_POWER_FAIL       = 4'hB; // vbatt_ok deasserted (advisory)

    localparam logic [DATA_WIDTH-1:0] GAMMA_SAT_THRESH = 16'd8192;

    localparam logic [DATA_WIDTH-1:0] GAMMA_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_GAMMA  * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] C_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_C      * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] BIAS0_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_BIAS_0 * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] BIAS1_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_BIAS_1 * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] BIAS2_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_BIAS_2 * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] BIAS3_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_BIAS_3 * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] BIAS4_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_BIAS_4 * (2.0 ** FRAC_BITS)));

    // =========================================================================
    // Internal signals
    // =========================================================================
    logic [DATA_WIDTH-1:0] gamma_int;
    logic [DATA_WIDTH-1:0] gamma_latched;
    logic [DATA_WIDTH-1:0] c_int;
    logic [DATA_WIDTH-1:0] bias_int [5];
    logic [3:0]            err_detect;

    // Distance matrix
    logic                    dist_start;
    logic                    dist_done;
    logic [DATA_WIDTH-1:0]   dist_feature_in;
    logic [DATA_WIDTH-1:0]   dist_sv_in;
    logic                    dist_valid_in;
    logic [DIST_WIDTH-1:0]   dist_out;
    logic                    dist_valid_out;
    logic                    dist_valid_latch;

    // Horner engine
    logic                    horner_start;
    logic                    horner_done;
    logic [DIST_WIDTH-1:0]   horner_dist_in;
    logic                    horner_valid_in;
    logic [DATA_WIDTH-1:0]   horner_kernel_out;
    logic                    horner_valid_out;

    // Battery synchronizers
    logic vbatt_ok_s;
    logic vbatt_warn_s;

    // Feature bank  (holds current sample's 256-dim input vector)
    // feature_bank is an on-chip SRAM macro (feature_sram_512x16 u_feat, below).
    // Caches the current beat's FEATURE_DIM features; sourced from the shared
    // off-chip RAM. Replaces the 4096-FF register array + 256:1 read mux.

    // Alpha dual coefficients (signed Q6.10, one per SV) live in an on-chip SRAM
    // MACRO instance (alpha_sram_1024x16 = 4x gf180mcu_fd_ip_sram__sram512x8m8wm1),
    // not flip-flops — see the u_alpha instance below. NOTE: unlike the old register
    // file there is no reset-to-1.0 default; the host MUST load all NUM_SV alphas via
    // alpha_write before classifying. Read latency is 1 cycle, hidden because sv_base
    // is stable for the whole per-SV compute (>256 cycles) before use.

    // Batch counters
    logic [9:0] num_samples_latched;
    logic [9:0] sample_counter;
    logic [7:0] sv_counter;
    logic [2:0] class_counter;
    logic [7:0] sv_count_reg [5];
    logic [6:0] heartbeat_count;

    // Feature bank write path  (LOAD_INPUT — from off-chip RAM, RAM_LATENCY cycles)
    // 9-bit address avoids 8-bit wrap at FEATURE_DIM = 256
    logic [8:0] feat_wr_addr;
    logic       feat_wr_en_r;
    logic [8:0] feat_wr_addr_r;
    logic [8:0] feat_wr_count;

    // Feature bank read path  (COMPUTE_DIST — local, 1-cycle latency)
    logic [8:0]              feat_rd_addr;
    logic                    feat_rd_en;
    logic                    feat_rd_en_r;
    logic [DATA_WIDTH-1:0]   feat_rd_data;

    // SV row index (sv_base = global SV index within the SV region)
    logic [9:0] sv_base;

    // Alpha SRAM macro, single port: write (alpha_write_en) shares the address with
    // the read (sv_base). Alpha loading precedes classification, so no collision.
    logic signed [DATA_WIDTH-1:0] alpha_rd;
    logic        [DATA_WIDTH-1:0] alpha_rd_w;
    logic [9:0]                   alpha_a;
    assign alpha_a  = alpha_write_en ? alpha_addr : sv_base[9:0];
    assign alpha_rd = $signed(alpha_rd_w);
    alpha_sram_1024x16 u_alpha (
    `ifdef USE_POWER_PINS
        .VDD  (VDD),
        .VSS  (VSS),
    `endif
        .clk  (clk),
        .ce   (1'b1),
        .we   (alpha_write_en),
        .addr (alpha_a),
        .wdata(alpha_data),
        .rdata(alpha_rd_w)
    );

    // Weighted kernel: alpha_rd (= alpha_mem[sv_base]) × kernel_out (Q6.10×Q6.10=Q12.20)
    logic signed [32:0] alpha_k_full;
    assign alpha_k_full = $signed(alpha_rd) * $signed({1'b0, kernel_out});

    // FSM
    typedef enum logic [2:0] {
        IDLE,
        LOAD_INPUT,
        COMPUTE_DIST,
        COMPUTE_KERNEL,
        OUTPUT_RESULT,
        WRITE_CLASS,
        ERROR_STATE
    } state_t;
    state_t state, next_state;

    // Off-chip RAM wait-state counter  (supports RAM_LATENCY >= 1)
    logic [3:0] ram_wait_cnt;   // counts 0 .. RAM_LATENCY-1
    wire        ram_beat = (ram_wait_cnt == 4'(RAM_LATENCY - 1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) ram_wait_cnt <= '0;
        else if (state == LOAD_INPUT || state == COMPUTE_DIST) begin
            if (ram_beat) ram_wait_cnt <= '0;
            else          ram_wait_cnt <= ram_wait_cnt + 1;
        end else
            ram_wait_cnt <= '0;
    end

    // Argmax accumulators (signed — alpha-weighted scores can be negative)
    logic signed [31:0] class_score_acc [5];
    wire signed [31:0] cs_acc0 = class_score_acc[0];
    wire signed [31:0] cs_acc1 = class_score_acc[1];
    wire signed [31:0] cs_acc2 = class_score_acc[2];
    wire signed [31:0] cs_acc3 = class_score_acc[3];
    wire signed [31:0] cs_acc4 = class_score_acc[4];
    logic [2:0]        argmax_class;
    logic signed [31:0] argmax_best;

    logic [10:0] total_sv_check;
    assign total_sv_check = sv_count_reg[0] + sv_count_reg[1]
                          + sv_count_reg[2] + sv_count_reg[3]
                          + sv_count_reg[4];

    wire last_sv        = (sv_counter >= sv_count_reg[class_counter] - 1);
    wire last_class     = (class_counter >= 3'd4);
    wire last_heartbeat = (sample_counter >= num_samples_latched - 1);

    // =========================================================================
    // Sub-module instances
    // =========================================================================
    distance_matrix #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .DIST_WIDTH(DIST_WIDTH), .FEATURE_DIM(FEATURE_DIM)
    ) u_distance_matrix (
        .clk(clk), .rst_n(rst_n),
        .start(dist_start),
        .feature_in(dist_feature_in), .sv_in(dist_sv_in), .valid_in(dist_valid_in),
        .dist_out(dist_out), .valid_out(dist_valid_out), .done(dist_done)
    );

    sync_ff #(.RESET_VAL(1'b1)) u_sync_vbatt_ok (
        .clk(clk), .rst_n(rst_n), .d(vbatt_ok),   .q(vbatt_ok_s)
    );
    sync_ff #(.RESET_VAL(1'b0)) u_sync_vbatt_warn (
        .clk(clk), .rst_n(rst_n), .d(vbatt_warn), .q(vbatt_warn_s)
    );

    horner_engine #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS), .DIST_WIDTH(DIST_WIDTH)
    ) u_horner_engine (
        .clk(clk), .rst_n(rst_n),
        .start(horner_start), .dist_in(horner_dist_in),
        .valid_in(horner_valid_in), .gamma(gamma_latched),
        .kernel_out(horner_kernel_out), .valid_out(horner_valid_out), .done(horner_done)
    );

    // =========================================================================
    // Parameter registers
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gamma_int   <= GAMMA_DEFAULT;
            c_int       <= C_DEFAULT;
            bias_int[0] <= BIAS0_DEFAULT;
            bias_int[1] <= BIAS1_DEFAULT;
            bias_int[2] <= BIAS2_DEFAULT;
            bias_int[3] <= BIAS3_DEFAULT;
            bias_int[4] <= BIAS4_DEFAULT;
            // alpha_mem is an SRAM: no reset init — host loads all coeffs via alpha_write
        end else begin
            if (param_write_en) begin
                case (param_addr)
                    3'b000: gamma_int   <= param_data;
                    3'b001: c_int       <= param_data;
                    3'b010: bias_int[0] <= param_data;
                    3'b011: bias_int[1] <= param_data;
                    3'b100: bias_int[2] <= param_data;
                    3'b101: bias_int[3] <= param_data;
                    3'b110: bias_int[4] <= param_data;
                    default: begin end
                endcase
            end
            // alpha writes handled by the alpha_mem SRAM block above
        end
    end

    assign gamma_reg = gamma_int;
    assign c_reg     = c_int;

    // =========================================================================
    // FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:           if (start && vbatt_ok_s) next_state = LOAD_INPUT;
            LOAD_INPUT:     if (feat_wr_count == FEATURE_DIM) next_state = COMPUTE_DIST;
            COMPUTE_DIST:   if (dist_done)  next_state = COMPUTE_KERNEL;
            COMPUTE_KERNEL: if (horner_done) next_state = OUTPUT_RESULT;
            OUTPUT_RESULT: begin
                if (kernel_ready && kernel_valid) begin
                    if (last_sv && last_class) next_state = WRITE_CLASS;
                    else                       next_state = COMPUTE_DIST;
                end
            end
            WRITE_CLASS: begin
                if (last_heartbeat) next_state = IDLE;
                else                next_state = LOAD_INPUT;
            end
            ERROR_STATE: next_state = IDLE;
            default:     next_state = ERROR_STATE;
        endcase
    end

    // =========================================================================
    // Off-chip RAM address mux
    //
    // Address layout: {row[10:0], col[7:0]} = 19 bits
    //   LOAD_INPUT:   row = NUM_SV + sample_counter  (input matrix region)
    //   COMPUTE_DIST: row = sv_base                  (SV matrix region)
    //
    // 9-bit addr counters prevent wrapping at FEATURE_DIM = 256 = 2^8,
    // ensuring no spurious reads after the 256th word.
    // =========================================================================
    logic [10:0] row_idx;
    logic [7:0]  col_idx;

    always_comb begin
        if (state == LOAD_INPUT) begin
            row_idx = 11'(NUM_SV) + 11'(sample_counter);
            col_idx = feat_wr_addr[7:0];
        end else begin
            row_idx = {1'b0, sv_base};
            col_idx = feat_rd_addr[7:0];
        end
    end

    assign ram_addr = {row_idx, col_idx};
    assign ram_ren  = (state == LOAD_INPUT)   ? (feat_wr_addr < 9'(FEATURE_DIM)) :
                      (state == COMPUTE_DIST) ? feat_rd_en : 1'b0;

    // SV data arrives on ram_rdata during COMPUTE_DIST (same bus, different row)
    assign dist_sv_in = ram_rdata;

    // =========================================================================
    // Feature bank — write path  (LOAD_INPUT)
    // ram_rdata valid 1 cycle after ram_ren/ram_addr; feat_wr_en_r captures this.
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feat_wr_en_r   <= 1'b0;
            feat_wr_addr_r <= '0;
        end else begin
            feat_wr_en_r   <= (state == LOAD_INPUT) && (feat_wr_addr < 9'(FEATURE_DIM)) && ram_beat;
            feat_wr_addr_r <= feat_wr_addr;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_wr_addr <= '0;
        else begin
            case (state)
                LOAD_INPUT: if (feat_wr_addr < 9'(FEATURE_DIM) && ram_beat)
                                feat_wr_addr <= feat_wr_addr + 1;
                default:    feat_wr_addr <= '0;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_wr_count <= '0;
        else begin
            case (state)
                LOAD_INPUT: if (feat_wr_en_r) feat_wr_count <= feat_wr_count + 1;
                default:    feat_wr_count <= '0;
            endcase
        end
    end

    // feature_bank write is handled by the feature SRAM (u_feat) below.

    // =========================================================================
    // Feature bank — read path  (COMPUTE_DIST)
    // =========================================================================
    assign feat_rd_en = (state == COMPUTE_DIST) && (feat_rd_addr < 9'(FEATURE_DIM));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_rd_en_r <= 1'b0;
        else        feat_rd_en_r <= feat_rd_en && ram_beat;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_rd_addr <= '0;
        else begin
            case (state)
                COMPUTE_DIST: if (feat_rd_en && ram_beat) feat_rd_addr <= feat_rd_addr + 1;
                default:      feat_rd_addr <= '0;
            endcase
        end
    end

    // Feature bank SRAM (single-port): write during LOAD_INPUT (feat_wr_en_r),
    // registered read during COMPUTE_DIST paced by ram_beat — disjoint phases, so
    // one shared port. Behavioral read latency is 1 cycle = matches the old register
    // read (testbenches stay valid); the macro's ~45ns physical access (> 40ns clk)
    // is covered by the RAM_LATENCY(=3) pacing — enforce with a multicycle path /
    // slower clock at harden.
    wire       feat_sram_ce = feat_wr_en_r || (feat_rd_en && ram_beat);
    wire       feat_sram_we = feat_wr_en_r;
    wire [8:0] feat_sram_a  = feat_wr_en_r ? {1'b0, feat_wr_addr_r[7:0]}
                                           : {1'b0, feat_rd_addr[7:0]};
    feature_sram_512x16 u_feat (
    `ifdef USE_POWER_PINS
        .VDD  (VDD),
        .VSS  (VSS),
    `endif
        .clk  (clk),
        .ce   (feat_sram_ce),
        .we   (feat_sram_we),
        .addr (feat_sram_a),
        .wdata(ram_rdata),
        .rdata(feat_rd_data)
    );

    // =========================================================================
    // SV row index  (cumulative sum of sv_count_reg for the current class)
    // =========================================================================
    always_comb begin
        sv_base = {2'b00, sv_counter};
        if (class_counter >= 1) sv_base = sv_base + {2'b00, sv_count_reg[0]};
        if (class_counter >= 2) sv_base = sv_base + {2'b00, sv_count_reg[1]};
        if (class_counter >= 3) sv_base = sv_base + {2'b00, sv_count_reg[2]};
        if (class_counter >= 4) sv_base = sv_base + {2'b00, sv_count_reg[3]};
    end

    // =========================================================================
    // Pipeline connections
    // =========================================================================
    assign dist_start      = (state == COMPUTE_DIST);
    assign dist_feature_in = feat_rd_data;
    assign dist_valid_in   = feat_rd_en_r;

    assign horner_start    = (state == COMPUTE_KERNEL);
    assign horner_dist_in  = dist_out;
    assign horner_valid_in = dist_valid_latch;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)              dist_valid_latch <= 1'b0;
        else if (dist_valid_out) dist_valid_latch <= 1'b1;
        else if (state == COMPUTE_KERNEL) dist_valid_latch <= 1'b0;
    end

    // =========================================================================
    // Counter management
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_counter      <= '0;
            sv_counter          <= '0;
            class_counter       <= '0;
            gamma_latched       <= GAMMA_DEFAULT;
            num_samples_latched <= 10'd1;
            for (int i = 0; i < 5; i++) sv_count_reg[i] <= '0;
        end else begin
            case (state)
                IDLE: begin
                    sample_counter <= '0;
                    sv_counter     <= '0;
                    class_counter  <= '0;
                    if (start && vbatt_ok_s) begin
                        for (int i = 0; i < 5; i++)
                            sv_count_reg[i] <= num_sv_per_class_flat[i*8 +: 8];
                        gamma_latched       <= gamma_int;
                        num_samples_latched <= num_samples;
                    end
                end
                LOAD_INPUT: begin
                    sv_counter    <= '0;
                    class_counter <= '0;
                end
                OUTPUT_RESULT: begin
                    if (kernel_ready && kernel_valid) begin
                        if (last_sv && last_class) begin
                            sv_counter    <= '0;
                            class_counter <= '0;
                        end else if (last_sv) begin
                            sv_counter    <= '0;
                            class_counter <= class_counter + 1;
                        end else begin
                            sv_counter    <= sv_counter + 1;
                        end
                    end
                end
                WRITE_CLASS: begin
                    sample_counter <= sample_counter + 1;
                end
                default: begin end
            endcase
        end
    end

    // =========================================================================
    // Warm-up counter  (saturates at 100; advisory while < 100)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            heartbeat_count <= 7'd0;
        else if ((state == WRITE_CLASS) && (heartbeat_count < 7'd100))
            heartbeat_count <= heartbeat_count + 7'd1;
    end

    // =========================================================================
    // Class score accumulation  (bias seeds each new sample's accumulators)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 5; i++) class_score_acc[i] <= '0;
        end else begin
            case (state)
                IDLE: begin
                    if (start && vbatt_ok_s) begin
                        class_score_acc[0] <= {{16{bias_int[0][15]}}, bias_int[0]};
                        class_score_acc[1] <= {{16{bias_int[1][15]}}, bias_int[1]};
                        class_score_acc[2] <= {{16{bias_int[2][15]}}, bias_int[2]};
                        class_score_acc[3] <= {{16{bias_int[3][15]}}, bias_int[3]};
                        class_score_acc[4] <= {{16{bias_int[4][15]}}, bias_int[4]};
                    end
                end
                OUTPUT_RESULT: begin
                    if (kernel_valid && kernel_ready)
                        class_score_acc[class_counter] <=
                            $signed(class_score_acc[class_counter])
                            + $signed(alpha_k_full[32:FRAC_BITS]);
                end
                WRITE_CLASS: begin
                    class_score_acc[0] <= {{16{bias_int[0][15]}}, bias_int[0]};
                    class_score_acc[1] <= {{16{bias_int[1][15]}}, bias_int[1]};
                    class_score_acc[2] <= {{16{bias_int[2][15]}}, bias_int[2]};
                    class_score_acc[3] <= {{16{bias_int[3][15]}}, bias_int[3]};
                    class_score_acc[4] <= {{16{bias_int[4][15]}}, bias_int[4]};
                end
                default: begin end
            endcase
        end
    end

    // =========================================================================
    // Argmax
    // =========================================================================
    always_comb begin
        argmax_class = 3'd0; argmax_best = cs_acc0;
        if ($signed(cs_acc1) > $signed(argmax_best)) begin argmax_class = 3'd1; argmax_best = cs_acc1; end
        if ($signed(cs_acc2) > $signed(argmax_best)) begin argmax_class = 3'd2; argmax_best = cs_acc2; end
        if ($signed(cs_acc3) > $signed(argmax_best)) begin argmax_class = 3'd3; argmax_best = cs_acc3; end
        if ($signed(cs_acc4) > $signed(argmax_best)) begin argmax_class = 3'd4; argmax_best = cs_acc4; end
    end

    assign class_scores_la = {cs_acc3, cs_acc2, cs_acc1, cs_acc0};

    // =========================================================================
    // Error priority encoder
    // =========================================================================
    always_comb begin
        if (state == ERROR_STATE)
            err_detect = ERR_ILLEGAL_STATE;
        else if ((state != IDLE) && (total_sv_check == 0))
            err_detect = ERR_SV_ZERO;
        else if ((state != IDLE) && (total_sv_check > NUM_SV))
            err_detect = ERR_SV_OVERFLOW;
        else if ((state != IDLE) && (num_samples == 0))
            err_detect = ERR_NUM_SAMPLES_ZERO;
        else if ((state != IDLE) && (gamma_int > GAMMA_SAT_THRESH))
            err_detect = ERR_GAMMA_SAT;
        else if ((state != IDLE) && (gamma_int == '0))
            err_detect = ERR_GAMMA_ZERO;
        else if (!vbatt_ok_s)
            err_detect = ERR_POWER_FAIL;
        else if (vbatt_warn_s)
            err_detect = ERR_LOW_BATTERY;
        else if (heartbeat_count > 0 && heartbeat_count < 7'd100)
            err_detect = ERR_WARMING_UP;
        else
            err_detect = ERR_NONE;
    end

    // =========================================================================
    // Output registers
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_rdy   <= 1'b0;
            class_out    <= 3'd0;
            done         <= 1'b0;
            error        <= 1'b0;
            error_code   <= ERR_NONE;
            kernel_out   <= '0;
            kernel_valid <= 1'b0;
        end else begin
            // Per-sample: fires once per WRITE_CLASS; class_out holds until next
            sample_rdy <= (state == WRITE_CLASS);
            class_out  <= argmax_class;

            // Batch done: fires on the last WRITE_CLASS only
            done <= (state == WRITE_CLASS) && last_heartbeat;

            // Error latch: sticky for codes < 0x8, advisory (auto-clear) for >= 0x8
            if (err_detect != ERR_NONE && err_detect < 4'h8) begin
                if (error_code == ERR_NONE || error_code >= 4'h8) begin
                    error      <= 1'b1;
                    error_code <= err_detect;
                end
            end else if (err_detect >= 4'h8) begin
                if (error_code == ERR_NONE || error_code >= 4'h8) begin
                    error      <= 1'b1;
                    error_code <= err_detect;
                end
            end else begin
                if (error_code >= 4'h8) begin
                    error      <= 1'b0;
                    error_code <= ERR_NONE;
                end
            end

            if (horner_valid_out)
                kernel_out <= horner_kernel_out;
            if (horner_valid_out)
                kernel_valid <= 1'b1;
            else if (kernel_valid && kernel_ready)
                kernel_valid <= 1'b0;
        end
    end

endmodule

// ===========================================================================
// Distance Matrix Engine  — unchanged
// ===========================================================================

module distance_matrix #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS  = 10,
    parameter int DIST_WIDTH = 20,
    parameter int FEATURE_DIM = 256
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start,
    output logic                    done,
    input  logic [DATA_WIDTH-1:0]   feature_in,
    input  logic [DATA_WIDTH-1:0]   sv_in,
    input  logic                    valid_in,
    output logic [DIST_WIDTH-1:0]   dist_out,
    output logic                    valid_out
);
    typedef enum logic [1:0] { IDLE, ACCUMULATE, OUTPUT, DONE_STATE } state_t;
    state_t state, next_state;

    logic [2*DATA_WIDTH-1:0]   diff;
    logic [2*DATA_WIDTH-1:0]   diff_squared;
    logic [2*DATA_WIDTH+8-1:0] accumulator;
    logic [8:0]                dim_counter;
    logic [1:0]                drain_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:       if (start) next_state = ACCUMULATE;
            ACCUMULATE: if (drain_cnt == 2'd2) next_state = OUTPUT;
            OUTPUT:     next_state = DONE_STATE;
            DONE_STATE: next_state = IDLE;
            default:    next_state = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) dim_counter <= '0;
        else begin
            case (state)
                IDLE:       dim_counter <= '0;
                ACCUMULATE: if (valid_in) dim_counter <= dim_counter + 1;
                default:    dim_counter <= dim_counter;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) drain_cnt <= 2'd0;
        else begin
            case (state)
                IDLE: drain_cnt <= 2'd0;
                ACCUMULATE: begin
                    if (dim_counter >= FEATURE_DIM - 1 && valid_in && drain_cnt == 2'd0)
                        drain_cnt <= 2'd1;
                    else if (drain_cnt != 2'd0)
                        drain_cnt <= drain_cnt + 2'd1;
                end
                default: drain_cnt <= 2'd0;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            diff         <= '0;
            diff_squared <= '0;
        end else if (state == IDLE) begin
            diff         <= '0;
            diff_squared <= '0;
        end else if (state == ACCUMULATE) begin
            if (valid_in)
                diff <= $signed(feature_in) - $signed(sv_in);
            else if (drain_cnt == 2'd1)
                diff_squared <= $signed(diff) * $signed(diff);
            if (valid_in)
                diff_squared <= $signed(diff) * $signed(diff);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) accumulator <= '0;
        else begin
            case (state)
                IDLE:       accumulator <= '0;
                ACCUMULATE: if (valid_in || drain_cnt != 2'd0)
                    accumulator <= accumulator + (diff_squared >>> FRAC_BITS);
                default: accumulator <= accumulator;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dist_out  <= '0;
            valid_out <= 1'b0;
            done      <= 1'b0;
        end else begin
            case (state)
                OUTPUT: begin
                    dist_out  <= (|accumulator[2*DATA_WIDTH+8-1:DIST_WIDTH])
                                  ? {DIST_WIDTH{1'b1}}
                                  : accumulator[DIST_WIDTH-1:0];
                    valid_out <= 1'b1;
                    done      <= 1'b0;
                end
                DONE_STATE: begin
                    valid_out <= 1'b0;
                    done      <= 1'b1;
                end
                default: begin
                    valid_out <= 1'b0;
                    done      <= 1'b0;
                end
            endcase
        end
    end

endmodule

// ===========================================================================
// Horner Engine  —  Range-Reduction LUT version  (unchanged)
// ===========================================================================

module horner_engine #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS  = 10,
    parameter int DIST_WIDTH = 20
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start,
    output logic                    done,
    input  logic [DATA_WIDTH-1:0]   gamma,
    input  logic [DIST_WIDTH-1:0]   dist_in,
    input  logic                    valid_in,
    output logic [DATA_WIDTH-1:0]   kernel_out,
    output logic                    valid_out
);
    localparam logic [DATA_WIDTH-1:0] COEFF_00 = (1 << FRAC_BITS);
    localparam logic [DATA_WIDTH-1:0] COEFF_01 = (1 << FRAC_BITS);
    localparam logic [DATA_WIDTH-1:0] COEFF_02 = (1 << (FRAC_BITS-1));
    localparam logic [DATA_WIDTH-1:0] COEFF_03 = ((1 << FRAC_BITS) / 6);
    localparam logic [DATA_WIDTH-1:0] COEFF_04 = ((1 << FRAC_BITS) / 24);
    localparam logic [DATA_WIDTH-1:0] COEFF_05 = ((1 << FRAC_BITS) / 120);
    localparam logic [DATA_WIDTH-1:0] COEFF_06 = ((1 << FRAC_BITS) / 720);
    localparam logic [DATA_WIDTH-1:0] COEFF_07 = ((1 << FRAC_BITS) / 5040);
    localparam logic [DATA_WIDTH-1:0] COEFF_08 = ((1 << FRAC_BITS) / 40320);
    localparam logic [DATA_WIDTH-1:0] COEFF_09 = ((1 << FRAC_BITS) / 362880);
    localparam logic [DATA_WIDTH-1:0] COEFF_10 = ((1 << FRAC_BITS) / 3628800);
    localparam logic [DATA_WIDTH-1:0] COEFF_11 = ((1 << FRAC_BITS) / 39916800);
    localparam logic [DATA_WIDTH-1:0] COEFF_12 = ((1 << FRAC_BITS) / 479001600);
    localparam logic [DATA_WIDTH-1:0] COEFF_13 = 1;
    localparam logic [DATA_WIDTH-1:0] COEFF_14 = 1;
    localparam logic [DATA_WIDTH-1:0] COEFF_15 = 1;

    function automatic logic [DATA_WIDTH-1:0] exp_int_lut(input logic [3:0] idx);
        case (idx)
            4'd0: exp_int_lut = 16'd1024;
            4'd1: exp_int_lut = 16'd377;
            4'd2: exp_int_lut = 16'd139;
            4'd3: exp_int_lut = 16'd51;
            4'd4: exp_int_lut = 16'd19;
            4'd5: exp_int_lut = 16'd7;
            4'd6: exp_int_lut = 16'd3;
            4'd7: exp_int_lut = 16'd1;
            default: exp_int_lut = 16'd0;
        endcase
    endfunction

    typedef enum logic [4:0] {
        IDLE,
        SCALE, SCALE2,
        HORNER_14, HORNER_13, HORNER_12, HORNER_11, HORNER_10,
        HORNER_9,  HORNER_8,  HORNER_7,  HORNER_6,  HORNER_5,
        HORNER_4,  HORNER_3,  HORNER_2,  HORNER_1,  HORNER_0,
        OUTPUT
    } state_t;
    state_t state, next_state;

    logic [DATA_WIDTH+DIST_WIDTH-1:0] temp_p;
    logic signed [2*DATA_WIDTH-1:0]   temp_h;
    logic signed [DATA_WIDTH-1:0]     x;
    logic [DATA_WIDTH-1:0]            lut_val;
    logic signed [DATA_WIDTH-1:0]     result;
    logic signed [DATA_WIDTH-1:0]     result_next;

    wire signed [DATA_WIDTH-1:0] temp_h_shifted;
    assign temp_h_shifted = temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS];

    wire [DATA_WIDTH-1:0]   horner_clamp;
    wire [2*DATA_WIDTH-1:0] lut_product;
    assign horner_clamp = ($signed(result_next) < 0)                 ? '0       :
                          ($signed(result_next) > $signed(COEFF_00)) ? COEFF_00 :
                                                                        DATA_WIDTH'(result_next);
    assign lut_product  = lut_val * horner_clamp;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:      if (start && valid_in) next_state = SCALE;
            SCALE:     next_state = SCALE2;
            SCALE2:    next_state = HORNER_14;
            HORNER_14: next_state = HORNER_13;
            HORNER_13: next_state = HORNER_12;
            HORNER_12: next_state = HORNER_11;
            HORNER_11: next_state = HORNER_10;
            HORNER_10: next_state = HORNER_9;
            HORNER_9:  next_state = HORNER_8;
            HORNER_8:  next_state = HORNER_7;
            HORNER_7:  next_state = HORNER_6;
            HORNER_6:  next_state = HORNER_5;
            HORNER_5:  next_state = HORNER_4;
            HORNER_4:  next_state = HORNER_3;
            HORNER_3:  next_state = HORNER_2;
            HORNER_2:  next_state = HORNER_1;
            HORNER_1:  next_state = HORNER_0;
            HORNER_0:  next_state = OUTPUT;
            OUTPUT:    next_state = IDLE;
            default:   next_state = IDLE;
        endcase
    end

    always_comb begin
        case (state)
            HORNER_14: result_next = COEFF_15;
            HORNER_13: result_next = $signed(COEFF_14) + $signed(temp_h_shifted);
            HORNER_12: result_next = $signed(COEFF_13) + $signed(temp_h_shifted);
            HORNER_11: result_next = $signed(COEFF_12) + $signed(temp_h_shifted);
            HORNER_10: result_next = $signed(COEFF_11) + $signed(temp_h_shifted);
            HORNER_9:  result_next = $signed(COEFF_10) + $signed(temp_h_shifted);
            HORNER_8:  result_next = $signed(COEFF_09) + $signed(temp_h_shifted);
            HORNER_7:  result_next = $signed(COEFF_08) + $signed(temp_h_shifted);
            HORNER_6:  result_next = $signed(COEFF_07) + $signed(temp_h_shifted);
            HORNER_5:  result_next = $signed(COEFF_06) + $signed(temp_h_shifted);
            HORNER_4:  result_next = $signed(COEFF_05) + $signed(temp_h_shifted);
            HORNER_3:  result_next = $signed(COEFF_04) + $signed(temp_h_shifted);
            HORNER_2:  result_next = $signed(COEFF_03) + $signed(temp_h_shifted);
            HORNER_1:  result_next = $signed(COEFF_02) + $signed(temp_h_shifted);
            HORNER_0:  result_next = $signed(COEFF_01) + $signed(temp_h_shifted);
            OUTPUT:    result_next = $signed(COEFF_00) + $signed(temp_h_shifted);
            default:   result_next = result;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x       <= '0;
            temp_p  <= '0;
            temp_h  <= '0;
            lut_val <= '0;
            result  <= '0;
        end else begin
            case (state)
                IDLE:   result <= COEFF_00;
                SCALE:  temp_p <= gamma * dist_in;
                SCALE2: begin
                    x       <= -$signed({6'b0, temp_p[FRAC_BITS+9:FRAC_BITS]});
                    lut_val <= (|temp_p[DATA_WIDTH+DIST_WIDTH-1 : DIST_WIDTH+4])
                               ? '0
                               : exp_int_lut(temp_p[DIST_WIDTH+3 : DIST_WIDTH]);
                end
                HORNER_14: begin temp_h <= $signed(x) * $signed(result_next); result <= result_next; end
                HORNER_13: begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_14) + $signed(temp_h_shifted); end
                HORNER_12: begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_13) + $signed(temp_h_shifted); end
                HORNER_11: begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_12) + $signed(temp_h_shifted); end
                HORNER_10: begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_11) + $signed(temp_h_shifted); end
                HORNER_9:  begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_10) + $signed(temp_h_shifted); end
                HORNER_8:  begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_09) + $signed(temp_h_shifted); end
                HORNER_7:  begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_08) + $signed(temp_h_shifted); end
                HORNER_6:  begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_07) + $signed(temp_h_shifted); end
                HORNER_5:  begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_06) + $signed(temp_h_shifted); end
                HORNER_4:  begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_05) + $signed(temp_h_shifted); end
                HORNER_3:  begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_04) + $signed(temp_h_shifted); end
                HORNER_2:  begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_03) + $signed(temp_h_shifted); end
                HORNER_1:  begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_02) + $signed(temp_h_shifted); end
                HORNER_0:  begin temp_h <= $signed(x) * $signed(result_next); result <= $signed(COEFF_01) + $signed(temp_h_shifted); end
                default: begin end
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kernel_out <= '0;
            valid_out  <= 1'b0;
            done       <= 1'b0;
        end else begin
            case (state)
                OUTPUT: begin
                    kernel_out <= lut_product[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS];
                    valid_out  <= 1'b1;
                    done       <= 1'b1;
                end
                default: begin
                    valid_out <= 1'b0;
                    done      <= 1'b0;
                end
            endcase
        end
    end

endmodule

// ===========================================================================
// 2-FF Synchronizer  (unchanged)
// ===========================================================================

module sync_ff #(
    parameter int STAGES    = 2,
    parameter bit RESET_VAL = 1'b0
) (
    input  logic clk,
    input  logic rst_n,
    input  logic d,
    output logic q
);
    logic [STAGES-1:0] pipe;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) pipe <= {STAGES{RESET_VAL}};
        else        pipe <= {pipe[STAGES-2:0], d};
    end
    assign q = pipe[STAGES-1];
endmodule

`default_nettype wire
