# SRAM macros — 6x gf180mcu_fd_ip_sram__sram512x8m8wm1 forming
# u_core.u_alpha (alpha_sram_1024x16, alpha table) + u_core.u_feat
# (feature_sram_512x16, the current-feature bank).
# All four are orientation N (stacked vertically in the 0p5x1 core), so a single
# macro PDN grid covers them.
#
# NOTE: the Metal4 edge-stripe offsets/pitch/number below are carried over from
# the template's single-SRAM example and MUST be re-tuned to the actual macro
# power-pin geometry + final placement on Orca (check_power_grid must pass for
# every VDD/VSS net after PnR).

define_pdn_grid \
    -macro \
    -instances "i_chip_core.u_core.u_alpha.u0lo \
                i_chip_core.u_core.u_alpha.u0hi \
                i_chip_core.u_core.u_alpha.u1lo \
                i_chip_core.u_core.u_alpha.u1hi \
                i_chip_core.u_core.u_feat.ulo \
                i_chip_core.u_core.u_feat.uhi" \
    -name sram_macros \
    -starts_with POWER \
    -halo "$::env(PDN_HORIZONTAL_HALO) $::env(PDN_VERTICAL_HALO)"

add_pdn_connect \
    -grid sram_macros \
    -layers "$::env(PDN_VERTICAL_LAYER) $::env(PDN_HORIZONTAL_LAYER)"

add_pdn_connect \
    -grid sram_macros \
    -layers "$::env(PDN_VERTICAL_LAYER) Metal3"

# Stripes over the SRAM rows to bind macro power pins into the grid.
# (Re-tune -offset/-pitch/-number_of_straps to the sram512x8 power-pin pitch.)
add_pdn_stripe \
    -grid sram_macros \
    -layer Metal4 \
    -width 2.36 \
    -offset 1.18 \
    -spacing 0.28 \
    -pitch 426.86 \
    -starts_with GROUND \
    -number_of_straps 2

# Extra Metal4 stripes to restore top-level PDN integrity where the above block it.
add_pdn_stripe \
    -grid sram_macros \
    -layer Metal4 \
    -width 4.00 \
    -offset 65.93 \
    -spacing 0.28 \
    -pitch 50 \
    -starts_with GROUND \
    -number_of_straps 7
