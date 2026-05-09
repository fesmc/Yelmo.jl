# benchmarks/initmip-grl/run.jl
#
# Steady-state Greenland simulation driven by present-day climate and
# topography. Forcing from MAR (smb, T_srf), topography from Morlighem
# et al. 2017 (M17), GHF from Shapiro & Ritzwoller 2004 (S04).
# bmb_shlf held constant at -0.5 m/yr. Thermodynamics fully active
# (method="temp", rf_method=1). Initialized with a robin-cold state.
#
# Settings follow the Fortran reference yelmo/tests/yelmo_initmip.f90
# and yelmo/par/yelmo_initmip.nml as closely as the current port allows.
#
# Outputs (under output/, gitignored):
#   - region_domain.nc    — whole-domain time-series via regions API.
#   - snapshots.nc        — 2D snapshots every SNAPSHOT_DT_YR.
#   - restart_final.nc    — full model state at T_END_YR.
#
# Run from this directory:
#   julia --project=. run.jl

cd(@__DIR__)   # data paths in the nml are relative to this directory

using IceSheetBenchmarks
using Yelmo
using Yelmo: step!, init_state!, init_topo_load!
using Yelmo: init_regions, update_regions!, write_regions!
using Yelmo: init_output, write_output!, OutputSelection
using Yelmo: print_timings
using Yelmo.YelmoModelPar: YelmoModelParameters
using Oceananigans: interior
using NCDatasets
using Statistics
using Printf

# ----------------------------------------------------------------------
# Configuration (edit in place).
# ----------------------------------------------------------------------

const T_END_YR       = 10.0     # total simulated time [yr]
const DT_OUTER_YR    = 1.0      # outer-loop dt [yr]; adaptive PC sub-steps inside
const SNAPSHOT_DT_YR = 10.0     # 2D snapshot cadence [yr]

const BMB_SHLF_CONST = -0.5     # [m/yr] constant basal melt under shelves

const NAMELIST_PATH = abspath(joinpath(@__DIR__, "yelmo_initmip_grl.nml"))
const DATA_DIR      = abspath(joinpath(@__DIR__, "data", "GRL-16KM"))

const OUTPUT_DIR    = abspath(joinpath(@__DIR__, "output"))
const SNAPSHOTS_NC  = joinpath(OUTPUT_DIR, "snapshots.nc")
const RESTART_FINAL = joinpath(OUTPUT_DIR, "restart_final.nc")

# ----------------------------------------------------------------------
# Build the model.
# ----------------------------------------------------------------------

function _build()
    b = InitMIPGRLBenchmark(joinpath(DATA_DIR, "GRL-16KM_REGIONS.nc"))
    p = YelmoModelParameters(NAMELIST_PATH, "initmip_grl")
    y = YelmoModel(b, 0.0; p = p, boundaries = :bounded)

    # --- Topography from M17 ---
    # init_topo_load! reads H_ice/z_bed/z_bed_sd from the nml path,
    # applies z_bed_f_sd scaling, removes englacial lakes, and adjusts
    # bedrock gradients. grad_lim_zb = 0.5 matches ytopo nml.
    init_topo_load!(y; grad_lim_zb = 0.5)

    # --- Climate and forcing: load directly into bnd fields ---
    #
    # Unit conversions match Fortran yelmo_data.f90:285-331:
    #   - T_srf: °C → K (add 273.15) when minimum < 100.
    #   - smb:   mm w.e./yr → m i.e./yr  (× 1e-3 · ρ_w/ρ_ice).
    smb_conv = 1.0e-3 * y.c.rho_w / y.c.rho_ice

    NCDataset(joinpath(DATA_DIR, "GRL-16KM_MARv3.11-ERA_annmean_1961-1990.nc")) do ds
        smb_raw   = ds["smb"][:, :]
        T_srf_raw = ds["T_srf"][:, :]
        interior(y.bnd.smb_ref)[:, :, 1] .= smb_raw .* smb_conv
        T_srf_field = @view interior(y.bnd.T_srf)[:, :, 1]
        T_srf_field .= T_srf_raw
        if minimum(T_srf_field) < 100.0
            T_srf_field .+= 273.15
        end
    end

    # Ice-free cells receive an additional -2 m/yr SMB penalty to
    # prevent spurious ice growth outside the present-day margin.
    # Mirrors Fortran yelmo_initmip.f90:183.
    smb   = @view interior(y.bnd.smb_ref)[:, :, 1]
    H_ice = @view interior(y.tpo.H_ice)[:, :, 1]
    @. smb = ifelse(H_ice <= 0.0, smb - 2.0, smb)

    # GHF from Shapiro & Ritzwoller 2004.
    NCDataset(joinpath(DATA_DIR, "GRL-16KM_GHF-S04.nc")) do ds
        interior(y.bnd.Q_geo)[:, :, 1] .= ds["ghf"][:, :]
    end

    # Constant shelf basal melt and present-day sea level.
    fill!(interior(y.bnd.bmb_shlf), BMB_SHLF_CONST)
    fill!(interior(y.bnd.z_sl), 0.0)

    # --- Initialize state (thermodynamics with robin-cold profile) ---
    init_state!(y, 0.0; thrm_method = "robin-cold")

    return y
end

# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------

function main()
    mkpath(OUTPUT_DIR)
    @info "initmip-grl — t_end=$(T_END_YR) yr, dt_outer=$(DT_OUTER_YR) yr, bmb_shlf=$(BMB_SHLF_CONST) m/yr"

    y = _build()

    n_steps            = Int(round(T_END_YR / DT_OUTER_YR))
    steps_per_snapshot = max(Int(round(SNAPSHOT_DT_YR / DT_OUTER_YR)), 1)

    # --- Whole-domain regions time series ---
    regs = init_regions(y; outdir = OUTPUT_DIR)

    # --- 2D snapshot file ---
    # Groups match Fortran write_step_2D: tpo, dyn, mat, thrm, bnd, dta.
    snap_out = init_output(y, SNAPSHOTS_NC;
                           selection = OutputSelection(groups = [:tpo, :dyn, :mat, :thrm, :bnd, :dta]))

    # Write t = 0 records.
    update_regions!(regs, y)
    write_regions!(regs, y, y.time)
    write_output!(snap_out, y)

    @info "t=0 initialized" V_ice_km3=regs[1].diag.V_ice V_sle_m=regs[1].diag.V_sle

    # Per-outer-step counters for the adaptive PC + SSA Picard loops.
    # `pc_scratch[]` is allocated lazily on the first step!, so read
    # cumulative `n_steps_taken / n_rejections` from it and diff per
    # outer step. `ssa_iter_now[]` is the last Picard iteration count
    # from the most recent SSA solve.
    pc_taken_prev   = 0
    pc_reject_prev  = 0

    @printf("  %6s  %10s  %10s  %8s  %5s  %5s  %5s\n",
            "t[yr]", "V_ice[km³]", "V_sle[m]", "max_H[m]",
            "PCsub", "PCrej", "SSAit")
    flush(stdout)

    for k in 1:n_steps
        step!(y, DT_OUTER_YR)

        update_regions!(regs, y)
        write_regions!(regs, y, y.time)

        if k % steps_per_snapshot == 0
            write_output!(snap_out, y)
        end

        # Per-outer-step diagnostics: PC sub-steps + rejections (delta from
        # cumulative counters), last SSA Picard iteration count.
        pc        = y.dyn.scratch.pc_scratch[]
        pc_sub    = pc === nothing ? 0 : pc.n_steps_taken - pc_taken_prev
        pc_rej    = pc === nothing ? 0 : pc.n_rejections - pc_reject_prev
        pc_taken_prev  = pc === nothing ? 0 : pc.n_steps_taken
        pc_reject_prev = pc === nothing ? 0 : pc.n_rejections
        ssa_it    = Int(y.dyn.scratch.ssa_iter_now[])

        d = regs[1].diag
        @printf("  %6.0f  %10.4e  %10.4f  %8.1f  %5d  %5d  %5d\n",
                y.time, d.V_ice, d.V_sle, d.H_ice_max, pc_sub, pc_rej, ssa_it)
        flush(stdout)
    end

    close(snap_out.ds)

    # --- Restart file (full model state for continuation) ---
    restart_out = init_output(y, RESTART_FINAL)
    write_output!(restart_out, y)
    close(restart_out.ds)

    @info "done" snapshots=SNAPSHOTS_NC restart=RESTART_FINAL

    # --- Per-section timings (yelmo.timing = True in nml) ---
    if y.timer.enabled
        println("\n--- Section timings ---")
        print_timings(y)
        flush(stdout)
    end
end

main()
