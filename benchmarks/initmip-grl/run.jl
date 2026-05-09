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
# Backend: select with the INITMIP_BACKEND environment variable.
#   - "yelmo"  (default) — pure Julia `YelmoModel`.
#   - "mirror"           — `YelmoMirror` over the Fortran-yelmo C-API.
# Both backends share the same `yelmo_initmip_grl.nml`, the same
# forcing-data path, and the same time-stepping configuration; the
# build path differs (Yelmo loads topo/masks via Yelmo.jl helpers,
# Mirror leans on Fortran's own loaders triggered by `yelmo_init`).
#
# Outputs (under output/, gitignored):
#   - region_domain.nc    — whole-domain time-series (yelmo backend only).
#   - snapshots.nc        — 2D snapshots every SNAPSHOT_DT_YR.
#   - restart_final.nc    — full model state at T_END_YR.
#
# Run from this directory:
#   julia --project=. run.jl                       # yelmo backend
#   INITMIP_BACKEND=mirror julia --project=. run.jl  # mirror backend

cd(@__DIR__)   # data paths in the nml are relative to this directory

using IceSheetBenchmarks
using Yelmo
using Yelmo: step!, init_state!, init_topo_load!, init_masks!
using Yelmo: init_regions, update_regions!, write_regions!
using Yelmo: init_output, write_output!, OutputSelection
using Yelmo: print_timings
using Yelmo: YelmoMirror
using Yelmo.YelmoModelPar: YelmoModelParameters
using Yelmo.YelmoPar: YelmoParameters
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

const BACKEND = lowercase(get(ENV, "INITMIP_BACKEND", "yelmo"))
BACKEND in ("yelmo", "mirror") ||
    error("Unsupported INITMIP_BACKEND=$(BACKEND); choose 'yelmo' or 'mirror'.")

# ----------------------------------------------------------------------
# Backend-specific build.
# ----------------------------------------------------------------------

# Apply the MAR / S04 forcing fields to a freshly-constructed model
# (either YelmoModel or YelmoMirror). Mutating `y.bnd.*` directly works
# for both backends — for Mirror the values get pushed to Fortran on
# the next `init_state!` / `step!` via `yelmo_sync!`.
#
# Unit conversions match Fortran yelmo_data.f90:285-331:
#   - T_srf: °C → K (add 273.15) when minimum < 100.
#   - smb:   mm w.e./yr → m i.e./yr  (× 1e-3 · ρ_w/ρ_ice).
#
# rho_ice / rho_w are constants of the ice-sheet model — read from the
# backend's own constants struct so YelmoMirror picks up Fortran's
# `Earth` defaults rather than Yelmo.jl's.
function _apply_forcing!(y, rho_ice::Real, rho_w::Real)
    smb_conv = 1.0e-3 * rho_w / rho_ice

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

    return y
end

function _build_yelmo()
    b = InitMIPGRLBenchmark(joinpath(DATA_DIR, "GRL-16KM_REGIONS.nc"))
    p = YelmoModelParameters(NAMELIST_PATH, "initmip_grl")
    y = YelmoModel(b, 0.0; p = p, boundaries = :bounded)

    # Topography from M17 — init_topo_load! reads H_ice/z_bed/z_bed_sd
    # from the nml path, applies z_bed_f_sd scaling, removes englacial
    # lakes, and adjusts bedrock gradients. grad_lim_zb = 0.5 matches
    # ytopo nml.
    init_topo_load!(y; grad_lim_zb = 0.5)

    # Basins / regions (used by data_compare!'s regional masks).
    init_masks!(y)

    # Climate, forcing, BCs.
    _apply_forcing!(y, y.c.rho_ice, y.c.rho_w)

    # Initialize state (thermodynamics with robin-cold profile).
    init_state!(y, 0.0; thrm_method = "robin-cold")

    return y
end

function _build_mirror()
    # YelmoMirror reads the same nml; Fortran-side `yelmo_init` loads
    # the topography (yelmo_init_topo block) and masks (yelmo_masks
    # block) on its own. We then overwrite the climate / forcing
    # fields on the Julia mirror — `init_state!` syncs them to
    # Fortran before the analytic state init.
    p = YelmoParameters(NAMELIST_PATH, "initmip_grl")
    y = YelmoMirror(p, 0.0; rundir = OUTPUT_DIR, overwrite = true)

    # YelmoMirror keeps physical constants on the Fortran side rather
    # than as a Julia-side `y.c`; pull them from the parsed nml's
    # `&phys` block (rho_ice = 910, rho_w = 1000 for `phys_const="Earth"`).
    _apply_forcing!(y, p.phys.rho_ice, p.phys.rho_w)

    init_state!(y, 0.0; thrm_method = "robin-cold")

    return y
end

_build() = BACKEND == "mirror" ? _build_mirror() : _build_yelmo()

# ----------------------------------------------------------------------
# Per-step diagnostics.
#
# YelmoModel backend: PC sub-step / Picard counters live on
# `y.dyn.scratch.*`, read directly from the model.
#
# Mirror backend: time-stepping is internal to Fortran and the C-API
# we use doesn't surface per-step counters. The Fortran-yelmo
# `yelmo:: timelog:` lines (yelmo_ice.f90:538) DO carry max_dt /
# min_dt / n_dtmin per outer step, and they flow to stdout naturally
# alongside our diagnostics — `grep '^ yelmo:: timelog:' run.log`
# to see them post-hoc. Capturing and parsing them per-step from
# within Julia turned out to be fiddly (libc-side stdout buffering,
# fd dup interaction with `redirect_stdout`), so the Mirror columns
# below stay at 0 for now.
# ----------------------------------------------------------------------

mutable struct StepCounters
    pc_taken_prev::Int
    pc_reject_prev::Int
end
StepCounters() = StepCounters(0, 0)

function _step_diagnostics(y, ctrs::StepCounters)
    if BACKEND == "yelmo"
        pc        = y.dyn.scratch.pc_scratch[]
        pc_sub    = pc === nothing ? 0 : pc.n_steps_taken - ctrs.pc_taken_prev
        pc_rej    = pc === nothing ? 0 : pc.n_rejections - ctrs.pc_reject_prev
        ctrs.pc_taken_prev  = pc === nothing ? 0 : pc.n_steps_taken
        ctrs.pc_reject_prev = pc === nothing ? 0 : pc.n_rejections
        ssa_it    = Int(y.dyn.scratch.ssa_iter_now[])
        return (pc_sub = pc_sub, pc_rej = pc_rej, ssa_it = ssa_it)
    else
        # Mirror: see comment block above. Counters stay 0; read
        # Fortran's `yelmo:: timelog:` lines in run.log instead.
        return (pc_sub = 0, pc_rej = 0, ssa_it = 0)
    end
end

# ----------------------------------------------------------------------
# Domain-mean diagnostics. For the YelmoModel backend we use the
# regions API (whole-domain region) for the printed line; for Mirror
# we compute the same scalars directly from the Field-mirror state
# since the regions API isn't generalised to Mirror yet.
# ----------------------------------------------------------------------

# Lightweight whole-domain stats for the Mirror backend, since the
# regions API (`regions/calc_region.jl`) is currently YelmoModel-only.
# Only the printed-line scalars are computed here; the snapshots and
# restart files still carry the full state.
function _domain_diag_mirror(y)
    H     = interior(y.tpo.H_ice)[:, :, 1]
    z_bed = interior(y.bnd.z_bed)[:, :, 1]
    z_sl  = interior(y.bnd.z_sl)[:, :, 1]

    dx = abs(Float64(y.g.Δxᶜᵃᵃ))
    dy = abs(Float64(y.g.Δyᵃᶜᵃ))

    # Mirror keeps physical constants on the Fortran side; pull them
    # from the parsed nml's `&phys` block (rho_ice = 910, rho_sw = 1028
    # for `phys_const="Earth"`).
    rho_ice = y.p.phys.rho_ice
    rho_sw  = y.p.phys.rho_sw

    # V_ice in km³ (matches regions API convention).
    V_ice = sum(H) * dx * dy * 1e-9

    # H_af = max(0, H_ice + min(0, z_bed - z_sl) · ρ_sw/ρ_ice). Mirrors
    # `_calc_H_af` in `regions/calc_region.jl`. Then V_sl in km³ and
    # V_sle in m s.l.e. via `1e-3 / 394.7` (Fortran yelmo_boundaries.f90:74).
    sum_H_af = 0.0
    @inbounds for i in eachindex(H)
        z_diff = min(0.0, z_bed[i] - z_sl[i])
        sum_H_af += max(0.0, H[i] + z_diff * (rho_sw / rho_ice))
    end
    V_sl  = sum_H_af * dx * dy * 1e-9               # km³ above flotation
    V_sle = V_sl * (1.0e-3 / 394.7)                  # m s.l.e.

    return (V_ice = V_ice, V_sle = V_sle, H_ice_max = maximum(H))
end

function _domain_diag_yelmo(regs)
    d = regs[1].diag
    return (V_ice = d.V_ice, V_sle = d.V_sle, H_ice_max = d.H_ice_max)
end

# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------

function main()
    mkpath(OUTPUT_DIR)

    # Clear stale outputs from a previous run before producing fresh
    # ones. Only files this script writes — leave any user files (e.g.
    # the shell-redirected `run.log`) untouched.
    for fname in ("region_domain.nc", "snapshots.nc", "restart_final.nc",
                  "initmip_grl.nml")
        path = joinpath(OUTPUT_DIR, fname)
        isfile(path) && rm(path)
    end

    @info "initmip-grl — backend=$(BACKEND), t_end=$(T_END_YR) yr, dt_outer=$(DT_OUTER_YR) yr, bmb_shlf=$(BMB_SHLF_CONST) m/yr"

    y = _build()

    n_steps            = Int(round(T_END_YR / DT_OUTER_YR))
    steps_per_snapshot = max(Int(round(SNAPSHOT_DT_YR / DT_OUTER_YR)), 1)

    # Whole-domain regions time series — YelmoModel only.
    regs = BACKEND == "yelmo" ? init_regions(y; outdir = OUTPUT_DIR) : nothing

    # 2D snapshot file. Groups match Fortran write_step_2D.
    # `init_output` / `write_output!` branch internally on
    # `uses_split_boundary_storage(y)`: Yelmo uses split-boundary
    # storage (Nz_file = Nz + 2 with `_b` / `_s` glue), Mirror uses
    # interior-extended (Nz_file = Nz, write as-is).
    snap_out = init_output(y, SNAPSHOTS_NC;
                           selection = OutputSelection(groups = [:tpo, :dyn, :mat, :thrm, :bnd, :dta]))

    # Write t = 0 records.
    if regs !== nothing
        update_regions!(regs, y)
        write_regions!(regs, y, y.time)
    end
    write_output!(snap_out, y)

    diag0 = regs === nothing ? _domain_diag_mirror(y) : _domain_diag_yelmo(regs)
    @info "t=0 initialized" V_ice_km3=diag0.V_ice V_sle_m=diag0.V_sle

    ctrs = StepCounters()
    @printf("  %6s  %10s  %10s  %8s  %5s  %5s  %5s\n",
            "t[yr]", "V_ice[km³]", "V_sle[m]", "max_H[m]",
            "PCsub", "PCrej", "SSAit")
    flush(stdout)

    for k in 1:n_steps
        step!(y, DT_OUTER_YR)

        if regs !== nothing
            update_regions!(regs, y)
            write_regions!(regs, y, y.time)
        end

        if k % steps_per_snapshot == 0
            write_output!(snap_out, y)
        end

        d  = regs === nothing ? _domain_diag_mirror(y) : _domain_diag_yelmo(regs)
        sd = _step_diagnostics(y, ctrs)
        @printf("  %6.0f  %10.4e  %10.4f  %8.1f  %5d  %5d  %5d\n",
                y.time, d.V_ice, d.V_sle, d.H_ice_max,
                sd.pc_sub, sd.pc_rej, sd.ssa_it)
        flush(stdout)
    end

    close(snap_out.ds)

    # Restart file (full model state for continuation). Same writer
    # path as snapshots; both backends supported via the
    # `uses_split_boundary_storage` branch inside `init_output` /
    # `write_output!`.
    restart_out = init_output(y, RESTART_FINAL)
    write_output!(restart_out, y)
    close(restart_out.ds)

    @info "done" backend=BACKEND snapshots=SNAPSHOTS_NC restart=RESTART_FINAL

    # Per-section timings — YelmoModel only (Mirror has no Julia-side timer).
    if BACKEND == "yelmo" && y.timer.enabled
        println("\n--- Section timings ---")
        print_timings(y)
        flush(stdout)
    end
end

main()
