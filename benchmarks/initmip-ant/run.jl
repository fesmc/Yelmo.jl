# benchmarks/initmip-ant/run.jl
#
# Steady-state Antarctica simulation driven by present-day climate and
# topography. Forcing from RACMO2.3 (smb, T_srf), topography from
# BedMachine, GHF from Shapiro & Ritzwoller 2004 (S04). bmb_shlf held
# constant at -0.2 m/yr. Thermodynamics fully active (method="temp",
# rf_method=1). Initialized with a robin-cold state.
#
# Settings follow the Fortran reference yelmo/tests/yelmo_initmip.f90
# and yelmo/par/yelmo_initmip.nml (`set_ant_pd` case) as closely as the
# current port allows.
#
# Backend: select with the INITMIP_BACKEND environment variable.
#   - "yelmo"  (default) — pure Julia `YelmoModel`.
#   - "mirror"           — `YelmoMirror` over the Fortran-yelmo C-API.
# Both backends share the same `yelmo_initmip_ant.nml`, the same
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
using Yelmo.YelmoPar: YelmoParameters
using Yelmo.YelmoMirrorPar: YelmoMirrorParameters
using Yelmo: SSASolver
using Oceananigans: interior
using NCDatasets
using Statistics
using Printf

# ----------------------------------------------------------------------
# Configuration (edit in place).
# ----------------------------------------------------------------------

const T_END_YR       = parse(Float64, get(ENV, "INITMIP_T_END_YR", "20.0"))
const DT_OUTER_YR    = parse(Float64, get(ENV, "INITMIP_DT_OUTER_YR", "1.0"))
const SNAPSHOT_DT_YR = 10.0     # 2D snapshot cadence [yr]

const BMB_SHLF_CONST = -0.2     # [m/yr] constant basal melt under shelves (set_ant_pd)

# Namelist file. Defaults to the standard initmip-ant configuration.
const NAMELIST_PATH = abspath(joinpath(@__DIR__,
    get(ENV, "INITMIP_NML", "yelmo_initmip_ant.nml")))
const DATA_DIR      = abspath(joinpath(@__DIR__, "data", "ANT-32KM"))

# Override the SSA assembly formulation. Defaults to the namelist /
# Julia default (:residual). Set to "energy_quadratic" to use the
# symmetric viscous-energy Hessian assembly (CG inner solve).
# Yelmo-backend only — ignored under INITMIP_BACKEND=mirror.
const SSA_METHOD = Symbol(get(ENV, "INITMIP_SSA_METHOD", "residual"))

const BACKEND = lowercase(get(ENV, "INITMIP_BACKEND", "yelmo"))
BACKEND in ("yelmo", "mirror") ||
    error("Unsupported INITMIP_BACKEND=$(BACKEND); choose 'yelmo' or 'mirror'.")

const _OUTPUT_SUBDIR = get(ENV, "INITMIP_OUTPUT_SUBDIR", "")
const OUTPUT_DIR     = isempty(_OUTPUT_SUBDIR) ?
    abspath(joinpath(@__DIR__, BACKEND == "mirror" ? "output-mirror" : "output")) :
    abspath(joinpath(@__DIR__, _OUTPUT_SUBDIR))
const SNAPSHOTS_NC  = joinpath(OUTPUT_DIR, "snapshots.nc")
const RESTART_FINAL = joinpath(OUTPUT_DIR, "restart_final.nc")

const RACMO_FILE = "ANT-32KM_RACMO23-ERAINT-HYBRID_1981-2010.nc"
const GHF_FILE   = "ANT-32KM_GHF-S04.nc"

# ----------------------------------------------------------------------
# Backend-specific build.
# ----------------------------------------------------------------------

# Apply the RACMO / S04 forcing fields to a freshly-constructed model
# (either YelmoModel or YelmoMirror). Mutating `y.bnd.*` directly works
# for both backends.
#
# RACMO fields are monthly (nx, ny, 12); take the annual mean (Fortran
# yelmo_data.f90:317,323). Unit conversions:
#   - smb:   kg m-2 d-1 (≡ mm w.e./d) → m i.e./yr via
#            1e-3 · 365 · ρ_w/ρ_ice  (Fortran conv_mmdwe_maie,
#            yelmo_boundaries.f90:66), then |smb| < 1e-3 → 0
#            (yelmo_data.f90:331).
#   - T_srf: already in Kelvin (RACMO); °C guard kept for safety.
#
# Note: unlike initmip-grl, no ice-free SMB penalty is applied for
# Antarctica.
function _apply_forcing!(y, rho_ice::Real, rho_w::Real)
    smb_conv = 1.0e-3 * 365.0 * rho_w / rho_ice

    NCDataset(joinpath(DATA_DIR, RACMO_FILE)) do ds
        smb_m = Float64.(ds["smb"][:, :, :])      # (nx,ny,12) kg m-2 d-1
        T_m   = Float64.(ds["T_srf"][:, :, :])    # (nx,ny,12) K
        smb_ann = dropdims(sum(smb_m; dims = 3); dims = 3) ./ 12.0
        T_ann   = dropdims(sum(T_m;   dims = 3); dims = 3) ./ 12.0

        smb_field = @view interior(y.bnd.smb_ref)[:, :, 1]
        smb_field .= smb_ann .* smb_conv
        @. smb_field = ifelse(abs(smb_field) < 1.0e-3, 0.0, smb_field)

        T_srf_field = @view interior(y.bnd.T_srf)[:, :, 1]
        T_srf_field .= T_ann
        if minimum(T_srf_field) < 100.0
            T_srf_field .+= 273.15
        end
    end

    # GHF from Shapiro & Ritzwoller 2004.
    NCDataset(joinpath(DATA_DIR, GHF_FILE)) do ds
        interior(y.bnd.Q_geo)[:, :, 1] .= ds["ghf"][:, :]
    end

    # Constant shelf basal melt and present-day sea level.
    fill!(interior(y.bnd.bmb_shlf), BMB_SHLF_CONST)
    fill!(interior(y.bnd.z_sl), 0.0)

    return y
end

"""
    _override_field(s, f::Symbol, v) -> s′

Reconstruct `s` with field `f` replaced by `v`. Dep-free alternative to
`Setfield.@set`; uses the all-positional default inner constructor that
every struct has (works for both `Base.@kwdef` and plain structs).
"""
function _override_field(s, f::Symbol, v)
    vals = (n === f ? v : getfield(s, n) for n in fieldnames(typeof(s)))
    return typeof(s)(vals...)
end

function _build_yelmo()
    b = InitMIPBenchmark(joinpath(DATA_DIR, "ANT-32KM_REGIONS.nc"))
    p = YelmoParameters(NAMELIST_PATH, "initmip_ant")

    if SSA_METHOD !== :residual
        new_ssa = SSASolver(method = SSA_METHOD)
        new_ydyn = _override_field(p.ydyn, :ssa_solver, new_ssa)
        p = _override_field(p, :ydyn, new_ydyn)
        @info "overriding ssa_solver" method=SSA_METHOD
    end

    y = YelmoModel(b, 0.0; p = p, boundaries = :bounded)

    # Topography from BedMachine — init_topo_load! reads
    # H_ice/z_bed/z_bed_sd from the nml path, applies z_bed_f_sd scaling,
    # removes englacial lakes, and adjusts bedrock gradients.
    init_topo_load!(y; grad_lim_zb = 0.5)

    # Basins / regions (also paints bnd.mask_ice via define_mask_ice!).
    init_masks!(y)

    # Climate, forcing, BCs.
    _apply_forcing!(y, y.c.rho_ice, y.c.rho_w)

    # Initialize state (thermodynamics with robin-cold profile).
    init_state!(y, 0.0; thrm_method = "robin-cold")

    return y
end

function _build_mirror()
    p = YelmoMirrorParameters(NAMELIST_PATH, "initmip_ant")
    y = YelmoMirror(p, 0.0; rundir = OUTPUT_DIR, overwrite = true)

    _apply_forcing!(y, p.phys.rho_ice, p.phys.rho_w)

    init_state!(y, 0.0; thrm_method = "robin-cold")

    return y
end

_build() = BACKEND == "mirror" ? _build_mirror() : _build_yelmo()

# ----------------------------------------------------------------------
# Per-step diagnostics.
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
        return (pc_sub = 0, pc_rej = 0, ssa_it = 0)
    end
end

# ----------------------------------------------------------------------
# Domain-mean diagnostics.
# ----------------------------------------------------------------------

function _domain_diag_mirror(y)
    H     = interior(y.tpo.H_ice)[:, :, 1]
    z_bed = interior(y.bnd.z_bed)[:, :, 1]
    z_sl  = interior(y.bnd.z_sl)[:, :, 1]

    dx = abs(Float64(y.g.Δxᶜᵃᵃ))
    dy = abs(Float64(y.g.Δyᵃᶜᵃ))

    rho_ice = y.p.phys.rho_ice
    rho_sw  = y.p.phys.rho_sw

    V_ice = sum(H) * dx * dy * 1e-9

    sum_H_af = 0.0
    @inbounds for i in eachindex(H)
        z_diff = min(0.0, z_bed[i] - z_sl[i])
        sum_H_af += max(0.0, H[i] + z_diff * (rho_sw / rho_ice))
    end
    V_sl  = sum_H_af * dx * dy * 1e-9
    V_sle = V_sl * (1.0e-3 / 394.7)

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

    for fname in ("region_domain.nc", "snapshots.nc", "restart_final.nc",
                  "initmip_ant.nml")
        path = joinpath(OUTPUT_DIR, fname)
        isfile(path) && rm(path)
    end

    @info "initmip-ant — backend=$(BACKEND), nml=$(basename(NAMELIST_PATH)), t_end=$(T_END_YR) yr, dt_outer=$(DT_OUTER_YR) yr, bmb_shlf=$(BMB_SHLF_CONST) m/yr, ssa_method=$(SSA_METHOD)"

    y = _build()

    n_steps            = Int(round(T_END_YR / DT_OUTER_YR))
    steps_per_snapshot = max(Int(round(SNAPSHOT_DT_YR / DT_OUTER_YR)), 1)

    regs = BACKEND == "yelmo" ? init_regions(y; outdir = OUTPUT_DIR) : nothing

    snap_out = init_output(y, SNAPSHOTS_NC;
                           selection = OutputSelection(groups = [:tpo, :dyn, :mat, :thrm, :bnd, :dta]))

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

    pc_taken_total  = 0
    pc_reject_total = 0
    t_loop_start = time()
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
        pc_taken_total  += sd.pc_sub
        pc_reject_total += sd.pc_rej
        @printf("  %6.0f  %10.4e  %10.4f  %8.1f  %5d  %5d  %5d\n",
                y.time, d.V_ice, d.V_sle, d.H_ice_max,
                sd.pc_sub, sd.pc_rej, sd.ssa_it)
        flush(stdout)
    end
    t_loop_end = time()
    wall_s = t_loop_end - t_loop_start
    @info "wall-clock loop time" seconds=wall_s n_outer=n_steps pc_substeps_total=pc_taken_total pc_rejects_total=pc_reject_total

    close(snap_out.ds)

    if BACKEND == "yelmo"
        log = y.dyn.scratch.timestep_log[]
        log === nothing || close(log)
    end

    restart_out = init_output(y, RESTART_FINAL)
    write_output!(restart_out, y)
    close(restart_out.ds)

    @info "done" backend=BACKEND snapshots=SNAPSHOTS_NC restart=RESTART_FINAL

    if BACKEND == "yelmo" && y.timer.enabled
        println("\n--- Section timings ---")
        print_timings(y)
        flush(stdout)
    end
end

main()
