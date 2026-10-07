# benchmarks/initmip-grl/run.jl
#
# Steady-state Greenland benchmark, designed pure-Julia-first.
#
# The configuration is a native `YelmoParameters` built in `build_params`
# (no namelist input file). The pure-Julia `YelmoModel` is initialised
# directly from it. Selecting `backend = :mirror` instead derives a
# `YelmoMirrorParameters` via `to_mirror` (which keeps Fortran-native
# values for backend-divergent timestepping options) and runs the
# Fortran model through `YelmoMirror`.
#
# Forcing: MAR (smb, T_srf), topography from Morlighem 2017 (M17), GHF
# from Shapiro & Ritzwoller 2004 (S04), bmb_shlf = -0.5 m/yr. Initialised
# from topography (no restart) + a robin-cold thermal state. Follows the
# Fortran reference `set_grl_pd` case.
#
# Usage (from this directory):
#   julia --project=. -e 'include("run.jl"); main()'
#   julia --project=. -e 'include("run.jl"); main(t_end=1.0)'
#   julia --project=. -e 'include("run.jl"); main(backend=:mirror)'

using IceSheetBenchmarks            # InitMIPBenchmark
using Yelmo                         # everything else, incl. `interior`
using NCDatasets
using Printf

const DATA_DIR    = joinpath("data", "GRL-16KM")
const REGIONS_NC  = joinpath(@__DIR__, DATA_DIR, "GRL-16KM_REGIONS.nc")
const MAR_FILE    = joinpath(@__DIR__, DATA_DIR, "GRL-16KM_MARv3.11-ERA_annmean_1961-1990.nc")
const GHF_FILE    = joinpath(@__DIR__, DATA_DIR, "GRL-16KM_GHF-S04.nc")

const BMB_SHLF_CONST = -0.5         # [m/yr] constant basal melt under shelves

# ----------------------------------------------------------------------
# Canonical configuration — pure-Julia YelmoParameters.
# Anything left unset keeps its Yelmo.jl default; only non-default values
# appear here. The pc settings (HEUN, pc_tol 5, pc_eps 1) are YelmoModel's
# (`backend = :yelmo`); the Mirror backend keeps its Fortran-native values
# (`MIRROR_DIVERGENT_YELMO`).
# ----------------------------------------------------------------------
function build_params(backend::Symbol = :yelmo)
    pc = backend === :yelmo ? (pc_method = "HEUN", pc_tol = 5.0, pc_eps = 1.0) : (;)
    return with_ported_options(YelmoParameters("initmip_grl";
        yelmo = yelmo_params(;
            domain       = "Greenland",
            grid_name    = "GRL-16KM",
            grid_path    = joinpath(DATA_DIR, "GRL-16KM_REGIONS.nc"),
            dt_method    = 2,             # adaptive predictor-corrector
            timing       = true,
            log_timestep = true,
            pc...,
        ),
        ycalv = ycalv_params(use_lsf = false, calv_flt_method = "vm-l19", calv_grnd_method = "zero", H_min_grnd = 0.0, H_min_flt = 0.0),
        ydyn = ydyn_params(ssa_solver = SSASolver(method = :residual), ssa_lat_bc = "floating", ssa_vel_max = 5000.0, ssa_iter_max = 50),
        yhyd = yhyd_params(bkt_N_closure = 3, till_N0 = 1000.0, till_delta = 0.04, till_e0 = 0.69, till_Cc = 0.12,
                           bkt_till_rate = 0.001, W_til_max = 2.0),
        ymat = ymat_params(rf_method = 1, de_max = 2.0),   # standard rate-factor function
        ytherm = ytherm_params(method = "temp", solver_advec = "impl-upwind",
                               rock_method = "equil", nzr_aa = 5, H_rock = 2000.0),
        yelmo_masks = yelmo_masks_params(
            basins_load  = true,
            basins_path  = joinpath(DATA_DIR, "GRL-16KM_BASINS-nasa.nc"),
            basins_nms   = ["basin", "basin_mask"],
            regions_load = true,
            regions_path = joinpath(DATA_DIR, "GRL-16KM_REGIONS.nc"),
            regions_nms  = ["mask", "None"],
        ),
        yelmo_init_topo = yelmo_init_topo_params(
            init_topo_load  = true,
            init_topo_path  = joinpath(DATA_DIR, "GRL-16KM_TOPO-M17-v5.nc"),
            init_topo_names = ["H_ice", "z_bed", "z_bed_sd", "z_srf"],
            init_topo_state = 0,
            z_bed_f_sd      = -1.0,
        ),
        # No present-day comparison data loaded (data_load! is not called).
        yelmo_data = yelmo_data_params(
            pd_topo_load = false, pd_tsrf_load = false,
            pd_smb_load = false, pd_vel_load = false,
        ),
    ))
end

# ----------------------------------------------------------------------
# Forcing. MAR fields are 2D annual means. smb is mm w.e./yr → m i.e./yr
# via 1e-3·ρ_w/ρ_ice (Fortran conv_mmawe_maie). T_srf is °C → K. Ice-free
# cells get an extra -2 m/yr SMB penalty to suppress spurious growth
# outside the present-day margin (Fortran yelmo_initmip.f90:183).
# ----------------------------------------------------------------------
function apply_forcing!(y, rho_ice, rho_w)
    smb_conv = 1.0e-3 * rho_w / rho_ice

    NCDataset(MAR_FILE) do ds
        smb = @view interior(y.bnd.smb_ref)[:, :, 1]
        smb .= ds["smb"][:, :] .* smb_conv

        T = @view interior(y.bnd.T_srf)[:, :, 1]
        T .= ds["T_srf"][:, :]
        minimum(T) < 100.0 && (T .+= 273.15)
    end

    # Ice-free SMB penalty.
    smb   = @view interior(y.bnd.smb_ref)[:, :, 1]
    H_ice = @view interior(y.tpo.H_ice)[:, :, 1]
    @. smb = ifelse(H_ice <= 0.0, smb - 2.0, smb)

    NCDataset(GHF_FILE) do ds
        interior(y.bnd.Q_geo)[:, :, 1] .= ds["ghf"][:, :]
    end

    fill!(interior(y.bnd.bmb_shlf), BMB_SHLF_CONST)
    fill!(interior(y.bnd.z_sl), 0.0)
    return y
end

# ----------------------------------------------------------------------
# Backend builds. Both share `apply_forcing!` + robin-cold init.
# ----------------------------------------------------------------------
function build_yelmo(p)
    y = YelmoModel(InitMIPBenchmark(REGIONS_NC), 0.0; p = p, boundaries = :bounded)
    init_topo_load!(y; grad_lim_zb = 0.5)   # M17 topo, no restart
    init_masks!(y)                          # basins/regions + bnd.mask_ice
    apply_forcing!(y, y.c.rho_ice, y.c.rho_w)
    init_state!(y, 0.0; thrm_method = "robin-cold")
    return y
end

function build_mirror(p, outdir)
    # Divergent timestepping options stay Fortran-native; Fortran's timestep
    # log cannot run inside Julia (see `_check_log_timestep`).
    mp = YelmoMirrorParameters(to_mirror(p); yelmo = (log_timestep = false,))
    y = YelmoMirror(mp, 0.0; rundir = outdir, overwrite = true)
    apply_forcing!(y, mp.phys.rho_ice, mp.phys.rho_w)
    init_state!(y, 0.0; thrm_method = "robin-cold")
    return y
end

# ----------------------------------------------------------------------
# Whole-domain scalars. YelmoModel uses the regions API; Mirror computes
# them inline (regions API is YelmoModel-only).
# ----------------------------------------------------------------------
_diag(regs, y) = (V_ice = regs[1].diag.V_ice, V_sle = regs[1].diag.V_sle,
                  H_ice_max = regs[1].diag.H_ice_max)
_diag(::Nothing, y) = _diag_mirror(y)

function _diag_mirror(y)
    H     = interior(y.tpo.H_ice)[:, :, 1]
    z_bed = interior(y.bnd.z_bed)[:, :, 1]
    z_sl  = interior(y.bnd.z_sl)[:, :, 1]
    dx = abs(Float64(y.g.Δxᶜᵃᵃ)); dy = abs(Float64(y.g.Δyᵃᶜᵃ))
    rho_ice = y.p.phys.rho_ice; rho_sw = y.p.phys.rho_sw
    V_ice = sum(H) * dx * dy * 1e-9
    sum_H_af = 0.0
    @inbounds for i in eachindex(H)
        sum_H_af += max(0.0, H[i] + min(0.0, z_bed[i] - z_sl[i]) * (rho_sw / rho_ice))
    end
    V_sle = sum_H_af * dx * dy * 1e-9 * (1.0e-3 / 394.7)
    return (V_ice = V_ice, V_sle = V_sle, H_ice_max = maximum(H))
end

# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
function main(; t_end = 20.0, dt_outer = 1.0, backend = :yelmo,
                snapshot_dt = 10.0, outdir = nothing)
    backend in (:yelmo, :mirror) || error("backend must be :yelmo or :mirror")
    cd(@__DIR__)
    outdir = something(outdir,
        joinpath(@__DIR__, backend === :mirror ? "output-mirror" : "output"))
    mkpath(outdir)
    for f in ("region_domain.nc", "snapshots.nc", "restart_final.nc", "initmip_grl.nml")
        isfile(joinpath(outdir, f)) && rm(joinpath(outdir, f))
    end

    p = build_params(backend)
    @info "initmip-grl" backend t_end dt_outer bmb_shlf=BMB_SHLF_CONST pc_method=p.yelmo.pc_method
    y = backend === :mirror ? build_mirror(p, outdir) : build_yelmo(p)

    n_steps    = Int(round(t_end / dt_outer))
    snap_every = max(Int(round(snapshot_dt / dt_outer)), 1)
    regs = backend === :yelmo ? init_regions(y; outdir = outdir) : nothing
    snap = init_output(y, joinpath(outdir, "snapshots.nc");
                       selection = OutputSelection(groups = [:tpo, :dyn, :mat, :thrm, :bnd, :dta]))

    if regs !== nothing
        update_regions!(regs, y); write_regions!(regs, y, y.time)
    end
    write_output!(snap, y)
    d0 = _diag(regs, y)
    @info "t=0 initialized" V_ice_km3=d0.V_ice V_sle_m=d0.V_sle

    @printf("  %6s  %12s  %10s  %8s\n", "t[yr]", "V_ice[km³]", "V_sle[m]", "max_H[m]")
    flush(stdout)
    t0 = time()
    for k in 1:n_steps
        step!(y, dt_outer)
        if regs !== nothing
            update_regions!(regs, y); write_regions!(regs, y, y.time)
        end
        k % snap_every == 0 && write_output!(snap, y)
        d = _diag(regs, y)
        @printf("  %6.0f  %12.4e  %10.4f  %8.1f\n", y.time, d.V_ice, d.V_sle, d.H_ice_max)
        flush(stdout)
    end
    @info "wall-clock loop time" seconds=(time() - t0) n_outer=n_steps

    close(snap.ds)
    if backend === :yelmo
        log = y.dyn.scratch.timestep_log[]
        log === nothing || close(log)
    end
    restart = init_output(y, joinpath(outdir, "restart_final.nc"))
    write_output!(restart, y); close(restart.ds)
    @info "done" backend outdir

    if backend === :yelmo && y.timer.enabled
        println("\n--- Section timings ---"); print_timings(y); flush(stdout)
    end
    return y
end
