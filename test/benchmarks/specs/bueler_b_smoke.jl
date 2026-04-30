# ----------------------------------------------------------------------
# BUELER-B smoke benchmark.
#
# Halfar similarity solution (Bueler et al. 2005), pure decay (λ = 0):
# initial dome H₀ = 3600 m, R₀ = 750 km, evolved at fixed-temperature
# isothermal SIA from t = 0 yr (analytical IC) to t = 1000 yr.
# Produces one fixture: `bueler_b_smoke__t1000.nc`.
#
# Grid: 31×31 cells at 50 km resolution, spanning ±750 km from the
# dome centre. Total ~50 KB fixture.
#
# This is the smoke benchmark for the scaffolding PR — the first
# fixture under `test/benchmarks/fixtures/`. Real EISMINT and
# ISMIP-HOM benchmarks (with full lockstep against YelmoModel's
# velocity solver) land with milestone 3c.
# ----------------------------------------------------------------------

using Oceananigans: interior

# Synthetic 31×31 grid, dx = 50 km, centered on the dome at (0, 0).
# Cell centres span -750 km to +750 km in 50 km steps.
const _BUELER_B_DX_KM = 50.0
const _BUELER_B_R0_KM = 750.0
const _BUELER_B_NX    = Int(2 * _BUELER_B_R0_KM / _BUELER_B_DX_KM) + 1   # 31

# Halfar / BUELER-B physical parameters (mirror the Fortran defaults
# in `yelmo_benchmarks.f90:233-238`).
const _BUELER_B_H0      = 3600.0    # initial dome thickness [m]
const _BUELER_B_LAMBDA  = 0.0       # no mass balance
const _BUELER_B_N       = 3.0
const _BUELER_B_A       = 1e-16     # [Pa^-3 yr^-1]
const _BUELER_B_RHO_ICE = 910.0
const _BUELER_B_G       = 9.81

function _bueler_b_axes()
    half_km = _BUELER_B_R0_KM
    xc = collect(range(-half_km * 1e3, half_km * 1e3; length=_BUELER_B_NX))
    return xc, copy(xc)   # symmetric in y
end

function _bueler_b_setup!(ymirror, t0::Float64)
    # Compute the analytical Halfar IC at t = 0 yr (the BUELER-B spec
    # treats `time_init` as elapsed-since-IC, so we always seed at 0).
    xc_m = collect(Yelmo.Oceananigans.Grids.xnodes(ymirror.g, Yelmo.Oceananigans.Grids.Center()))
    yc_m = collect(Yelmo.Oceananigans.Grids.ynodes(ymirror.g, Yelmo.Oceananigans.Grids.Center()))

    H = @view interior(ymirror.tpo.H_ice)[:, :, 1]
    smb_view = @view interior(ymirror.bnd.smb_ref)[:, :, 1]
    bueler_test_BC!(H, smb_view, xc_m, yc_m, 0.0;
                    R0 = _BUELER_B_R0_KM, H0 = _BUELER_B_H0,
                    lambda = _BUELER_B_LAMBDA, n = _BUELER_B_N,
                    A = _BUELER_B_A,
                    rho_ice = _BUELER_B_RHO_ICE, g = _BUELER_B_G)

    # Boundary fields: flat bed, no ocean, no shelf, cold surface.
    fill!(interior(ymirror.bnd.z_bed),       0.0)
    fill!(interior(ymirror.bnd.z_sl),        0.0)
    fill!(interior(ymirror.bnd.bmb_shlf),    0.0)
    fill!(interior(ymirror.bnd.fmb_shlf),    0.0)
    fill!(interior(ymirror.bnd.T_shlf),    273.15)
    fill!(interior(ymirror.bnd.H_sed),       0.0)
    fill!(interior(ymirror.bnd.T_srf),     223.15)
    fill!(interior(ymirror.bnd.Q_geo),      42.0)
    # smb_ref already populated by bueler_test_BC! (= 0 for λ = 0).

    # Push everything we just set back to the Fortran side.
    yelmo_sync!(ymirror)

    return nothing
end

const BUELER_B_SMOKE_SPEC = let
    xc, yc = _bueler_b_axes()
    BenchmarkSpec(
        name           = "bueler_b_smoke",
        namelist_path  = abspath(joinpath(@__DIR__, "yelmo_BUELER-B.nml")),
        grid           = (xc = xc, yc = yc, grid_name = "EISMINT-EXT"),
        end_time       = 1000.0,            # yr
        output_times   = [1000.0],          # final-snapshot only
        setup_initial_state! = _bueler_b_setup!,
        time_init      = 0.0,
        dt             = 10.0,
    )
end
