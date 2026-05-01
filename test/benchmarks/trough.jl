# ----------------------------------------------------------------------
# TroughBenchmark — Feldmann-Levermann (2017) trough domain.
#
# Unlike `BuelerBenchmark` (which has a closed-form Halfar solution),
# `TroughBenchmark` does **not** carry an analytical reference. Its
# regression target is a YelmoMirror-produced fixture committed under
# `test/benchmarks/fixtures/`. The benchmark thus uses the
# `AbstractBenchmark` interface but routes `state(b, t)` through a
# fixture-loading helper (so the in-memory `YelmoModel(b, t)` builds
# from the committed fixture) and uses `write_fixture!(b, path; times)`
# as a thin wrapper around `run_mirror_benchmark!` from `helpers.jl`.
#
# The Fortran reference for the F17 setup is
# `/Users/alrobi001/models/yelmo/tests/yelmo_trough.f90` —
# `trough_f17_topo_init` (line 357) writes the initial bedrock /
# ice-thickness fields, and the namelist
# `/Users/alrobi001/models/yelmo/par/yelmo_TROUGH-F17.nml` carries the
# domain-shape parameters (lx, ly, fc, dc, wc, x_cf).
# ----------------------------------------------------------------------

export TroughBenchmark

"""
    TroughBenchmark(variant::Symbol; dx_km=4.0, namelist_path,
                    lx_km=700.0, ly_km=160.0, fc_km=16.0,
                    dc_m=500.0, wc_km=24.0, x_cf_km=640.0,
                    Tsrf_const=-20.0, smb_const=0.3, Qgeo_const=70.0,
                    rho_ice=910.0, g=9.81)

Feldmann-Levermann (2017) "TROUGH-F17" trough benchmark.

`variant`:
  - `:F17` — the standard Feldmann-Levermann (2017, TC) setup. The
    only variant supported in this milestone.

`dx_km` is the grid resolution in km. Defaults to the namelist value
`4.0`. The grid has `Nx = lx/dx + 1` points spanning
`[0, lx]` in x and `Ny = ly/dx + 1` points spanning `[-ly/2, +ly/2]`
in y, matching the Fortran `yelmo_init_grid` call at
`yelmo_trough.f90:100-102`.

The trough geometry parameters (`fc, dc, wc, x_cf`) and forcing
(`Tsrf, smb, Q_geo`) default to the values in
`yelmo_TROUGH-F17.nml`. `namelist_path` defaults to
`test/benchmarks/specs/yelmo_TROUGH.nml` — a Yelmo-namelist that
disables all file loads so initial state comes from the Julia
`_setup_trough_initial_state!` callback.

Per the locked-in milestone-3d design, this benchmark is
single-time (snapshot at `times = [t_out]`) and does **not** carry
an `analytical_velocity` method.
"""
struct TroughBenchmark <: AbstractBenchmark
    variant::Symbol
    xc::Vector{Float64}
    yc::Vector{Float64}
    dx_km::Float64

    # Trough geometry parameters (from yelmo_TROUGH-F17.nml /
    # yelmo_trough.f90:357 trough_f17_topo_init).
    lx_km::Float64
    ly_km::Float64
    fc_km::Float64        # characteristic side-wall width
    dc_m::Float64         # depth of bed trough below side walls
    wc_km::Float64        # half-width of bed trough
    x_cf_km::Float64      # calving-front x-position

    # Forcing (uniform).
    Tsrf_const::Float64   # [degC] surface T
    smb_const::Float64    # [m/yr] surface mass balance
    Qgeo_const::Float64   # [mW/m²] geothermal flux

    # Physical constants (echoed from yelmo_phys_const TROUGH section).
    rho_ice::Float64
    g::Float64

    # Path to the Yelmo Fortran namelist that YelmoMirror reads.
    namelist_path::String
end

# Default namelist path, relative to this file. Created in commit 2.
const _DEFAULT_TROUGH_NAMELIST = abspath(joinpath(@__DIR__, "specs",
                                                   "yelmo_TROUGH.nml"))

function TroughBenchmark(variant::Symbol;
                         dx_km::Real      = 4.0,
                         lx_km::Real      = 700.0,
                         ly_km::Real      = 160.0,
                         fc_km::Real      = 16.0,
                         dc_m::Real       = 500.0,
                         wc_km::Real      = 24.0,
                         x_cf_km::Real    = 640.0,
                         Tsrf_const::Real = -20.0,
                         smb_const::Real  = 0.3,
                         Qgeo_const::Real = 70.0,
                         rho_ice::Real    = 910.0,
                         g::Real          = 9.81,
                         namelist_path::AbstractString = _DEFAULT_TROUGH_NAMELIST)
    variant === :F17 ||
        error("TroughBenchmark: only :F17 variant is implemented in milestone 3d (got $variant).")

    # Mirror the Fortran grid construction in yelmo_trough.f90:96-102:
    #   xmax =  lx                             # km
    #   ymax =  ly/2,  ymin = -ly/2            # km
    #   nx = int(xmax/dx)+1                    # cell-centre count
    #   ny = int((ymax-ymin)/dx)+1
    #   x[i] = 0   + (i-1)*dx                  # i = 1..nx, in km
    #   y[j] = ymin + (j-1)*dx                 # j = 1..ny, in km
    nx = Int(floor(lx_km / dx_km)) + 1
    ymax_km =  ly_km / 2.0
    ymin_km = -ly_km / 2.0
    ny = Int(floor((ymax_km - ymin_km) / dx_km)) + 1

    # Fortran convention: cell centres at [0, dx, 2dx, ...] km in x and
    # [-ly/2, -ly/2 + dx, ..., +ly/2] km in y. Convert to metres.
    xc_m = collect(range(0.0, (nx - 1) * dx_km * 1e3; length=nx))
    yc_m = collect(range(ymin_km * 1e3, ymax_km * 1e3; length=ny))

    return TroughBenchmark(variant, xc_m, yc_m, Float64(dx_km),
                           Float64(lx_km), Float64(ly_km),
                           Float64(fc_km), Float64(dc_m),
                           Float64(wc_km), Float64(x_cf_km),
                           Float64(Tsrf_const), Float64(smb_const),
                           Float64(Qgeo_const),
                           Float64(rho_ice), Float64(g),
                           String(namelist_path))
end

# Resolve the per-spec name used for fixture filenames. Mirrors
# `_spec_name` for `BuelerBenchmark` so `regenerate.jl` can dispatch
# uniformly. Single-time fixtures use `<name>_t<int>.nc`.
_spec_name(b::TroughBenchmark) = "trough_$(lowercase(string(b.variant)))"

# `analytical_velocity` is NOT defined for TroughBenchmark — falls
# through to the default error stub in `benchmarks.jl`.
