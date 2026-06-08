# ----------------------------------------------------------------------
# TroughBenchmark — Feldmann-Levermann (2017) trough domain.
#
# Reference: Feldmann, J. & Levermann, A. (2017). "From cyclic ice
# streaming to Heinrich-like events: the grow-and-surge instability
# in the Parallel Ice Sheet Model", The Cryosphere 11, 1913-1932.
#
# Fortran reference: `yelmo/tests/yelmo_trough.f90` (`trough_f17_topo_init`
# at line 357) for the closed-form bed geometry and
# `yelmo/par/yelmo_TROUGH-F17.nml` for the domain-shape parameters
# (lx, ly, fc, dc, wc, x_cf).
#
# The closed-form geometry (`_trough_f17_zbed`) and the spec struct
# live here. The benchmark has **no** closed-form transient solution;
# host-side code is expected to provide `state(b, t)` (e.g. by reading
# a host-produced reference fixture) and `write_fixture!(b, …)` (host-
# specific fixture generation).
# ----------------------------------------------------------------------

"""
    TroughBenchmark(variant::Symbol; dx_km=4.0,
                    lx_km=700.0, ly_km=160.0, fc_km=16.0,
                    dc_m=500.0, wc_km=24.0, x_cf_km=640.0,
                    Tsrf_const=-20.0, smb_const=0.3, Qgeo_const=70.0,
                    rho_ice=910.0, g=9.81)

Feldmann-Levermann (2017) "TROUGH-F17" trough benchmark spec.

`variant`:
  - `:F17` — the standard Feldmann-Levermann (2017, TC) setup. The
    only variant supported at present.

`dx_km` is the grid resolution in km. Default `4.0` matches the
namelist value. The grid has `Nx = lx/dx + 1` points spanning
`[0, lx]` in x and `Ny = ly/dx + 1` points spanning `[-ly/2, +ly/2]`
in y, matching the Fortran `yelmo_init_grid` call at
`yelmo_trough.f90:100-102`.

The trough geometry parameters (`fc, dc, wc, x_cf`) and forcing
(`Tsrf, smb, Q_geo`) default to the values in
`yelmo_TROUGH-F17.nml`.

This benchmark has no closed-form transient solution; the
`AbstractBenchmark` `state(b, t)` and `write_fixture!(b, …)`
methods are expected to be implemented by the host (e.g. Yelmo.jl
reads a reference fixture produced by its Fortran backend).
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
end

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
                         g::Real          = 9.81)
    variant === :F17 ||
        error("TroughBenchmark: only :F17 variant is implemented (got $variant).")

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

    xc_m = collect(range(0.0, (nx - 1) * dx_km * 1e3; length=nx))
    yc_m = collect(range(ymin_km * 1e3, ymax_km * 1e3; length=ny))

    return TroughBenchmark(variant, xc_m, yc_m, Float64(dx_km),
                           Float64(lx_km), Float64(ly_km),
                           Float64(fc_km), Float64(dc_m),
                           Float64(wc_km), Float64(x_cf_km),
                           Float64(Tsrf_const), Float64(smb_const),
                           Float64(Qgeo_const),
                           Float64(rho_ice), Float64(g))
end

# F17 closed-form bedrock elevation at one (x_km, y_km) point.
# Ports `yelmo_trough.f90:382-396`. Pure math; used by host-side IC
# setters to fill `z_bed` from the analytical formula.
@inline function _trough_f17_zbed(x_km::Real, y_km::Real,
                                  fc_km::Real, dc_m::Real, wc_km::Real;
                                  zb_deep::Real = -720.0)
    zb_x = -150.0 - 0.84 * abs(Float64(x_km))                   # [m]
    e1 = -2.0 * (Float64(y_km) - Float64(wc_km)) / Float64(fc_km)
    e2 =  2.0 * (Float64(y_km) + Float64(wc_km)) / Float64(fc_km)
    zb_y = (Float64(dc_m) / (1.0 + exp(e1))) +
           (Float64(dc_m) / (1.0 + exp(e2)))                    # [m]
    return max(zb_x + zb_y, Float64(zb_deep))
end
