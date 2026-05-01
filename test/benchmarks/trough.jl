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

# ----------------------------------------------------------------------
# Fixture path resolution.
# ----------------------------------------------------------------------

# Resolve the canonical fixture path for `b` at time `t`, relative to
# the committed `test/benchmarks/fixtures/` directory. Single-time
# fixtures only — multi-time deferred per locked-in milestone-3d
# decision Q11.
const _TROUGH_FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

function _trough_fixture_path(b::TroughBenchmark, t::Real;
                              fixtures_dir::AbstractString = _TROUGH_FIXTURES_DIR)
    return joinpath(fixtures_dir,
                    "$(_spec_name(b))_t$(Int(round(Float64(t)))).nc")
end

# ----------------------------------------------------------------------
# state(b, t): TroughBenchmark has NO closed-form solution. We read
# the YelmoMirror-produced fixture from disk and route it through the
# same in-memory YelmoModel(b, t) constructor BUELER uses. If the
# fixture is missing, error with a hint to run regenerate.jl.
# ----------------------------------------------------------------------

"""
    state(b::TroughBenchmark, t::Real) -> NamedTuple

Read the committed YelmoMirror fixture for `b` at time `t` and return
a NamedTuple whose keys map onto Yelmo schema variables (and are
routed into the appropriate component group by the generic
`YelmoModel(::AbstractBenchmark, t)` constructor in `benchmarks.jl`).

Errors with a hint to run `regenerate.jl` if the fixture is missing.

Returned keys: `xc`, `yc`, plus whichever schema-Center fields the
fixture carries (`H_ice`, `z_bed`, `f_grnd`, `smb_ref`, `T_srf`,
`Q_geo`).

Face-staggered fields (`ux_b`, `uy_b`, `ux_bar`, `uy_bar`, `ATT`)
are *deliberately not* returned by `state()`: their on-disk layout
matches Yelmo Fortran's `(xc, yc)` cell-centred convention, but the
in-memory `YelmoModel` allocates them as `XFaceField`/`YFaceField`
with halo-included sizes `(Nx+1, Ny)` / `(Nx, Ny+1)`. Loading them
through `_assign_field!` would shape-mismatch. The regression test
loads the file-based `YelmoModel(restart_file, time)` for
face-staggered comparisons and uses `state(b, t)` only for the
Center-aligned state.

Yelmo Fortran restarts store Float32 — values are promoted to
Float64 here to match Yelmo.jl's Field eltypes.
"""
function state(b::TroughBenchmark, t::Real)
    path = _trough_fixture_path(b, t)
    isfile(path) || error(
        "TroughBenchmark.state: fixture missing at $path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl " *
        "$(_spec_name(b)) --overwrite` first.")

    NCDataset(path, "r") do ds
        # Always forward the grid axes from the benchmark struct (so
        # the in-memory and file-based loads use bit-identical xc/yc
        # in metres, regardless of whether the fixture stored them in
        # km or m).
        out = Dict{Symbol,Any}(:xc => b.xc, :yc => b.yc)

        # Optional Center-aligned fields: forward only those present
        # in the fixture. Yelmo Fortran restarts have a trailing
        # singleton time-dim on 2D fields → drop it via [:, :, 1].
        for name in ("H_ice", "z_bed", "f_grnd",
                     "smb_ref", "T_srf", "Q_geo")
            if haskey(ds, name)
                raw = ds[name][:, :, :]
                arr2d = ndims(raw) == 3 ? raw[:, :, 1] : raw
                out[Symbol(name)] = Array{Float64}(arr2d)
            end
        end

        return NamedTuple(out)
    end
end

# ----------------------------------------------------------------------
# Trough F17 initial-state callback (the YelmoMirror dispatch input).
# Mirrors `trough_f17_topo_init` (yelmo_trough.f90:357) faithfully:
# compute z_bed from the F17 formula, set H_ice = 50 m everywhere
# inside the calving front (xc < x_cf), zero beyond. Plus the
# uniform forcing (T_srf, smb, Q_geo, z_sl, T_shlf, H_sed, calv_mask)
# from yelmo_trough.f90:111-119 + define_calving_front:306-321.
# ----------------------------------------------------------------------

# Compute F17 z_bed for one (x_km, y_km) point. Ports
# yelmo_trough.f90:382-396.
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

# Apply the F17 initial state to a YelmoMirror in-place. Reads
# `b.xc`/`b.yc` (metres) → converts to km for the F17 formula → fills
# z_bed, H_ice, surface forcing, calving mask. Then `yelmo_sync!`s
# Julia state back to Fortran so `init_state!` sees the trough
# topography.
#
# `time` is the simulation time at the call (passed through by
# `BenchmarkSpec.setup_initial_state!`; unused for the F17 setup
# since the IC has no time dependence).
function _setup_trough_initial_state!(ymirror, b::TroughBenchmark, time::Real)
    Nx = length(b.xc)
    Ny = length(b.yc)

    # Scratch arrays for the (Nx, Ny) interior fields. Build in pure
    # Julia, then copy into the YelmoMirror Field interiors.
    z_bed = zeros(Nx, Ny)
    H_ice = zeros(Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        x_km = b.xc[i] * 1e-3
        y_km = b.yc[j] * 1e-3
        z_bed[i, j] = _trough_f17_zbed(x_km, y_km,
                                        b.fc_km, b.dc_m, b.wc_km)
        H_ice[i, j] = (x_km <= b.x_cf_km) ? 50.0 : 0.0
    end

    # === Push 2D fields into the YelmoMirror — mirrors the Fortran
    # block at yelmo_trough.f90:206-211 plus the boundary loads at
    # 112-119 plus the calving-front mask at 237.
    _assign_field!(ymirror.bnd.z_bed,    z_bed)
    _assign_field!(ymirror.tpo.H_ice,    H_ice)

    # Boundary forcing (uniform). z_sl=0, bmb_shlf=0, H_sed=0.
    fill!(interior(ymirror.bnd.z_sl),     0.0)
    fill!(interior(ymirror.bnd.bmb_shlf), 0.0)
    fill!(interior(ymirror.bnd.H_sed),    0.0)

    # T_shlf = T0 (using the pressure-melting reference from
    # YelmoConstants — match the Fortran `yelmo1%bnd%c%T0`
    # = 273.15 K).
    T0 = 273.15
    fill!(interior(ymirror.bnd.T_shlf), T0)

    # T_srf = T0 + Tsrf_const [K]
    fill!(interior(ymirror.bnd.T_srf), T0 + b.Tsrf_const)

    # smb (m/yr), Q_geo (mW/m²)
    fill!(interior(ymirror.bnd.smb_ref), b.smb_const)
    fill!(interior(ymirror.bnd.Q_geo),   b.Qgeo_const)

    # Calving mask: `True` where x ≥ x_cf  (i.e. ice will be calved
    # beyond the calving front). Stored as Float64 (0/1) in the
    # YelmoMirror — mirrors the Fortran logical → C-API double
    # round-trip.
    calv_mask = zeros(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        x_km = b.xc[i] * 1e-3
        calv_mask[i, j] = (x_km >= b.x_cf_km) ? 1.0 : 0.0
    end
    _assign_field!(ymirror.bnd.calv_mask, calv_mask)

    # Sync Julia → Fortran so init_state! sees the trough topography.
    yelmo_sync!(ymirror)

    return ymirror
end

# ----------------------------------------------------------------------
# write_fixture!(b, path; times) — single-time wrapper around
# `run_mirror_benchmark!`. Builds a `BenchmarkSpec` from the trough
# parameters, drives YelmoMirror to `times[1]`, writes the restart,
# moves the resulting NetCDF to `path` if needed.
# ----------------------------------------------------------------------

"""
    write_fixture!(b::TroughBenchmark, path::AbstractString;
                   times = [1000.0]) -> Vector{String}

Drive YelmoMirror through the F17 initial-state setup and a single
end-time integration, writing the resulting restart NetCDF to `path`.

Single-time only in this milestone — multi-time fixtures (a `time`
dimension with multiple snapshots) are deferred per the locked-in
milestone-3d decision Q11.

Returns a 1-element `Vector{String}` containing `path`.
"""
function write_fixture!(b::TroughBenchmark, path::AbstractString;
                        times::AbstractVector{<:Real} = [1000.0])
    length(times) == 1 ||
        error("write_fixture!(TroughBenchmark, …): multi-time fixtures " *
              "deferred to a future milestone (got $(length(times)) times).")
    t_out = Float64(first(times))

    isfile(b.namelist_path) || error(
        "TroughBenchmark.write_fixture!: namelist not found at " *
        "$(b.namelist_path).")

    spec = BenchmarkSpec(
        name           = _spec_name(b),
        namelist_path  = b.namelist_path,
        grid           = (xc = b.xc, yc = b.yc,
                          grid_name = "TROUGH-F17"),
        time_init      = 0.0,
        end_time       = t_out,
        output_times   = [t_out],
        dt             = 5.0,
        setup_initial_state! = (ymirror, t) ->
            _setup_trough_initial_state!(ymirror, b, t),
    )

    fixtures_dir = dirname(path)
    mkpath(fixtures_dir)
    isfile(path) && rm(path)

    paths = run_mirror_benchmark!(spec; fixtures_dir = fixtures_dir,
                                  overwrite = true)
    src = paths[1]
    if src != path
        mv(src, path; force = true)
        paths = [path]
    end
    return paths
end
