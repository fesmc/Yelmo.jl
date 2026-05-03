# ----------------------------------------------------------------------
# MISMIP3DBenchmark — Marine Ice-Sheet Model Intercomparison Project,
# experiment "Stnd" (steady-state buildup phase).
#
# Reference:
#   - Pattyn et al. (2013), "Grounding-line migration in plan-view marine
#     ice-sheet models", J. Glaciol. 59(215), 410-422.
#   - Yelmo Fortran implementation: `yelmo/tests/mismip3D.f90` and
#     `yelmo/tests/yelmo_mismip.f90` (program), `yelmo/par/yelmo_MISMIP3D.nml`.
#
# Geometry (Stnd):
#
#   - Domain: x ∈ [0, 800] km (Bounded), y ∈ [-50, +50] km (Periodic).
#   - Bed:    z_bed(x, y) = -100 - x_km   (m, y-invariant; slope -1/1000).
#   - IC:     H_ice = 10 m where z_bed ≥ -500 m, else 0
#             (i.e. the ice-bearing region is x_km ≤ 400).
#
# Boundaries (Stnd, all constant in time):
#
#   - smb_ref = 0.5 m/yr
#   - T_srf   = 273.15 K
#   - Q_geo   = 42 mW/m²
#   - calv_mask(nx, :) = .TRUE. (only the eastern column allows calving).
#     Yelmo.jl approximation: `bnd.ice_allowed[Nx, :] = 0` so the residual
#     cleanup keeps ice from accumulating in the calving column.
#
# Validation strategy:
#
#   This benchmark serves as a multi-step time-evolution exercise of the
#   `step!(YelmoModel, dt)` orchestrator (topo + dyn) under :periodic_y
#   boundaries. There is no closed-form velocity or H_ice solution.
#   `analytical_velocity` is intentionally NOT implemented.
#
#   The fixture written by `write_fixture!` captures the analytical IC at
#   t=0 (single-time, no Halfar-style time evolution at the benchmark
#   level). Multi-step dynamics are exercised by the test, not by the
#   fixture file.
# ----------------------------------------------------------------------

export MISMIP3DBenchmark

"""
    MISMIP3DBenchmark(variant::Symbol = :Stnd; dx_km=16.0,
                       xmax_km=800.0, Ly_km=100.0,
                       H0=10.0, z_bed_floor=-500.0,
                       bed_intercept=-100.0, bed_slope=1.0,
                       smb_const=0.5, T_srf_const=273.15, Q_geo_const=42.0,
                       rho_ice=910.0, rho_sw=1028.0, g=9.81,
                       n_glen=3.0, A_glen=3.1536e-18,
                       cf_ref=3.165176e4, N_eff_const=1.0)

MISMIP3D Stnd benchmark struct. Carries domain axes, bed geometry, IC
slab thickness, surface forcing, and Glen-flow / friction parameters.

`dx_km = 16.0` gives Nx = 51, Ny = 7 — odd Ny yields a centerline cell
at j = 4 for stronger y-symmetry tests. `dx_km = 20.0` (Fortran default)
gives Nx = 41, Ny = 6 (even Ny, no centerline).

Only `variant = :Stnd` is supported in this milestone.
"""
struct MISMIP3DBenchmark <: AbstractBenchmark
    variant::Symbol
    xc::Vector{Float64}
    yc::Vector{Float64}
    xmax_km::Float64
    Ly_km::Float64
    dx_km::Float64
    H0::Float64
    z_bed_floor::Float64
    bed_intercept::Float64
    bed_slope::Float64
    smb_const::Float64
    T_srf_const::Float64
    Q_geo_const::Float64
    rho_ice::Float64
    rho_sw::Float64
    g::Float64
    n_glen::Float64
    A_glen::Float64
    cf_ref::Float64
    N_eff_const::Float64
    namelist_path::String
end

# Default namelist path, relative to this file. Mirrors the
# `_DEFAULT_TROUGH_NAMELIST` pattern.
const _DEFAULT_MISMIP3D_NAMELIST = abspath(joinpath(@__DIR__, "specs",
                                                     "yelmo_MISMIP3D.nml"))

function MISMIP3DBenchmark(variant::Symbol = :Stnd;
                            dx_km::Real         = 16.0,
                            xmax_km::Real       = 800.0,
                            Ly_km::Real         = 100.0,
                            H0::Real            = 10.0,
                            z_bed_floor::Real   = -500.0,
                            bed_intercept::Real = -100.0,
                            bed_slope::Real     = 1.0,
                            smb_const::Real     = 0.5,
                            T_srf_const::Real   = 273.15,
                            Q_geo_const::Real   = 42.0,
                            rho_ice::Real       = 910.0,
                            rho_sw::Real        = 1028.0,
                            g::Real             = 9.81,
                            n_glen::Real        = 3.0,
                            A_glen::Real        = 3.1536e-18,
                            cf_ref::Real        = 3.165176e4,
                            N_eff_const::Real   = 1.0,
                            namelist_path::AbstractString = _DEFAULT_MISMIP3D_NAMELIST)
    variant === :Stnd || error(
        "MISMIP3DBenchmark: only variant :Stnd is supported (got $(variant)).")

    dx_km_f  = Float64(dx_km)
    xmax_km_f = Float64(xmax_km)
    Ly_km_f  = Float64(Ly_km)

    # Nx / Ny follow the Fortran node-count convention
    # `int(extent/dx)+1` (`yelmo_mismip.f90:132`), reused here to size
    # the cell-centred Yelmo.jl grid. With dx_km=16 (recommended):
    # Nx = int(800/16)+1 = 51, Ny = int(100/16)+1 = 7 (odd → centerline
    # cell at j=4). With dx_km=20 (Fortran default): Nx = 41, Ny = 6.
    Nx = Int(floor(xmax_km_f / dx_km_f)) + 1
    Ny = Int(floor(Ly_km_f / dx_km_f)) + 1

    # Cell-centre axes (Yelmo.jl convention):
    #   - x ∈ [0, Nx·dx]: cell-centre at (i - 0.5)·dx, i = 1..Nx.
    #   - y ∈ [-Ny·dx/2, +Ny·dx/2]: cell-centre at -Ny·dx/2 + (j-0.5)·dx,
    #     j = 1..Ny. For odd Ny the centerline cell j = (Ny+1)/2 sits
    #     exactly at y = 0.
    #
    # Note: with dx_km=16 / Ny=7 the y-extent is 112 km, slightly wider
    # than the Fortran [-50, +50] = 100 km range. This is an unavoidable
    # consequence of the cell-centred Yelmo.jl grid + the Fortran int+1
    # node convention, and is harmless because the Stnd IC is y-invariant.
    dx_m = dx_km_f * 1e3
    xc = collect(range(0.5 * dx_m, (Nx - 0.5) * dx_m; length=Nx))
    yc = collect(range(-0.5 * Ny * dx_m + 0.5 * dx_m,
                        0.5 * Ny * dx_m - 0.5 * dx_m; length=Ny))

    return MISMIP3DBenchmark(variant,
                              xc, yc,
                              xmax_km_f, Ly_km_f, dx_km_f,
                              Float64(H0),
                              Float64(z_bed_floor),
                              Float64(bed_intercept),
                              Float64(bed_slope),
                              Float64(smb_const),
                              Float64(T_srf_const),
                              Float64(Q_geo_const),
                              Float64(rho_ice),
                              Float64(rho_sw),
                              Float64(g),
                              Float64(n_glen),
                              Float64(A_glen),
                              Float64(cf_ref),
                              Float64(N_eff_const),
                              String(namelist_path))
end

"""
    state(b::MISMIP3DBenchmark, t::Real) -> NamedTuple

Dual-path: at `t = 0`, returns the analytical Stnd IC (closed-form
NamedTuple, no NetCDF round-trip). For `t > 0`, reads the YelmoMirror
reference fixture at `_mismip3d_fixture_path(b, t)` (mirrors the
`TroughBenchmark` pattern). The fixture file must already exist —
errors with a hint to run `regenerate.jl` if missing.

Analytical (`t = 0`) returned NamedTuple keys:

  - `xc`, `yc`     — grid axes (metres).
  - `H_ice`        — 10 m where z_bed ≥ -500 m, else 0 (extends to x_km ≤ 400
                     under the default geometry).
  - `z_bed`        — `-100 - x_km` (m, y-invariant).
  - `z_sl`         — 0 m (sea level).
  - `smb_ref`      — 0.5 m/yr (constant; eastern column kept ice-free via
                     `ice_allowed`).
  - `T_srf`        — 273.15 K (constant).
  - `Q_geo`        — 42 mW/m² (constant).
  - `bmb_shlf`     — 0 m/yr.
  - `T_shlf`       — 273.15 K.
  - `H_sed`        — 0 m.
  - `ice_allowed`  — 1 everywhere except `[Nx, :] = 0` (kill column at
                     calving boundary).

Fixture-loaded (`t > 0`) returned keys: `xc`, `yc`, plus whichever
schema-Center fields the fixture carries (`H_ice`, `z_bed`, `f_grnd`,
`smb_ref`, `T_srf`, `Q_geo`). Face-staggered fields (`ux_b`, `uy_b`,
…) are deliberately not returned — see the `TroughBenchmark.state`
docstring for rationale.
"""
function state(b::MISMIP3DBenchmark, t::Real)
    t_f = Float64(t)

    if t_f == 0.0
        return _mismip3d_analytical_state(b)
    end

    # Fixture-backed path (YelmoMirror reference).
    path = _mismip3d_fixture_path(b, t_f)
    isfile(path) || error(
        "MISMIP3DBenchmark.state: fixture missing at $path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl " *
        "$(_spec_name(b)) --overwrite` first.")

    NCDataset(path, "r") do ds
        out = Dict{Symbol,Any}(:xc => b.xc, :yc => b.yc)
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

# Analytical Stnd IC NamedTuple (extracted so `state(b, 0.0)` returns
# the historical analytical content unchanged).
function _mismip3d_analytical_state(b::MISMIP3DBenchmark)
    Nx, Ny = length(b.xc), length(b.yc)

    # z_bed = bed_intercept - bed_slope·x_km (y-invariant).
    z_bed = [b.bed_intercept - b.bed_slope * (b.xc[i] / 1e3) for i in 1:Nx, j in 1:Ny]

    # H_ice = H0 where z_bed ≥ z_bed_floor, else 0.
    H_ice = [z_bed[i, j] >= b.z_bed_floor ? b.H0 : 0.0 for i in 1:Nx, j in 1:Ny]

    # ice_allowed: 1 everywhere except the eastern (calving) column.
    ice_allowed = ones(Float64, Nx, Ny)
    ice_allowed[Nx, :] .= 0.0

    smb_ref  = fill(b.smb_const,   Nx, Ny)
    T_srf    = fill(b.T_srf_const, Nx, Ny)
    Q_geo    = fill(b.Q_geo_const, Nx, Ny)
    z_sl     = zeros(Nx, Ny)
    bmb_shlf = zeros(Nx, Ny)
    T_shlf   = fill(b.T_srf_const, Nx, Ny)
    H_sed    = zeros(Nx, Ny)

    return (xc = b.xc, yc = b.yc,
            H_ice = H_ice, z_bed = z_bed, z_sl = z_sl,
            smb_ref = smb_ref, T_srf = T_srf, Q_geo = Q_geo,
            bmb_shlf = bmb_shlf, T_shlf = T_shlf, H_sed = H_sed,
            ice_allowed = ice_allowed)
end

# Spec name used by `regenerate.jl` and the fixture filename.
_spec_name(b::MISMIP3DBenchmark) = "mismip3d_$(lowercase(string(b.variant)))"

# Resolve the canonical fixture path for `b` at time `t`. Mirrors the
# `_trough_fixture_path` helper. Single-time fixtures only; Phase 2
# emits two: the analytical IC at t=0 and the YelmoMirror reference
# at the requested end-time.
const _MISMIP3D_FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

function _mismip3d_fixture_path(b::MISMIP3DBenchmark, t::Real;
                                fixtures_dir::AbstractString = _MISMIP3D_FIXTURES_DIR)
    return joinpath(fixtures_dir,
                    "$(_spec_name(b))_t$(Int(round(Float64(t)))).nc")
end

# Default zeta axes for MISMIP3D analytical fixture. Match BUELER /
# HOM-C (uniform 11-point ice / 5-point rock) for consistency with
# the file-based loader's default schema.
const _MISMIP3D_DEFAULT_ZETA_AC      = collect(range(0.0, 1.0; length=11))
const _MISMIP3D_DEFAULT_ZETA_ROCK_AC = collect(range(0.0, 1.0; length=5))

"""
    write_fixture!(b::MISMIP3DBenchmark, path::AbstractString;
                   times = [0.0]) -> Vector{String}

Dual-mode fixture writer. For each `t` in `times`:

  - **t == 0.0**: serialise the analytical MISMIP3D Stnd IC to
    `_mismip3d_fixture_path(b, 0.0)` (or `path` when `times = [0.0]`).
    No YelmoMirror; closed-form NamedTuple → NetCDF.
  - **t > 0.0**: drive YelmoMirror from the analytical IC through
    the `_setup_mismip3d_initial_state!` callback + the namelist at
    `b.namelist_path`, integrate to `t`, write the restart to
    `_mismip3d_fixture_path(b, t)` (or `path` when `times = [t]`).

Returns a `Vector{String}` containing one path per requested time
(in the same order as `times`).

When called with the default single-element `times = [0.0]` the
behaviour is identical to the original analytical-only contract:
the file is written verbatim to `path`. When `times` has more than
one entry, only `path` for `times == [first(times)]`-style call sites
makes sense; `regenerate.jl` dispatches `[0.0, 500.0]` via the
multi-call wrapper below, where each call passes its own `path`.
"""
function write_fixture!(b::MISMIP3DBenchmark, path::AbstractString;
                        times::AbstractVector{<:Real} = [0.0])
    length(times) == 1 ||
        error("write_fixture!(MISMIP3DBenchmark, …): pass `times` with " *
              "exactly one entry per call (got $(length(times))). " *
              "regenerate.jl loops over [0.0, 500.0] one t at a time.")
    t = Float64(first(times))

    if t > 0.0
        return _write_mismip3d_mirror_fixture!(b, path, t)
    end

    s = _mismip3d_analytical_state(b)
    mkpath(dirname(path))
    isfile(path) && rm(path)

    NCDataset(path, "c") do ds
        Nx = length(b.xc)
        Ny = length(b.yc)
        zeta_ac      = _MISMIP3D_DEFAULT_ZETA_AC
        zeta_rock_ac = _MISMIP3D_DEFAULT_ZETA_ROCK_AC
        Nz_ac      = length(zeta_ac)
        Nz_rock_ac = length(zeta_rock_ac)

        defDim(ds, "xc",          Nx)
        defDim(ds, "yc",          Ny)
        defDim(ds, "zeta",        Nz_ac - 1)
        defDim(ds, "zeta_ac",     Nz_ac)
        defDim(ds, "zeta_rock",   Nz_rock_ac - 1)
        defDim(ds, "zeta_rock_ac", Nz_rock_ac)

        xv = defVar(ds, "xc", Float64, ("xc",))
        xv[:] = b.xc ./ 1e3
        xv.attrib["units"] = "km"

        yv = defVar(ds, "yc", Float64, ("yc",))
        yv[:] = b.yc ./ 1e3
        yv.attrib["units"] = "km"

        zc = defVar(ds, "zeta", Float64, ("zeta",))
        zc[:] = 0.5 .* (zeta_ac[1:end-1] .+ zeta_ac[2:end])
        zc.attrib["units"] = "1"

        zac = defVar(ds, "zeta_ac", Float64, ("zeta_ac",))
        zac[:] = zeta_ac
        zac.attrib["units"] = "1"

        zrc = defVar(ds, "zeta_rock", Float64, ("zeta_rock",))
        zrc[:] = 0.5 .* (zeta_rock_ac[1:end-1] .+ zeta_rock_ac[2:end])
        zrc.attrib["units"] = "1"

        zrac = defVar(ds, "zeta_rock_ac", Float64, ("zeta_rock_ac",))
        zrac[:] = zeta_rock_ac
        zrac.attrib["units"] = "1"

        Hv = defVar(ds, "H_ice", Float64, ("xc", "yc"))
        Hv[:, :] = s.H_ice
        Hv.attrib["units"]     = "m"
        Hv.attrib["long_name"] = "Ice thickness (MISMIP3D Stnd 10 m slab)"

        zb = defVar(ds, "z_bed", Float64, ("xc", "yc"))
        zb[:, :] = s.z_bed
        zb.attrib["units"]     = "m"
        zb.attrib["long_name"] = "Bedrock elevation (-100 - x_km)"

        zslv = defVar(ds, "z_sl", Float64, ("xc", "yc"))
        zslv[:, :] = s.z_sl
        zslv.attrib["units"]     = "m"
        zslv.attrib["long_name"] = "Sea level"

        smbv = defVar(ds, "smb_ref", Float64, ("xc", "yc"))
        smbv[:, :] = s.smb_ref
        smbv.attrib["units"]     = "m/yr"
        smbv.attrib["long_name"] = "Surface mass balance (MISMIP3D Stnd: 0.5 m/yr)"

        Tv = defVar(ds, "T_srf", Float64, ("xc", "yc"))
        Tv[:, :] = s.T_srf
        Tv.attrib["units"]     = "K"
        Tv.attrib["long_name"] = "Surface temperature (273.15 K)"

        Qv = defVar(ds, "Q_geo", Float64, ("xc", "yc"))
        Qv[:, :] = s.Q_geo
        Qv.attrib["units"]     = "mW m^-2"
        Qv.attrib["long_name"] = "Geothermal flux (42 mW/m²)"

        bmbv = defVar(ds, "bmb_shlf", Float64, ("xc", "yc"))
        bmbv[:, :] = s.bmb_shlf
        bmbv.attrib["units"]     = "m/yr"
        bmbv.attrib["long_name"] = "Basal mass balance (shelf, zero)"

        Tshv = defVar(ds, "T_shlf", Float64, ("xc", "yc"))
        Tshv[:, :] = s.T_shlf
        Tshv.attrib["units"]     = "K"
        Tshv.attrib["long_name"] = "Shelf base temperature"

        Hsv = defVar(ds, "H_sed", Float64, ("xc", "yc"))
        Hsv[:, :] = s.H_sed
        Hsv.attrib["units"]     = "m"
        Hsv.attrib["long_name"] = "Sediment thickness (zero)"

        iav = defVar(ds, "ice_allowed", Float64, ("xc", "yc"))
        iav[:, :] = s.ice_allowed
        iav.attrib["units"]     = "1"
        iav.attrib["long_name"] = "Ice-allowed mask (eastern col = 0, " *
                                  "approximating Fortran calv_mask kill-pos)"

        ds.attrib["benchmark"]      = "MISMIP3D-$(string(b.variant))"
        ds.attrib["solution_type"]  = "analytical-IC"
        ds.attrib["time_yr"]        = t
        ds.attrib["xmax_km"]        = b.xmax_km
        ds.attrib["Ly_km"]          = b.Ly_km
        ds.attrib["dx_km"]          = b.dx_km
        ds.attrib["H0_m"]           = b.H0
        ds.attrib["bed_intercept"]  = b.bed_intercept
        ds.attrib["bed_slope"]      = b.bed_slope
        ds.attrib["z_bed_floor_m"]  = b.z_bed_floor
        ds.attrib["smb_const_m_yr"] = b.smb_const
        ds.attrib["T_srf_K"]        = b.T_srf_const
        ds.attrib["Q_geo_mWm2"]     = b.Q_geo_const
        ds.attrib["n_glen"]         = b.n_glen
        ds.attrib["A_glen_Pa-3yr-1"] = b.A_glen
        ds.attrib["cf_ref"]         = b.cf_ref
        ds.attrib["N_eff_Pa"]       = b.N_eff_const
    end

    return [path]
end

# `analytical_velocity` is intentionally not implemented (no closed-form
# velocity solution for MISMIP3D Stnd). Falls through to the
# `AbstractBenchmark` default error stub.

# ----------------------------------------------------------------------
# YelmoMirror initial-state callback for MISMIP3D Stnd.
#
# Mirrors the Yelmo.jl-side test override (test_mismip3d_stnd.jl
# lines 184-196): apply the thicker grounded-IC variant from the
# Fortran source's commented-out section in mismip3D.f90:62-64
#   `H_ice = max(0, 1000 - 0.9·z_bed)` for z_bed < 0
# so the SSA system is well-posed from step 1 (the literal Fortran
# 10 m all-floating IC is rank-deficient — see test header for the
# full discussion). z_bed below `b.z_bed_floor` stays ice-free.
#
# Also sets the constant Stnd boundary forcing (T_srf, smb_ref, Q_geo,
# z_sl, bmb_shlf, H_sed) plus `calv_mask[Nx, :] = 1` for the Fortran
# kill-pos calving of the eastern column.
# ----------------------------------------------------------------------

# `time` is unused (Stnd IC has no time dependence); kept for the
# `(ymirror, time)` callback signature `BenchmarkSpec.setup_initial_state!`
# expects.
function _setup_mismip3d_initial_state!(ymirror, b::MISMIP3DBenchmark, time::Real)
    Nx = length(b.xc)
    Ny = length(b.yc)

    # Compute z_bed and the thicker grounded H_ice override on the
    # Julia side, then assign into YelmoMirror Field interiors.
    z_bed = zeros(Nx, Ny)
    H_ice = zeros(Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        x_km = b.xc[i] / 1e3
        zb   = b.bed_intercept - b.bed_slope * x_km
        z_bed[i, j] = zb
        H_ice[i, j] = (zb < b.z_bed_floor) ? 0.0 :
                                              max(0.0, 1000.0 - 0.9 * zb)
    end

    _assign_field!(ymirror.bnd.z_bed, z_bed)
    _assign_field!(ymirror.tpo.H_ice, H_ice)

    # Stnd boundary forcing (uniform).
    fill!(interior(ymirror.bnd.z_sl),     0.0)
    fill!(interior(ymirror.bnd.bmb_shlf), 0.0)
    fill!(interior(ymirror.bnd.H_sed),    0.0)
    fill!(interior(ymirror.bnd.T_shlf),   b.T_srf_const)
    fill!(interior(ymirror.bnd.T_srf),    b.T_srf_const)
    fill!(interior(ymirror.bnd.smb_ref),  b.smb_const)
    fill!(interior(ymirror.bnd.Q_geo),    b.Q_geo_const)

    # Calving mask: True only on the eastern (i=Nx) column (Fortran's
    # kill-pos calv_mask). Stored as Float64 (0/1) in the YelmoMirror.
    calv_mask = zeros(Nx, Ny)
    calv_mask[Nx, :] .= 1.0
    _assign_field!(ymirror.bnd.calv_mask, calv_mask)

    # Sync Julia → Fortran so `init_state!` sees the override.
    yelmo_sync!(ymirror)

    return ymirror
end

# ----------------------------------------------------------------------
# write_fixture! YelmoMirror branch (t > 0). Builds a `BenchmarkSpec`,
# drives YelmoMirror to `t`, writes the restart, moves into place.
# Mirrors the trough.jl `write_fixture!` body verbatim.
# ----------------------------------------------------------------------

function _write_mismip3d_mirror_fixture!(b::MISMIP3DBenchmark,
                                         path::AbstractString,
                                         t_out::Float64)
    isfile(b.namelist_path) || error(
        "MISMIP3DBenchmark.write_fixture!: namelist not found at " *
        "$(b.namelist_path).")

    spec = BenchmarkSpec(
        name           = _spec_name(b),
        namelist_path  = b.namelist_path,
        grid           = (xc = b.xc, yc = b.yc,
                          grid_name = "MISMIP3D"),
        time_init      = 0.0,
        end_time       = t_out,
        output_times   = [t_out],
        dt             = 1.0,
        setup_initial_state! = (ymirror, t) ->
            _setup_mismip3d_initial_state!(ymirror, b, t),
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

# ----------------------------------------------------------------------
# Multi-phase ATT-ramp fixture writer.
#
# Drives YelmoMirror through a sequence of phases, each with its own
# constant `rf_const` (Glen rate factor). The Fortran-side mechanism
# requires a one-shot namelist patch (`rf_method = -1`, "external rate
# factor") so Fortran's `calc_ymat` (yelmo_material.f90:201-211)
# does not overwrite `mat%now%ATT` from the scalar `mat%par%rf_const`
# on every step. With `rf_method = -1`, Fortran trusts whatever's in
# the `mat.ATT` field; we push the new uniform value through the C-API
# at each phase boundary.
#
# Outputs ONE NetCDF restart at `times_phase_end[end]`. Filename is the
# user-supplied `path`; intermediate phase boundaries are not snapshot.
# ----------------------------------------------------------------------

function _write_mismip3d_mirror_fixture_att!(b::MISMIP3DBenchmark,
                                             path::AbstractString,
                                             times_phase_end::AbstractVector{<:Real},
                                             rf_phases::AbstractVector{<:Real})
    @assert length(times_phase_end) == length(rf_phases) (
        "phase-count mismatch: times=$(length(times_phase_end)) " *
        "rf=$(length(rf_phases))")
    @assert !isempty(times_phase_end) "need at least one phase"
    @assert all(times_phase_end .> 0) "phase end times must be positive"
    @assert issorted(times_phase_end) "phase end times must be sorted"
    isfile(b.namelist_path) || error(
        "_write_mismip3d_mirror_fixture_att!: namelist not found at " *
        "$(b.namelist_path).")

    # Patch the source namelist: switch `rf_method` to -1 (external).
    src = read(b.namelist_path, String)
    patched = replace(src,
                      r"rf_method\s*=\s*-?\d+" => "rf_method            = -1")
    if !occursin("rf_method            = -1", patched)
        error("_write_mismip3d_mirror_fixture_att!: failed to patch " *
              "rf_method in $(b.namelist_path) — check the namelist " *
              "still has an `rf_method = N` line.")
    end
    tmpdir = mktempdir(; prefix = "mismip3d_att_nml_")
    patched_nml = joinpath(tmpdir, "yelmo_MISMIP3D_att.nml")
    write(patched_nml, patched)

    # Build a BenchmarkSpec scaffold for the helpers. We bypass
    # `run_mirror_benchmark!` because we need to inject ATT mutation
    # between phases. The spec is reused only for paths / naming /
    # rundir setup.
    spec = BenchmarkSpec(
        name           = _spec_name(b) * "_att",
        namelist_path  = patched_nml,
        grid           = (xc = b.xc, yc = b.yc, grid_name = "MISMIP3D"),
        time_init      = 0.0,
        end_time       = Float64(times_phase_end[end]),
        output_times   = [Float64(times_phase_end[end])],
        dt             = 1.0,
        setup_initial_state! = (ymirror, t) ->
            _setup_mismip3d_initial_state!(ymirror, b, t),
    )

    fixtures_dir = dirname(path)
    mkpath(fixtures_dir)
    isfile(path) && rm(path)

    p = Yelmo.YelmoPar.read_nml(spec.namelist_path)
    rundir = mktempdir(; prefix = "bench_$(spec.name)_")
    yelmo_input = abspath(joinpath(dirname(Yelmo.YelmoMirrorCore.yelmolib),
                                   "..", "..", "input"))
    isdir(yelmo_input) || error(
        "_write_mismip3d_mirror_fixture_att!: yelmo input dir not found " *
        "at $yelmo_input.")
    symlink(yelmo_input, joinpath(rundir, "input"))

    cwd_orig = pwd()
    cd(rundir)
    ymirror = try
        YelmoMirror(p, spec.time_init;
                    grid      = spec.grid,
                    alias     = "ylmo1",
                    rundir    = rundir,
                    overwrite = true)
    catch
        cd(cwd_orig)
        rethrow()
    end

    try
        spec.setup_initial_state!(ymirror, spec.time_init)

        # Pre-fill mat.ATT to Phase-0 rf BEFORE init_state! so the
        # rf_method=-1 path sees a valid value when Fortran's init
        # runs `calc_ymat`.
        fill!(interior(ymirror.mat.ATT), Float64(rf_phases[1]))
        Yelmo.YelmoMirrorCore._set_var!(ymirror.mat.ATT, ymirror.v.mat.ATT,
                                        ymirror.buffers, ymirror.calias)

        init_state!(ymirror, spec.time_init)

        # Re-push baseline ATT after init_state! in case it cleared
        # the field during initialisation.
        fill!(interior(ymirror.mat.ATT), Float64(rf_phases[1]))
        Yelmo.YelmoMirrorCore._set_var!(ymirror.mat.ATT, ymirror.v.mat.ATT,
                                        ymirror.buffers, ymirror.calias)

        t = spec.time_init
        for (phase_idx, t_target) in enumerate(times_phase_end)
            rf_val = Float64(rf_phases[phase_idx])

            # Set rate factor for this phase. (Phase 1 was already
            # pushed above; this is a no-op for phase 1 and the actual
            # mutation for phases ≥ 2.)
            fill!(interior(ymirror.mat.ATT), rf_val)
            Yelmo.YelmoMirrorCore._set_var!(ymirror.mat.ATT,
                                            ymirror.v.mat.ATT,
                                            ymirror.buffers,
                                            ymirror.calias)

            while t < t_target - 1e-9
                dt_step = min(spec.dt, t_target - t)
                step!(ymirror, dt_step)
                t = ymirror.time
            end
        end

        yelmo_write_restart!(ymirror, path; time = t)
    finally
        cd(cwd_orig)
    end

    return [path]
end
