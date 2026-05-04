# ----------------------------------------------------------------------
# EISMINT1MovingBenchmark — EISMINT-1 moving-margin experiment.
#
# Reference:
#   - Huybrechts et al. (1996) "The EISMINT benchmarks for testing
#     ice-sheet models", Ann. Glaciol. 23, 1-12.
#   - Yelmo Fortran reference: `yelmo/tests/ice_benchmarks.f90:323-349`
#     (`eismint_boundaries`, "moving" branch) and
#     `yelmo/par-gmd/yelmo_EISMINT_moving.nml`.
#
# Geometry (constant forcing):
#
#   - Domain: 1500 × 1500 km nominal. Yelmo.jl cell-centred grid uses
#     `Nx = Ny = floor(L/dx) + 1` (mirror of `yelmo_mismip.f90:132`).
#     With `dx_km = 50` (EISMINT-1 default): `Nx = Ny = 31`, extent
#     1550 km, cell summit at i = j = 16 → (775, 775) km. The 50 km
#     overshoot vs the nominal 1500 km is the same cell-centred /
#     node-centred mismatch documented in MISMIP3D.
#   - Bed: flat (`z_bed = 0` everywhere).
#   - IC: zero ice (`H_ice = 0`), ramps up from radial smb forcing.
#
# Forcing (constant in time, `period = 0` in the Fortran spec):
#
#   - `T_srf = 270 - 0.01·H_ice` [K]. Therm-disabled in this benchmark
#     so `T_srf` only matters cosmetically; we set it to 270 K once and
#     leave it (since `H_ice = 0` initially anyway).
#   - `smb = min(smb_max, s·(R_el − dist_km))` [m/yr] with
#     `R_el = 450 km`, `s = 0.01 m/yr/km`, `smb_max = 0.5 m/yr`.
#   - `Q_geo = 42 mW/m²` (irrelevant; therm disabled).
#
# Validation strategy:
#
#   This benchmark exercises the SIA velocity, vertical velocity (uz)
#   from continuity, the explicit advection scheme, and the adaptive
#   predictor-corrector timestepping — all four of the kinematic /
#   integration pieces that landed in milestone 3h. There is no
#   closed-form ice-thickness solution; comparison data lives in the
#   `ice-benchmarks` repo (Huybrechts EISMINT-1 figures) and is plotted
#   externally (see `examples/eismint_moving_compare.jl`).
#
#   The fixture written by `write_fixture!(t = 0)` captures the (trivial)
#   IC; the YelmoMirror reference at `t = 25 kyr` is regenerated
#   separately via `regenerate.jl`.
# ----------------------------------------------------------------------

export EISMINT1MovingBenchmark

"""
    EISMINT1MovingBenchmark(; dx_km=50.0, L_km=1500.0,
                              R_el_km=450.0, smb_max=0.5,
                              smb_grad=0.01, T_srf_const=270.0,
                              Q_geo_const=42.0,
                              n_glen=3.0, A_glen=1e-16,
                              namelist_path=_DEFAULT_EISMINT_MOVING_NAMELIST)

EISMINT-1 moving-margin benchmark. Carries grid axes, smb forcing
parameters, surface temperature, and Glen-flow parameters.

`dx_km = 50.0` is the standard EISMINT-1 grid (Nx = Ny = 31, cell
summit at (775, 775) km). Setup is square; default `L_km = 1500`.

`A_glen = 1e-16 Pa⁻³ yr⁻¹` matches `rf_const` in
`par-gmd/yelmo_EISMINT_moving.nml`.
"""
struct EISMINT1MovingBenchmark <: AbstractBenchmark
    xc::Vector{Float64}
    yc::Vector{Float64}
    L_km::Float64
    dx_km::Float64
    x_summit::Float64
    y_summit::Float64
    R_el_km::Float64
    smb_max::Float64
    smb_grad::Float64        # [m/yr / km]
    T_srf_const::Float64
    Q_geo_const::Float64
    n_glen::Float64
    A_glen::Float64
    namelist_path::String
end

const _DEFAULT_EISMINT_MOVING_NAMELIST = abspath(joinpath(@__DIR__, "specs",
                                                            "yelmo_EISMINT_moving.nml"))

function EISMINT1MovingBenchmark(; dx_km::Real        = 50.0,
                                   L_km::Real         = 1500.0,
                                   R_el_km::Real      = 450.0,
                                   smb_max::Real      = 0.5,
                                   smb_grad::Real     = 0.01,
                                   T_srf_const::Real  = 270.0,
                                   Q_geo_const::Real  = 42.0,
                                   n_glen::Real       = 3.0,
                                   A_glen::Real       = 1e-16,
                                   namelist_path::AbstractString = _DEFAULT_EISMINT_MOVING_NAMELIST)
    dx_f = Float64(dx_km); L_f = Float64(L_km)
    Nx = Int(floor(L_f / dx_f)) + 1   # 31 for dx=50, L=1500
    Ny = Nx                            # square domain

    dx_m = dx_f * 1e3
    xc = collect(range(0.5 * dx_m, (Nx - 0.5) * dx_m; length=Nx))
    yc = collect(range(0.5 * dx_m, (Ny - 0.5) * dx_m; length=Ny))

    # Geometric centre cell (odd Nx → exact centre cell at i=(Nx+1)/2).
    x_summit = 0.5 * (xc[1] + xc[end])
    y_summit = 0.5 * (yc[1] + yc[end])

    return EISMINT1MovingBenchmark(xc, yc,
                                    L_f, dx_f,
                                    x_summit, y_summit,
                                    Float64(R_el_km),
                                    Float64(smb_max),
                                    Float64(smb_grad),
                                    Float64(T_srf_const),
                                    Float64(Q_geo_const),
                                    Float64(n_glen),
                                    Float64(A_glen),
                                    String(namelist_path))
end

"""
    eismint_moving_smb(b::EISMINT1MovingBenchmark, x_m, y_m) -> Float64

Surface mass balance at point (x, y) in metres, per the EISMINT-1
moving-margin spec (`ice_benchmarks.f90:332-345`):

    smb = min(smb_max, smb_grad · (R_el_km − dist_km))

where `dist_km` is the radial distance from the dome centre.
"""
@inline function eismint_moving_smb(b::EISMINT1MovingBenchmark, x_m::Real, y_m::Real)
    dist_km = sqrt((x_m - b.x_summit)^2 + (y_m - b.y_summit)^2) / 1e3
    return min(b.smb_max, b.smb_grad * (b.R_el_km - dist_km))
end

# ----------------------------------------------------------------------
# state(b, t) — analytical IC (t = 0) + fixture-loaded snapshots (t > 0).
# ----------------------------------------------------------------------

function state(b::EISMINT1MovingBenchmark, t::Real)
    t_f = Float64(t)
    if t_f == 0.0
        return _eismint_moving_analytical_state(b)
    end
    path = _eismint_moving_fixture_path(b, t_f)
    isfile(path) || error(
        "EISMINT1MovingBenchmark.state: fixture missing at $path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl " *
        "$(_spec_name(b)) --overwrite` first.")
    NCDataset(path, "r") do ds
        out = Dict{Symbol,Any}(:xc => b.xc, :yc => b.yc)
        for name in ("H_ice", "z_bed", "smb_ref", "T_srf", "Q_geo")
            if haskey(ds, name)
                raw  = ds[name][:, :, :]
                arr2 = ndims(raw) == 3 ? raw[:, :, 1] : raw
                out[Symbol(name)] = Array{Float64}(arr2)
            end
        end
        return NamedTuple(out)
    end
end

function _eismint_moving_analytical_state(b::EISMINT1MovingBenchmark)
    Nx, Ny = length(b.xc), length(b.yc)

    # IC: zero ice, flat terrestrial bed at +1000 m. EISMINT-1 is a
    # purely continental setup — the absolute bed elevation doesn't
    # affect SIA dynamics (only slopes matter), but keeping `z_bed`
    # safely above `z_sl = 0` ensures `f_grnd = 1` from step 1, which
    # `mbal_tendency!` requires before it will apply positive smb to
    # ice-free cells (`mass_balance.jl:106-108`).
    H_ice = zeros(Nx, Ny)
    z_bed = fill(1000.0, Nx, Ny)
    z_sl  = zeros(Nx, Ny)

    # Radial smb pattern (constant in time per the Fortran "moving" branch).
    smb_ref = [eismint_moving_smb(b, b.xc[i], b.yc[j]) for i in 1:Nx, j in 1:Ny]

    # Constant fields (therm-disabled — these don't feed back to dynamics).
    T_srf  = fill(b.T_srf_const, Nx, Ny)
    Q_geo  = fill(b.Q_geo_const, Nx, Ny)
    T_shlf = fill(b.T_srf_const, Nx, Ny)

    bmb_shlf    = zeros(Nx, Ny)
    H_sed       = zeros(Nx, Ny)
    ice_allowed = ones(Nx, Ny)
    calv_mask   = zeros(Nx, Ny)

    return (xc = b.xc, yc = b.yc,
            H_ice = H_ice, z_bed = z_bed, z_sl = z_sl,
            smb_ref = smb_ref, T_srf = T_srf, Q_geo = Q_geo,
            bmb_shlf = bmb_shlf, T_shlf = T_shlf, H_sed = H_sed,
            ice_allowed = ice_allowed, calv_mask = calv_mask)
end

_spec_name(b::EISMINT1MovingBenchmark) = "eismint_moving"

const _EISMINT_MOVING_FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

function _eismint_moving_fixture_path(b::EISMINT1MovingBenchmark, t::Real;
                                       fixtures_dir::AbstractString = _EISMINT_MOVING_FIXTURES_DIR)
    return joinpath(fixtures_dir, "$(_spec_name(b))_t$(Int(round(Float64(t)))).nc")
end

# Default zeta axes for the analytical fixture writer.
#
# Fortran's `par-gmd/yelmo_EISMINT_moving.nml` uses `nz_aa = 31`,
# `zeta_scale = "exp"`, `zeta_exp = 2.0` (layer centres concentrated
# toward the bed). For our analytical IC there is no temperature-
# dependent ATT to resolve, so the dome dynamics are insensitive to
# the vertical-stretching choice. Keep the fixture writer simple with
# a uniform zeta_aa — the standalone trajectory test calls
# `YelmoModel(b, t)` directly and does not round-trip through this
# fixture, so the zeta choice here only affects the on-disk file used
# by the C5b lockstep test (which can override it via the YelmoMirror
# Fortran namelist anyway).
const _EISMINT_MOVING_NZ_AA = 31
function _eismint_moving_zeta_ac()
    nz_aa = _EISMINT_MOVING_NZ_AA
    zeta_aa = collect(range(0.5/nz_aa, 1.0 - 0.5/nz_aa; length=nz_aa))
    zeta_ac = vcat(0.0, 0.5*(zeta_aa[1:end-1].+zeta_aa[2:end]), 1.0)
    return zeta_aa, zeta_ac
end
const _EISMINT_MOVING_DEFAULT_ZETA_ROCK_AC = collect(range(0.0, 1.0; length=5))

"""
    write_fixture!(b::EISMINT1MovingBenchmark, path; times=[0.0]) -> Vector{String}

Write the analytical zero-ice IC at `t = 0` to `path` (NetCDF restart).
Multi-time fixtures (t > 0) require `regenerate.jl`'s YelmoMirror branch.
"""
function write_fixture!(b::EISMINT1MovingBenchmark, path::AbstractString;
                        times::AbstractVector{<:Real} = [0.0])
    length(times) == 1 ||
        error("write_fixture!(EISMINT1MovingBenchmark, …): pass `times` " *
              "with exactly one entry per call (got $(length(times))).")
    t = Float64(first(times))
    if t > 0.0
        return _write_eismint_moving_mirror_fixture!(b, path, t)
    end

    s = _eismint_moving_analytical_state(b)
    mkpath(dirname(path))
    isfile(path) && rm(path)

    _, zeta_ac = _eismint_moving_zeta_ac()
    zeta_rock_ac = _EISMINT_MOVING_DEFAULT_ZETA_ROCK_AC
    Nz_ac      = length(zeta_ac)
    Nz_rock_ac = length(zeta_rock_ac)
    Nx, Ny = length(b.xc), length(b.yc)

    NCDataset(path, "c") do ds
        defDim(ds, "xc", Nx); defDim(ds, "yc", Ny)
        defDim(ds, "zeta", Nz_ac - 1); defDim(ds, "zeta_ac", Nz_ac)
        defDim(ds, "zeta_rock", Nz_rock_ac - 1)
        defDim(ds, "zeta_rock_ac", Nz_rock_ac)

        xv = defVar(ds, "xc", Float64, ("xc",))
        xv[:] = b.xc ./ 1e3; xv.attrib["units"] = "km"
        yv = defVar(ds, "yc", Float64, ("yc",))
        yv[:] = b.yc ./ 1e3; yv.attrib["units"] = "km"
        zc = defVar(ds, "zeta", Float64, ("zeta",))
        zc[:] = 0.5 .* (zeta_ac[1:end-1] .+ zeta_ac[2:end]); zc.attrib["units"] = "1"
        zac = defVar(ds, "zeta_ac", Float64, ("zeta_ac",))
        zac[:] = zeta_ac; zac.attrib["units"] = "1"
        zrc = defVar(ds, "zeta_rock", Float64, ("zeta_rock",))
        zrc[:] = 0.5 .* (zeta_rock_ac[1:end-1] .+ zeta_rock_ac[2:end])
        zrc.attrib["units"] = "1"
        zrac = defVar(ds, "zeta_rock_ac", Float64, ("zeta_rock_ac",))
        zrac[:] = zeta_rock_ac; zrac.attrib["units"] = "1"

        for (name, data, units, longname) in (
            ("H_ice",       s.H_ice,       "m",        "Ice thickness (zero IC)"),
            ("z_bed",       s.z_bed,       "m",        "Bedrock elevation (flat)"),
            ("z_sl",        s.z_sl,        "m",        "Sea level"),
            ("smb_ref",     s.smb_ref,     "m/yr",     "Surface mass balance (radial moving-margin pattern)"),
            ("T_srf",       s.T_srf,       "K",        "Surface temperature"),
            ("Q_geo",       s.Q_geo,       "mW m^-2",  "Geothermal flux"),
            ("bmb_shlf",    s.bmb_shlf,    "m/yr",     "Shelf bmb (zero)"),
            ("T_shlf",      s.T_shlf,      "K",        "Shelf base temperature"),
            ("H_sed",       s.H_sed,       "m",        "Sediment thickness (zero)"),
            ("ice_allowed", s.ice_allowed, "1",        "Ice-allowed mask"),
            ("calv_mask",   s.calv_mask,   "1",        "Calving mask (zero)"),
        )
            v = defVar(ds, name, Float64, ("xc", "yc"))
            v[:, :] = data
            v.attrib["units"]     = units
            v.attrib["long_name"] = longname
        end

        ds.attrib["benchmark"]     = "EISMINT1-moving"
        ds.attrib["solution_type"] = "analytical-IC"
        ds.attrib["time_yr"]       = t
        ds.attrib["L_km"]          = b.L_km
        ds.attrib["dx_km"]         = b.dx_km
        ds.attrib["R_el_km"]       = b.R_el_km
        ds.attrib["smb_max"]       = b.smb_max
        ds.attrib["smb_grad"]      = b.smb_grad
        ds.attrib["T_srf_K"]       = b.T_srf_const
        ds.attrib["Q_geo_mWm2"]    = b.Q_geo_const
        ds.attrib["n_glen"]        = b.n_glen
        ds.attrib["A_glen_Pa-3yr-1"] = b.A_glen
    end
    return [path]
end

# Stub — YelmoMirror path lands in C5b alongside the lockstep test.
function _write_eismint_moving_mirror_fixture!(b::EISMINT1MovingBenchmark,
                                                 path::AbstractString,
                                                 t_out::Float64)
    error("_write_eismint_moving_mirror_fixture!: YelmoMirror branch not yet " *
          "implemented (lands with the lockstep test in C5b).")
end

# ----------------------------------------------------------------------
# YelmoMirror initial-state callback for EISMINT-moving.
# ----------------------------------------------------------------------
function _setup_eismint_moving_initial_state!(ymirror, b::EISMINT1MovingBenchmark, time::Real)
    Nx, Ny = length(b.xc), length(b.yc)

    # Zero IC + flat bed.
    fill!(interior(ymirror.tpo.H_ice),     0.0)
    fill!(interior(ymirror.bnd.z_bed),     0.0)
    fill!(interior(ymirror.bnd.z_sl),      0.0)
    fill!(interior(ymirror.bnd.bmb_shlf),  0.0)
    fill!(interior(ymirror.bnd.H_sed),     0.0)
    fill!(interior(ymirror.bnd.T_shlf),    b.T_srf_const)
    fill!(interior(ymirror.bnd.T_srf),     b.T_srf_const)
    fill!(interior(ymirror.bnd.Q_geo),     b.Q_geo_const)

    # Radial smb pattern.
    smb_int = interior(ymirror.bnd.smb_ref)
    @inbounds for j in 1:Ny, i in 1:Nx
        smb_int[i, j, 1] = eismint_moving_smb(b, b.xc[i], b.yc[j])
    end

    # No calving.
    fill!(interior(ymirror.bnd.calv_mask), 0.0)

    yelmo_sync!(ymirror)
    return ymirror
end
