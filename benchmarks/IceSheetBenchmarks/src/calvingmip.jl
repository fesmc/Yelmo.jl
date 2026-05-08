# ----------------------------------------------------------------------
# CalvingMIPBenchmark — circular- and Thule-domain benchmarks from the
# CalvingMIP intercomparison (https://github.com/JRowanJordan/CalvingMIP).
#
# References:
#   - CalvingMIP wiki: https://github.com/JRowanJordan/CalvingMIP/wiki
#   - Fortran driver:  yelmo/tests/yelmo_calving.f90
#   - Fortran geometry: yelmo/tests/calving_benchmarks.f90
#   - Calving laws:    yelmo/src/physics/calving/calving_ac.f90
#
# Domain (circular, exp1/2):
#   x, y ∈ [-800, 800] km  (1600 × 1600 km square)
#   Bed:  z_bed(r) = Bc − (Bc − Bl) · r² / R0²
#         R0 = 800 km, Bc = 900 m, Bl = −2000 m  (parabolic bowl)
#   Land cells (above SL) are pinned by the calving driver to lsf=−1.
#   Initial H_ice = 0 everywhere; ice grows from constant SMB.
#
# Boundary forcing (all experiments, constant in time):
#   smb       = 0.3 m/yr
#   T_srf     = 223.15 K (−50 °C)
#   Q_geo     = 42 mW/m²
#   z_sl      = 0 m
#   bmb_shlf  = 0 m/yr
#
# This file is the model-agnostic spec — `state(b, t)` returns a
# NamedTuple of fields and `write_fixture!` writes the same fields to
# NetCDF. Calving-law hooks (which require ice-model field types) live
# in the `YelmoBenchmarks` package extension.
# ----------------------------------------------------------------------

export CalvingMIPBenchmark
export calvmip_bed_circular, calvmip_bed_thule

# -----------------------------------------------------------------------
# Bed geometry (port of calving_benchmarks.f90)
# -----------------------------------------------------------------------

"""
    calvmip_bed_circular(x, y) -> Float64

CalvingMIP circular-domain bed elevation (m) at point (x, y) in metres.
Parabolic bowl: z_bed = Bc − (Bc − Bl) · r² / R0².

Parameters match the CalvingMIP spec (calving_benchmarks.f90:63-87):
  R0 = 800 km, Bc = 900 m, Bl = −2000 m.
"""
function calvmip_bed_circular(x::Real, y::Real)
    R0 = 800e3
    Bc = 900.0
    Bl = -2000.0
    r = sqrt(x^2 + y^2)
    return Bc - (Bc - Bl) * r^2 / R0^2
end

"""
    calvmip_bed_thule(x, y) -> Float64

CalvingMIP Thule-domain bed elevation (m) at point (x, y) in metres.
Parabolic bowl with cosine undulations:
  a(r) = Bc − (Bc − Bl) · r² / R0²
  l(θ) = R0 − cos(2θ) · R0/2
  z_bed = Ba · cos(3π r / l(θ)) + a(r)

Parameters: R0 = 800 km, Bc = 900 m, Bl = −2000 m, Ba = 1100 m.
"""
function calvmip_bed_thule(x::Real, y::Real)
    R0 = 800e3
    Bc = 900.0
    Bl = -2000.0
    Ba = 1100.0
    r = sqrt(x^2 + y^2)
    θ = atan(y, x)
    l = R0 - cos(2θ) * R0 / 2
    a = Bc - (Bc - Bl) * r^2 / R0^2
    return Ba * cos(3π * r / l) + a
end

# -----------------------------------------------------------------------
# CalvingMIPBenchmark struct
# -----------------------------------------------------------------------

"""
    CalvingMIPBenchmark(exp::Symbol = :exp1; dx_km=25.0, ...)

CalvingMIP benchmark on the circular (exp1/2) or Thule (exp3/4/5) domain.

Currently supported:
  - `:exp1` — equilibrium calving pinning the front at r = 750 km (circular).
  - `:exp2` — oscillating calving front, chained from an Exp1 state (circular).
  - `:exp3` — equilibrium calving pinning the front at r = 750 km (Thule).
  - `:exp4` — oscillating calving front, chained from an Exp3 state (Thule).

The calving law itself is supplied by the host model via a hook (e.g.
`YelmoHooks.calv_flt`); this struct only carries the model-agnostic
geometry and forcing.
"""
struct CalvingMIPBenchmark <: AbstractBenchmark
    exp           ::Symbol
    domain        ::Symbol
    xc            ::Vector{Float64}
    yc            ::Vector{Float64}
    dx_km         ::Float64
    smb_const     ::Float64
    T_srf_const   ::Float64
    Q_geo_const   ::Float64
end

function CalvingMIPBenchmark(exp::Symbol = :exp1;
                              dx_km::Real        = 25.0,
                              smb_const::Real    = 0.3,
                              T_srf_const::Real  = 223.15,
                              Q_geo_const::Real  = 42.0)
    exp in (:exp1, :exp2, :exp3, :exp4) || error(
        "CalvingMIPBenchmark: unsupported exp = $exp. Supported: :exp1, :exp2, :exp3, :exp4.")

    domain = exp in (:exp1, :exp2) ? :circular : :thule

    dx_m  = Float64(dx_km) * 1e3
    # Domain: x, y ∈ [−800, 800] km  → 1600 km extent → 64 cells at 25 km.
    extent_m = 1600e3
    N = Int(round(extent_m / dx_m))
    xc = collect(range(-extent_m/2 + dx_m/2,  extent_m/2 - dx_m/2; length=N))
    yc = collect(range(-extent_m/2 + dx_m/2,  extent_m/2 - dx_m/2; length=N))

    return CalvingMIPBenchmark(exp, domain,
                                xc, yc, Float64(dx_km),
                                Float64(smb_const),
                                Float64(T_srf_const),
                                Float64(Q_geo_const))
end

# -----------------------------------------------------------------------
# Analytical t = 0 IC
# -----------------------------------------------------------------------

"""
    state(b::CalvingMIPBenchmark, t) -> NamedTuple

Analytical CalvingMIP IC at `t = 0`: H_ice = 0 everywhere, z_bed from
the circular bowl formula, lsf = +1 everywhere.

Non-zero times require a model-driven trajectory (e.g. via a YelmoMirror
fixture or an Exp1 restart) and are out of scope for this spec package.
"""
function state(b::CalvingMIPBenchmark, t::Real)
    Float64(t) == 0.0 || error(
        "CalvingMIPBenchmark.state: only t = 0 is supported (got t = $t). " *
        "Run a forward simulation or load a restart for non-zero times.")
    return _calvingmip_analytical_state(b)
end

function _calvingmip_analytical_state(b::CalvingMIPBenchmark)
    Nx = length(b.xc); Ny = length(b.yc)

    bed_fn = b.domain == :thule ? calvmip_bed_thule : calvmip_bed_circular
    z_bed  = [bed_fn(b.xc[i], b.yc[j]) for i in 1:Nx, j in 1:Ny]
    H_ice = zeros(Nx, Ny)
    # lsf = +1 (all ocean) — ice will grow from SMB; the calving step's
    # above-SL pin will force lsf = −1 over land each step.
    lsf   = ones(Nx, Ny)

    smb_ref     = fill(b.smb_const,   Nx, Ny)
    T_srf       = fill(b.T_srf_const, Nx, Ny)
    Q_geo       = fill(b.Q_geo_const, Nx, Ny)
    z_sl        = zeros(Nx, Ny)
    bmb_shlf    = zeros(Nx, Ny)
    T_shlf      = fill(b.T_srf_const, Nx, Ny)
    H_sed       = zeros(Nx, Ny)
    ice_allowed = ones(Nx, Ny)
    calv_mask   = zeros(Nx, Ny)

    return (xc = b.xc, yc = b.yc,
            H_ice = H_ice, z_bed = z_bed, z_sl = z_sl,
            smb_ref = smb_ref, T_srf = T_srf, Q_geo = Q_geo,
            bmb_shlf = bmb_shlf, T_shlf = T_shlf, H_sed = H_sed,
            ice_allowed = ice_allowed, calv_mask = calv_mask,
            lsf = lsf)
end

# Default zeta axes for the analytical fixture writer.
const _CALVINGMIP_NZ_AA = 11
function _calvingmip_zeta_ac()
    nz_aa = _CALVINGMIP_NZ_AA
    zeta_aa = collect(range(0.5/nz_aa, 1.0 - 0.5/nz_aa; length=nz_aa))
    zeta_ac = vcat(0.0, 0.5*(zeta_aa[1:end-1].+zeta_aa[2:end]), 1.0)
    return zeta_aa, zeta_ac
end
const _CALVINGMIP_DEFAULT_ZETA_ROCK_AC = collect(range(0.0, 1.0; length=5))

"""
    write_fixture!(b::CalvingMIPBenchmark, path; times=[0.0]) -> Vector{String}

Write the analytical zero-ice IC at `t = 0` to `path` as a NetCDF
restart. Multi-time fixtures (`t > 0`) require a forward simulation;
those live next to the host-model integration (e.g. the YelmoMirror
fixture writer in `test/benchmarks/calvingmip.jl`).
"""
function write_fixture!(b::CalvingMIPBenchmark, path::AbstractString;
                        times::AbstractVector{<:Real} = [0.0])
    length(times) == 1 ||
        error("write_fixture!(CalvingMIPBenchmark, …): pass `times` " *
              "with exactly one entry per call (got $(length(times))).")
    t = Float64(first(times))
    t == 0.0 ||
        error("CalvingMIPBenchmark.write_fixture!: only t = 0 supported " *
              "(got t = $t). Use a model-side fixture writer for t > 0.")

    s = _calvingmip_analytical_state(b)
    bed_longname = b.domain == :thule ? "Bedrock elevation (Thule bowl + undulations)" :
                                        "Bedrock elevation (parabolic bowl)"
    mkpath(dirname(path))
    isfile(path) && rm(path)

    _, zeta_ac = _calvingmip_zeta_ac()
    zeta_rock_ac = _CALVINGMIP_DEFAULT_ZETA_ROCK_AC
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
        zc[:] = 0.5 .* (zeta_ac[1:end-1] .+ zeta_ac[2:end])
        zc.attrib["units"] = "1"
        zac = defVar(ds, "zeta_ac", Float64, ("zeta_ac",))
        zac[:] = zeta_ac; zac.attrib["units"] = "1"
        zrc = defVar(ds, "zeta_rock", Float64, ("zeta_rock",))
        zrc[:] = 0.5 .* (zeta_rock_ac[1:end-1] .+ zeta_rock_ac[2:end])
        zrc.attrib["units"] = "1"
        zrac = defVar(ds, "zeta_rock_ac", Float64, ("zeta_rock_ac",))
        zrac[:] = zeta_rock_ac; zrac.attrib["units"] = "1"

        for (name, data, units, longname) in (
            ("H_ice",       s.H_ice,       "m",       "Ice thickness (zero IC)"),
            ("z_bed",       s.z_bed,       "m",       bed_longname),
            ("z_sl",        s.z_sl,        "m",       "Sea level"),
            ("smb_ref",     s.smb_ref,     "m/yr",    "Surface mass balance (constant)"),
            ("T_srf",       s.T_srf,       "K",       "Surface temperature"),
            ("Q_geo",       s.Q_geo,       "mW m^-2", "Geothermal flux"),
            ("bmb_shlf",    s.bmb_shlf,    "m/yr",    "Shelf bmb (zero)"),
            ("T_shlf",      s.T_shlf,      "K",       "Shelf base temperature"),
            ("H_sed",       s.H_sed,       "m",       "Sediment thickness (zero)"),
            ("ice_allowed", s.ice_allowed, "1",       "Ice-allowed mask"),
            ("calv_mask",   s.calv_mask,   "1",       "Calving mask (zero)"),
            ("lsf",         s.lsf,         "1",       "Level-set function (+1 ocean, −1 ice)"),
        )
            v = defVar(ds, name, Float64, ("xc", "yc"))
            v[:, :] = data
            v.attrib["units"]     = units
            v.attrib["long_name"] = longname
        end

        ds.attrib["benchmark"]      = "CalvingMIP-$(b.exp)"
        ds.attrib["solution_type"]  = "analytical-IC"
        ds.attrib["time_yr"]        = t
        ds.attrib["dx_km"]          = b.dx_km
        ds.attrib["smb_const"]      = b.smb_const
        ds.attrib["T_srf_K"]        = b.T_srf_const
        ds.attrib["Q_geo_mWm2"]     = b.Q_geo_const
    end
    return [path]
end
