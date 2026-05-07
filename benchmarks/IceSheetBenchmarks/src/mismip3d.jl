# ----------------------------------------------------------------------
# MISMIP3DBenchmark — Marine Ice-Sheet Model Intercomparison Project
# Phase 3D, experiment "Stnd" (steady-state buildup phase).
#
# Reference:
#   Pattyn, F., et al. (2013). "Grounding-line migration in plan-view
#   marine ice-sheet models", J. Glaciol. 59(215), 410-422.
#
# Geometry (Stnd):
#   - Domain: x ∈ [0, 800] km (Bounded), y ∈ [-50, +50] km (Periodic).
#   - Bed:    z_bed(x, y) = -100 - x_km   (m, y-invariant; slope -1/1000).
#   - IC:     H_ice = 10 m where z_bed ≥ -500 m, else 0
#             (i.e. ice-bearing region is x_km ≤ 400).
#
# Boundaries (Stnd, all constant in time):
#   - smb_ref = 0.5 m/yr
#   - T_srf   = 273.15 K
#   - Q_geo   = 42 mW/m²
#   - calv_mask(nx, :) = .TRUE. (only the eastern column allows calving).
#     Approximated here via `ice_allowed[Nx, :] = 0`.
#
# This is the model-agnostic spec. The literal Fortran 10 m all-floating
# IC is preserved here for fidelity. A consumer (e.g. Yelmo.jl run.jl)
# may swap in the thicker grounded variant `H_ice = max(0, 1000 - 0.9·z_bed)`
# documented in `mismip3D.f90:62-64` to keep its SSA solver well-posed
# from step 1 — that's a model concern, not a benchmark concern.
# ----------------------------------------------------------------------

"""
    MISMIP3DBenchmark(variant::Symbol = :Stnd;
                       dx_km=16.0, xmax_km=800.0, Ly_km=100.0,
                       H0=10.0, z_bed_floor=-500.0,
                       bed_intercept=-100.0, bed_slope=1.0,
                       smb_const=0.5, T_srf_const=273.15,
                       Q_geo_const=42.0,
                       n_glen=3.0, A_glen=3.1536e-18,
                       cf_ref=3.165176e4, N_eff_const=1.0)

MISMIP3D Stnd benchmark spec. Carries domain axes, bed geometry, IC
slab thickness, surface forcing, and Glen-flow / friction parameters.

`dx_km = 16.0` gives Nx = 51, Ny = 7 — odd Ny yields a centerline cell
at j = 4 for stronger y-symmetry tests. `dx_km = 20.0` (Fortran default)
gives Nx = 41, Ny = 6 (even Ny, no centerline).

Only `variant = :Stnd` is supported.
"""
struct MISMIP3DBenchmark <: AbstractBenchmark
    variant::Symbol
    xc::Vector{Float64}      # cell-centre x [m]
    yc::Vector{Float64}      # cell-centre y [m]
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
    n_glen::Float64
    A_glen::Float64
    cf_ref::Float64
    N_eff_const::Float64
end

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
                            n_glen::Real        = 3.0,
                            A_glen::Real        = 3.1536e-18,
                            cf_ref::Real        = 3.165176e4,
                            N_eff_const::Real   = 1.0)
    variant === :Stnd || error(
        "MISMIP3DBenchmark: only variant :Stnd is supported (got $(variant)).")

    dx_km_f   = Float64(dx_km)
    xmax_km_f = Float64(xmax_km)
    Ly_km_f   = Float64(Ly_km)

    # Nx / Ny follow the Fortran node-count convention `int(extent/dx)+1`.
    # With dx_km=16: Nx = 51, Ny = 7 (odd → centerline cell at j=4).
    Nx = Int(floor(xmax_km_f / dx_km_f)) + 1
    Ny = Int(floor(Ly_km_f / dx_km_f)) + 1

    dx_m = dx_km_f * 1e3
    xc = collect(range(0.5 * dx_m, (Nx - 0.5) * dx_m; length=Nx))
    yc = collect(range(-0.5 * Ny * dx_m + 0.5 * dx_m,
                        0.5 * Ny * dx_m - 0.5 * dx_m; length=Ny))

    return MISMIP3DBenchmark(variant, xc, yc,
                              xmax_km_f, Ly_km_f, dx_km_f,
                              Float64(H0),
                              Float64(z_bed_floor),
                              Float64(bed_intercept),
                              Float64(bed_slope),
                              Float64(smb_const),
                              Float64(T_srf_const),
                              Float64(Q_geo_const),
                              Float64(n_glen),
                              Float64(A_glen),
                              Float64(cf_ref),
                              Float64(N_eff_const))
end

"""
    state(b::MISMIP3DBenchmark, t) -> NamedTuple

Analytical Stnd IC at `t = 0`. Returns a NamedTuple keyed by
`:xc, :yc, :H_ice, :z_bed, :z_sl, :smb_ref, :T_srf, :Q_geo,
:bmb_shlf, :T_shlf, :H_sed, :ice_allowed`.

Only `t = 0` is supported here; non-zero times require a forward
simulation and are out of scope for this spec package.
"""
function state(b::MISMIP3DBenchmark, t::Real)
    Float64(t) == 0.0 || error(
        "MISMIP3DBenchmark.state: only t = 0 is supported " *
        "(got t = $t). Run a forward simulation to obtain non-zero times.")
    return _mismip3d_analytical_state(b)
end

function _mismip3d_analytical_state(b::MISMIP3DBenchmark)
    Nx, Ny = length(b.xc), length(b.yc)

    # z_bed = bed_intercept - bed_slope·x_km (y-invariant).
    z_bed = [b.bed_intercept - b.bed_slope * (b.xc[i] / 1e3) for i in 1:Nx, j in 1:Ny]

    # H_ice = H0 where z_bed ≥ z_bed_floor, else 0 (literal Fortran IC).
    H_ice = [z_bed[i, j] >= b.z_bed_floor ? b.H0 : 0.0 for i in 1:Nx, j in 1:Ny]

    ice_allowed = ones(Float64, Nx, Ny)
    ice_allowed[Nx, :] .= 0.0    # eastern column = calving boundary

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

const _MISMIP3D_DEFAULT_ZETA_AC      = collect(range(0.0, 1.0; length=11))
const _MISMIP3D_DEFAULT_ZETA_ROCK_AC = collect(range(0.0, 1.0; length=5))

"""
    write_fixture!(b::MISMIP3DBenchmark, path; times=[0.0]) -> Vector{String}

Write the analytical Stnd IC at `t = 0` to `path` as a NetCDF restart.
Only `t = 0` is supported.
"""
function write_fixture!(b::MISMIP3DBenchmark, path::AbstractString;
                        times::AbstractVector{<:Real} = [0.0])
    length(times) == 1 ||
        error("write_fixture!(MISMIP3DBenchmark, …): pass `times` " *
              "with exactly one entry per call (got $(length(times))).")
    t = Float64(first(times))
    t == 0.0 ||
        error("MISMIP3DBenchmark.write_fixture!: only t = 0 is supported " *
              "(got t = $t).")

    s = _mismip3d_analytical_state(b)
    mkpath(dirname(path))
    isfile(path) && rm(path)

    Nx = length(b.xc); Ny = length(b.yc)
    zeta_ac      = _MISMIP3D_DEFAULT_ZETA_AC
    zeta_rock_ac = _MISMIP3D_DEFAULT_ZETA_ROCK_AC
    Nz_ac        = length(zeta_ac)
    Nz_rock_ac   = length(zeta_rock_ac)

    NCDataset(path, "c") do ds
        defDim(ds, "xc",          Nx)
        defDim(ds, "yc",          Ny)
        defDim(ds, "zeta",        Nz_ac - 1)
        defDim(ds, "zeta_ac",     Nz_ac)
        defDim(ds, "zeta_rock",   Nz_rock_ac - 1)
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
            ("H_ice",       s.H_ice,       "m",        "Ice thickness (Stnd 10 m slab)"),
            ("z_bed",       s.z_bed,       "m",        "Bedrock elevation (-100 - x_km)"),
            ("z_sl",        s.z_sl,        "m",        "Sea level"),
            ("smb_ref",     s.smb_ref,     "m/yr",     "Surface mass balance (Stnd: 0.5 m/yr)"),
            ("T_srf",       s.T_srf,       "K",        "Surface temperature"),
            ("Q_geo",       s.Q_geo,       "mW m^-2",  "Geothermal flux"),
            ("bmb_shlf",    s.bmb_shlf,    "m/yr",     "Shelf bmb (zero)"),
            ("T_shlf",      s.T_shlf,      "K",        "Shelf base temperature"),
            ("H_sed",       s.H_sed,       "m",        "Sediment thickness (zero)"),
            ("ice_allowed", s.ice_allowed, "1",        "Ice-allowed mask (eastern column = 0)"),
        )
            v = defVar(ds, name, Float64, ("xc", "yc"))
            v[:, :] = data
            v.attrib["units"]     = units
            v.attrib["long_name"] = longname
        end

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
