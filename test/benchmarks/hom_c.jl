# ----------------------------------------------------------------------
# HOMCBenchmark — ISMIP-HOM Experiment C ("ice stream over a sloping bed
# with spatially-varying basal friction").
#
# Reference: Pattyn et al. (2008), "Benchmark experiments for higher-
# order and full-Stokes ice sheet models (ISMIP-HOM)", The Cryosphere
# 2, 95-108. Yelmo Fortran implementation: `yelmo/tests/yelmo_ismiphom.f90`
# (the `EXPC` branch, lines 131-146) and `yelmo/par/yelmo_ISMIPHOM.nml`.
#
# Geometry (Yelmo Fortran convention):
#
#   alpha = 0.1° = 0.1 π / 180 rad
#   omega = 2π / L                 # β-perturbation wavenumber
#
#   z_srf = -x · tan(alpha)
#   z_bed =  z_srf - 1000 m         (uniform 1000 m thick slab over a
#                                    sloping bed)
#   H_ice = 1000 m
#
# Basal friction (Pa·yr·m^-1):
#
#   β(x, y) = β₀ + β_amp · sin(omega · x) · sin(omega · y)
#
# with β₀ = 1000 and β_amp = 1000. Note: this is the **Yelmo Fortran**
# convention (perturbation amplitude 1.0). The published Pattyn 2008
# formula uses amplitude 0.9 → β oscillates over [100, 1900]; the Yelmo
# Fortran version oscillates over [0, 2000]. We mirror the Fortran
# reference (amplitude exposed as a struct field for future flexibility).
#
# Boundary conditions: fully periodic in x and y. The Fortran reference
# pads the domain with `f_extend = 0.5` (half a period on each side) to
# minimise edge effects under clamped BC. Yelmo.jl uses fully-periodic
# BC directly so f_extend is unnecessary.
#
# Validation strategy: HOM-C has no closed-form velocity solution; the
# Pattyn 2008 paper publishes inter-model centre-line velocities for
# the HOM intercomparison. The Yelmo.jl regression target is **180°
# rotational anti-symmetry** of the SSA solution — under fully-periodic
# BC the (x, y) → (L-x, L-y) rotation flips the slope-driven driving
# stress sign while preserving β / H / ATT / forcing, so the velocity
# must satisfy ux(x, y) = -ux(L-x, L-y) and uy(x, y) = -uy(L-x, L-y).
#
# `analytical_velocity` is intentionally NOT implemented (falls through
# to the default error stub).
# ----------------------------------------------------------------------

export HOMCBenchmark

"""
    HOMCBenchmark(variant::Symbol = :C; L_km=80.0, dx_km=L_km*0.025,
                  alpha_deg=0.1, A_glen=1e-16, n_glen=3.0,
                  beta0=1000.0, beta_amp=1000.0,
                  rho_ice=910.0, g=9.81)

ISMIP-HOM Experiment C benchmark struct. Carries domain axes, slope,
material parameters, and basal-friction perturbation parameters.

Only `variant = :C` is supported in this milestone.

`dx_km` defaults to `0.25 · (L_km / 10)` (the Yelmo Fortran convention,
`yelmo_ismiphom.f90:60`). For the canonical L = 80 km case this gives
`dx_km = 2.0` and Nx = Ny = 40.

The benchmark deliberately uses an unpadded `[0, L_km] × [0, L_km]`
domain with fully-periodic BC; the Fortran `f_extend = 0.5` padding is
unnecessary under periodic BC.
"""
struct HOMCBenchmark <: AbstractBenchmark
    variant::Symbol
    xc::Vector{Float64}
    yc::Vector{Float64}
    L_km::Float64
    dx_km::Float64
    H::Float64
    alpha_rad::Float64
    A_glen::Float64
    n_glen::Float64
    beta0::Float64
    beta_amp::Float64
    rho_ice::Float64
    g::Float64
end

function HOMCBenchmark(variant::Symbol = :C;
                       L_km::Real      = 80.0,
                       dx_km::Real     = 0.25 * (Float64(L_km) / 10.0),
                       alpha_deg::Real = 0.1,
                       H::Real         = 1000.0,
                       A_glen::Real    = 1e-16,
                       n_glen::Real    = 3.0,
                       beta0::Real     = 1000.0,
                       beta_amp::Real  = 1000.0,
                       rho_ice::Real   = 910.0,
                       g::Real         = 9.81)
    variant === :C || error(
        "HOMCBenchmark: only variant :C is supported (got $(variant)).")

    L_km_f  = Float64(L_km)
    dx_km_f = Float64(dx_km)
    Nx_f = L_km_f / dx_km_f
    isinteger(Nx_f) || error(
        "HOMCBenchmark: L_km/dx_km must be integer (got $Nx_f).")
    Nx = Int(Nx_f)

    # Cell-centre axes on [0, L] × [0, L]. Under fully-periodic BC the
    # cell at i = 1 has centre dx/2 and the cell at i = Nx has centre
    # L - dx/2 (the "missing" cell at L is the periodic copy of cell 1).
    dx_m = dx_km_f * 1e3
    xc = collect(range(0.5 * dx_m, (Nx - 0.5) * dx_m; length=Nx))
    yc = copy(xc)

    return HOMCBenchmark(variant,
                         xc, yc,
                         L_km_f, dx_km_f,
                         Float64(H),
                         Float64(alpha_deg) * π / 180.0,
                         Float64(A_glen),
                         Float64(n_glen),
                         Float64(beta0),
                         Float64(beta_amp),
                         Float64(rho_ice),
                         Float64(g))
end

"""
    state(b::HOMCBenchmark, t::Real) -> NamedTuple

Analytical IC at time `t`. Returns a NamedTuple with:
  - `xc`, `yc`     — grid axes (metres).
  - `H_ice`        — 1000 m uniform.
  - `z_bed`        — `-x · tan α - H` (linear in x).
  - `smb_ref`      — zero (HOM-C has no mass-balance forcing).
  - `T_srf`        — 263.15 K (arbitrary; HOM-C is isothermal).
  - `Q_geo`        — 50.0 mW/m² (arbitrary).

The basal-friction perturbation β is **not** returned in `state` — it
lives on `dyn.beta` / `dyn.beta_acx` / `dyn.beta_acy` (which are not
part of the schema-routed Center-aligned state). Use
`_setup_hom_c_beta!` after constructing the YelmoModel to fill the
β fields from the analytical formula.
"""
function state(b::HOMCBenchmark, t::Real)
    Nx, Ny = length(b.xc), length(b.yc)
    H_ice = fill(b.H, Nx, Ny)
    z_srf = [-b.xc[i] * tan(b.alpha_rad) for i in 1:Nx, j in 1:Ny]
    z_bed = z_srf .- b.H
    smb   = zeros(Nx, Ny)
    Tsrf  = fill(263.15, Nx, Ny)
    Qgeo  = fill(50.0,   Nx, Ny)
    return (xc = b.xc, yc = b.yc,
            H_ice = H_ice, z_bed = z_bed,
            smb_ref = smb, T_srf = Tsrf, Q_geo = Qgeo)
end

# Spec name used by `regenerate.jl` and the fixture filename.
# `ismiphom_c_l<L_km>` matches the trough naming convention.
_spec_name(b::HOMCBenchmark) =
    "ismiphom_c_l$(round(Int, b.L_km))"

# Default zeta axes for HOM-C analytical fixture. Match the BUELER
# default (uniform 11-point ice / 5-point rock) so the file-based
# constructor uses identical layer geometry.
const _HOMC_DEFAULT_ZETA_AC      = collect(range(0.0, 1.0; length=11))
const _HOMC_DEFAULT_ZETA_ROCK_AC = collect(range(0.0, 1.0; length=5))

"""
    write_fixture!(b::HOMCBenchmark, path::AbstractString;
                   times = [0.0]) -> Vector{String}

Serialise the analytical HOM-C IC at time `t = first(times)` to a
NetCDF restart at `path`. Single-time only (HOM-C has no time
evolution at the IC level — β is steady, geometry is steady).

Returns a 1-element `Vector{String}` containing `path`.
"""
function write_fixture!(b::HOMCBenchmark, path::AbstractString;
                        times::AbstractVector{<:Real} = [0.0])
    length(times) == 1 ||
        error("write_fixture!(HOMCBenchmark, …): multi-time fixtures " *
              "not supported (got $(length(times)) times).")
    t = Float64(first(times))

    s = state(b, t)
    mkpath(dirname(path))
    isfile(path) && rm(path)

    NCDataset(path, "c") do ds
        Nx = length(b.xc)
        Ny = length(b.yc)
        zeta_ac      = _HOMC_DEFAULT_ZETA_AC
        zeta_rock_ac = _HOMC_DEFAULT_ZETA_ROCK_AC
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
        Hv.attrib["long_name"] = "Ice thickness (HOM-C uniform slab)"

        zb = defVar(ds, "z_bed", Float64, ("xc", "yc"))
        zb[:, :] = s.z_bed
        zb.attrib["units"]     = "m"
        zb.attrib["long_name"] = "Bedrock elevation (sloping bed, alpha=0.1°)"

        smbv = defVar(ds, "smb_ref", Float64, ("xc", "yc"))
        smbv[:, :] = s.smb_ref
        smbv.attrib["units"]     = "m/yr"
        smbv.attrib["long_name"] = "Surface mass balance (zero for HOM-C)"

        Tv = defVar(ds, "T_srf", Float64, ("xc", "yc"))
        Tv[:, :] = s.T_srf
        Tv.attrib["units"]     = "K"
        Tv.attrib["long_name"] = "Surface temperature (HOM-C is isothermal)"

        Qv = defVar(ds, "Q_geo", Float64, ("xc", "yc"))
        Qv[:, :] = s.Q_geo
        Qv.attrib["units"]     = "mW m^-2"
        Qv.attrib["long_name"] = "Geothermal flux (arbitrary; isothermal test)"

        ds.attrib["benchmark"]      = "ISMIPHOM-$(string(b.variant))"
        ds.attrib["solution_type"]  = "analytical-IC"
        ds.attrib["time_yr"]        = t
        ds.attrib["L_km"]           = b.L_km
        ds.attrib["dx_km"]          = b.dx_km
        ds.attrib["alpha_deg"]      = b.alpha_rad * 180.0 / π
        ds.attrib["H_m"]            = b.H
        ds.attrib["A_glen_Pa-3yr-1"] = b.A_glen
        ds.attrib["n_glen"]         = b.n_glen
        ds.attrib["beta0"]          = b.beta0
        ds.attrib["beta_amp"]       = b.beta_amp
    end

    return [path]
end

# ----------------------------------------------------------------------
# β perturbation helpers — direct fill of dyn.beta / dyn.beta_acx /
# dyn.beta_acy on a constructed YelmoModel.
#
# `beta_method = -1` (`src/dyn/basal_dragging.jl:601`) is a no-op and
# `beta_gl_stag = -1` (`basal_dragging.jl:987`) bypasses the standard
# acx/acy staggering AND the GL block — so pre-filled β fields survive
# every Picard iteration.
# ----------------------------------------------------------------------

# Closed-form β at an (x_m, y_m) point (metres):
#   β = β₀ + β_amp · sin(2π x / L) · sin(2π y / L)
@inline function _hom_c_beta(b::HOMCBenchmark, x_m::Real, y_m::Real)
    omega = 2π / (b.L_km * 1e3)
    return b.beta0 + b.beta_amp * sin(omega * Float64(x_m)) * sin(omega * Float64(y_m))
end

"""
    _setup_hom_c_beta!(y, b::HOMCBenchmark) -> y

Fill `y.dyn.beta`, `y.dyn.beta_acx`, and `y.dyn.beta_acy` with the
HOM-C analytical β perturbation. Call after `YelmoModel(b, t;
boundaries=:periodic)` and before `dyn_step!` for tests that use
`beta_method = -1` (external β).

Stagger conventions (matching `_assemble_ssa_matrix!`'s reads):

  - `dyn.beta`     — Center, slot `[i, j, 1]` at cell-centre `(xc[i], yc[j])`.
  - `dyn.beta_acx` — XFace under Periodic-x (interior shape `(Nx, Ny, 1)`).
                     Slot `[ip1f, j, 1]` (where `ip1f = mod1(i+1, Nx)`)
                     holds the face-east of cell `(i, j)` at position
                     `0.5·(xc[i] + xc[i+1])` (with periodic wrap).
  - `dyn.beta_acy` — YFace under Periodic-y, analogous.
"""
function _setup_hom_c_beta!(y, b::HOMCBenchmark)
    Nx, Ny = length(b.xc), length(b.yc)
    L_m    = b.L_km * 1e3
    dx_m   = b.dx_km * 1e3

    # aa-cell β at xc[i], yc[j].
    Bi = interior(y.dyn.beta)
    @inbounds for j in 1:Ny, i in 1:Nx
        Bi[i, j, 1] = _hom_c_beta(b, b.xc[i], b.yc[j])
    end

    # acx-face β at face-east of cell (i, j). Under fully-periodic the
    # face position is xc[i] + dx/2 (cell-centre + half spacing) — for
    # cell i = Nx this is L - dx/2 + dx/2 = L, equivalent to 0 by
    # periodicity. The slot under Periodic-x is `mod1(i+1, Nx)` (so
    # cell i = Nx writes to slot 1, the wrapped neighbour).
    Bx = interior(y.dyn.beta_acx)
    @inbounds for j in 1:Ny, i in 1:Nx
        x_face = b.xc[i] + 0.5 * dx_m
        # No need to mod x_face since sin is L-periodic.
        ip1f = mod1(i + 1, Nx)
        Bx[ip1f, j, 1] = _hom_c_beta(b, x_face, b.yc[j])
    end

    # acy-face β at face-north of cell (i, j). Analogous to acx.
    By = interior(y.dyn.beta_acy)
    @inbounds for j in 1:Ny, i in 1:Nx
        y_face = b.yc[j] + 0.5 * dx_m
        jp1f = mod1(j + 1, Ny)
        By[i, jp1f, 1] = _hom_c_beta(b, b.xc[i], y_face)
    end

    return y
end
