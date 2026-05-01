# ----------------------------------------------------------------------
# Bueler analytical ice-flow solutions, ported from Yelmo Fortran
# (`yelmo/tests/ice_benchmarks.f90`), and the corresponding
# `BuelerBenchmark <: AbstractBenchmark` implementation of the
# benchmark interface from `benchmarks.jl`.
#
# Math layer:
#   - `bueler_gamma`    : SIA prefactor `γ = 2 A (ρ_i g)^n / (n + 2)`.
#   - `bueler_test_BC!` : Halfar (1981) similarity solution for the
#       time-dependent isothermal-SIA dome (Bueler et al. 2005,
#       Eqs. 10–11). Sets `H_ice` and `mbal` in-place at every
#       (xc, yc, time) point. `lambda = 0` reproduces pure decay
#       (BUELER-B); `lambda > 0` gives the BUELER-C variant with an
#       analytical mass balance.
#
# Benchmark layer:
#   - `BuelerBenchmark`  : carries grid axes + Halfar parameters,
#       implements `state` / `write_fixture!` / `analytical_velocity`.
#
# Units: `xc`/`yc` in metres, `H` in metres, `mbal` in m/yr,
# `time` in years, `R0` in km, `H0` in metres, `A` in Pa^-3 yr^-1
# (the Yelmo Fortran convention; the Halfar formula's `time` in the
# similarity exponents is in the same units as `A`'s time, hence yr).
# ----------------------------------------------------------------------

export bueler_gamma, bueler_test_BC!
export BuelerBenchmark

# ----------------------------------------------------------------------
# Math layer (ported verbatim from the previous bueler.jl).
# ----------------------------------------------------------------------

"""
    bueler_gamma(A, n, rho_ice, g) -> γ

SIA flow prefactor `γ = 2 A (ρ_i g)^n / (n + 2)`. With Yelmo defaults
(`A = 1e-16` Pa⁻³ yr⁻¹, `n = 3`, `ρ_i = 910`, `g = 9.81`) this gives
`γ ≈ 2.84 × 10⁻²` m⁻³ yr⁻¹ (Bueler et al. 2005 default).
"""
@inline bueler_gamma(A, n, rho_ice, g) = 2.0 * A * (rho_ice * g)^n / (n + 2.0)

"""
    bueler_test_BC!(H_ice, mbal, xc, yc, time;
                    R0=750.0, H0=3600.0, lambda=0.0,
                    n=3.0, A=1e-16, rho_ice=910.0, g=9.81)

Halfar similarity solution for the isothermal-SIA dome. Mutates
`H_ice` (m) and `mbal` (m/yr) in-place at every (xc[i], yc[j]) point.

`H_ice` and `mbal` must be 2D arrays of size `(length(xc), length(yc))`.
`xc` and `yc` are cell-centre coordinates in **metres**.

`time` is the elapsed time **after** the analytical reference
(`t = 0` corresponds to the initial dome at the start of the
simulation, **not** an absolute Halfar `t₀`). The internal `t₀`
shift is added inside the formula via the (R0, H0) dome scale.

`lambda = 0` reproduces BUELER-B (pure Halfar decay, no mass balance).
`lambda > 0` reproduces BUELER-C (with analytical `mbal = λ H / t`).

Port of `yelmo/tests/ice_benchmarks.f90:167 bueler_test_BC`.
"""
function bueler_test_BC!(H_ice::AbstractMatrix{<:Real},
                         mbal::AbstractMatrix{<:Real},
                         xc::AbstractVector{<:Real},
                         yc::AbstractVector{<:Real},
                         time::Real;
                         R0::Real=750.0, H0::Real=3600.0,
                         lambda::Real=0.0, n::Real=3.0,
                         A::Real=1e-16,
                         rho_ice::Real=910.0, g::Real=9.81)
    Nx, Ny = length(xc), length(yc)
    size(H_ice) == (Nx, Ny) ||
        error("bueler_test_BC!: H_ice has shape $(size(H_ice)); expected ($Nx, $Ny).")
    size(mbal) == (Nx, Ny) ||
        error("bueler_test_BC!: mbal has shape $(size(mbal)); expected ($Nx, $Ny).")

    # Convert R0 from km → m to match the Cartesian xc/yc grid.
    R0_m = Float64(R0) * 1e3

    # Halfar similarity exponents.
    α = (2.0 - (n + 1.0) * lambda) / (5.0 * n + 3.0)
    β = (1.0 + (2.0 * n + 1.0) * lambda) / (5.0 * n + 3.0)
    γ = bueler_gamma(A, n, rho_ice, g)
    t0 = (β / γ) * ((2.0 * n + 1.0) / (n + 1.0))^n *
         (R0_m^(n + 1) / H0^(2.0 * n + 1.0))
    t1 = Float64(time) + t0

    @inbounds for j in 1:Ny, i in 1:Nx
        r = sqrt(Float64(xc[i])^2 + Float64(yc[j])^2)
        # Halfar profile, Bueler 2005 Eqs. 10–11. The `max(0, ⋯)`
        # clamps the flank to zero outside the moving margin.
        fac = max(0.0, 1.0 - ((t1 / t0)^(-β) * r / R0_m)^((n + 1.0) / n))
        H_ice[i, j] = H0 * (t1 / t0)^(-α) * fac^(n / (2.0 * n + 1.0))
        mbal[i, j]  = (lambda / t1) * H_ice[i, j]
    end

    return H_ice, mbal
end

# ----------------------------------------------------------------------
# BuelerBenchmark — concrete AbstractBenchmark implementation.
# ----------------------------------------------------------------------

"""
    BuelerBenchmark(variant::Symbol; dx_km, R0_km=750.0, H0=3600.0,
                    lambda=nothing, n=3.0, A=1e-16, rho_ice=910.0, g=9.81)

Halfar dome benchmark (Bueler et al. 2005). `variant`:

  - `:B` — pure Halfar decay (no mass balance, `λ = 0`). The default.
  - `:C` — Halfar + analytical mass balance `mbal = λ H / t`. Requires
           `lambda` to be passed explicitly.

`dx_km` is the cell width in km; the grid is square and centered on
the dome with `Nx = Ny = 2 R0_km / dx_km + 1` points spanning
`[-R0_km e3, +R0_km e3]` in metres.

Other physical constants default to the Yelmo Fortran defaults from
`yelmo_benchmarks.f90:233-238`.
"""
struct BuelerBenchmark <: AbstractBenchmark
    variant::Symbol
    xc::Vector{Float64}
    yc::Vector{Float64}
    R0_km::Float64
    H0::Float64
    lambda::Float64
    n::Float64
    A::Float64
    rho_ice::Float64
    g::Float64
end

function BuelerBenchmark(variant::Symbol;
                         dx_km::Real,
                         R0_km::Real = 750.0,
                         H0::Real    = 3600.0,
                         lambda      = nothing,
                         n::Real     = 3.0,
                         A::Real     = 1e-16,
                         rho_ice::Real = 910.0,
                         g::Real     = 9.81)
    variant in (:B, :C) || error(
        "BuelerBenchmark: variant must be :B or :C, got $(variant).")

    if variant === :B
        lambda === nothing || lambda == 0.0 ||
            error("BuelerBenchmark(:B): variant B has lambda=0 by definition; do not pass lambda kwarg.")
        lam = 0.0
    else  # :C
        lambda === nothing &&
            error("BuelerBenchmark(:C): variant C requires `lambda` keyword (mass-balance scale).")
        lambda > 0.0 ||
            error("BuelerBenchmark(:C): lambda must be > 0; got $(lambda).")
        lam = Float64(lambda)
    end

    Nx = Int(2 * R0_km / dx_km) + 1
    isinteger(2 * R0_km / dx_km) ||
        error("BuelerBenchmark: 2*R0_km/dx_km must be integer (got $(2*R0_km/dx_km)).")
    half_m = R0_km * 1e3
    xc = collect(range(-half_m, half_m; length=Nx))
    yc = copy(xc)

    return BuelerBenchmark(variant, xc, yc, Float64(R0_km), Float64(H0),
                           lam, Float64(n), Float64(A),
                           Float64(rho_ice), Float64(g))
end

"""
    state(b::BuelerBenchmark, t::Real) -> NamedTuple

Analytical Halfar state at time `t`. Returns a NamedTuple with:

  - `xc`, `yc`     : grid axes in metres (echoes `b.xc` / `b.yc`).
  - `H_ice`        : 2D analytical ice thickness from `bueler_test_BC!`.
  - `z_bed`        : 2D bedrock elevation (flat at zero).
  - `smb_ref`      : 2D analytical surface mass balance (zero for
                     `variant = :B`, `λ H / t` for `:C`).

The keys other than `xc` / `yc` map onto Yelmo schema variables and
are routed into the appropriate component group by the generic
`YelmoModel(::AbstractBenchmark, t)` constructor in `benchmarks.jl`.
"""
function state(b::BuelerBenchmark, t::Real)
    Nx = length(b.xc)
    Ny = length(b.yc)
    H   = zeros(Nx, Ny)
    smb = zeros(Nx, Ny)
    bueler_test_BC!(H, smb, b.xc, b.yc, Float64(t);
                    R0      = b.R0_km,
                    H0      = b.H0,
                    lambda  = b.lambda,
                    n       = b.n,
                    A       = b.A,
                    rho_ice = b.rho_ice,
                    g       = b.g)
    z_bed = zeros(Nx, Ny)
    return (xc = b.xc, yc = b.yc,
            H_ice = H, z_bed = z_bed, smb_ref = smb)
end

# Default coordinate axes and provenance attributes to embed in the
# fixture NetCDF. Matches the schema baked into the previously-committed
# `bueler_b_smoke__t1000.nc` so the rename is variable-equivalent.
const _BUELER_DEFAULT_ZETA_AC      = collect(range(0.0, 1.0; length=11))
const _BUELER_DEFAULT_ZETA_ROCK_AC = collect(range(0.0, 1.0; length=5))

"""
    write_fixture!(b::BuelerBenchmark, path::AbstractString;
                   times = [0.0]) -> Vector{String}

Serialize the analytical Halfar state at each `t` in `times` to a
NetCDF restart at `path`. Single-time only in this PR — multi-time
fixtures (a `time` dimension with multiple snapshots) are deferred to
a future milestone.

Returns a 1-element `Vector{String}` containing `path`.
"""
function write_fixture!(b::BuelerBenchmark, path::AbstractString;
                        times::AbstractVector{<:Real} = [0.0])
    length(times) == 1 ||
        error("write_fixture!(BuelerBenchmark, …): multi-time fixtures " *
              "deferred to a future milestone (got $(length(times)) times).")
    t = Float64(first(times))

    s = state(b, t)
    mkpath(dirname(path))
    isfile(path) && rm(path)

    NCDataset(path, "c") do ds
        Nx = length(b.xc)
        Ny = length(b.yc)
        zeta_ac      = _BUELER_DEFAULT_ZETA_AC
        zeta_rock_ac = _BUELER_DEFAULT_ZETA_ROCK_AC
        Nz_ac      = length(zeta_ac)
        Nz_rock_ac = length(zeta_rock_ac)

        # Dimensions.
        defDim(ds, "xc",          Nx)
        defDim(ds, "yc",          Ny)
        defDim(ds, "zeta",        Nz_ac - 1)
        defDim(ds, "zeta_ac",     Nz_ac)
        defDim(ds, "zeta_rock",   Nz_rock_ac - 1)
        defDim(ds, "zeta_rock_ac", Nz_rock_ac)

        # Coordinate variables.
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

        # State variables.
        Hv = defVar(ds, "H_ice", Float64, ("xc", "yc"))
        Hv[:, :] = s.H_ice
        Hv.attrib["units"]     = "m"
        Hv.attrib["long_name"] = "Ice thickness (analytical Halfar)"

        smbv = defVar(ds, "smb_ref", Float64, ("xc", "yc"))
        smbv[:, :] = s.smb_ref
        smbv.attrib["units"]     = "m/yr"
        smbv.attrib["long_name"] = "Surface mass balance (analytical)"

        zb = defVar(ds, "z_bed", Float64, ("xc", "yc"))
        zb[:, :] = s.z_bed
        zb.attrib["units"]     = "m"
        zb.attrib["long_name"] = "Bedrock elevation (flat)"

        # Provenance.
        ds.attrib["benchmark"]     = "BUELER-$(string(b.variant))"
        ds.attrib["solution_type"] = "analytical-halfar"
        ds.attrib["time_yr"]       = t
        ds.attrib["R0_km"]         = b.R0_km
        ds.attrib["H0_m"]          = b.H0
        ds.attrib["lambda"]        = b.lambda
        ds.attrib["n_glen"]        = b.n
        ds.attrib["A_Pa-3yr-1"]    = b.A
    end

    return [path]
end

# ----------------------------------------------------------------------
# Private helpers for the closed-form Halfar depth-averaged velocity.
#
# Derivation: for the BUELER-B (lambda = 0) Halfar dome (Bueler et al.
# 2005, J. Glaciol. 51 (173), Eqs. 9-11; original similarity solution
# Halfar 1981, JGR 86 (C11), pp. 11065-11072):
#
#   alpha = (2 - (n+1) lambda) / (5 n + 3),
#   beta  = (1 + (2 n + 1) lambda) / (5 n + 3),
#   gamma = 2 A (rho_i g)^n / (n + 2),
#   t0    = (beta / gamma) * ((2 n + 1)/(n + 1))^n * R0^(n+1) / H0^(2 n + 1).
#
# Following the Yelmo Fortran convention in `bueler_test_BC` (and
# matching Bueler 2005 Eq. 11 with the time variable shifted to
# `t -> t + t0`), the Halfar profile is parameterised by
#
#   u(r, t) = (t1/t0)^(-beta) * r / R0,    with t1 = t + t0,
#
# and for u < 1
#
#   H(r, t) = H0 * (t1/t0)^(-alpha) * [1 - u^((n+1)/n)]^(n/(2 n + 1)),
#
# giving (by direct r-differentiation, q = n/(2n+1), p = (n+1)/n,
# u_r = (t1/t0)^(-beta)/R0)
#
#   dH/dr(r, t) = -H0 * (t1/t0)^(-alpha) * (t1/t0)^(-beta) / R0 *
#                  (n+1)/(2 n + 1) * u^(1/n) *
#                  [1 - u^((n+1)/n)]^(n/(2 n + 1) - 1).
#
# At u >= 1 the dome margin has been crossed: H = dH/dr = 0.
#
# The depth-averaged SIA velocity for a flat-bed Halfar dome is
#
#   bar(u_r) = -gamma * H^(n+1) * |dH/dr|^(n-1) * dH/dr   (radial),
#
# converted to Cartesian as bar(u_x) = (x/r) * bar(u_r), bar(u_y) =
# (y/r) * bar(u_r). At r = 0 the symmetry forces bar(u) = 0.
# ----------------------------------------------------------------------

@inline function _halfar_exponents(n::Real, lambda::Real)
    α = (2.0 - (Float64(n) + 1.0) * Float64(lambda)) / (5.0 * Float64(n) + 3.0)
    β = (1.0 + (2.0 * Float64(n) + 1.0) * Float64(lambda)) / (5.0 * Float64(n) + 3.0)
    return α, β
end

@inline function _halfar_t0(b::BuelerBenchmark)
    R0_m = b.R0_km * 1e3
    α, β = _halfar_exponents(b.n, b.lambda)
    γ = bueler_gamma(b.A, b.n, b.rho_ice, b.g)
    t0 = (β / γ) * ((2.0 * b.n + 1.0) / (b.n + 1.0))^b.n *
         (R0_m^(b.n + 1) / b.H0^(2.0 * b.n + 1.0))
    return α, β, γ, t0, R0_m
end

# Closed-form (H, dH/dr) at radius `r` (m) and time `t` (yr) for the
# Halfar dome described by `b`. Returns `(0.0, 0.0)` outside the moving
# margin (u >= 1).
function _halfar_HR_dHdr(b::BuelerBenchmark, r::Real, t::Real)
    α, β, γ, t0, R0_m = _halfar_t0(b)
    t1 = Float64(t) + t0
    # Yelmo Fortran convention (matches `bueler_test_BC!` above):
    # `u = (t1/t0)^(-β) * r / R0`. As t increases, (t1/t0)^(-β)
    # decreases (β > 0), so the margin radius r_margin = R0 (t1/t0)^β
    # increases — the Halfar dome spreads with time.
    u  = (t1 / t0)^(-β) * Float64(r) / R0_m
    if u >= 1.0
        return (0.0, 0.0)
    end
    H = b.H0 * (t1 / t0)^(-α) * (1.0 - u^((b.n + 1.0) / b.n))^(b.n / (2.0 * b.n + 1.0))
    prefac = -b.H0 * (t1 / t0)^(-α) * (t1 / t0)^(-β) / R0_m *
             (b.n + 1.0) / (2.0 * b.n + 1.0)
    dHdr = prefac * u^(1.0 / b.n) *
           (1.0 - u^((b.n + 1.0) / b.n))^(b.n / (2.0 * b.n + 1.0) - 1.0)
    return (H, dHdr)
end

# Convenience: closed-form dH/dr only (used by the unit test that
# cross-checks against numerical centred-differences of H).
_halfar_dHdr_closed(b::BuelerBenchmark, r::Real, t::Real) =
    _halfar_HR_dHdr(b, r, t)[2]

"""
    analytical_velocity(b::BuelerBenchmark, t::Real) -> (ux_bar, uy_bar)

Closed-form depth-averaged Halfar velocity at time `t` (years) for the
Halfar dome described by `b`. Returns face-staggered 2D arrays:

  - `ux_bar` of shape `(Nx + 1, Ny)` — values at x-faces, layout
    matching `interior(y.dyn.ux_bar)[:, :, 1]` for an `XFaceField` on a
    `(Nx, Ny)`-Center grid.
  - `uy_bar` of shape `(Nx, Ny + 1)` — values at y-faces, matching
    `interior(y.dyn.uy_bar)[:, :, 1]` for a `YFaceField`.

The first/last x-face row (`ux_bar[1, :]`, `ux_bar[Nx+1, :]`) and the
first/last y-face row (`uy_bar[:, 1]`, `uy_bar[:, Ny+1]`) are left at
zero — these face cells lie outside the dome margin at all benchmark
resolutions and are not part of any margin-masked error metric.

Derivation: see the comment block above this method. References:

  - Halfar, P. 1981. On the dynamics of the ice sheets. JGR 86 (C11),
    pp. 11065-11072.
  - Bueler, E. et al. 2005. Exact solutions and verification of
    numerical models for isothermal ice sheets. J. Glaciol. 51 (173),
    pp. 291-306. Eqs. 9-11 and Section 4.
"""
function analytical_velocity(b::BuelerBenchmark, t::Real)
    Nx, Ny = length(b.xc), length(b.yc)
    ux_bar = zeros(Nx + 1, Ny)
    uy_bar = zeros(Nx, Ny + 1)

    α, β, γ, t0, R0_m = _halfar_t0(b)

    # ux_bar at face position (xc[i] + xc[i+1])/2, yc[j].
    for j in 1:Ny, i in 1:Nx-1
        x_f = 0.5 * (b.xc[i] + b.xc[i+1])
        y_f = b.yc[j]
        r   = sqrt(x_f^2 + y_f^2)
        H, dHdr = _halfar_HR_dHdr(b, r, t)
        if r < 1e-9 || H == 0.0
            ux_bar[i+1, j] = 0.0
        else
            grad_mag = abs(dHdr)
            ux_bar[i+1, j] = -γ * H^(b.n + 1.0) * grad_mag^(b.n - 1.0) *
                             (x_f / r) * dHdr
        end
    end

    # uy_bar at face position xc[i], (yc[j] + yc[j+1])/2.
    for j in 1:Ny-1, i in 1:Nx
        x_f = b.xc[i]
        y_f = 0.5 * (b.yc[j] + b.yc[j+1])
        r   = sqrt(x_f^2 + y_f^2)
        H, dHdr = _halfar_HR_dHdr(b, r, t)
        if r < 1e-9 || H == 0.0
            uy_bar[i, j+1] = 0.0
        else
            grad_mag = abs(dHdr)
            uy_bar[i, j+1] = -γ * H^(b.n + 1.0) * grad_mag^(b.n - 1.0) *
                             (y_f / r) * dHdr
        end
    end

    return (ux_bar, uy_bar)
end

# Resolve the per-spec name used for fixture filenames. Mirrors
# `_spec_name` for `BenchmarkSpec` (the YelmoMirror path) so
# `regenerate.jl` can dispatch on either backend uniformly.
_spec_name(b::BuelerBenchmark) = "bueler_$(lowercase(string(b.variant)))"
