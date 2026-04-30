# ----------------------------------------------------------------------
# Bueler analytical ice-flow solutions, ported from Yelmo Fortran
# (`yelmo/tests/ice_benchmarks.f90`).
#
#   - `bueler_gamma`   : SIA prefactor `γ = 2 A (ρ_i g)^n / (n + 2)`.
#   - `bueler_test_BC!`: Halfar (1981) similarity solution for the
#     time-dependent isothermal-SIA dome (Bueler et al. 2005, Eqs.
#     10–11). Sets `H_ice` and `mbal` in-place at every (xc, yc, time)
#     point. `lambda = 0` reproduces pure decay (BUELER-B); `lambda > 0`
#     gives the BUELER-C variant with an analytical mass balance.
#
# Units: `xc`/`yc` in metres, `H` in metres, `mbal` in m/yr,
# `time` in years, `R0` in km, `H0` in metres, `A` in Pa^-3 yr^-1
# (the Yelmo Fortran convention; the Halfar formula's `time` in the
# similarity exponents is in the same units as `A`'s time, hence yr).
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
