# ----------------------------------------------------------------------
# Grounded-state helpers for the `tpo` step.
#
#   - `calc_H_grnd!`                 — flotation diagnostic [m]
#   - `determine_grounded_fractions!` — CISM bilinear-interpolation
#     subgrid-grounded-fraction kernel (Leguy et al. 2021), ported
#     from `physics/topography.f90:1896` (and helpers below it).
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calc_H_grnd!, determine_grounded_fractions!

"""
    calc_H_grnd!(H_grnd, H_ice, z_bed, z_sl, rho_ice, rho_sw) -> H_grnd

Flotation diagnostic: `H_grnd = H_ice - max(z_sl - z_bed, 0) * rho_sw / rho_ice`.
Positive ⇒ grounded (anchored to bed); negative ⇒ floating.

Lightweight helper; no neighbour stencils. Mirrors the scattered
formula used throughout `yelmo_topography.f90`.
"""
function calc_H_grnd!(H_grnd, H_ice, z_bed, z_sl,
                      rho_ice::Real, rho_sw::Real)
    Hg  = interior(H_grnd)
    H   = interior(H_ice)
    Zb  = interior(z_bed)
    Zsl = interior(z_sl)
    rho_ratio = rho_sw / rho_ice
    @inbounds for j in axes(Hg, 2), i in axes(Hg, 1)
        depth   = Zsl[i, j, 1] - Zb[i, j, 1]
        H_float = depth > 0.0 ? depth * rho_ratio : 0.0
        Hg[i, j, 1] = H[i, j, 1] - H_float
    end
    return H_grnd
end

# Out-of-domain reads of the flotation field resolve to 0 (open ocean —
# no ice, bed at sea level by construction). Matches the Dirichlet-zero
# halo used elsewhere in the module.
@inline function _f_or_zero(F::AbstractMatrix, i::Int, j::Int, nx::Int, ny::Int)
    return (1 <= i <= nx && 1 <= j <= ny) ? F[i, j] : 0.0
end

"""
    determine_grounded_fractions!(f_grnd, H_grnd;
                                  f_grnd_acx=nothing,
                                  f_grnd_acy=nothing,
                                  f_grnd_ab=nothing) -> f_grnd

Compute the grounded fraction at aa-nodes (cell centres) and optionally
at acx-, acy-, and ab-nodes from the flotation diagnostic `H_grnd`.

Algorithm (Leguy et al. 2021, ported via IMAU-ICE v2.0):

  1. Define `f_flt = -H_grnd` (positive ⇒ floating, negative ⇒ grounded).
  2. For each cell, compute four quadrant-grounded-fractions
     (`f_NW`, `f_NE`, `f_SW`, `f_SE`) using `_cism_quads!`. Each
     quadrant is itself a unit square with corner values drawn from a
     9-point corner stencil of `f_flt`.
  3. The aa-fraction is the mean of the four quadrants. The optional
     acx-/acy-/ab-fractions are quadrant means at the corresponding
     face-/corner-staggered positions.

Out-of-domain neighbour reads of `f_flt` resolve to 0.

Port of `physics/topography.f90:determine_grounded_fractions`.
"""
function determine_grounded_fractions!(f_grnd, H_grnd;
                                       f_grnd_acx = nothing,
                                       f_grnd_acy = nothing,
                                       f_grnd_ab  = nothing)
    Hg = interior(H_grnd)
    nx = size(Hg, 1)
    ny = size(Hg, 2)

    f_flt = Matrix{Float64}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        f_flt[i, j] = -Hg[i, j, 1]
    end

    f_NW = Matrix{Float64}(undef, nx, ny)
    f_NE = Matrix{Float64}(undef, nx, ny)
    f_SW = Matrix{Float64}(undef, nx, ny)
    f_SE = Matrix{Float64}(undef, nx, ny)
    _cism_quads!(f_NW, f_NE, f_SW, f_SE, f_flt, nx, ny)

    Fa = interior(f_grnd)
    @inbounds for j in 1:ny, i in 1:nx
        Fa[i, j, 1] = 0.25 * (f_NW[i, j] + f_NE[i, j] +
                              f_SW[i, j] + f_SE[i, j])
    end

    if f_grnd_acx !== nothing
        Facx = interior(f_grnd_acx)
        @inbounds for j in 1:ny, i in 1:nx
            ip1 = min(i + 1, nx)
            Facx[i, j, 1] = 0.25 * (f_NE[i,   j] + f_SE[i,   j] +
                                    f_NW[ip1, j] + f_SW[ip1, j])
        end
    end
    if f_grnd_acy !== nothing
        Facy = interior(f_grnd_acy)
        @inbounds for j in 1:ny, i in 1:nx
            jp1 = min(j + 1, ny)
            Facy[i, j, 1] = 0.25 * (f_NE[i, j  ] + f_NW[i, j  ] +
                                    f_SE[i, jp1] + f_SW[i, jp1])
        end
    end
    if f_grnd_ab !== nothing
        Fab = interior(f_grnd_ab)
        @inbounds for j in 1:ny, i in 1:nx
            ip1 = min(i + 1, nx)
            jp1 = min(j + 1, ny)
            Fab[i, j, 1] = 0.25 * (f_NE[i,   j  ] + f_NW[ip1, j  ] +
                                   f_SE[i,   jp1] + f_SW[ip1, jp1])
        end
    end

    return f_grnd
end

# Per-cell quadrant-grounded-fraction kernel.
# Mirrors `determine_grounded_fractions_CISM_quads` in
# `physics/topography.f90:1978`.
function _cism_quads!(f_NW, f_NE, f_SW, f_SE, f_flt, nx::Int, ny::Int)
    @inbounds for j in 1:ny, i in 1:nx
        f_m  = f_flt[i, j]
        fW   = 0.5 * (_f_or_zero(f_flt, i-1, j,   nx, ny) + f_m)
        fE   = 0.5 * (f_m + _f_or_zero(f_flt, i+1, j, nx, ny))
        fN   = 0.5 * (f_m + _f_or_zero(f_flt, i,   j+1, nx, ny))
        fS   = 0.5 * (f_m + _f_or_zero(f_flt, i,   j-1, nx, ny))
        fNW  = 0.25 * (_f_or_zero(f_flt, i-1, j+1, nx, ny) +
                       _f_or_zero(f_flt, i,   j+1, nx, ny) +
                       _f_or_zero(f_flt, i-1, j,   nx, ny) + f_m)
        fNE  = 0.25 * (_f_or_zero(f_flt, i,   j+1, nx, ny) +
                       _f_or_zero(f_flt, i+1, j+1, nx, ny) +
                       f_m + _f_or_zero(f_flt, i+1, j, nx, ny))
        fSW  = 0.25 * (_f_or_zero(f_flt, i-1, j,   nx, ny) + f_m +
                       _f_or_zero(f_flt, i-1, j-1, nx, ny) +
                       _f_or_zero(f_flt, i,   j-1, nx, ny))
        fSE  = 0.25 * (f_m + _f_or_zero(f_flt, i+1, j, nx, ny) +
                       _f_or_zero(f_flt, i,   j-1, nx, ny) +
                       _f_or_zero(f_flt, i+1, j-1, nx, ny))

        f_NW[i, j] = _calc_fraction_above_zero(fNW,  fN, fW,  f_m)
        f_NE[i, j] = _calc_fraction_above_zero(fN,  fNE, f_m, fE)
        f_SW[i, j] = _calc_fraction_above_zero(fW, f_m,  fSW, fS)
        f_SE[i, j] = _calc_fraction_above_zero(f_m, fE,  fS,  fSE)
    end
    return nothing
end

# Snap-to-tolerance for the fraction-above-zero analytical kernel.
# Matches `ftol` in `calc_fraction_above_zero` (Fortran).
const _FRAC_FTOL = 1e-4

@inline function _snap(f::Real)
    if f == 0.0
        return _FRAC_FTOL
    elseif f > 0.0
        return max(_FRAC_FTOL, f)
    else
        return min(-_FRAC_FTOL, f)
    end
end

# Anticlockwise corner-label rotation:
#   new_NW = old_NE, new_NE = old_SE, new_SE = old_SW, new_SW = old_NW
# (Mirrors Fortran `rotate_quad`.)
@inline function _rotate_quad(fNW, fNE, fSW, fSE)
    return fNE, fSE, fNW, fSW   # (NW, NE, SW, SE)
end

# Rotate a four-corner square anticlockwise (≤ 4 times) until the
# (sign of NW, NE, SW, SE) matches one of the four canonical scenarios:
#   1: SW grounded, rest floating
#   2: SW floating, rest grounded
#   3: south grounded, north floating
#   4: SW & NE grounded, SE & NW floating
# Returns the rotated tuple plus the scenario id.
function _rotate_quad_until_match(fNW0, fNE0, fSW0, fSE0)
    fNW, fNE, fSW, fSE = fNW0, fNE0, fSW0, fSE0
    @inbounds for _ in 1:4
        fNW, fNE, fSW, fSE = _rotate_quad(fNW, fNE, fSW, fSE)
        if     fSW < 0.0 && fSE > 0.0 && fNE > 0.0 && fNW > 0.0
            return fNW, fNE, fSW, fSE, 1
        elseif fSW > 0.0 && fSE < 0.0 && fNE < 0.0 && fNW < 0.0
            return fNW, fNE, fSW, fSE, 2
        elseif fSW < 0.0 && fSE < 0.0 && fNE > 0.0 && fNW > 0.0
            return fNW, fNE, fSW, fSE, 3
        elseif fSW < 0.0 && fSE > 0.0 && fNE < 0.0 && fNW > 0.0
            return fNW, fNE, fSW, fSE, 4
        end
    end
    error("rotate_quad_until_match: no matching scenario for " *
          "(NW=$fNW0, NE=$fNE0, SW=$fSW0, SE=$fSE0)")
end

# Fraction of a unit square where the bilinear interpolant of the four
# corner values is positive. Mirrors `calc_fraction_above_zero` in
# `physics/topography.f90:2059`. By convention here, "positive" means
# "grounded" (since `f_flt = -H_grnd`, grounded points have f_flt < 0
# — see the sign handling in the four scenarios below).
function _calc_fraction_above_zero(f_NW, f_NE, f_SW, f_SE)
    fNW = _snap(f_NW)
    fNE = _snap(f_NE)
    fSW = _snap(f_SW)
    fSE = _snap(f_SE)

    if fNW <= 0.0 && fNE <= 0.0 && fSW <= 0.0 && fSE <= 0.0
        return 1.0
    elseif fNW >= 0.0 && fNE >= 0.0 && fSW >= 0.0 && fSE >= 0.0
        return 0.0
    end

    fNW, fNE, fSW, fSE, scen = _rotate_quad_until_match(fNW, fNE, fSW, fSE)

    aa = fSW
    bb = fSE - fSW
    cc = fNW - fSW
    dd = fNE + fSW - fNW - fSE
    if abs(dd) < _FRAC_FTOL
        fSW = fSW > 0.0 ? fSW + 0.1 : fSW - 0.1
        aa = fSW
        bb = fSE - fSW
        cc = fNW - fSW
        dd = fNE + fSW - fNW - fSE
    end

    phi = 0.0
    if scen == 1
        # SW grounded, rest floating
        phi = ((bb*cc - aa*dd) * log(abs(1.0 - (aa*dd)/(bb*cc))) +
               aa*dd) / (dd^2)

    elseif scen == 2
        # SW floating, rest grounded — solve for floating fraction,
        # return its complement.
        aa = -fSW
        bb = -(fSE - fSW)
        cc = -(fNW - fSW)
        dd = -(fNE + fSW - fNW - fSE)
        if abs(dd) < _FRAC_FTOL
            fSW = fSW > 0.0 ? fSW + 0.1 : fSW - 0.1
            aa = -fSW
            bb = -(fSE - fSW)
            cc = -(fNW - fSW)
            dd = -(fNE + fSW - fNW - fSE)
        end
        phi = 1.0 - ((bb*cc - aa*dd) * log(abs(1.0 - (aa*dd)/(bb*cc))) +
                     aa*dd) / (dd^2)

    elseif scen == 3
        # South grounded, north floating
        if abs(1.0 - fNW/fNE) < 1e-6 && abs(1.0 - fSW/fSE) < 1e-6
            # GL parallel to x-axis — closed-form
            phi = fSW / (fSW - fNW)
        else
            x = 0.0
            f1 = ((bb*cc - aa*dd) * log(abs(cc + dd*x)) - bb*dd*x) / (dd^2)
            x = 1.0
            f2 = ((bb*cc - aa*dd) * log(abs(cc + dd*x)) - bb*dd*x) / (dd^2)
            phi = f2 - f1
        end

    elseif scen == 4
        # SW & NE grounded, SE & NW floating — sum of two opposite
        # corner subscenarios.
        aa = fSW
        bb = fSE - fSW
        cc = fNW - fSW
        dd = fNE + fSW - fNW - fSE
        phi = ((bb*cc - aa*dd) * log(abs(1.0 - (aa*dd)/(bb*cc))) +
               aa*dd) / (dd^2)

        # 180° rotation lands on the NE-corner subscenario.
        fNW, fNE, fSW, fSE = _rotate_quad(fNW, fNE, fSW, fSE)
        fNW, fNE, fSW, fSE = _rotate_quad(fNW, fNE, fSW, fSE)
        aa = fSW
        bb = fSE - fSW
        cc = fNW - fSW
        dd = fNE + fSW - fNW - fSE
        phi += ((bb*cc - aa*dd) * log(abs(1.0 - (aa*dd)/(bb*cc))) +
                aa*dd) / (dd^2)

    else
        error("calc_fraction_above_zero: unknown scenario $scen")
    end

    return clamp(phi, 0.0, 1.0)
end
