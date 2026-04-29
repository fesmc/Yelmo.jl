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

# Tolerances for the stable fraction-above-zero kernel. These bound
# the regions where direct evaluation of the analytical formula
# becomes ill-conditioned, and trigger a switch to the corresponding
# closed-form limit.
#
#   - `_LINEAR_DD_TOL`: the bilinear cross-coefficient `dd = NE+SW-NW-SE`
#     measures how strongly the level set bends inside the cell. When
#     `|dd|` falls below this tolerance the interpolant is effectively
#     a linear function, and we compute the grounded area exactly via
#     half-plane / unit-square clipping (no `1/dd^2` blow-up).
#
#   - `_SADDLE_DELTA_TOL`: in the bilinear branch, the scenario formula
#     has a removable 0/0 singularity when `δ = bb*cc - aa*dd → 0`
#     (the level set passes through the corner being integrated to).
#     The formula evaluates to `aa/dd` in the limit; we switch to
#     this closed form when `|δ|` falls below the tolerance, scaled
#     by the magnitudes of `bb*cc` and `aa*dd` to be invariant under
#     uniform rescaling of the corner values.
#
# Both tolerances are well above floating-point noise (~1e-15) and
# well below the magnitudes typical for ice-sheet flotation diagnostics
# (|H_grnd| in metres).
const _LINEAR_DD_TOL    = 1e-12
const _SADDLE_DELTA_TOL = 1e-12

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
# By the "0 = grounded" convention used throughout this kernel, a
# corner exactly equal to zero is treated as ≤ 0 (grounded side) for
# scenario assignment. The actual analytical area is invariant under
# this choice (a corner at exactly zero contributes a measure-zero
# point to the level set), so the convention only affects the
# scenario routing — the formula still evaluates the correct limit.
# Returns the rotated tuple plus the scenario id.
function _rotate_quad_until_match(fNW0, fNE0, fSW0, fSE0)
    fNW, fNE, fSW, fSE = fNW0, fNE0, fSW0, fSE0
    @inbounds for _ in 1:4
        fNW, fNE, fSW, fSE = _rotate_quad(fNW, fNE, fSW, fSE)
        if     fSW <= 0.0 && fSE >  0.0 && fNE >  0.0 && fNW >  0.0
            return fNW, fNE, fSW, fSE, 1
        elseif fSW >  0.0 && fSE <= 0.0 && fNE <= 0.0 && fNW <= 0.0
            return fNW, fNE, fSW, fSE, 2
        elseif fSW <= 0.0 && fSE <= 0.0 && fNE >  0.0 && fNW >  0.0
            return fNW, fNE, fSW, fSE, 3
        elseif fSW <= 0.0 && fSE >  0.0 && fNE <= 0.0 && fNW >  0.0
            return fNW, fNE, fSW, fSE, 4
        end
    end
    error("rotate_quad_until_match: no matching scenario for " *
          "(NW=$fNW0, NE=$fNE0, SW=$fSW0, SE=$fSE0)")
end

# Exact area of `{(u,v) ∈ [0,1]^2 : aa + bb·u + cc·v ≤ 0}`. Uses
# Sutherland-Hodgman clipping of the unit square against the half-
# plane, then the shoelace formula on the resulting (≤ 5-vertex)
# polygon. Returns 0 when the half-plane misses the square.
function _grounded_area_linear_unit_square(aa, bb, cc)
    @inline f(p) = aa + bb*p[1] + cc*p[2]
    poly = NTuple{2,Float64}[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    out  = NTuple{2,Float64}[]
    n = length(poly)
    @inbounds for k in 1:n
        p1 = poly[k]
        p2 = poly[mod1(k + 1, n)]
        f1 = f(p1)
        f2 = f(p2)
        in1 = f1 <= 0.0      # "≤ 0" = grounded, matching the convention
        in2 = f2 <= 0.0
        if in1 && in2
            push!(out, p2)
        elseif in1 && !in2
            t = f1 / (f1 - f2)
            push!(out, (p1[1] + t*(p2[1] - p1[1]),
                        p1[2] + t*(p2[2] - p1[2])))
        elseif !in1 && in2
            t = f1 / (f1 - f2)
            push!(out, (p1[1] + t*(p2[1] - p1[1]),
                        p1[2] + t*(p2[2] - p1[2])))
            push!(out, p2)
        end
    end
    isempty(out) && return 0.0
    A = 0.0
    m = length(out)
    @inbounds for k in 1:m
        x1, y1 = out[k]
        x2, y2 = out[mod1(k + 1, m)]
        A += x1*y2 - x2*y1
    end
    return abs(A) / 2.0
end

# Stable evaluation of the scenario-1 grounded-area formula
#
#     φ(aa, bb, cc, dd) = ((bb·cc − aa·dd)·log|1 − aa·dd/(bb·cc)|
#                          + aa·dd) / dd²
#
# i.e. the area of the SW-corner grounded triangle for a bilinear
# interpolant `f = aa + bb·u + cc·v + dd·u·v`. The formula has a
# removable 0/0 singularity at `δ ≡ bb·cc − aa·dd = 0`; in that limit
# `δ·log|δ| → 0` and the whole expression collapses to `aa/dd`.
# Caller must guarantee `|dd| ≥ _LINEAR_DD_TOL`.
@inline function _phi_corner_grounded(aa, bb, cc, dd)
    bbcc = bb * cc
    aadd = aa * dd
    delta = bbcc - aadd
    scale = max(abs(bbcc), abs(aadd), 1.0)
    if abs(delta) < _SADDLE_DELTA_TOL * scale
        return aa / dd
    end
    return (delta * log(abs(delta / bbcc)) + aadd) / (dd * dd)
end

"""
    _calc_fraction_above_zero(f_NW, f_NE, f_SW, f_SE) -> phi ∈ [0,1]

Grounded fraction of a unit square `[0,1]²` containing a bilinear
interpolant defined by the four corner values. By convention, "≤ 0"
is grounded and "> 0" is floating — corners with the value zero are
treated as grounded (sit on the level set, contribute measure zero).

The kernel has two branches:

  1. **Linear branch.** When the bilinear cross-coefficient
     `dd = NE+SW-NW-SE` is small (`|dd| < _LINEAR_DD_TOL`), the
     interpolant has no `u·v` term and the level set is a straight
     line. We compute the grounded area exactly by clipping the unit
     square against the half-plane `aa + bb·u + cc·v ≤ 0`.

  2. **Bilinear branch.** Otherwise, rotate to one of four canonical
     sign patterns and evaluate the corresponding analytical area:
     scenarios 1, 2, 4 (SW corner / SW-only-floating / saddle) all
     reduce to one or two evaluations of `_phi_corner_grounded`.
     Scenario 3 (south grounded, north floating) is integrated
     directly across the cell.

This is a stable replacement for `physics/topography.f90:2059
calc_fraction_above_zero`. It removes both numerical bandages from
the Fortran reference: (i) the `±_FRAC_FTOL` snap on near-zero corner
values, and (ii) the `±0.1` perturbation when `|dd|` is small. The
linear branch makes (ii) unnecessary, and the analytical limit
inside `_phi_corner_grounded` makes (i) unnecessary.
"""
function _calc_fraction_above_zero(f_NW, f_NE, f_SW, f_SE)
    if f_NW <= 0.0 && f_NE <= 0.0 && f_SW <= 0.0 && f_SE <= 0.0
        return 1.0
    elseif f_NW > 0.0 && f_NE > 0.0 && f_SW > 0.0 && f_SE > 0.0
        return 0.0
    end

    # Linear branch: |dd| ≈ 0 means the bilinear is effectively a
    # plane and the level set is a straight line. Polygon clip is
    # exact and avoids the 1/dd^2 blow-up of the bilinear formula.
    dd0 = f_NE + f_SW - f_NW - f_SE
    if abs(dd0) < _LINEAR_DD_TOL
        aa = f_SW
        bb = f_SE - f_SW
        cc = f_NW - f_SW
        return _grounded_area_linear_unit_square(aa, bb, cc)
    end

    fNW, fNE, fSW, fSE, scen = _rotate_quad_until_match(f_NW, f_NE, f_SW, f_SE)
    aa = fSW
    bb = fSE - fSW
    cc = fNW - fSW
    dd = fNE + fSW - fNW - fSE

    phi = if scen == 1
        # SW grounded, rest floating.
        _phi_corner_grounded(aa, bb, cc, dd)

    elseif scen == 2
        # SW floating, rest grounded — flip signs to use the
        # corner-grounded formula on the floating piece, then take
        # the complement.
        1.0 - _phi_corner_grounded(-aa, -bb, -cc, -dd)

    elseif scen == 3
        # South grounded, north floating. Integrate v_c(u) over u ∈ [0,1]:
        #   F(x) = ((bb·cc − aa·dd) log|cc + dd·x| − bb·dd·x) / dd²
        #   φ    = F(1) − F(0)
        # `cc` and `cc + dd` are both > 0 within scenario 3 (top edge
        # floating, north corners > 0), so the log arguments are positive
        # and the formula is stable for any |dd| ≥ _LINEAR_DD_TOL.
        bbcc_aadd = bb*cc - aa*dd
        dd2 = dd * dd
        f1 = (bbcc_aadd * log(abs(cc))           ) / dd2
        f2 = (bbcc_aadd * log(abs(cc + dd)) - bb*dd) / dd2
        f2 - f1

    elseif scen == 4
        # Saddle: SW & NE grounded, SE & NW floating. Sum the two
        # opposite-corner sub-areas. The 180° rotation lands the
        # NE-grounded sub-piece into a scenario-1-shaped square.
        phi_SW = _phi_corner_grounded(aa, bb, cc, dd)
        fNW2, fNE2, fSW2, fSE2 = _rotate_quad(_rotate_quad(fNW, fNE, fSW, fSE)...)
        aa2 = fSW2
        bb2 = fSE2 - fSW2
        cc2 = fNW2 - fSW2
        dd2 = fNE2 + fSW2 - fNW2 - fSE2
        phi_SW + _phi_corner_grounded(aa2, bb2, cc2, dd2)

    else
        error("calc_fraction_above_zero: unknown scenario $scen")
    end

    return clamp(phi, 0.0, 1.0)
end
