# ----------------------------------------------------------------------
# Grounded-state helpers for the `tpo` step.
#
#   - `calc_H_grnd!`                 — flotation diagnostic [m]
#   - `determine_grounded_fractions!` — CISM bilinear-interpolation
#     subgrid-grounded-fraction kernel (Leguy et al. 2021), ported
#     from `physics/topography.f90:1896` (and helpers below it).
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.BoundaryConditions: fill_halo_regions!

using ..YelmoCore: fill_corner_halos!

export calc_H_grnd!, determine_grounded_fractions!,
       calc_f_grnd_subgrid_linear!, calc_f_grnd_subgrid_area!,
       calc_f_grnd_pinning_points!,
       calc_grounded_fractions!

"""
    calc_H_grnd!(H_grnd, H_ice, z_bed, z_sl, rho_ice, rho_sw) -> H_grnd

Flotation diagnostic. Two-branch formula matching
`physics/topography.f90:806 calc_H_grnd`:

  - Bed below sea level (`z_sl > z_bed`):
    `H_grnd = H_ice - (z_sl - z_bed) · rho_sw/rho_ice`
    — overburden minus water-column thickness.
  - Bed at or above sea level (`z_sl ≤ z_bed`):
    `H_grnd = H_ice + (z_bed - z_sl)`
    — ice thickness plus bed elevation above sea level. Ensures
    ice-free land also has `H_grnd > 0` so it classifies as
    grounded in subsequent kernels.

Positive ⇒ grounded (anchored to bed or above-SL land); negative ⇒
floating. Lightweight helper; no neighbour stencils.
"""
function calc_H_grnd!(H_grnd, H_ice, z_bed, z_sl,
                      rho_ice::Real, rho_sw::Real)
    Hg  = interior(H_grnd)
    H   = interior(H_ice)
    Zb  = interior(z_bed)
    Zsl = interior(z_sl)
    rho_sw_ice = rho_sw / rho_ice
    @inbounds for j in axes(Hg, 2), i in axes(Hg, 1)
        depth = Zsl[i, j, 1] - Zb[i, j, 1]
        Hg[i, j, 1] = depth > 0.0 ?
            H[i, j, 1] - rho_sw_ice * depth :
            H[i, j, 1] + (Zb[i, j, 1] - Zsl[i, j, 1])
    end
    return H_grnd
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

Halo handling: `H_grnd`'s halos are filled via `fill_halo_regions!`,
so neighbour reads honour the field's grid topology and boundary
conditions automatically (Neumann-zero clamp by default; Periodic
wrap on Periodic axes).

Port of `physics/topography.f90:determine_grounded_fractions`.
"""
function determine_grounded_fractions!(f_grnd, H_grnd;
                                       f_grnd_acx = nothing,
                                       f_grnd_acy = nothing,
                                       f_grnd_ab  = nothing)
    fill_halo_regions!(H_grnd)
    fill_corner_halos!(H_grnd)

    Hg = interior(H_grnd)
    nx = size(Hg, 1)
    ny = size(Hg, 2)

    f_NW = Matrix{Float64}(undef, nx, ny)
    f_NE = Matrix{Float64}(undef, nx, ny)
    f_SW = Matrix{Float64}(undef, nx, ny)
    f_SE = Matrix{Float64}(undef, nx, ny)
    _cism_quads!(f_NW, f_NE, f_SW, f_SE, H_grnd, nx, ny)

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
# `physics/topography.f90:1978`. Reads `H_grnd` halos directly (filled
# upstream) and computes `f_flt = -H_grnd` inline — saves an
# allocation versus materialising a separate `f_flt` field.
function _cism_quads!(f_NW, f_NE, f_SW, f_SE, H_grnd, nx::Int, ny::Int)
    @inbounds for j in 1:ny, i in 1:nx
        f_m  = -H_grnd[i,   j,   1]
        fW   = 0.5  * (-H_grnd[i-1, j,   1] + f_m)
        fE   = 0.5  * (f_m + -H_grnd[i+1, j,   1])
        fN   = 0.5  * (f_m + -H_grnd[i,   j+1, 1])
        fS   = 0.5  * (f_m + -H_grnd[i,   j-1, 1])
        fNW  = 0.25 * (-H_grnd[i-1, j+1, 1] +
                       -H_grnd[i,   j+1, 1] +
                       -H_grnd[i-1, j,   1] + f_m)
        fNE  = 0.25 * (-H_grnd[i,   j+1, 1] +
                       -H_grnd[i+1, j+1, 1] +
                       f_m + -H_grnd[i+1, j, 1])
        fSW  = 0.25 * (-H_grnd[i-1, j,   1] + f_m +
                       -H_grnd[i-1, j-1, 1] +
                       -H_grnd[i,   j-1, 1])
        fSE  = 0.25 * (f_m + -H_grnd[i+1, j,   1] +
                       -H_grnd[i,   j-1, 1] +
                       -H_grnd[i+1, j-1, 1])

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

# ----------------------------------------------------------------------
# `gl_sep = 1` — linear interpolation between grounded/floating cells
# ----------------------------------------------------------------------

"""
    calc_f_grnd_subgrid_linear!(f_grnd, f_grnd_acx, f_grnd_acy, H_grnd)
        -> f_grnd

Compute the binary aa-node `f_grnd` (1 if `H_grnd > 0`, else 0) and
the linearly-interpolated face fractions `f_grnd_acx`, `f_grnd_acy`
from the flotation diagnostic `H_grnd`. At a grounding-line face the
fraction is `f = -H_grnd₁ / (H_grnd₂ - H_grnd₁)` where `H_grnd₁` is
the grounded-side value and `H_grnd₂` the floating-side value
(zero crossing of a linear interpolant).

Halo handling: `H_grnd` halos are filled via `fill_halo_regions!` so
boundary faces honour the grid topology + BCs. The Fortran's edge
fix-up (`f_grnd_x[nx,:] = f_grnd_x[nx-1,:]`) is replaced by halo
reads on the eastern/northern boundary face.

Port of `physics/topography.f90:1108 calc_f_grnd_subgrid_linear`.
This is the `gl_sep = 1` variant.
"""
function calc_f_grnd_subgrid_linear!(f_grnd, f_grnd_acx, f_grnd_acy, H_grnd)
    fill_halo_regions!(H_grnd)

    Fa  = interior(f_grnd)
    Fx  = interior(f_grnd_acx)
    Fy  = interior(f_grnd_acy)
    Hg  = interior(H_grnd)
    nx, ny = size(Fa, 1), size(Fa, 2)

    # aa-node: binary f_grnd (matches the Fortran where ... 1.0 / 0.0).
    @inbounds for j in 1:ny, i in 1:nx
        Fa[i, j, 1] = Hg[i, j, 1] > 0.0 ? 1.0 : 0.0
    end

    # x-face: linear interp of H_grnd between aa-cells (i-1) and (i).
    @inbounds for j in axes(Fx, 2), i in axes(Fx, 1)
        Fx[i, j, 1] = _linear_face_fraction(H_grnd[i-1, j, 1], H_grnd[i, j, 1])
    end

    # y-face.
    @inbounds for j in axes(Fy, 2), i in axes(Fy, 1)
        Fy[i, j, 1] = _linear_face_fraction(H_grnd[i, j-1, 1], H_grnd[i, j, 1])
    end

    return f_grnd
end

# Linear face-fraction between two aa-node H_grnd values. Cells with
# the same flotation sign give 0 (both floating) or 1 (both
# grounded); a sign change linearises through the zero crossing.
@inline function _linear_face_fraction(H1::Real, H2::Real)
    if H1 > 0.0 && H2 <= 0.0
        return -H1 / (H2 - H1)
    elseif H1 <= 0.0 && H2 > 0.0
        # Symmetric formula with `H_grnd_1` taken from the grounded side.
        return -H2 / (H1 - H2)
    elseif H1 <= 0.0 && H2 <= 0.0
        return 0.0
    else
        return 1.0
    end
end

# ----------------------------------------------------------------------
# `gl_sep = 2` — area-based subgrid evaluation with `gz_nx` interp pts
# ----------------------------------------------------------------------

"""
    calc_f_grnd_subgrid_area!(f_grnd, f_grnd_acx, f_grnd_acy, H_grnd;
                              gz_nx=15) -> f_grnd

Compute aa-node `f_grnd` by evaluating the bilinear interpolant of
`H_grnd` at the corners (ab-nodes) of each cell, then sampling the
interpolant on a `gz_nx × gz_nx` regular sub-grid and counting the
fraction above zero. Face fractions `f_grnd_acx` / `f_grnd_acy` are
the average of the aa fractions on each side (the Fortran routine
overrides the per-face subgrid sampling with this average — line
1083+ — and we follow that active variant).

Halo handling: `H_grnd` halos are filled via `fill_halo_regions!`
plus `fill_corner_halos!` (the 4-corner ab-node averages reach into
diagonals).

Port of `physics/topography.f90:959 calc_f_grnd_subgrid_area`.
This is the `gl_sep = 2` variant.
"""
function calc_f_grnd_subgrid_area!(f_grnd, f_grnd_acx, f_grnd_acy, H_grnd;
                                   gz_nx::Int = 15)
    gz_nx > 1 || error("calc_f_grnd_subgrid_area!: gz_nx must be > 1 (got $gz_nx).")

    fill_halo_regions!(H_grnd)
    fill_corner_halos!(H_grnd)

    Fa = interior(f_grnd)
    Fx = interior(f_grnd_acx)
    Fy = interior(f_grnd_acy)
    nx, ny = size(Fa, 1), size(Fa, 2)

    inv_n = 1.0 / (gz_nx * gz_nx)

    # Phase 1: aa-node fraction via 4-corner ab-node averaging then
    # bilinear sub-cell sampling.
    @inbounds for j in 1:ny, i in 1:nx
        # H_grnd at the four ab-corners (NE, NW, SW, SE).
        Hg_NE = 0.25 * (H_grnd[i,   j,   1] + H_grnd[i+1, j,   1] +
                        H_grnd[i+1, j+1, 1] + H_grnd[i,   j+1, 1])
        Hg_NW = 0.25 * (H_grnd[i,   j,   1] + H_grnd[i-1, j,   1] +
                        H_grnd[i-1, j+1, 1] + H_grnd[i,   j+1, 1])
        Hg_SW = 0.25 * (H_grnd[i,   j,   1] + H_grnd[i-1, j,   1] +
                        H_grnd[i-1, j-1, 1] + H_grnd[i,   j-1, 1])
        Hg_SE = 0.25 * (H_grnd[i,   j,   1] + H_grnd[i+1, j,   1] +
                        H_grnd[i+1, j-1, 1] + H_grnd[i,   j-1, 1])

        Hg_min = min(Hg_NE, Hg_NW, Hg_SW, Hg_SE)
        Hg_max = max(Hg_NE, Hg_NW, Hg_SW, Hg_SE)

        if Hg_max >= 0.0 && Hg_min < 0.0
            # Subgrid GL — sample the bilinear interpolant on a
            # gz_nx × gz_nx sub-cell grid and count points >= 0.
            Fa[i, j, 1] = _bilinear_grounded_fraction(Hg_NW, Hg_NE,
                                                      Hg_SW, Hg_SE,
                                                      gz_nx, inv_n)
        elseif Hg_min >= 0.0
            Fa[i, j, 1] = 1.0
        else
            Fa[i, j, 1] = 0.0
        end
    end

    # Phase 2: face fractions = arithmetic mean of the two adjacent aa
    # values (Fortran's active `if (.TRUE.)` branch at line 1083+).
    # Boundary face values come from halo reads on the aa-fraction
    # field — but f_grnd's halos aren't filled here yet, so we read
    # interior with clamping for now. The Fortran's `f_grnd_acx[nx,:]
    # = f_grnd_acx[nx-1,:]` edge fix is reproduced by clamping below.
    @inbounds for j in axes(Fx, 2), i in axes(Fx, 1)
        i_l = max(i - 1, 1)
        i_r = min(i,     nx)
        Fx[i, j, 1] = 0.5 * (Fa[i_l, j, 1] + Fa[i_r, j, 1])
    end
    @inbounds for j in axes(Fy, 2), i in axes(Fy, 1)
        j_l = max(j - 1, 1)
        j_r = min(j,     ny)
        Fy[i, j, 1] = 0.5 * (Fa[i, j_l, 1] + Fa[i, j_r, 1])
    end

    return f_grnd
end

# Sub-cell sampling of the bilinear interpolant defined by four
# ab-corner values. Returns `count(samples >= 0) / gz_nx²`. The
# corners are passed in (NW, NE, SW, SE) order.
function _bilinear_grounded_fraction(Hg_NW::Real, Hg_NE::Real,
                                     Hg_SW::Real, Hg_SE::Real,
                                     gz_nx::Int, inv_n::Float64)
    n_grnd = 0
    @inbounds for jj in 1:gz_nx, ii in 1:gz_nx
        u = (ii - 0.5) / gz_nx           # 0 < u < 1, `u=0` ↔ west edge
        v = (jj - 0.5) / gz_nx           # 0 < v < 1, `v=0` ↔ south edge
        H = (1 - u) * (1 - v) * Hg_SW +
            u       * (1 - v) * Hg_SE +
            (1 - u) * v       * Hg_NW +
            u       * v       * Hg_NE
        H >= 0.0 && (n_grnd += 1)
    end
    return n_grnd * inv_n
end

# ----------------------------------------------------------------------
# Subgrid pinning-point fraction
# ----------------------------------------------------------------------

"""
    calc_f_grnd_pinning_points!(f_grnd_pin, H_ice, f_ice,
                                z_bed, z_bed_sd, z_sl, rho_ice, rho_sw)
        -> f_grnd_pin

For each floating ice cell, compute the fraction of the cell whose
sub-grid bed could touch the base of the ice shelf, given a
distribution of bed elevations `N(z_bed, z_bed_sd)`. Uses the
Pollard & DeConto (2012, Eq. 13) approximation:

    f = 0.5 · max(0, 1 − (z_base − z_bed) / σ_bed)

with `z_base = z_sl − rho_ice/rho_sw · H_eff` (the ice-shelf draft).
Cells where the bed is shallower than the draft are floating-by-
default in the cell-mean sense but may have grounded sub-cell
patches. Cells with `σ_bed = 0` are always 0.

Per-cell, no neighbour reads — no halo handling needed.

Port of `physics/topography.f90:1224 calc_f_grnd_pinning_points`.
"""
function calc_f_grnd_pinning_points!(f_grnd_pin, H_ice, f_ice,
                                     z_bed, z_bed_sd, z_sl,
                                     rho_ice::Real, rho_sw::Real)
    Fp  = interior(f_grnd_pin)
    H   = interior(H_ice)
    Fi  = interior(f_ice)
    Zb  = interior(z_bed)
    Zsd = interior(z_bed_sd)
    Zsl = interior(z_sl)

    rho_ice_sw = rho_ice / rho_sw

    @inbounds for j in axes(Fp, 2), i in axes(Fp, 1)
        h = H[i, j, 1]
        f = Fi[i, j, 1]

        # Effective ice thickness (no `set_frac_zero` here — match
        # Fortran's `calc_H_eff(...)` without the kwarg).
        H_eff = f > 0.0 ? h / f : h

        # Ice-shelf draft assuming floating.
        z_base = Zsl[i, j, 1] - rho_ice_sw * H_eff

        zb = Zb[i, j, 1]
        if z_base > zb
            sigma = Zsd[i, j, 1]
            if sigma == 0.0
                Fp[i, j, 1] = 0.0
            else
                # Pollard & DeConto (2012), Eq. 13.
                Fp[i, j, 1] = 0.5 * max(0.0, 1.0 - (z_base - zb) / sigma)
            end
        else
            Fp[i, j, 1] = 0.0
        end
    end
    return f_grnd_pin
end

# ----------------------------------------------------------------------
# `gl_sep` dispatch — pick the right f_grnd variant
# ----------------------------------------------------------------------

"""
    calc_grounded_fractions!(f_grnd, f_grnd_acx, f_grnd_acy, f_grnd_ab,
                             H_grnd, gl_sep; gz_nx=15) -> f_grnd

Dispatch over the `ytopo.gl_sep` parameter:

  - `gl_sep == 1` → `calc_f_grnd_subgrid_linear!` (linear interp).
    Writes aa, acx, acy. `f_grnd_ab` is left untouched.
  - `gl_sep == 2` → `calc_f_grnd_subgrid_area!` with `gz_nx`
    sub-cell points per side. Writes aa, acx, acy.
  - `gl_sep == 3` → `determine_grounded_fractions!` (CISM-quad,
    Leguy et al. 2021). Writes aa, acx, acy, ab.

`f_grnd_ab` is populated only by the CISM-quad variant; pass
`nothing` to skip it for the other branches.

Port of `physics/topography.f90:949 select case(tpo%par%gl_sep)`.
"""
function calc_grounded_fractions!(f_grnd, f_grnd_acx, f_grnd_acy, f_grnd_ab,
                                  H_grnd, gl_sep::Integer;
                                  gz_nx::Int = 15)
    if gl_sep == 1
        calc_f_grnd_subgrid_linear!(f_grnd, f_grnd_acx, f_grnd_acy, H_grnd)
    elseif gl_sep == 2
        calc_f_grnd_subgrid_area!(f_grnd, f_grnd_acx, f_grnd_acy, H_grnd;
                                  gz_nx = gz_nx)
    elseif gl_sep == 3
        determine_grounded_fractions!(f_grnd, H_grnd;
                                      f_grnd_acx = f_grnd_acx,
                                      f_grnd_acy = f_grnd_acy,
                                      f_grnd_ab  = f_grnd_ab)
    else
        error("calc_grounded_fractions!: gl_sep=$(gl_sep) not supported. " *
              "Use 1 (linear), 2 (area), or 3 (CISM-quad).")
    end
    return f_grnd
end
