# ----------------------------------------------------------------------
# Mass-balance helpers for the `tpo` step.
#
# These mirror the Fortran subroutines in
# `yelmo/src/physics/mass_conservation.f90`:
#   - `apply_tendency!`  ↔ `apply_tendency`  (line 112)
#   - `mbal_tendency!`   ↔ `calc_G_mbal`     (line 486)
#   - `resid_tendency!`  ↔ `calc_G_boundaries` (line 609)
# All three are in-place mutators that accept and return Oceananigans
# `Field`s (or any array-like with an `interior(...)` view).
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export apply_tendency!, mbal_tendency!

# Tolerance for "ice is effectively zero". Mirrors the `TOL` parameter in
# `mass_conservation.f90` (controls the `is_equal` test inside Fortran's
# `apply_tendency`, and the explicit small-thickness zeroing).
const _APPLY_TOL = 1e-9

"""
    apply_tendency!(H_ice, mb_dot, dt; adjust_mb=false, mb_lim=9999.0) -> H_ice

Apply the per-cell mass-balance tendency `mb_dot` [m/yr] to the ice
thickness `H_ice` [m] over time interval `dt` [yr]. Per cell:

  1. Clip `mb_dot[i,j]` to `[-mb_lim, +mb_lim]`.
  2. `H_ice[i,j] += dt * mb_dot[i,j]`.
  3. Clamp `H_ice` to `≥ 0` and zero values thinner than `1e-9 m`.
  4. If `adjust_mb` is `true`, rewrite `mb_dot[i,j]` to the realized
     rate `(H_new - H_prev) / dt` so the tendency reflects what was
     actually applied (after the clamp / tolerance zeroing).

No-op when `dt ≤ 0`. Port of `apply_tendency` in
`yelmo/src/physics/mass_conservation.f90:112`.
"""
function apply_tendency!(H_ice, mb_dot, dt::Real;
                         adjust_mb::Bool = false,
                         mb_lim::Float64 = 9999.0)
    dt > 0 || return H_ice

    H = interior(H_ice)
    G = interior(mb_dot)

    @inbounds for j in axes(H, 2), i in axes(H, 1)
        H_prev = H[i, j, 1]

        # Clip the requested tendency.
        g = G[i, j, 1]
        if g >  mb_lim;  g =  mb_lim;  end
        if g < -mb_lim;  g = -mb_lim;  end

        H_new = H_prev + dt * g

        # Clamp to non-negative and zero out tiny residuals.
        if H_new < 0.0
            H_new = 0.0
        elseif abs(H_new) < _APPLY_TOL
            H_new = 0.0
        end

        H[i, j, 1] = H_new

        if adjust_mb
            G[i, j, 1] = (H_new - H_prev) / dt
        end
    end

    return H_ice
end

"""
    mbal_tendency!(G_mb, H_ice, f_grnd, mbal, dt) -> G_mb

Pre-clip a raw mass-balance forcing `mbal` [m/yr] into a realisable
tendency `G_mb` [m/yr] given the current ice state. In place:

  1. `G_mb .= mbal`.
  2. Where `G_mb < 0` and `H_ice == 0`: zero `G_mb` (no melt without ice).
  3. Where `f_grnd == 0` and `H_ice == 0`: zero `G_mb` (no accumulation
     in open ocean cells).
  4. Where `H_ice + dt * G_mb < 0`: set `G_mb = -H_ice / dt`
     (don't melt more ice than is present).

Port of `calc_G_mbal` in `yelmo/src/physics/mass_conservation.f90:486`.
"""
function mbal_tendency!(G_mb, H_ice, f_grnd, mbal, dt::Real)
    G = interior(G_mb)
    H = interior(H_ice)
    Fg = interior(f_grnd)
    M = interior(mbal)

    @inbounds for j in axes(G, 2), i in axes(G, 1)
        g = M[i, j, 1]
        h = H[i, j, 1]
        fg = Fg[i, j, 1]

        # No melt where there is no ice.
        if g < 0.0 && h == 0.0
            g = 0.0
        end

        # No accumulation in open ocean (un-grounded, ice-free cells).
        if fg == 0.0 && h == 0.0
            g = 0.0
        end

        # Don't over-melt the available ice in this step.
        if dt != 0.0 && (h + dt * g) < 0.0
            g = -h / dt
        end

        G[i, j, 1] = g
    end

    return G_mb
end
