# ----------------------------------------------------------------------
# Ice-thickness relaxation toward a reference state.
#
# Two helpers, mirroring `physics/mass_conservation.f90`:
#
#   - `set_tau_relax!`     ↔ `set_tau_relax`     (line 819)
#   - `calc_G_relaxation!` ↔ `calc_G_relaxation` (line 916)
#
# `set_tau_relax!` builds the per-cell timescale field according to
# the spatial-mask selector `topo_rel` (1..3 supported; 4 errors
# pending the `mask_grz` port). `calc_G_relaxation!` converts that
# timescale into a `dHdt`-shaped tendency.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export set_tau_relax!, calc_G_relaxation!

# Marker meaning "this cell is not relaxed". Matches Fortran convention
# (set_tau_relax fills cells outside the relaxation mask with -1.0).
const _TAU_OFF = -1.0

# Saturated lookup with a sentinel: if (i,j) is out of bounds, return
# `out_of_bounds_value`. Used to read `f_grnd` neighbours for the
# topo_rel == 2 grounding-line "edge" detection, where domain edges
# should NOT be treated as floating neighbours (they're outside the
# physical mesh, not unrooted ice).
@inline function _f_grnd_or(F::AbstractArray, i::Int, j::Int,
                            nx::Int, ny::Int, default::Float64)
    return (1 <= i <= nx && 1 <= j <= ny) ? F[i, j, 1] : default
end

"""
    set_tau_relax!(tau_relax, H_ice, f_grnd, mask_grz, H_ref,
                   topo_rel, tau) -> tau_relax

Build the per-cell relaxation timescale `tau_relax` according to a
spatial mask selector:

  - `topo_rel == 1`: floating ice and ice-free points → `tau`.
  - `topo_rel == 2`: floating + grounding-line ice (a grounded cell
    with at least one floating orthogonal neighbour) → `tau`.
  - `topo_rel == 3`: every cell → `tau`.
  - `topo_rel == 4`: grounding-zone points keyed off `mask_grz` →
    not yet ported (errors).

All other cells are set to `_TAU_OFF` (-1.0), the "no-relaxation"
sentinel that `calc_G_relaxation!` recognises.

Port of `physics/mass_conservation.f90:set_tau_relax`.
"""
function set_tau_relax!(tau_relax, H_ice, f_grnd, mask_grz, H_ref,
                        topo_rel::Integer, tau::Real)
    Tau = interior(tau_relax)
    Fg  = interior(f_grnd)
    Hr  = interior(H_ref)
    nx  = size(Tau, 1)
    ny  = size(Tau, 2)

    if topo_rel == 1
        @inbounds for j in 1:ny, i in 1:nx
            Tau[i, j, 1] = (Fg[i, j, 1] == 0.0 || Hr[i, j, 1] == 0.0) ?
                           tau : _TAU_OFF
        end

    elseif topo_rel == 2
        @inbounds for j in 1:ny, i in 1:nx
            fg_here = Fg[i, j, 1]
            hr_here = Hr[i, j, 1]
            if fg_here == 0.0 || hr_here == 0.0
                Tau[i, j, 1] = tau
            elseif fg_here > 0.0
                # Read out-of-bounds as `1.0` (grounded) so domain
                # edges don't spuriously trigger the grounding-zone
                # rule. Mirrors get_neighbor_indices_bc_codes default.
                fW = _f_grnd_or(Fg, i-1, j,   nx, ny, 1.0)
                fE = _f_grnd_or(Fg, i+1, j,   nx, ny, 1.0)
                fS = _f_grnd_or(Fg, i,   j-1, nx, ny, 1.0)
                fN = _f_grnd_or(Fg, i,   j+1, nx, ny, 1.0)
                edge = (fW == 0.0) | (fE == 0.0) | (fS == 0.0) | (fN == 0.0)
                Tau[i, j, 1] = edge ? tau : _TAU_OFF
            else
                Tau[i, j, 1] = _TAU_OFF
            end
        end

    elseif topo_rel == 3
        fill!(Tau, tau)

    elseif topo_rel == 4
        error("set_tau_relax!: topo_rel = 4 not yet ported (depends on " *
              "mask_grz from the grounding-zone diagnostic, which the " *
              "Yelmo.jl v1 port has not yet computed).")

    else
        # Fortran default branch: no relaxation anywhere.
        fill!(Tau, _TAU_OFF)
    end

    return tau_relax
end

"""
    calc_G_relaxation!(dHdt, H_ice, H_ref, tau_relax, dt) -> dHdt

Compute the relaxation tendency `dHdt = (H_ref - H_ice) / tau` per
cell, with two special cases:

  - `tau_relax[i,j] == 0` ⇒ "impose ice thickness" (single-step):
    `dHdt = (H_ref - H_ice) / dt`, or `(H_ref - H_ice) / 1.0` for
    `dt ≤ 0`.
  - `tau_relax[i,j] < 0`  ⇒ no relaxation (`dHdt = 0`).

Output is fully overwritten — caller does not need to zero `dHdt`
beforehand. Port of `physics/mass_conservation.f90:calc_G_relaxation`.
"""
function calc_G_relaxation!(dHdt, H_ice, H_ref, tau_relax, dt::Real)
    G   = interior(dHdt)
    H   = interior(H_ice)
    Hr  = interior(H_ref)
    Tau = interior(tau_relax)

    fill!(G, 0.0)

    @inbounds for j in axes(G, 2), i in axes(G, 1)
        tau = Tau[i, j, 1]
        if tau == 0.0
            denom = dt > 0.0 ? dt : 1.0
            G[i, j, 1] = (Hr[i, j, 1] - H[i, j, 1]) / denom
        elseif tau > 0.0
            G[i, j, 1] = (Hr[i, j, 1] - H[i, j, 1]) / tau
        end
        # tau < 0 → leave at 0.
    end

    return dHdt
end
