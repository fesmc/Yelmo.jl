# ----------------------------------------------------------------------
# Online ice-age tracer (`dep_time` deposition-time field).
#
# Port of `yelmo/src/physics/ice_tracer.f90` — scope (A): the
# implicit ("impl") column solver path only. Specifically:
#
#   - `calc_tracer_3D!`              ← `calc_tracer_3D` (driver)
#   - `_calc_tracer_column_impl!`    ← `calc_tracer_column` (impl branch)
#   - `_calc_X_base`                 ← `calc_X_base`
#   - `_calc_advec_horizontal_col!`  ← `calc_advec_horizontal_column`
#   - `_fix_tracer_violation!`       ← `fix_tracer_violation`
#
# Out of scope for this commit (deferred):
#
#   - explicit solver `calc_tracer_column_expl`
#   - `calc_isochrones` (`y.mat.depth_iso` stays at its loaded value)
#   - the `*-tracer` enhancement-method paths in `mat_step!`
#     (`enh_method ∈ {simple-tracer, shear2D-tracer, shear3D-tracer}`)
#
# Yelmo.jl convention (Path B): `zeta_aa` is *interior* layer centres
# only (length `Nz_aa`, excludes `z = 0` and `z = 1`). `zeta_ac` is
# the face axis (length `Nz_aa + 1`, includes both endpoints). The
# Fortran tracer-column kernel uses Dirichlet pins at `k = 1` and
# `k = nz_aa` because Fortran's `zeta_aa` includes the boundary
# nodes; under Path B we mirror the thrm column-solver pattern
# instead — fold `X_base` / `X_srf` into the RHS of the interior
# stencil at `k = 1` / `k = Nz_aa` via the half-cell `dzeta_a[1]` /
# `dzeta_b[nz_aa]` weights. See `src/thrm/column_solver.jl` for the
# template.
#
# Boundary topology: this commit supports `Bounded × Bounded` grids
# only (matches Fortran's implicit assumption — the border-fill
# step `X[1, :] = X[3, :]` etc. clobbers wraparound). Periodic
# domains error explicitly.
# ----------------------------------------------------------------------

using Oceananigans.Grids: topology, Bounded

export calc_tracer_3D!

# Numerical safety knobs — match Fortran `ice_tracer.f90:55-57`.
const _TRACER_H_ICE_MIN     = 100.0     # [m] skip very thin ice
const _TRACER_UZ_MAX        =   5.0     # [m/yr] skip high-uz cells
const _TRACER_BMB_THINNING  = -1e-3     # [m/yr] keep solution bounded
const _TRACER_HORIZ_ULIM    = 5000.0    # [m/yr] horizontal velocity clip
const _TRACER_FDIFF_MARGIN  = 0.05      # bound-rate at margin
const _TRACER_FDIFF_INTERIOR = 0.1      # bound-rate elsewhere

"""
    calc_tracer_3D!(X_ice, X_srf, ux, uy, uz, H_ice, bmb,
                    zeta_aa, zeta_ac, dx, dt;
                    kappa, mask=nothing) -> X_ice

Advance a 3D age-tracer field `X_ice` (a Center 3D field) by `dt` years
under the velocity field `(ux, uy, uz)`, with surface boundary value
`X_srf` (scalar, typically `time`) and a per-column basal value
derived from `bmb` and the column state.

Mirrors Fortran `ice_tracer.f90:19-157 calc_tracer_3D` for the `impl`
solver branch. Border cells `i ∈ {1, 2, Nx-1, Nx}` and `j ∈ {1, 2,
Ny-1, Ny}` are skipped by the column loop (the horizontal upwind
stencil reaches `i ± 2`) and filled afterwards by direct copy from
the nearest interior column.

Tracer cells that fail the safety mask (`H_ice < $(_TRACER_H_ICE_MIN)`,
`|uz_basal| > $(_TRACER_UZ_MAX)`, or `mask[i, j] == false`) are
overwritten with `X_srf` everywhere in the column.

`kappa` is the artificial vertical diffusivity (m²/yr) — pass
`y.p.ymat.tracer_impl_kappa`. `mask` may be a 2D `Bool` array of the
same shape as `H_ice` to disable additional cells (Fortran exposes
this as the `mask` optional argument); pass `nothing` to use the
intrinsic safety mask only.
"""
function calc_tracer_3D!(X_ice, X_srf::Real,
                         ux, uy, uz, H_ice, bmb,
                         zeta_aa::AbstractVector{Float64},
                         zeta_ac::AbstractVector{Float64},
                         dx::Real, dt::Real;
                         kappa::Real,
                         mask = nothing)
    # Topology guard. The Fortran kernel + border-fill assume Bounded
    # x and y; under Periodic, the border-fill step would clobber the
    # wraparound copy. Error explicitly to surface this rather than
    # silently giving wrong results.
    Tx_top = topology(X_ice.grid, 1)
    Ty_top = topology(X_ice.grid, 2)
    (Tx_top === Bounded && Ty_top === Bounded) ||
        error("calc_tracer_3D!: only Bounded × Bounded x/y topology is " *
              "supported (got $(Tx_top) × $(Ty_top)).")

    X = interior(X_ice)
    Ux = interior(ux)
    Uy = interior(uy)
    Uz = interior(uz)
    Hi = interior(H_ice)
    Bm = interior(bmb)

    Nx    = size(X, 1)
    Ny    = size(X, 2)
    Nz_aa = size(X, 3)
    @assert length(zeta_aa) == Nz_aa "zeta_aa length must equal Nz_aa"
    @assert length(zeta_ac) == Nz_aa + 1 "zeta_ac must have length Nz_aa + 1"

    # ----- Snapshot for fix_tracer_violation -----
    X_prev = copy(X)

    # ----- Reusable column scratch -----
    advecxy  = Vector{Float64}(undef, Nz_aa)
    dzeta_a  = Vector{Float64}(undef, Nz_aa)
    dzeta_b  = Vector{Float64}(undef, Nz_aa)
    subd     = Vector{Float64}(undef, Nz_aa)
    diag     = Vector{Float64}(undef, Nz_aa)
    supd     = Vector{Float64}(undef, Nz_aa)
    rhs      = Vector{Float64}(undef, Nz_aa)
    sol      = Vector{Float64}(undef, Nz_aa)
    cp_buf   = Vector{Float64}(undef, Nz_aa)
    dp_buf   = Vector{Float64}(undef, Nz_aa)

    # `dzeta_a` / `dzeta_b` depend only on the (uniform) sigma axis,
    # so compute once outside the column loop. Path B handles the
    # half-cell weights at k = 1 and k = Nz_aa internally.
    calc_dzeta_terms!(dzeta_a, dzeta_b, zeta_aa, zeta_ac)

    dt_f = Float64(dt)
    dx_f = Float64(dx)
    kappa_f = Float64(kappa)
    X_srf_f = Float64(X_srf)

    @inbounds for j in 3:(Ny - 2), i in 3:(Nx - 2)
        H = Hi[i, j, 1]

        # Basal-uz extracted once for the safety mask check.
        uz_base = Uz[i, j, 1]

        active =
            H >= _TRACER_H_ICE_MIN &&
            abs(uz_base) <= _TRACER_UZ_MAX &&
            (mask === nothing || mask[i, j])

        if !active
            for k in 1:Nz_aa
                X[i, j, k] = X_srf_f
            end
            continue
        end

        # --- Horizontal advection contribution per layer ---
        _calc_advec_horizontal_col!(advecxy, X, Ux, Uy, dx_f, i, j;
                                    ulim = _TRACER_HORIZ_ULIM)

        # --- Basal boundary value (Rybak & Huybrechts 2003) ---
        X_base = _calc_X_base(view(X, i, j, :), H, advecxy[1],
                              Bm[i, j, 1], _TRACER_BMB_THINNING,
                              zeta_aa, dt_f)

        # --- Implicit vertical advection-diffusion column solve ---
        _calc_tracer_column_impl!(view(X, i, j, :),
                                  view(Uz, i, j, :),
                                  advecxy, X_srf_f, X_base, H,
                                  zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                                  kappa_f, dt_f,
                                  subd, diag, supd, rhs, sol,
                                  cp_buf, dp_buf)

        # --- Bounds enforcement (per-column rate-of-change clamp) ---
        # Margin cells (any neighbour in 3x3 stencil is ice-free) get
        # a tighter clamp to avoid spurious tracer overshoots.
        is_margin = false
        for jj in (j - 1):(j + 1), ii in (i - 1):(i + 1)
            if Hi[ii, jj, 1] == 0.0
                is_margin = true
                break
            end
        end
        f_diff = is_margin ? _TRACER_FDIFF_MARGIN : _TRACER_FDIFF_INTERIOR

        _fix_tracer_violation!(view(X, i, j, :), view(X_prev, i, j, :),
                               X_base, X_srf_f, dt_f, f_diff)
    end

    # ----- Border fill -----
    # Fortran lines 145-153: copy the nearest interior column outwards
    # so the i ∈ {1, 2} and j ∈ {1, 2} (and mirror at Nx, Ny) ghost
    # cells get sensible tracer values. The 2-cell-deep frame matches
    # the 2nd-order-upwind reach.
    @inbounds for k in 1:Nz_aa, j in 1:Ny
        X[1,    j, k] = X[3,    j, k]
        X[2,    j, k] = X[3,    j, k]
        X[Nx-1, j, k] = X[Nx-2, j, k]
        X[Nx,   j, k] = X[Nx-2, j, k]
    end
    @inbounds for k in 1:Nz_aa, i in 1:Nx
        X[i, 1,    k] = X[i, 3,    k]
        X[i, 2,    k] = X[i, 3,    k]
        X[i, Ny-1, k] = X[i, Ny-2, k]
        X[i, Ny,   k] = X[i, Ny-2, k]
    end
    return X_ice
end

# ----------------------------------------------------------------------
# Per-column kernels (no field types — pure array indexing on the
# already-extracted `interior(...)` slabs).
# ----------------------------------------------------------------------

# Implicit (Crank-Nicolson backward-Euler) advection-diffusion column
# solver. Path B convention — `zeta_aa` is interior centres only and
# the boundary values `X_base` (z=0) / `X_srf` (z=1) enter via the
# half-cell `dzeta_a[1]` / `dzeta_b[nz_aa]` weights folded into the
# RHS. Mirrors `src/thrm/column_solver.jl`'s pattern. Faithful port
# of Fortran `ice_tracer.f90:215-360 calc_tracer_column` for the
# implicit branch (no test_expl_advecz, no Q_strn, no T_ref).
#
# `kappa` is uniform across the column (constant artificial diffusivity
# for the age tracer), so we don't carry a per-layer kappa array —
# the harmonic-mean staggering collapses to `kappa_a = kappa_b = kappa`.
function _calc_tracer_column_impl!(X_col::AbstractVector,
                                   uz_col::AbstractVector,
                                   advecxy::Vector{Float64},
                                   X_srf::Float64, X_base::Float64,
                                   H::Float64,
                                   zeta_aa::AbstractVector{Float64},
                                   zeta_ac::AbstractVector{Float64},
                                   dzeta_a::Vector{Float64},
                                   dzeta_b::Vector{Float64},
                                   kappa::Float64, dt::Float64,
                                   subd::Vector{Float64},
                                   diag::Vector{Float64},
                                   supd::Vector{Float64},
                                   rhs::Vector{Float64},
                                   sol::Vector{Float64},
                                   cp_buf::Vector{Float64},
                                   dp_buf::Vector{Float64})
    nz_aa = length(zeta_aa)
    H2 = H * H

    @inbounds begin
        # ----- k = 1 (basal interior centre, X_base absorbed) -----
        # uz on ac-nodes; index 1 = z=0 face, index 2 = first interior
        # face. uz_aa is the centred-difference average at this aa-node.
        uz_aa = 0.5 * (uz_col[1] + uz_col[2])
        # Vertical-advection denominator: distance from z=0 to the first
        # *interior+1* centre (zeta_aa[2]). For uz_max safety this falls
        # back to the half-cell distance if zeta_aa has only one layer
        # above the base, but in practice nz_aa ≥ 2 always holds.
        dz_adv = H * (zeta_aa[2] - 0.0)

        fac_a = -kappa * dzeta_a[1] * dt / H2
        fac_b = -kappa * dzeta_b[1] * dt / H2

        absorb = fac_a - uz_aa * dt / dz_adv
        subd[1] = 0.0
        supd[1] = fac_b + uz_aa * dt / dz_adv
        diag[1] = 1.0 - fac_a - fac_b
        rhs[1]  = X_col[1] - dt * advecxy[1] - absorb * X_base

        # ----- Interior layers 2..nz_aa-1 -----
        for k in 2:(nz_aa - 1)
            uz_aa = 0.5 * (uz_col[k] + uz_col[k + 1])

            fac_a = -kappa * dzeta_a[k] * dt / H2
            fac_b = -kappa * dzeta_b[k] * dt / H2

            dzeta_adv = zeta_aa[k + 1] - zeta_aa[k - 1]
            dz_adv    = H * dzeta_adv

            subd[k] = fac_a - uz_aa * dt / dz_adv
            supd[k] = fac_b + uz_aa * dt / dz_adv
            diag[k] = 1.0 - fac_a - fac_b
            rhs[k]  = X_col[k] - dt * advecxy[k]
        end

        # ----- k = nz_aa (top interior centre, X_srf absorbed) -----
        uz_aa = 0.5 * (uz_col[nz_aa] + uz_col[nz_aa + 1])
        dz_adv = H * (1.0 - zeta_aa[nz_aa - 1])

        fac_a = -kappa * dzeta_a[nz_aa] * dt / H2
        fac_b = -kappa * dzeta_b[nz_aa] * dt / H2

        absorb = fac_b + uz_aa * dt / dz_adv
        subd[nz_aa] = fac_a - uz_aa * dt / dz_adv
        supd[nz_aa] = 0.0
        diag[nz_aa] = 1.0 - fac_a - fac_b
        rhs[nz_aa]  = X_col[nz_aa] - dt * advecxy[nz_aa] - absorb * X_srf

        # ----- Tridiag solve -----
        solve_tridiag!(sol, subd, diag, supd, rhs, cp_buf, dp_buf)

        # ----- Copy back -----
        for k in 1:nz_aa
            X_col[k] = sol[k]
        end
    end
    return X_col
end

# Basal boundary condition (Rybak & Huybrechts 2003 Eq. 3). With basal
# melting (bmb_tot < 0) the basal value is extrapolated downward from
# the lowest two interior centres; with freezing (bmb_tot > 0) the
# basal value is held at the lowest interior centre.
#
# Faithful port of Fortran `ice_tracer.f90:666-709 calc_X_base`.
@inline function _calc_X_base(X_col::AbstractVector, H::Float64,
                              advecxy_1::Float64, bmb::Float64,
                              bmb_thinning::Float64,
                              zeta_aa::AbstractVector{Float64},
                              dt::Float64)
    bmb_tot = bmb + bmb_thinning
    if bmb_tot <= 0.0
        # Basal melting: extrapolate down from (X[1], X[2]).
        dz   = H * (zeta_aa[2] - zeta_aa[1])
        # Limit weight to ≤ 1 to avoid extrapolating into a phantom
        # below-base layer (Fortran lines 696-698).
        f_wt = min(1.0, -dt * bmb_tot / dz)
        return X_col[1] + f_wt * (X_col[2] - X_col[1]) - dt * advecxy_1
    else
        # Basal freezing: keep the lowest interior centre.
        return X_col[1]
    end
end

# 2nd-order upwind horizontal advection per layer (`advecxy[1:nz_aa]`).
# Faithful port of Fortran `ice_tracer.f90:711-829`. `var_ice` is the
# Yelmo.jl Center 3D interior; `Ux` / `Uy` are the corresponding
# face-staggered velocity interiors.
#
# Yelmo.jl ↔ Fortran face-staggering map:
#   Fortran ux(i, j, k)   = east face of cell (i, j) → Yelmo.jl Ux[i+1, j, k]
#   Fortran ux(i-1, j, k) = west face of cell (i, j) → Yelmo.jl Ux[i,   j, k]
#   (similarly for uy on the y-axis)
#
# So the Fortran reads `ux(i-1, j, k)` becomes `Ux[i, j, k]`, and
# `ux(i, j, k)` becomes `Ux[i+1, j, k]`.
function _calc_advec_horizontal_col!(advecxy::Vector{Float64},
                                     X::AbstractArray{Float64,3},
                                     Ux::AbstractArray{Float64,3},
                                     Uy::AbstractArray{Float64,3},
                                     dx::Float64, i::Int, j::Int;
                                     ulim::Float64 = _TRACER_HORIZ_ULIM)
    Nx, Ny, Nz_aa = size(X)
    inv_2dx = 1.0 / (2.0 * dx)

    fill!(advecxy, 0.0)

    @inbounds for k in 1:Nz_aa
        # aa-staggered velocity for the upwind switch. For k = 1 use
        # the current vertical layer; for k > 1 average between k and
        # k - 1 (matches Fortran 750-758).
        if k == 1
            ux_aa = 0.5 * (Ux[i + 1, j, k]     + Ux[i,     j, k])
            uy_aa = 0.5 * (Uy[i,     j + 1, k] + Uy[i, j,     k])
        else
            ux_aa = 0.25 * (Ux[i + 1, j, k]     + Ux[i,     j, k] +
                            Ux[i + 1, j, k - 1] + Ux[i,     j, k - 1])
            uy_aa = 0.25 * (Uy[i,     j + 1, k] + Uy[i, j,     k] +
                            Uy[i,     j + 1, k - 1] + Uy[i, j,     k - 1])
        end

        advecx = 0.0
        if ux_aa > 0.0 && i >= 3
            # Flow to right: read west face (Yelmo.jl Ux[i, j, k] =
            # Fortran ux(i-1, j, k)). 2nd-order upwind in x.
            u_now = Ux[i, j, k]
            u_now = sign(u_now) * min(abs(u_now), ulim)
            advecx = inv_2dx * u_now *
                     (-(4.0 * X[i - 1, j, k] - X[i - 2, j, k] -
                        3.0 * X[i, j, k]))
        elseif ux_aa < 0.0 && i <= Nx - 2
            # Flow to left: read east face (Ux[i+1, j, k] = Fortran
            # ux(i, j, k)).
            u_now = Ux[i + 1, j, k]
            u_now = sign(u_now) * min(abs(u_now), ulim)
            advecx = inv_2dx * u_now *
                     ((4.0 * X[i + 1, j, k] - X[i + 2, j, k] -
                       3.0 * X[i, j, k]))
        end

        advecy = 0.0
        if uy_aa > 0.0 && j >= 3
            u_now = Uy[i, j, k]
            u_now = sign(u_now) * min(abs(u_now), ulim)
            advecy = inv_2dx * u_now *
                     (-(4.0 * X[i, j - 1, k] - X[i, j - 2, k] -
                        3.0 * X[i, j, k]))
        elseif uy_aa < 0.0 && j <= Ny - 2
            u_now = Uy[i, j + 1, k]
            u_now = sign(u_now) * min(abs(u_now), ulim)
            advecy = inv_2dx * u_now *
                     ((4.0 * X[i, j + 1, k] - X[i, j + 2, k] -
                       3.0 * X[i, j, k]))
        end

        advecxy[k] = advecx + advecy
    end
    return advecxy
end

# Per-column rate-of-change limiter. Caps `(X - X_prev) / dt` to
# `±f_diff · |mean(X_prev)|`; resets the column to `X_srf` if any
# absurd value (`> 1e10`) survives. Faithful port of Fortran
# `ice_tracer.f90:159-213 fix_tracer_violation`.
@inline function _fix_tracer_violation!(X_col::AbstractVector,
                                        X_prev_col::AbstractVector,
                                        X_base::Float64, X_srf::Float64,
                                        dt::Float64, f_diff::Float64)
    nz = length(X_col)

    # If the column was already pinned at X_srf everywhere (inactive
    # mask cell on the previous step), skip — there's nothing to limit.
    is_active = false
    @inbounds for k in 1:nz
        if X_col[k] != X_srf
            is_active = true
            break
        end
    end
    is_active || return X_col

    # Mean magnitude of the previous column.
    sum_prev = 0.0
    @inbounds for k in 1:nz
        sum_prev += X_prev_col[k]
    end
    X_mean = abs(sum_prev / nz)

    if X_mean != 0.0
        cap = f_diff * X_mean * dt
        @inbounds for k in 1:nz
            d = X_col[k] - X_prev_col[k]
            if d > cap
                X_col[k] = X_prev_col[k] + cap
            elseif d < -cap
                X_col[k] = X_prev_col[k] - cap
            end
        end
    end

    # Last-resort: any survivor with absurd magnitude → reset whole
    # column to X_srf.
    blew_up = false
    @inbounds for k in 1:nz
        if abs(X_col[k]) > 1e10
            blew_up = true
            break
        end
    end
    if blew_up
        @inbounds for k in 1:nz
            X_col[k] = X_srf
        end
    end
    return X_col
end
