const OPT_MV = -9999.0   # Fortran yelmo_defs.f90: MISSING_VALUE_DEFAULT

_opt_at(A::AbstractMatrix, i, j) = @inbounds A[i, j]
_opt_at(a::Real, i, j) = a

"""
    calc_magnitude_from_staggered_ice(u, v, H; boundaries=nothing) -> umag

Centered (aa-node) magnitude of a vector field given as staggered
(ac-node) components `u`, `v`, weighted by whether the neighboring cell
has ice (`H != 0`) so margin cells don't pull in a zero from outside the
ice sheet. `boundaries = "periodic"` wraps the outermost row/column as
in the Fortran original; any other value (including `nothing`) leaves
domain edges as computed (clamped-index) rather than wrapped.

Port of `calc_magnitude_from_staggered_ice` in `ice_optimization.f90`.
"""
function calc_magnitude_from_staggered_ice(u::AbstractMatrix, v::AbstractMatrix,
                                            H::AbstractMatrix; boundaries=nothing)
    nx, ny = size(u)
    umag = zeros(Float64, nx, ny)

    @inbounds for j in 1:ny, i in 1:nx
        im1, ip1 = max(i - 1, 1), min(i + 1, nx)
        jm1, jp1 = max(j - 1, 1), min(j + 1, ny)

        H1x = 0.5 * (H[im1, j] + H[i, j])
        H2x = 0.5 * (H[i, j] + H[ip1, j])
        f1x = H1x == 0.0 ? 0.0 : 0.5
        f2x = H2x == 0.0 ? 0.0 : 0.5
        unow = (f1x + f2x) > 0.0 ? (f1x * u[im1, j] + f2x * u[i, j]) / (f1x + f2x) : 0.0

        H1y = 0.5 * (H[i, jm1] + H[i, j])
        H2y = 0.5 * (H[i, j] + H[i, jp1])
        f1y = H1y == 0.0 ? 0.0 : 0.5
        f2y = H2y == 0.0 ? 0.0 : 0.5
        vnow = (f1y + f2y) > 0.0 ? (f1y * v[i, jm1] + f2y * v[i, j]) / (f1y + f2y) : 0.0

        umag[i, j] = sqrt(unow^2 + vnow^2)
    end

    if boundaries == "periodic"
        umag[1, :]  .= umag[nx - 1, :]
        umag[nx, :] .= umag[2, :]
        umag[:, 1]  .= umag[:, ny - 1]
        umag[:, ny] .= umag[:, 2]
    end

    return umag
end

"""
    optimize_cb_ref!(cb_ref, H_ice, dHdt, ux, uy, H_obs, uxy_obs, H_grnd_obs,
                      cf_min, cf_max, dx, tau_c, H0, dt;
                      fill_method="cf_min", cb_tgt=nothing) -> cb_ref

In-place friction-coefficient (`cb_ref`) nudge following Lipscomb et al.
(2021, TC) — the "L21" method in the colleague's ISMIP7 `opt_ant.sh`
(`opt.tau_c`, `opt.H0`). Relaxes `cb_ref` so the modelled steady-state
ice thickness/velocity moves toward `H_obs`/`uxy_obs`.

`cf_min`/`cf_max` may be a scalar (uniform bound) or a matrix matching
`cb_ref`'s shape (spatially varying bound, e.g. from `ytill`'s own
`cf_min`/`cf_ref` fields) — the caller decides which, per
`opt.use_yelmo_cf_min` in the nml.

`fill_method` selects how cells outside the optimizable region (floating
per `H_grnd_obs`, or ice-free per `H_obs`) are filled:
- `"cf_min"` — set to `cf_min` (matches the colleague's config).
- `"target"` — set to `cb_tgt` (must be supplied).
Fortran also defines `"nearest"`/`"analog"` fill methods, but both are
marked "not working right now" upstream (the routine `stop`s
immediately if selected) and are not ported here.

Two upstream behaviors are preserved rather than "fixed": the
`sigma_err`/`sigma_vel` Gaussian pre-smoothing of the thickness error is
dead code in the Fortran source (`if (.FALSE.)`) and is not applied
here either; and cells with zero velocity that are neither floating nor
ice-free are left at their previous `cb_ref` value rather than updated
or filled (same gap as upstream — `cb_ref` is only ever written by the
velocity-gated loop below or by the `fill_method` where-clause).

Port of `optimize_cb_ref` in `ice_optimization.f90` (the `z_bed`/`z_sl`
arguments and `sigma_err`/`sigma_vel`/`fill_dist` are dropped: none are
read on any code path that isn't a `stop`).
"""
function optimize_cb_ref!(cb_ref::AbstractMatrix, H_ice::AbstractMatrix, dHdt::AbstractMatrix,
                           ux::AbstractMatrix, uy::AbstractMatrix,
                           H_obs::AbstractMatrix, uxy_obs::AbstractMatrix, H_grnd_obs::AbstractMatrix,
                           cf_min::Union{Real,AbstractMatrix}, cf_max::Union{Real,AbstractMatrix},
                           dx::Real, tau_c::Real, H0::Real, dt::Real;
                           fill_method::String="cf_min",
                           cb_tgt::Union{Nothing,AbstractMatrix}=nothing)

    fill_method in ("cf_min", "target") ||
        error("optimize_cb_ref!: fill_method=\"$fill_method\" is not implemented " *
              "(matches upstream Fortran, where \"nearest\"/\"analog\" `stop` immediately).")
    fill_method == "target" && cb_tgt === nothing &&
        error("optimize_cb_ref!: fill_method=\"target\" requires cb_tgt.")

    nx, ny = size(cb_ref)
    f_damp  = 2.0
    f_tgt   = 0.05 * H0
    tau_tgt = tau_c
    scaleH  = H0 <= 0.0
    eps     = 1.0e-12

    cb_prev = copy(cb_ref)
    uxy = calc_magnitude_from_staggered_ice(ux, uy, H_ice)

    uxy_err = fill(OPT_MV, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        if uxy_obs[i, j] != OPT_MV && uxy_obs[i, j] != 0.0
            uxy_err[i, j] = uxy[i, j] - uxy_obs[i, j]
        end
    end

    H_err = H_ice .- H_obs

    fill!(cb_ref, OPT_MV)

    @inbounds for j in 1:ny, i in 1:nx
        im1, ip1 = max(i - 1, 1), min(i + 1, nx)
        jm1, jp1 = max(j - 1, 1), min(j + 1, ny)

        ux_aa = 0.5 * (ux[i, j] + ux[im1, j])
        uy_aa = 0.5 * (uy[i, j] + uy[i, jm1])

        (uxy[i, j] != 0.0 && H_grnd_obs[i, j] > 0.0) || continue

        i1 = ux_aa >= 0.0 ? im1 : ip1
        j1 = uy_aa >= 0.0 ? jm1 : jp1

        if uxy_err[i, j] == OPT_MV
            xwt, ywt = 0.5, 0.5
        else
            xywt = abs(ux_aa) + abs(uy_aa)
            if xywt > 0.0
                xwt, ywt = abs(ux_aa) / xywt, abs(uy_aa) / xywt
            else
                xwt, ywt = 0.5, 0.5
            end
        end

        H_err_now = xwt * H_err[i1, j] + ywt * H_err[i, j1]
        dHdt_now  = xwt * dHdt[i1, j]  + ywt * dHdt[i, j1]

        cb_tgt_fac = cb_tgt === nothing ? 0.0 :
            log((cb_prev[i, j] + eps) / (cb_tgt[i, j] + eps))

        scale = scaleH ? cb_prev[i, j] / max(50.0, H_obs[i, j]) : cb_prev[i, j] / H0
        cb_ref_dot = -scale * (H_err_now / tau_c + f_damp * dHdt_now + (f_tgt / tau_tgt) * cb_tgt_fac)

        cb_ref[i, j] = cb_prev[i, j] + cb_ref_dot * dt
    end

    @inbounds for j in 1:ny, i in 1:nx
        if H_grnd_obs[i, j] <= 0.0 || H_obs[i, j] == 0.0
            cb_ref[i, j] = fill_method == "target" ? cb_tgt[i, j] : _opt_at(cf_min, i, j)
        end
    end

    @inbounds for j in 1:ny, i in 1:nx
        lo, hi = _opt_at(cf_min, i, j), _opt_at(cf_max, i, j)
        cb_ref[i, j] < lo && (cb_ref[i, j] = lo)
        cb_ref[i, j] > hi && (cb_ref[i, j] = hi)
    end

    return cb_ref
end
