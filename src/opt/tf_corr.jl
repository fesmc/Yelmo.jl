"""
    optimize_tf_corr!(tf_corr, H_ice, H_grnd, dHicedt, H_obs, H_grnd_lim,
                       dx, tau_m, m_temp, tf_min, tf_max, dt;
                       sigma=0.0, basins=nothing, basin_fill=false) -> tf_corr

In-place ocean thermal-forcing correction, nudged so the modelled
floating-ice thickness relaxes toward `H_obs` (paired with
`optimize_cb_ref!` for the grounded-ice friction coefficient — both are
driven together in the colleague's ISMIP7 `opt_ant.sh`, `opt.opt_tf`).

`tf_corr` is not a field yelmo itself carries (this refactored fork has
no `tf_corr` state at all — the classic driver applied it directly to
the ocean thermal-forcing boundary condition it constructed each step).
The caller owns the `tf_corr` array across iterations and is
responsible for folding it into whatever feeds `bnd_T_shlf` (or the
melt scheme's ocean-temperature input) before the next `yelmo_step!`.

With `basin_fill=true`, `tf_corr` is additionally basin-averaged: for
each basin id in `1:maximum(basins)`, cells near the grounding line
(`H_grnd < H_grnd_lim`, with an observed or modeled ice presence) set
the basin's mean `tf_corr`, which is then broadcast onto that basin's
open-ocean cells (`H_grnd > H_grnd_lim`) — propagating the
near-grounding-line correction into the interior shelf. `basins` is
required when `basin_fill=true`.

Port of `optimize_tf_corr` in `ice_optimization.f90` (`optimize_tf_corr_basin`,
a variant not used by any config here, is not ported; `H_grnd_obs` is
dropped from the Fortran signature — it's declared but never read there).
"""
function optimize_tf_corr!(tf_corr::AbstractMatrix, H_ice::AbstractMatrix, H_grnd::AbstractMatrix,
                            dHicedt::AbstractMatrix, H_obs::AbstractMatrix,
                            H_grnd_lim::Real, dx::Real, tau_m::Real, m_temp::Real,
                            tf_min::Real, tf_max::Real, dt::Real;
                            sigma::Real=0.0,
                            basins::Union{Nothing,AbstractMatrix}=nothing,
                            basin_fill::Bool=false)

    basin_fill && basins === nothing &&
        error("optimize_tf_corr!: basin_fill=true requires basins.")

    nx, ny = size(tf_corr)
    f_damp      = 2.0
    tau_tgt     = 500.0
    tf_corr_tgt = 0.0

    H_err = H_ice .- H_obs
    sigma > 0.0 && gaussian_filter!(H_err; sigma=Float64(sigma), dx=Float64(dx))

    @inbounds for j in 1:ny, i in 1:nx
        H_grnd[i, j] > H_grnd_lim && (H_err[i, j] = 0.0)
    end

    @inbounds for j in 1:ny, i in 1:nx
        tf_corr_dot = 1.0 / (tau_m * m_temp) * (H_err[i, j] / tau_m + f_damp * dHicedt[i, j]) -
                      (1.0 / tau_tgt) * (tf_corr[i, j] - tf_corr_tgt)
        tf_corr[i, j] += tf_corr_dot * dt
        tf_corr[i, j] < tf_min && (tf_corr[i, j] = tf_min)
        tf_corr[i, j] > tf_max && (tf_corr[i, j] = tf_max)
    end

    if basin_fill
        nb = Int(maximum(basins))
        tol = 1.0e-5
        for b in 1:nb
            mask = (abs.(basins .- b) .< tol) .& (H_grnd .< H_grnd_lim) .& ((H_obs .> 0.0) .| (H_ice .> 0.0))
            n = count(mask)
            tf_corr_bar = n > 0 ? sum(tf_corr[mask]) / n : 0.0

            @inbounds for j in 1:ny, i in 1:nx
                if H_grnd[i, j] > H_grnd_lim && abs(basins[i, j] - b) < tol
                    tf_corr[i, j] = tf_corr_bar
                end
            end
        end
    end

    return tf_corr
end
