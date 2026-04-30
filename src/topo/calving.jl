# ----------------------------------------------------------------------
# Calving driver — phase 7 of `topo_step!`.
#
# `calving_step!(y, dt)` orchestrates the full level-set calving phase:
#
#   1. Snapshot `lsf_n ← lsf` (for `dlsfdt`).
#   2. Refresh grounding fractions on aa/acx/acy nodes.
#   3. Dispatch `calv_flt_method`  → `cmb_flt_acx, cmb_flt_acy`.
#   4. Dispatch `calv_grnd_method` → `cmb_grnd_acx, cmb_grnd_acy`.
#   5. `merge_calving_rates!` → `cr_acx, cr_acy`.
#   6. `lsf_update!` advances `lsf` at `w = u_bar + cr`.
#   7. Above-SL pin: `lsf = -1` over land.
#   8. Periodic `lsf_redistance!` per `ycalv.dt_lsf`.
#   9. Build kill-rate `cmb = -H/dt` where `lsf > 0`; populate the
#      aa-stagger magnitude diagnostics `cmb_flt`, `cmb_grnd`.
#   10. `apply_tendency!(H, cmb, dt; adjust_mb=true)`; refresh `f_ice`.
#   11. Post-kill consistency: where `H ≤ 0` and bed is below SL, set
#       `lsf = 1` (gated off for `calv_flt_method == "equil"` so the
#       front stays pinned in equilibrium runs).
#   12. `dlsfdt = (lsf - lsf_n) / dt`.
#
# Dispatch covers four laws: `"none"`/`"zero"`, `"equil"`,
# `"threshold"`, and `"vm-m16"`. The last errors at call time because
# it needs `mat`'s 1st principal stress, not yet ported. The aa-form
# (vertical mb) calving path from Fortran is not ported.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calving_step!

# Decide whether redistancing fires on this step.
#   dt_lsf < 0  → every step
#   dt_lsf == 0 → never
#   dt_lsf > 0  → fire when [time, time+dt] crosses an integer multiple
#                 of dt_lsf. Robust to non-integer dt.
@inline function _redist_trigger(time::Real, dt::Real, dt_lsf::Real)
    dt_lsf < 0 && return true
    dt_lsf == 0 && return false
    return floor((time + dt) / dt_lsf) > floor(time / dt_lsf)
end

# Dispatch table for one direction (floating or grounded).
function _dispatch_calving!(cr_x, cr_y, method::AbstractString,
                            u_bar, v_bar, H_ice, f_ice,
                            Hc::Real, tau_ice::Real,
                            tag::AbstractString)
    fill!(interior(cr_x), 0.0)
    fill!(interior(cr_y), 0.0)
    if method == "zero" || method == "none"
        return cr_x, cr_y
    elseif method == "equil"
        return calc_calving_equil_ac!(cr_x, cr_y, u_bar, v_bar)
    elseif method == "threshold"
        return calc_calving_threshold_ac!(cr_x, cr_y, u_bar, v_bar,
                                          H_ice, f_ice, Hc)
    elseif method == "vm-m16"
        return calc_calving_vonmises_m16_ac!(cr_x, cr_y, u_bar, v_bar,
                                             nothing, f_ice, tau_ice)
    else
        error("calving_step!: unknown $tag = \"$method\". " *
              "Supported: \"none\"/\"zero\", \"equil\", \"threshold\", \"vm-m16\".")
    end
end

"""
    calving_step!(y::YelmoModel, dt) -> y

Phase-7 calving driver. Runs only when `y.p.ycalv.use_lsf == true`;
otherwise leaves `cmb`, `lsf`, and the calving rate diagnostics at
their entering values (zeros if just allocated).

See the file header for the full pipeline. Mutates the entire ytopo
calving group plus `H_grnd`, `f_grnd`, `f_grnd_acx`, `f_grnd_acy`
(refreshed at the start of the step) and `f_ice` (after the kill).
Reads `dyn.ux_bar`, `dyn.uy_bar`, `bnd.z_bed`, `bnd.z_sl`.
"""
function calving_step!(y::YelmoModel, dt::Float64)
    # Always zero the calving outputs on entry — restart files can carry
    # stale `cmb` from a previous run, and a no-op calving step must
    # contribute zero to `mb_net`.
    fill!(interior(y.tpo.cmb),       0.0)
    fill!(interior(y.tpo.cmb_flt),   0.0)
    fill!(interior(y.tpo.cmb_grnd),  0.0)
    fill!(interior(y.tpo.dlsfdt),    0.0)

    y.p.ycalv.use_lsf || return y

    # 1. Snapshot lsf_n.
    interior(y.tpo.lsf_n) .= interior(y.tpo.lsf)

    # 2. Refresh grounding fractions including ac-stagger.
    calc_H_grnd!(y.tpo.H_grnd, y.tpo.H_ice, y.bnd.z_bed, y.bnd.z_sl,
                 y.c.rho_ice, y.c.rho_sw)
    determine_grounded_fractions!(y.tpo.f_grnd, y.tpo.H_grnd;
                                  f_grnd_acx = y.tpo.f_grnd_acx,
                                  f_grnd_acy = y.tpo.f_grnd_acy)

    # 3 + 4. Per-direction calving rates.
    _dispatch_calving!(y.tpo.cmb_flt_acx,  y.tpo.cmb_flt_acy,
                       y.p.ycalv.calv_flt_method,
                       y.dyn.ux_bar, y.dyn.uy_bar,
                       y.tpo.H_ice, y.tpo.f_ice,
                       y.p.ycalv.Hc_ref_flt, y.p.ycalv.tau_ice,
                       "calv_flt_method")
    _dispatch_calving!(y.tpo.cmb_grnd_acx, y.tpo.cmb_grnd_acy,
                       y.p.ycalv.calv_grnd_method,
                       y.dyn.ux_bar, y.dyn.uy_bar,
                       y.tpo.H_ice, y.tpo.f_ice,
                       y.p.ycalv.Hc_ref_grnd, y.p.ycalv.tau_ice,
                       "calv_grnd_method")

    # 5. Merge floating + grounded into single front velocity.
    merge_calving_rates!(y.tpo.cr_acx, y.tpo.cr_acy,
                         y.tpo.cmb_flt_acx,  y.tpo.cmb_flt_acy,
                         y.tpo.cmb_grnd_acx, y.tpo.cmb_grnd_acy,
                         y.dyn.ux_bar, y.dyn.uy_bar,
                         y.tpo.f_grnd_acx, y.tpo.f_grnd_acy,
                         y.bnd.z_bed, y.bnd.z_sl)

    # 6. Advect lsf at w = u_bar + cr.
    lsf_update!(y.tpo.lsf,
                y.dyn.ux_bar, y.dyn.uy_bar,
                y.tpo.cr_acx, y.tpo.cr_acy, dt;
                cfl_safety = y.p.yelmo.cfl_max)

    # 7. Above-SL pin.
    L  = interior(y.tpo.lsf)
    Zb = interior(y.bnd.z_bed)
    Zs = interior(y.bnd.z_sl)
    @inbounds for j in axes(L, 2), i in axes(L, 1)
        if Zb[i, j, 1] > Zs[i, j, 1]
            L[i, j, 1] = -1.0
        end
    end

    # 8. Optional redistancing.
    if _redist_trigger(y.time, dt, y.p.ycalv.dt_lsf)
        lsf_redistance!(y.tpo.lsf, _dx(y.g), _dy(y.g))
    end

    # 9. Kill-rate cmb + aa-stagger magnitude diagnostics.
    H  = interior(y.tpo.H_ice)
    C  = interior(y.tpo.cmb)
    Cf = interior(y.tpo.cmb_flt)
    Cg = interior(y.tpo.cmb_grnd)
    Fx = interior(y.tpo.cmb_flt_acx)
    Fy = interior(y.tpo.cmb_flt_acy)
    Gx = interior(y.tpo.cmb_grnd_acx)
    Gy = interior(y.tpo.cmb_grnd_acy)
    inv_dt = dt > 0 ? 1.0 / dt : 0.0
    @inbounds for j in axes(C, 2), i in axes(C, 1)
        # Kill rate where lsf > 0 (cell has retreated past the front).
        C[i, j, 1] = (L[i, j, 1] > 0.0) ? -H[i, j, 1] * inv_dt : 0.0

        # aa-stagger magnitude of (cmb_flt_acx, cmb_flt_acy).
        # XFace face indices around aa-cell (i,j) are i and i+1; YFace
        # face indices around aa-cell (i,j) are j and j+1.
        fx = 0.5 * (Fx[i, j, 1] + Fx[i + 1, j, 1])
        fy = 0.5 * (Fy[i, j, 1] + Fy[i, j + 1, 1])
        Cf[i, j, 1] = sqrt(fx * fx + fy * fy)

        gx = 0.5 * (Gx[i, j, 1] + Gx[i + 1, j, 1])
        gy = 0.5 * (Gy[i, j, 1] + Gy[i, j + 1, 1])
        Cg[i, j, 1] = sqrt(gx * gx + gy * gy)
    end

    # 10. Apply kill, refresh f_ice.
    apply_tendency!(y.tpo.H_ice, y.tpo.cmb, dt; adjust_mb = true)
    calc_f_ice!(y)

    # 11. Post-kill consistency: H ≤ 0 over below-SL ocean cells flips
    # lsf to ocean. Skipped for "equil" so the front stays pinned even
    # if external forcing thins ice transiently.
    if y.p.ycalv.calv_flt_method != "equil"
        @inbounds for j in axes(L, 2), i in axes(L, 1)
            if H[i, j, 1] <= 0.0 && L[i, j, 1] < 0.0 && Zb[i, j, 1] < Zs[i, j, 1]
                L[i, j, 1] = 1.0
            end
        end
    end

    # 12. dlsfdt diagnostic.
    Ln = interior(y.tpo.lsf_n)
    D  = interior(y.tpo.dlsfdt)
    @inbounds for k in eachindex(D)
        D[k] = inv_dt * (L[k] - Ln[k])
    end

    return y
end
