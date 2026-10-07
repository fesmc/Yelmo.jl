# ----------------------------------------------------------------------
# Ice-thickness fields exclusively for the dynamics solver.
#
# `tpo.H_ice_dyn` / `tpo.f_ice_dyn` are the thickness / cover fields
# handed to the SSA / DIVA / L1L2 dynamics step: the in-cell effective
# thickness `H_ice / f_ice` at partially covered margin cells, with a
# binary cover. Yelmo.jl-specific (v1.15 Fortran passed `H_ice` through);
# yelmo dev builds `H_ice_dyn = max(H_eff, H_ice)` from its subgrid front
# scheme (`ytopo.front_subgrid`), which will replace this when ported.
# The v1.15 `ssa_lat_bc = "slab"/"slab-ext"` modes (and
# `extend_floating_slab`) were removed in yelmo dev.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calc_dynamic_ice_fields!

"""
    calc_dynamic_ice_fields!(H_ice_dyn, f_ice_dyn, H_ice, f_ice) -> H_ice_dyn

Build the dynamics-only thickness `H_ice_dyn` and cover `f_ice_dyn`:
`H_ice_dyn = H_ice / f_ice` at sub-grid margin cells where
`0 < f_ice < 1`, and `f_ice_dyn` binary (`1.0` for any `f_ice > 0`). At
fully-covered cells (`f_ice ≥ 1`) `H_ice_dyn = H_ice` and
`f_ice_dyn = 1.0`. At ice-free cells (`f_ice = 0`) both are zero. This
carries the *effective* (in-cell) thickness into the dynamics matrix and
extends the binary cover to the whole partial cell, so the velocity
field is well-defined out to the prognostic margin — important for
sub-stepping schemes like `dt_method = 3` where the velocity is frozen
within an outer step while the front advances.
"""
function calc_dynamic_ice_fields!(H_ice_dyn, f_ice_dyn, H_ice, f_ice)
    Hd = interior(H_ice_dyn)
    Fd = interior(f_ice_dyn)
    H  = interior(H_ice)
    F  = interior(f_ice)

    @inbounds for j in axes(Hd, 2), i in axes(Hd, 1)
        f = F[i, j, 1]
        h = H[i, j, 1]
        if f <= 0.0
            Hd[i, j, 1] = 0.0
            Fd[i, j, 1] = 0.0
        elseif f >= 1.0
            Hd[i, j, 1] = h
            Fd[i, j, 1] = 1.0
        else
            # 0 < f < 1: in-cell effective thickness, binary cover.
            Hd[i, j, 1] = h / f
            Fd[i, j, 1] = 1.0
        end
    end

    return H_ice_dyn
end

calc_dynamic_ice_fields!(y::YelmoModel) =
    calc_dynamic_ice_fields!(y.tpo.H_ice_dyn, y.tpo.f_ice_dyn,
                             y.tpo.H_ice, y.tpo.f_ice)
