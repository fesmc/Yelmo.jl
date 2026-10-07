# ----------------------------------------------------------------------
# Ice-area-fraction calculation `f_ice`.
#
# Binary cover: `f_ice = 1` where `H_ice > 0`, else 0 — Fortran's
# `calc_ice_fraction` with `ytopo.front_subgrid = "none"`. The v1.15
# floating-margin sub-grid option (`ytopo.margin_flt_subgrid`) was
# replaced in yelmo dev by the subgrid front scheme (`front_subgrid =
# "floating"/"marine"`, not ported yet; `check_ported` rejects it).
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calc_f_ice!

"""
    calc_f_ice!(f_ice, H_ice) -> f_ice

Binary ice-area fraction: `f_ice = 1` where `H_ice > 0`, else 0
(Fortran `calc_ice_fraction` with `front_subgrid = "none"`).
"""
function calc_f_ice!(f_ice, H_ice)
    F = interior(f_ice)
    H = interior(H_ice)
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        F[i, j, 1] = H[i, j, 1] > 0.0 ? 1.0 : 0.0
    end
    return f_ice
end

# Convenience dispatch: refresh `tpo.f_ice` from a `YelmoModel`'s
# current state. Used inside `topo_step!` / `calving_step!`.
calc_f_ice!(y::YelmoModel) = calc_f_ice!(y.tpo.f_ice, y.tpo.H_ice)

