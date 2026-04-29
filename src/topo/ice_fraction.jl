# ----------------------------------------------------------------------
# Ice-area-fraction calculation `f_ice`.
#
# Milestone 2b uses a binary stub: `f_ice = 1` wherever `H_ice > 0`,
# else `0`. The schema field already supports fractional values, so a
# later milestone can replace the kernel without touching call sites.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calc_f_ice!

"""
    calc_f_ice!(f_ice, H_ice) -> f_ice

Set `f_ice[i,j] = 1.0` wherever `H_ice[i,j] > 0`, else `0.0`.

Binary stub for milestone 2b. The fractional Yelmo-Fortran version
(which derives `f_ice` from sub-grid pinning, grounding-line geometry,
and effective-thickness considerations) replaces this kernel in a
later milestone — the `tpo.f_ice` schema field already supports
fractional values, so wiring downstream of this function does not
need to change.
"""
function calc_f_ice!(f_ice, H_ice)
    F = interior(f_ice)
    H = interior(H_ice)
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        F[i, j, 1] = H[i, j, 1] > 0.0 ? 1.0 : 0.0
    end
    return f_ice
end
