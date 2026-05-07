# ----------------------------------------------------------------------
# YelmoHooks — per-model slots for user-supplied calving-rate functions.
#
# A hook, when set, is called by `calving_step!` *instead of* the
# method-string dispatch (`_dispatch_calving!`), letting users supply
# experiment-specific or prototype calving laws from outside Yelmo.jl
# without adding new method strings to the core.
#
# Usage:
#
#   y = YelmoModel(...)
#   y.hooks.calv_flt = (cr_x, cr_y, u_bar, v_bar, H_ice, f_ice, lsf, t) -> begin
#       calvmip_exp1!(cr_x, cr_y, u_bar, v_bar, xc, yc)   # captured xc, yc
#   end
#
# Hook signature (both slots):
#
#   f(cr_x, cr_y, u_bar, v_bar, H_ice, f_ice, lsf, time::Float64) -> nothing
#
# where:
#   cr_x, cr_y  — XFaceField / YFaceField to fill in-place with the
#                 calving rate (m/yr, signed opposite to ice flow)
#   u_bar, v_bar — depth-mean velocity (XFaceField / YFaceField)
#   H_ice       — ice thickness (CenterField, aa-nodes)
#   f_ice       — ice fraction   (CenterField, aa-nodes)
#   lsf         — level-set function φ (CenterField; φ<0 ice, φ>0 ocean)
#   time        — current model time (yr)
#
# Additional inputs (grid spacing, experiment parameters, …) should be
# captured via closure. See `test/benchmarks/calvingmip.jl` for an
# example.
#
# When a hook is set, the corresponding `calv_flt_method` /
# `calv_grnd_method` namelist string is ignored for that direction.
# Set `calv_flt_method = "custom"` in the namelist so the parameter
# validator does not reject the model; the actual law comes from the
# hook.
# ----------------------------------------------------------------------

module YelmoHooks

export YelmoHooks

"""
    YelmoHooks

Mutable container for user-supplied calving-rate hooks. Attach to a
`YelmoModel` via `y.hooks.calv_flt = f` after construction.

See `src/YelmoHooks.jl` for the hook function signature.
"""
mutable struct YelmoHooks
    calv_flt  ::Union{Nothing, Function}
    calv_grnd ::Union{Nothing, Function}
    YelmoHooks() = new(nothing, nothing)
end

end # module YelmoHooks
