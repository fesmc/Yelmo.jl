"""
    YelmoOpt

Spin-up friction-coefficient / thermal-forcing optimization, ported from
`libs/ice_optimization.f90` in the Fortran `yelmo` repo. This is
driver-level glue (not part of `yelmo_step!`'s fixed phase order): the
Fortran side only ever ran it from a standalone driver program
(`yelmo_opt.x` / the classic `yelmox_esm`), never from `yelmo_update`
itself, so it's ported here the same way — plain functions over 2D
arrays that a spin-up script calls once per iteration alongside
`yelmo_step!`/`yelmo_sync!`, not a method dispatched from `step!`.

Reusable by any spin-up (native `YelmoModel` or `YelmoMirror`), not
specific to hydrology coupling — see `Kryonomos.jl/examples/fasthydrology/`
for a driving loop that wires this up against real forcing.
"""
module YelmoOpt

using ..YelmoUtils: gaussian_filter!

export optimize_transient_param
export calc_magnitude_from_staggered_ice, optimize_cb_ref!
export optimize_tf_corr!

include("transient_param.jl")
include("cb_ref.jl")
include("tf_corr.jl")

end # module YelmoOpt
