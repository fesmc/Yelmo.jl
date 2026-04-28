module Yelmo

# Sub-modules
include("YelmoMeta.jl")
include("YelmoPar.jl")
include("YelmoModelPar.jl")
include("YelmoCore.jl")
#include("YelmoMirrorCoreMatrices.jl")
include("YelmoMirrorCoreFields.jl")
include("topo/YelmoModelTopo.jl")
include("YelmoIO.jl")

using .YelmoMeta
using .YelmoPar
using .YelmoModelPar
using .YelmoCore
using .YelmoMirrorCore
using .YelmoModelTopo
using .YelmoIO

# Re-export the public API at the package level

# YelmoMeta
export VariableMeta, parse_variable_table

# YelmoPar
export YelmoParameters
export yelmo_params, ytopo_params, ycalv_params, ydyn_params,
       ytill_params, yneff_params, ymat_params, ytherm_params,
       yelmo_masks_params, yelmo_init_topo_params, yelmo_data_params,
       phys_params, earth_params
export write_nml
export read_nml
export compare

# YelmoModelPar
export YelmoModelParameters

# YelmoCore
export AbstractYelmoModel, YelmoModel
export init_state!, step!, load_state!
export load_grids_from_restart, load_fields_from_restart
export load_field_from_dataset_2D, load_field_from_dataset_3D
export make_field, matches_patterns, yelmo_define_grids
export XFACE_VARIABLES, YFACE_VARIABLES, ZFACE_VARIABLES, VERTICAL_DIMS
export MASK_ICE_NONE, MASK_ICE_FIXED, MASK_ICE_DYNAMIC
export compare_state, StateComparison

# YelmoMirrorCore
export YelmoMirror, yelmo_sync!  # Public API
export yelmo_get_var2D, yelmo_get_var2D!    # Mainly internally used
export yelmo_get_var3D, yelmo_get_var3D!    # Mainly internally used
export yelmo_set_var2D!, yelmo_set_var3D!   # Mainly internally used

# YelmoModelTopo
export topo_step!

# YelmoIO
export init_output
export OutputSelection
export write_output!

end # module
