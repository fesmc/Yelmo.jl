module Yelmo

# Sub-modules
include("YelmoMeta.jl")
include("YelmoPar.jl")
#include("YelmoCore.jl")
include("YelmoCoreFields.jl")
include("YelmoIO.jl")

using .YelmoMeta
using .YelmoPar
using .YelmoCore
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

# YelmoCore
export YelmoMirror, init_state!, time_step!, sync!  # Public API
export yelmo_get_var2D, yelmo_get_var2D!    # Mainly internally used
export yelmo_get_var3D, yelmo_get_var3D!    # Mainly internally used
export yelmo_set_var2D!, yelmo_set_var3D!   # Mainly internally used

# YelmoIO
export load_grids_from_restart
export load_field_from_dataset_2D
export load_field_from_dataset_3D
export load_fields_from_restart
export init_output
export OutputSelection
export write_output!

end # module
