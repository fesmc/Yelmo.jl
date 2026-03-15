module Yelmo

# Sub-modules
include("YelmoMeta.jl")
include("YelmoCore.jl")
include("YelmoIO.jl")

using .YelmoMeta
using .YelmoCore
using .YelmoIO

# Re-export the public API

# YelmoMeta
export VariableMeta, parse_variable_table

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

end # module
