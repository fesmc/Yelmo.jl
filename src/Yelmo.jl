module Yelmo

# Sub-modules
include("YelmoMeta.jl")
include("YelmoConst.jl")
include("YelmoPar.jl")
include("YelmoModelPar.jl")
include("YelmoCore.jl")
#include("YelmoMirrorCoreMatrices.jl")
include("YelmoMirrorCoreFields.jl")
include("topo/YelmoModelTopo.jl")
include("dyn/YelmoModelDyn.jl")
include("YelmoIO.jl")

using .YelmoMeta
using .YelmoConst
using .YelmoPar
using .YelmoModelPar
using .YelmoCore
using .YelmoMirrorCore
using .YelmoModelTopo
using .YelmoModelDyn
using .YelmoIO

# Re-export the public API at the package level

# YelmoMeta
export VariableMeta, parse_variable_table

# YelmoConst
export YelmoConstants, yelmo_constants, earth_constants,
       eismint_constants, mismip3d_constants, trough_constants
# (MASK_ICE_* are also exported from YelmoCore for back-compat — see below.)

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
export MASK_BED_OCEAN, MASK_BED_LAND, MASK_BED_FROZEN, MASK_BED_STREAM,
       MASK_BED_GRLINE, MASK_BED_FLOAT, MASK_BED_ISLAND, MASK_BED_PARTIAL
export compare_state, StateComparison

# YelmoMirrorCore
export YelmoMirror, yelmo_sync!, yelmo_write_restart!  # Public API
export yelmo_get_var2D, yelmo_get_var2D!    # Mainly internally used
export yelmo_get_var3D, yelmo_get_var3D!    # Mainly internally used
export yelmo_set_var2D!, yelmo_set_var3D!   # Mainly internally used

# YelmoModelTopo
export topo_step!, advect_tracer!
export apply_tendency!, mbal_tendency!, resid_tendency!, calc_f_ice!
export calc_H_grnd!, determine_grounded_fractions!
export calc_bmb_total!, calc_fmb_total!, calc_mb_discharge!
export set_tau_relax!, calc_G_relaxation!
export calc_calving_equil_ac!, calc_calving_threshold_ac!,
       calc_calving_vonmises_m16_ac!, merge_calving_rates!
export lsf_init!, lsf_update!, lsf_redistance!,
       extrapolate_ocn_acx!, extrapolate_ocn_acy!
export calving_step!
export calc_distance_to_grounding_line!, calc_distance_to_ice_margin!,
       calc_grounding_line_zone!, gen_mask_bed!, calc_ice_front!,
       calc_z_srf!
export calc_gradient_acx!, calc_gradient_acy!
export calc_f_grnd_subgrid_linear!, calc_f_grnd_subgrid_area!,
       calc_f_grnd_pinning_points!, calc_grounded_fractions!
export extend_floating_slab!, calc_dynamic_ice_fields!
export update_diagnostics!

# YelmoModelDyn
export dyn_step!
export calc_driving_stress!, calc_driving_stress_gl!
export calc_lateral_bc_stress_2D!
export calc_ydyn_neff!
export calc_cb_ref!, calc_c_bed!
export calc_ice_flux!, calc_magnitude_from_staggered!, calc_vel_ratio!
export calc_shear_stress_3D!, calc_uxy_sia_3D!

# YelmoIO
export init_output
export OutputSelection
export write_output!

end # module
