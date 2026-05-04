module Yelmo

# Sub-modules
include("YelmoMeta.jl")
include("YelmoConst.jl")
include("YelmoPar.jl")
include("dyn/solvers.jl")
include("integration.jl")
include("YelmoModelPar.jl")
include("YelmoCore.jl")
#include("YelmoMirrorCoreMatrices.jl")
include("YelmoMirrorCoreFields.jl")
include("topo/YelmoModelTopo.jl")
include("dyn/YelmoModelDyn.jl")
include("mat/YelmoModelMat.jl")
# Adaptive timestepping (predictor-corrector): must be loaded AFTER
# topo + dyn + mat modules since it calls `topo_step!`, `mat_step!`,
# `dyn_step!`, and `update_diagnostics!`. Methods land into the
# top-level `Yelmo` namespace; `step!` (defined in YelmoCore.jl)
# dispatches into them at runtime via `_select_step!`.
include("timestepping.jl")
include("YelmoIO.jl")

using .YelmoMeta
using .YelmoConst
using .YelmoPar
using .YelmoSolvers
using .YelmoIntegration
using .YelmoModelPar
using .YelmoCore
using .YelmoMirrorCore
using .YelmoModelTopo
using .YelmoModelDyn
using .YelmoModelMat
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

# YelmoSolvers
export Solver, SSASolver

# YelmoIntegration
export vert_int_trapz_boundary!

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
export advect_tracer_upwind_explicit!, advect_tracer_upwind_implicit!
export ImplicitAdvectionCache, init_advection_cache,
       update_advection_matrix!, update_advection_operator!,
       solve_advection!
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
export calc_beta!, stagger_beta!
export calc_ice_flux!, calc_magnitude_from_staggered!, calc_vel_ratio!
export calc_shear_stress_3D!, calc_uxy_sia_3D!, calc_velocity_sia!
export gq2d_nodes
export calc_visc_eff_3D_aa!, calc_visc_eff_3D_nodes!, calc_visc_eff_int!
export stagger_visc_aa_ab!
export calc_jacobian_vel_3D_uxyterms!
export calc_jacobian_vel_3D_uzterms!
export calc_strain_rate_tensor_jac_quad3D!
export calc_uz_3D_jac!, calc_uz_3D!, calc_uz_3D_aa!
export set_ssa_masks!
export picard_relax_visc!, picard_relax_vel!
export picard_calc_convergence_l2, picard_calc_convergence_l1rel_matrix!
export set_inactive_margins!, calc_basal_stress!

# YelmoModelMat
export mat_step!

# YelmoIO
export init_output
export OutputSelection
export write_output!

# Adaptive timestepping (timestepping.jl)
export PCScheme, HEUN, FE_SBE, AB_SAM
export PIController, PI42

end # module
