# ----------------------------------------------------------------------
# Shared utilities for Yelmo.jl. Lives at the top level (not inside
# any per-phase sub-module) so any phase (`thrm`, `mat`, `dyn`,
# `topo`) can depend on it without an inter-module layering smell.
# Mirrors the `YelmoSolvers` (`dyn/solvers.jl`) and `YelmoIntegration`
# (`integration.jl`) patterns.
#
# Currently provides:
#
#   - `solve_tridiag!` — Thomas algorithm tridiagonal solver. Used by
#     every column-wise thermodynamics solver (temp, enthalpy,
#     bedrock, and eventually tracers).
#   - `_neighbor_*`, `_ip1_modular`, `_jp1_modular` — topology-aware
#     index helpers used by face-staggered kernels in `dyn` and
#     `thrm` (basal heating, advection).
#   - `gq2d_nodes`, `gq2d_nodes_2pt`, `gq2d_interp_to_node`,
#     `gq2d_shape_functions` — 2D Gauss-Legendre quadrature on the
#     reference square. Used by the SSA basal-drag kernels and the
#     thrm `qb_method = 2` basal-heating kernel.
#
# The kernels here take caller-supplied scratch buffers where
# applicable so the hot loops in 3D wrappers stay alloc-free.
# ----------------------------------------------------------------------

module YelmoUtils

using NCDatasets

export solve_tridiag!
export calc_dzeta_terms!
export gq2d_nodes, gq2d_nodes_2pt, gq2d_interp_to_node, gq2d_shape_functions
export gen_map_filename, map_scrip_load, map_scrip_field, vec_stat
export fill_weighted!, fill_nearest!, gaussian_filter!
export GridScaleWeights, map_field_to_lo, map_field_to_lo!,
       map_field_to_hi, map_field_to_hi!,
       refine_grid, coarsen_grid
export remove_englacial_lakes!, smooth_gauss_2D!, adjust_topography_gradients!

include("tridiag.jl")
include("topology.jl")
include("quadrature.jl")
include("dzeta.jl")
include("scrip_map.jl")
include("grid_scale.jl")
include("topography_helpers.jl")

end # module YelmoUtils
