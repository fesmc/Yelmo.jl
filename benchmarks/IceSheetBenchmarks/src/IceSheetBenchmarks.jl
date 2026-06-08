module IceSheetBenchmarks

using NCDatasets

# ----------------------------------------------------------------------
# AbstractBenchmark — interface contract.
#
# Concrete subtypes carry the parameters of a benchmark (grid axes,
# forcing parameters, Glen-flow parameters, …) and must implement:
#
#   - `state(b, t)` — analytical state at time `t`, as a NamedTuple
#     keyed by ice-sheet field names (`:xc`, `:yc`, `:H_ice`, `:z_bed`,
#     `:smb_ref`, `:T_srf`, `:Q_geo`, `:z_sl`, ...).
#   - `write_fixture!(b, path; times)` — serialize the analytical state
#     at one or more times to a NetCDF restart file.
#   - `analytical_velocity(b, t)` (optional) — closed-form
#     depth-averaged ice-velocity field `(ux_bar, uy_bar)`, when one
#     exists (e.g. Halfar dome). Benchmarks without an analytical
#     velocity solution leave this unimplemented.
# ----------------------------------------------------------------------

abstract type AbstractBenchmark end

function state end           # state(b::AbstractBenchmark, t::Real) -> NamedTuple
function write_fixture! end  # write_fixture!(b, path; times=[t]) -> Vector{String}
function analytical_velocity end

analytical_velocity(b::AbstractBenchmark, t::Real) = error(
    "analytical_velocity not implemented for $(typeof(b)). " *
    "Use a concrete benchmark subtype with a closed-form velocity solution.")

# Calving-law hooks. The model-agnostic skeleton is declared here so the
# `YelmoBenchmarks` package extension can extend them; a non-Yelmo host
# can extend them with array-only overloads.
function calvmip_exp1! end
function calvmip_exp2! end

export AbstractBenchmark
export state, write_fixture!, analytical_velocity
export calvmip_exp1!, calvmip_exp2!

include("eismint_moving.jl")
include("mismip3d.jl")
include("calvingmip.jl")
include("initmip.jl")

export EISMINT1MovingBenchmark, eismint_moving_smb
export MISMIP3DBenchmark
export CalvingMIPBenchmark, calvmip_bed_circular, calvmip_bed_thule
export InitMIPBenchmark

end # module
