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
# ----------------------------------------------------------------------

abstract type AbstractBenchmark end

function state end           # state(b::AbstractBenchmark, t::Real) -> NamedTuple
function write_fixture! end  # write_fixture!(b, path; times=[t]) -> Vector{String}

export AbstractBenchmark
export state, write_fixture!

include("eismint_moving.jl")
include("mismip3d.jl")

export EISMINT1MovingBenchmark, eismint_moving_smb
export MISMIP3DBenchmark

end # module
