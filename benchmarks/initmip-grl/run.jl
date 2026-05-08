# benchmarks/initmip-grl/run.jl
#
# Main run script for the initmip-grl benchmark.
#
# Flags are defined at the top of this file; edit them in place rather
# than passing CLI arguments. Outputs go to `output/` (gitignored).

using IceSheetBenchmarks
using Yelmo                # loading both activates the YelmoBenchmarks extension
using NCDatasets
using Statistics
using Printf

# ----------------------------------------------------------------------
# Configuration (edit in place).
# ----------------------------------------------------------------------

const T_END_YR    = 5000.0       # total simulated time [yr]
const DT_OUTER_YR = 100.0        # outer-loop dt [yr]; adaptive PC sub-steps inside
const SAMPLE_DT_YR = 100.0       # cadence for time-series snapshots [yr]

const OUTPUT_DIR = abspath(joinpath(@__DIR__, "output"))

# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------

function main()
    mkpath(OUTPUT_DIR)
    @info "initmip-grl benchmark — t_end = $(T_END_YR) yr, dt_outer = $(DT_OUTER_YR) yr"

    # TODO: pick a benchmark spec from IceSheetBenchmarks (or define a
    # new one inline below), build parameters + YelmoModel, run the
    # trajectory, and write outputs to OUTPUT_DIR.
    #
    # See benchmarks/eismint1-moving/run.jl for a worked example.

    @warn "initmip-grl/run.jl is a stub — fill in the run logic before using."
end

main()
