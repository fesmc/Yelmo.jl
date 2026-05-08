# benchmarks/initmip-grl/summary.jl
#
# Post-processing: read output/ NetCDFs and write summary.json with
# high-level statistics + metadata. Optionally also writes plots/.
#
# `summary.json` is committed — keep its contents physics-only and
# hardware-independent (no wall-clock times, no thread counts).

using NCDatasets
using JSON
using Statistics
using Pkg
using Dates

const OUTPUT_DIR  = abspath(joinpath(@__DIR__, "output"))
const SUMMARY_PATH = abspath(joinpath(@__DIR__, "summary.json"))

function _yelmo_version()
    deps = Pkg.dependencies()
    for (_, info) in deps
        info.name == "Yelmo" && return info.version === nothing ? "dev" : string(info.version)
    end
    return "unknown"
end

function main()
    isdir(OUTPUT_DIR) || error("output/ not found — run run.jl first.")

    # TODO: load the relevant NetCDFs from OUTPUT_DIR and compute
    # benchmark-specific statistics here.

    summary = Dict(
        "benchmark"      => "initmip-grl",
        "date"           => string(Dates.now()),
        "yelmo_version"  => _yelmo_version(),
        # "max_H_m"      => ...,
        # "mean_H_m"     => ...,
    )

    open(SUMMARY_PATH, "w") do io
        JSON.print(io, summary, 2)
        write(io, "\n")
    end
    @info "wrote $(SUMMARY_PATH)"
end

main()
