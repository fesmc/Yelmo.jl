# benchmarks/calvingmip-exp2/summary.jl
#
# Post-processing for CalvingMIP Exp2: read output/timeseries.nc and
# write summary.json with the peak asymmetry, the time it crossed the
# threshold (if any), and a per-direction front-radius range.

using NCDatasets
using JSON
using Statistics
using Pkg
using Dates

const OUTPUT_DIR    = abspath(joinpath(@__DIR__, "output"))
const TIMESERIES_NC = joinpath(OUTPUT_DIR, "timeseries.nc")
const SUMMARY_PATH  = abspath(joinpath(@__DIR__, "summary.json"))

const _DIRS = (:E, :NE, :N, :NW, :W, :SW, :S, :SE)

function _yelmo_version()
    for (_, info) in Pkg.dependencies()
        info.name == "Yelmo" && return info.version === nothing ? "dev" : string(info.version)
    end
    return "unknown"
end

function main()
    isfile(TIMESERIES_NC) ||
        error("output/timeseries.nc not found — run run.jl first.")

    NCDataset(TIMESERIES_NC, "r") do ds
        time = ds["time"][:]
        asym = ds["asym"][:]
        threshold = Float64(ds.attrib["asym_threshold"])

        i_peak = argmax(asym)
        i_cross = findfirst(a -> !isnan(a) && a > threshold, asym)

        dir_ranges = Dict{String,Any}()
        for d in _DIRS
            v = ds["front_$(d)"][:]
            valid = filter(!isnan, v)
            dir_ranges[string(d)] = isempty(valid) ?
                Dict("min" => nothing, "max" => nothing) :
                Dict("min" => minimum(valid), "max" => maximum(valid))
        end

        summary = Dict(
            "benchmark"       => "calvingmip-exp2",
            "date"            => string(Dates.now()),
            "yelmo_version"   => _yelmo_version(),
            "n_samples"       => length(time),
            "asym_threshold"  => threshold,
            "asym_peak"       => asym[i_peak],
            "asym_peak_time"  => time[i_peak],
            "asym_threshold_time" => i_cross === nothing ? nothing : time[i_cross],
            "front_radius_range_km" => dir_ranges,
        )

        open(SUMMARY_PATH, "w") do io
            JSON.print(io, summary, 2)
            write(io, "\n")
        end
        @info "wrote $(SUMMARY_PATH)"
        for (k, v) in summary
            @info "  $k = $v"
        end
    end
end

main()
