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
        time  = ds["time"][:]
        asym4 = ds["asym4"][:]
        asym8 = ds["asym"][:]
        threshold = Float64(ds.attrib["asym_threshold"])

        # The threshold is checked against asym4 (the 4-fold-symmetry-
        # breaking metric — independent of Cartesian-grid anisotropy).
        # asym8 is reported alongside as the all-8 spread (which also
        # picks up the cap's "slightly square" shape on a finite grid).
        i_peak4  = argmax(asym4)
        i_peak8  = argmax(asym8)
        i_cross4 = findfirst(a -> !isnan(a) && a > threshold, asym4)

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
            "asym4_peak"      => asym4[i_peak4],
            "asym4_peak_time" => time[i_peak4],
            "asym4_threshold_time" => i_cross4 === nothing ? nothing : time[i_cross4],
            "asym8_peak"      => asym8[i_peak8],
            "asym8_peak_time" => time[i_peak8],
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
