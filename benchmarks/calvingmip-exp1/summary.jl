# benchmarks/calvingmip-exp1/summary.jl
#
# Post-processing: read output/timeseries.nc and write summary.json with
# physics-only statistics + metadata.

using NCDatasets
using JSON
using Pkg
using Dates

const OUTPUT_DIR    = abspath(joinpath(@__DIR__, "output"))
const TIMESERIES_NC = joinpath(OUTPUT_DIR, "timeseries.nc")
const SUMMARY_PATH  = abspath(joinpath(@__DIR__, "summary.json"))

function _yelmo_version()
    for (_, info) in Pkg.dependencies()
        info.name == "Yelmo" && return info.version === nothing ? "dev" : string(info.version)
    end
    return "unknown"
end

function main()
    isfile(TIMESERIES_NC) ||
        error("output/timeseries.nc not found — run run.jl first.")

    s = NCDataset(TIMESERIES_NC, "r") do ds
        (
            t_end_yr        = Float64(ds.attrib["t_end_yr"]),
            t_final_yr      = Float64(ds["time"][end]),
            max_H_final     = Float64(ds["max_H"][end]),
            mean_H_final    = Float64(ds["mean_H"][end]),
            volume_final    = Float64(ds["total_volume_m3"][end]),
            n_ice_cells_final   = Int(ds["n_ice_cells"][end]),
            n_ocean_cells_final = Int(ds["n_ocean_cells"][end]),
            n_samples       = length(ds["time"]),
        )
    end

    summary = Dict(
        "benchmark"               => "calvingmip-exp1",
        "date"                    => string(Dates.now()),
        "yelmo_version"           => _yelmo_version(),
        "t_end_yr"                => s.t_end_yr,
        "t_final_yr"              => s.t_final_yr,
        "n_samples"               => s.n_samples,
        "final_max_H_m"           => s.max_H_final,
        "final_mean_H_m"          => s.mean_H_final,
        "final_total_volume_m3"   => s.volume_final,
        "final_n_ice_cells"       => s.n_ice_cells_final,
        "final_n_ocean_cells"     => s.n_ocean_cells_final,
    )

    open(SUMMARY_PATH, "w") do io
        JSON.print(io, summary, 2)
        write(io, "\n")
    end
    @info "wrote $(SUMMARY_PATH)"
    foreach((k, v) -> @info("  $k = $v"), keys(summary), values(summary))
end

main()
