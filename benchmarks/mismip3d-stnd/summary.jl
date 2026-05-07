# benchmarks/mismip3d-stnd/summary.jl
#
# Post-processing: read output/timeseries.nc and write summary.json with
# physics-only statistics + metadata.

using NCDatasets
using JSON
using Statistics
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
            t_end_yr           = Float64(ds.attrib["t_end_yr"]),
            t_final_yr         = Float64(ds["time"][end]),
            n_samples          = length(ds["time"]),
            max_H_final        = Float64(ds["max_H"][end]),
            mean_H_final       = Float64(ds["mean_H"][end]),
            mean_fgrnd_final   = Float64(ds["mean_f_grnd"][end]),
            gl_x_km_final      = Float64(ds["gl_x_km"][end]),
            max_ux_abs_final   = Float64(ds["max_ux_abs"][end]),
            max_uy_cnt_final   = Float64(ds["max_uy_center_abs"][end]),
            picard_iters_final = Int(ds["picard_iters_last"][end]),
        )
    end

    summary = Dict(
        "benchmark"                       => "mismip3d-stnd",
        "date"                            => string(Dates.now()),
        "yelmo_version"                   => _yelmo_version(),
        "t_end_yr"                        => s.t_end_yr,
        "t_final_yr"                      => s.t_final_yr,
        "n_samples"                       => s.n_samples,
        "final_max_H_m"                   => s.max_H_final,
        "final_mean_H_m"                  => s.mean_H_final,
        "final_mean_f_grnd"               => s.mean_fgrnd_final,
        "final_gl_x_km"                   => s.gl_x_km_final,
        "final_max_ux_abs_m_per_yr"       => s.max_ux_abs_final,
        "final_max_uy_center_abs_m_per_yr" => s.max_uy_cnt_final,
        "final_picard_iters"              => s.picard_iters_final,
    )

    open(SUMMARY_PATH, "w") do io
        JSON.print(io, summary, 2)
        write(io, "\n")
    end
    @info "wrote $(SUMMARY_PATH)"
    foreach((k, v) -> @info("  $k = $v"), keys(summary), values(summary))
end

main()
