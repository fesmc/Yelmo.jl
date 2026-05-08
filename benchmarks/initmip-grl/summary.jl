# benchmarks/initmip-grl/summary.jl
#
# Post-processing: read output/region_domain.nc and write summary.json
# with high-level diagnostics + metadata.
#
# `summary.json` is committed — keep its contents physics-only and
# hardware-independent (no wall-clock times, no thread counts).
#
# Run from this directory after run.jl:
#   julia --project=. summary.jl

cd(@__DIR__)

using NCDatasets
using JSON
using Statistics
using Pkg
using Dates

const OUTPUT_DIR   = abspath(joinpath(@__DIR__, "output"))
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

    region_nc = joinpath(OUTPUT_DIR, "region_domain.nc")
    isfile(region_nc) || error("region_domain.nc not found in output/")

    summary = NCDataset(region_nc) do ds
        t      = Vector{Float64}(ds["time"][:])
        V_ice  = Vector{Float64}(ds["V_ice"][:])    # 1e6 km³
        A_ice  = Vector{Float64}(ds["A_ice"][:])    # 1e6 km²
        V_sle  = Vector{Float64}(ds["V_sle"][:])    # m sle
        H_ice  = Vector{Float64}(ds["H_ice"][:])    # m (mean)
        H_max  = Vector{Float64}(ds["H_ice_max"][:])

        n = length(t)
        Dict(
            "benchmark"        => "initmip-grl",
            "date"             => string(Dates.now()),
            "yelmo_version"    => _yelmo_version(),
            "t_end_yr"         => t[end],
            "n_timesteps"      => n,
            "final_V_ice_1e6km3" => V_ice[end],
            "final_A_ice_1e6km2" => A_ice[end],
            "final_V_sle_m"    => V_sle[end],
            "final_H_ice_mean_m" => H_ice[end],
            "final_H_ice_max_m"  => H_max[end],
            "dV_ice_1e6km3"    => V_ice[end] - V_ice[1],
        )
    end

    open(SUMMARY_PATH, "w") do io
        JSON.print(io, summary, 2)
        write(io, "\n")
    end
    @info "wrote $(SUMMARY_PATH)"
end

main()
