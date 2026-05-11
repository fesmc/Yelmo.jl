# Compare Yelmo.jl (masked + unmasked) vs YelmoMirror timestep logs.
# Run: julia --project=. compare_pc_eta.jl
using NCDatasets
using Printf
using Statistics

const FILES = [
    ("Yelmo.jl masked",   "output/yelmo_timesteps_masked.nc"),
    ("Yelmo.jl unmasked", "output/yelmo_timesteps_unmasked.nc"),
    ("YelmoMirror",       "output-mirror/yelmo_timesteps_mirror.nc"),
]

function _read_cols(path)
    NCDataset(path) do ds
        cols = Dict{String,Vector{Float64}}()
        for v in ("time", "dt_now", "pc_eta", "iter_redo")
            haskey(ds, v) && (cols[v] = Float64.(ds[v][:]))
        end
        return cols
    end
end

@printf("%-22s %-9s %-12s %-10s %-10s %-10s %-12s %-12s\n",
        "backend", "n_steps", "rejections",
        "dt_mean", "dt_min", "dt_max",
        "eta_mean", "eta_max")
println("-"^95)

for (label, path) in FILES
    isfile(path) || (println(rpad(label, 22), " (missing: $path)"); continue)
    c = _read_cols(path)
    n  = length(c["time"])
    dt = c["dt_now"]
    eta = c["pc_eta"]
    rj = haskey(c, "iter_redo") ? Int(sum(c["iter_redo"] .> 1)) : -1
    @printf("%-22s %-9d %-12d %-10.4f %-10.4f %-10.4f %-12.3e %-12.3e\n",
            label, n, rj,
            mean(dt), minimum(dt), maximum(dt),
            mean(eta), maximum(eta))
end

# Distribution sanity: dt percentiles + count of "almost saturated" steps.
println()
@printf("%-22s %-10s %-10s %-10s %-10s %-10s\n",
        "backend", "dt_p25", "dt_med", "dt_p75", "dt_p95", "≥0.5 yr [%]")
println("-"^75)
for (label, path) in FILES
    isfile(path) || continue
    c = _read_cols(path)
    dt = c["dt_now"]
    n = length(dt)
    p25, p50, p75, p95 = quantile(dt, (0.25, 0.5, 0.75, 0.95))
    big = 100.0 * sum(dt .>= 0.5) / n
    @printf("%-22s %-10.4f %-10.4f %-10.4f %-10.4f %-10.1f\n",
            label, p25, p50, p75, p95, big)
end
