# Diagnose the residual ~4× eta_max gap between Yelmo.jl masked and Mirror.
# Reads both timestep logs aligned by `time` and prints per-step ratios + the
# top-10 worst offenders.
using NCDatasets
using Printf
using Statistics

const MASKED = "output/yelmo_timesteps_masked.nc"
const MIRROR = "output-mirror/yelmo_timesteps_mirror.nc"

function _read(path)
    NCDataset(path) do ds
        (time = Float64.(ds["time"][:]),
         dt   = Float64.(ds["dt_now"][:]),
         eta  = Float64.(ds["pc_eta"][:]))
    end
end

m = _read(MASKED)
f = _read(MIRROR)

println("Yelmo.jl masked: $(length(m.time)) steps; Mirror: $(length(f.time)) steps.")
println()

# Per integer-year bucket: aggregate eta and dt to compare apples-to-apples.
years = 1:10
@printf("%-6s %-12s %-12s %-12s %-12s %-12s %-12s\n",
        "yr", "M_n_sub", "M_eta_max", "M_eta_mean", "F_n_sub", "F_eta_max", "F_eta_mean")
println("-"^80)
for yr in years
    m_in = findall(t -> (yr - 1) < t <= yr, m.time)
    f_in = findall(t -> (yr - 1) < t <= yr, f.time)
    m_eta = m.eta[m_in]
    f_eta = f.eta[f_in]
    me_max  = isempty(m_eta) ? NaN : maximum(m_eta)
    me_mean = isempty(m_eta) ? NaN : mean(m_eta)
    fe_max  = isempty(f_eta) ? NaN : maximum(f_eta)
    fe_mean = isempty(f_eta) ? NaN : mean(f_eta)
    @printf("%-6d %-12d %-12.3e %-12.3e %-12d %-12.3e %-12.3e\n",
            yr, length(m_in), me_max, me_mean, length(f_in), fe_max, fe_mean)
end

println()
println("Top-5 Yelmo.jl masked eta peaks (with the dt that produced them):")
top = sortperm(m.eta; rev = true)[1:min(5, length(m.eta))]
@printf("  %-10s %-10s %-12s\n", "t [yr]", "dt [yr]", "eta [m/yr]")
for i in top
    @printf("  %-10.4f %-10.4f %-12.3e\n", m.time[i], m.dt[i], m.eta[i])
end

println()
println("Top-5 Mirror eta peaks:")
ftop = sortperm(f.eta; rev = true)[1:min(5, length(f.eta))]
@printf("  %-10s %-10s %-12s\n", "t [yr]", "dt [yr]", "eta [m/yr]")
for i in ftop
    @printf("  %-10.4f %-10.4f %-12.3e\n", f.time[i], f.dt[i], f.eta[i])
end
