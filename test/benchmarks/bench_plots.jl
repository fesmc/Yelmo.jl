# Shared plotting helper for the bench_* scripts.
#
# `make_bench_plots(yjl_log, mirror_log, png_path; window, label)` reads
# two `yelmo_timesteps.nc`-style logs (Yelmo.jl + Fortran Mirror) and
# writes a 4-panel PNG with:
#
#   - dt vs time          (line, both backends overlaid)
#   - pc_eta vs time      (line, log-y, both backends overlaid)
#   - dt histogram        (overlaid)
#   - pc_eta histogram    (overlaid, log-x)
#
# `window` (optional) restricts the time series + histograms to a time
# range `(t_lo, t_hi)`. Useful for warm-window comparisons.

using NCDatasets
using CairoMakie
using Statistics

function _load_log(path::AbstractString; window = nothing)
    NCDataset(path, "r") do ds
        t   = Array{Float64}(ds["time"][:])
        dt  = Array{Float64}(ds["dt_now"][:])
        eta = Array{Float64}(ds["pc_eta"][:])
        if window !== nothing
            tlo, thi = window
            mask = (t .>= tlo) .& (t .<= thi + 1e-6)
            t   = t[mask]
            dt  = dt[mask]
            eta = eta[mask]
        end
        (t = t, dt = dt, eta = eta)
    end
end

# Histogram-from-Vector helper that plays nicely with log-x axes
# (zero / negative values are dropped before computing edges).
function _safe_log_edges(x::AbstractVector{<:Real}; nbins::Int = 30)
    xv = filter(>(0), x)
    isempty(xv) && return Float64[]
    lo = log10(minimum(xv))
    hi = log10(maximum(xv))
    hi <= lo && return Float64[lo, lo + 1]
    return 10 .^ range(lo, hi; length = nbins + 1)
end

"""
    make_bench_plots(yjl_log, mirror_log, png_path;
                     window=nothing, label="bench") -> png_path

Read both timestep logs (Yelmo.jl + Mirror), produce a 4-panel
comparison PNG at `png_path`. `window` filters to `(t_lo, t_hi)`
in model years.
"""
function make_bench_plots(yjl_log::AbstractString,
                          mirror_log::AbstractString,
                          png_path::AbstractString;
                          window::Union{Nothing,Tuple{Real,Real}} = nothing,
                          label::AbstractString = "bench")
    isfile(yjl_log)    || error("make_bench_plots: missing $yjl_log")
    isfile(mirror_log) || error("make_bench_plots: missing $mirror_log")

    yjl    = _load_log(yjl_log;    window = window)
    mirror = _load_log(mirror_log; window = window)

    fig = Figure(size = (1100, 800))

    title_w = window === nothing ? "" :
              "  (window $(round(window[1], digits=1)) – $(round(window[2], digits=1)) yr)"
    Label(fig[0, 1:2], "$label$title_w";
          fontsize = 16, font = :bold, halign = :left)

    # --- Panel (1, 1): dt vs time ---
    ax1 = Axis(fig[1, 1]; xlabel = "time (yr)", ylabel = "dt (yr)",
               title = "Sub-step dt over time")
    lines!(ax1, yjl.t,    yjl.dt;    label = "Yelmo.jl", color = :tomato)
    lines!(ax1, mirror.t, mirror.dt; label = "Mirror",   color = :steelblue)
    axislegend(ax1, position = :rb)

    # --- Panel (1, 2): eta vs time (log-y) ---
    ax2 = Axis(fig[1, 2]; xlabel = "time (yr)", ylabel = "pc_eta (m/yr)",
               title = "Truncation-error proxy η over time",
               yscale = log10)
    # Add a tiny floor so log10 of zero entries doesn't error.
    eta_floor = 1e-10
    lines!(ax2, yjl.t,    max.(yjl.eta,    eta_floor); label = "Yelmo.jl", color = :tomato)
    lines!(ax2, mirror.t, max.(mirror.eta, eta_floor); label = "Mirror",   color = :steelblue)
    axislegend(ax2, position = :rb)

    # --- Panel (2, 1): dt histogram ---
    ax3 = Axis(fig[2, 1]; xlabel = "dt (yr)", ylabel = "count",
               title = "dt distribution")
    nbins = 30
    if !isempty(yjl.dt) && !isempty(mirror.dt)
        edges = range(min(minimum(yjl.dt), minimum(mirror.dt)),
                      max(maximum(yjl.dt), maximum(mirror.dt));
                      length = nbins + 1)
        hist!(ax3, yjl.dt;    bins = collect(edges), color = (:tomato, 0.5),
              label = "Yelmo.jl")
        hist!(ax3, mirror.dt; bins = collect(edges), color = (:steelblue, 0.5),
              label = "Mirror")
    end
    axislegend(ax3, position = :rt)

    # --- Panel (2, 2): eta histogram (log-x) ---
    ax4 = Axis(fig[2, 2]; xlabel = "pc_eta (m/yr)", ylabel = "count",
               title = "η distribution",
               xscale = log10)
    edges_eta = _safe_log_edges(vcat(yjl.eta, mirror.eta); nbins = 30)
    if !isempty(edges_eta)
        eta_floor = 1e-10
        hist!(ax4, max.(yjl.eta,    eta_floor); bins = collect(edges_eta),
              color = (:tomato, 0.5),    label = "Yelmo.jl")
        hist!(ax4, max.(mirror.eta, eta_floor); bins = collect(edges_eta),
              color = (:steelblue, 0.5), label = "Mirror")
    end
    axislegend(ax4, position = :rt)

    save(png_path, fig)
    @info "wrote bench plot" png_path n_yjl=length(yjl.t) n_mirror=length(mirror.t)
    return png_path
end
