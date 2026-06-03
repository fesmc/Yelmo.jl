# benchmarks/initmip-grl/profile.jl
#
# Profiling and benchmarking harness for the initmip-grl benchmark.
# Sibling to `run.jl` — reuses its `build_yelmo` / `build_mirror` /
# `apply_forcing!` so the *physics configuration is identical* and only
# the timing/profile instrumentation differs.
#
# Usage from this directory:
#   julia --project=. -e 'include("profile.jl"); main_profile()'                  # :bench (default)
#   julia --project=. -e 'include("profile.jl"); main_profile(mode=:profile)'     # CPU flame samples
#   julia --project=. -e 'include("profile.jl"); main_profile(mode=:alloc)'       # allocation profile
#   julia --project=. -e 'include("profile.jl"); main_profile(mode=:typestab)'    # JET @report_opt step!
#   julia --project=. -e 'include("profile.jl"); main_profile(mode=:compare)'     # yelmo vs mirror wall-clock
#
# Common kwargs:
#   t_end          end time [yr], default 20.0
#   t_warmup       warmup time before timing window starts [yr], default 1.0
#   dt_outer       outer-loop dt [yr], default 1.0
#   backend        :yelmo (default) or :mirror (mirror has no Profile/JET modes)
#   logdir         where to dump logs/.pb.gz, default benchmarks/initmip-grl/logs/
#   dt_method      override dt_method (Int or nothing); useful for fixed-dt comparison
#
# All artifacts (timing tables, profile dumps) go under
# `benchmarks/initmip-grl/logs/profile-<mode>-<timestamp>.{log,txt,pb.gz}`
# per the project's "no /tmp scratch" rule.

using Dates
using Printf
using Profile
using Statistics
using BenchmarkTools

# Reuse run.jl's build_* helpers + apply_forcing! verbatim.
include(joinpath(@__DIR__, "run.jl"))

const LOGDIR_DEFAULT = joinpath(@__DIR__, "logs")

# ----------------------------------------------------------------------
# Build a model with optional dt_method override. Reuses run.jl's
# build_params() then patches `yelmo.dt_method` if requested.
# ----------------------------------------------------------------------
function _build_model(; backend::Symbol, dt_method=nothing, outdir=nothing)
    p = build_params()
    if dt_method !== nothing
        # YelmoParameters is immutable; reconstruct yelmo sub-block.
        yp_kwargs = Dict{Symbol,Any}()
        for fn in propertynames(p.yelmo)
            yp_kwargs[fn] = getproperty(p.yelmo, fn)
        end
        yp_kwargs[:dt_method] = dt_method
        new_yelmo = yelmo_params(; yp_kwargs...)
        # Reconstruct YelmoParameters with overridden yelmo block.
        p_kwargs = Dict{Symbol,Any}()
        for fn in propertynames(p)
            fn === :name && continue
            fn === :yelmo && continue
            p_kwargs[fn] = getproperty(p, fn)
        end
        p = YelmoParameters(p.name; yelmo = new_yelmo, p_kwargs...)
    end
    if backend === :mirror
        outdir === nothing && (outdir = joinpath(@__DIR__, "output-mirror"))
        mkpath(outdir)
        return build_mirror(p, outdir), p
    else
        return build_yelmo(p), p
    end
end

# ----------------------------------------------------------------------
# Step loop that respects an optional warmup window. Timer is reset
# after warmup so the recorded breakdown only covers the timed window.
# Returns (wall_seconds_total, wall_seconds_timed, n_outer_timed).
# ----------------------------------------------------------------------
function _run_loop!(y; t_end::Real, dt_outer::Real, t_warmup::Real=0.0,
                    quiet::Bool=true)
    n_warmup = Int(round(t_warmup / dt_outer))
    n_total  = Int(round(t_end / dt_outer))
    n_timed  = max(n_total - n_warmup, 0)

    t0_total = time()
    # Warmup phase — let JIT compile, but don't let timer accumulate.
    for k in 1:n_warmup
        step!(y, dt_outer)
        quiet || @printf("  warmup t=%.0f\n", y.time)
    end
    if hasproperty(y, :timer) && y.timer.enabled
        reset_timings!(y)
    end
    t0_timed = time()
    for k in 1:n_timed
        step!(y, dt_outer)
        quiet || @printf("  timed  t=%.0f\n", y.time)
    end
    return (wall_total = time() - t0_total,
            wall_timed = time() - t0_timed,
            n_warmup   = n_warmup,
            n_timed    = n_timed)
end

# ----------------------------------------------------------------------
# Mode :bench — wall-clock + per-section timer table on one backend.
# ----------------------------------------------------------------------
function _mode_bench(io; backend, t_end, t_warmup, dt_outer, dt_method)
    @info "bench mode" backend t_end t_warmup dt_outer dt_method
    y, _ = _build_model(; backend=backend, dt_method=dt_method)
    r = _run_loop!(y; t_end=t_end, dt_outer=dt_outer, t_warmup=t_warmup)

    println(io, "")
    println(io, "── wall-clock ──────────────────────────────────────────")
    @printf(io, "  backend       : %s\n", backend)
    @printf(io, "  t_end         : %.1f yr\n", t_end)
    @printf(io, "  t_warmup      : %.1f yr  (%d outer steps)\n", t_warmup, r.n_warmup)
    @printf(io, "  timed window  : %.1f yr  (%d outer steps)\n", t_end - t_warmup, r.n_timed)
    @printf(io, "  wall total    : %.3f s\n", r.wall_total)
    @printf(io, "  wall timed    : %.3f s\n", r.wall_timed)
    @printf(io, "  s / outer step (timed) : %.3f\n",
            r.n_timed > 0 ? r.wall_timed / r.n_timed : NaN)
    println(io, "")

    if backend === :yelmo && y.timer.enabled
        println(io, "── per-section timings (timed window only) ─────────────")
        print_timings(y; io=io)
    end
    return (y = y, result = r)
end

# ----------------------------------------------------------------------
# Mode :compare — run both backends back-to-back, identical config.
# ----------------------------------------------------------------------
function _mode_compare(io; t_end, t_warmup, dt_outer, dt_method)
    @info "compare mode" t_end t_warmup dt_outer dt_method
    println(io, "── yelmo backend ───────────────────────────────────────")
    r_y = _mode_bench(io; backend=:yelmo, t_end=t_end, t_warmup=t_warmup,
                          dt_outer=dt_outer, dt_method=dt_method)

    println(io, "")
    println(io, "── mirror backend ──────────────────────────────────────")
    r_m = try
        _mode_bench(io; backend=:mirror, t_end=t_end, t_warmup=t_warmup,
                        dt_outer=dt_outer, dt_method=dt_method)
    catch err
        println(io, "Mirror run failed (libyelmo_c_api.so missing?): $err")
        nothing
    end

    if r_m !== nothing
        println(io, "")
        println(io, "── ratio ───────────────────────────────────────────────")
        ratio = r_y.result.wall_timed / r_m.result.wall_timed
        @printf(io, "  yelmo / mirror wall-clock ratio : %.2fx\n", ratio)
    end
    return (yelmo = r_y, mirror = r_m)
end

# ----------------------------------------------------------------------
# Mode :profile — CPU sampling profile. Writes a text flat profile and
# a `.pb.gz` if PProf is available (optional, not required).
# ----------------------------------------------------------------------
function _mode_profile(io, logdir; t_end, t_warmup, dt_outer, dt_method,
                       stamp, n_samples_target=2_000_000)
    @info "profile mode" t_end t_warmup dt_outer dt_method
    y, _ = _build_model(; backend=:yelmo, dt_method=dt_method)

    # Warmup so JIT compilation is done before profile starts.
    n_warmup = Int(round(t_warmup / dt_outer))
    n_timed  = max(Int(round(t_end / dt_outer)) - n_warmup, 1)
    @info "profile warmup" n_warmup
    for _ in 1:n_warmup
        step!(y, dt_outer)
    end

    Profile.clear()
    Profile.init(n=n_samples_target, delay=0.001)
    t0 = time()
    @profile begin
        for _ in 1:n_timed
            step!(y, dt_outer)
        end
    end
    wall = time() - t0
    @printf(io, "Profile: %d timed outer steps in %.3f s (%.3f s/step)\n",
            n_timed, wall, wall / n_timed)

    # Flat text profile — sorted by exclusive time.
    flat_path = joinpath(logdir, "profile-cpu-flat-$(stamp).txt")
    open(flat_path, "w") do f
        println(f, "Flat CPU profile, sorted by total samples")
        println(f, "step!() x $(n_timed) outer steps; wall = $(round(wall, digits=3)) s")
        println(f, "")
        Profile.print(f; format=:flat, sortedby=:count, mincount=10)
    end
    println(io, "Wrote flat profile: $flat_path")

    # Tree profile — full call-tree, useful for spotting parents.
    tree_path = joinpath(logdir, "profile-cpu-tree-$(stamp).txt")
    open(tree_path, "w") do f
        println(f, "Tree CPU profile")
        println(f, "")
        Profile.print(f; format=:tree, maxdepth=30, mincount=20)
    end
    println(io, "Wrote tree profile: $tree_path")

    # Optional PProf export — only if user has it installed.
    pb_path = nothing
    try
        @eval Main using PProf
        pb_path = joinpath(logdir, "profile-cpu-$(stamp).pb.gz")
        Base.invokelatest(getfield(Main, :PProf).pprof; out=pb_path, web=false)
        println(io, "Wrote PProf bundle: $pb_path")
    catch
        println(io, "PProf not installed; skipping .pb.gz export " *
                    "(install with `Pkg.add(\"PProf\")` to enable)")
    end

    return (y = y, flat = flat_path, tree = tree_path, pb = pb_path)
end

# ----------------------------------------------------------------------
# Mode :alloc — allocation profile.
# ----------------------------------------------------------------------
function _mode_alloc(io, logdir; t_end, t_warmup, dt_outer, dt_method,
                     stamp, sample_rate=0.01)
    @info "alloc mode" t_end t_warmup dt_outer dt_method sample_rate
    y, _ = _build_model(; backend=:yelmo, dt_method=dt_method)

    n_warmup = Int(round(t_warmup / dt_outer))
    n_timed  = max(Int(round(t_end / dt_outer)) - n_warmup, 1)
    for _ in 1:n_warmup
        step!(y, dt_outer)
    end

    Profile.Allocs.clear()
    t0 = time()
    Profile.Allocs.@profile sample_rate=sample_rate begin
        for _ in 1:n_timed
            step!(y, dt_outer)
        end
    end
    wall = time() - t0

    alloc_path = joinpath(logdir, "profile-alloc-$(stamp).txt")
    results = Profile.Allocs.fetch()
    n_allocs = length(results.allocs)
    total_bytes = sum(a.size for a in results.allocs; init=0)

    # Aggregate by (top-of-stack) function name; show top 40.
    buckets = Dict{String, NamedTuple{(:n, :bytes), Tuple{Int, Int}}}()
    for a in results.allocs
        top = isempty(a.stacktrace) ? "<no stack>" :
              string(a.stacktrace[1].func, " @ ", a.stacktrace[1].file, ":", a.stacktrace[1].line)
        cur = get(buckets, top, (n=0, bytes=0))
        buckets[top] = (n = cur.n + 1, bytes = cur.bytes + a.size)
    end
    sorted = sort!(collect(buckets); by = kv -> -kv[2].bytes)

    open(alloc_path, "w") do f
        @printf(f, "Allocation profile (sample_rate=%.3g)\n", sample_rate)
        @printf(f, "%d outer steps in %.3f s\n", n_timed, wall)
        @printf(f, "%d sampled allocations, %.1f MB total\n",
                n_allocs, total_bytes / 1024^2)
        println(f, "")
        @printf(f, "%-12s %10s   %s\n", "bytes(MB)", "count", "site")
        println(f, "─"^120)
        for (site, info) in first(sorted, 40)
            @printf(f, "%12.3f %10d   %s\n",
                    info.bytes / 1024^2, info.n, site)
        end
    end
    println(io, "Wrote allocation profile: $alloc_path")
    @printf(io, "Top allocation sites (top 5):\n")
    for (site, info) in first(sorted, 5)
        @printf(io, "  %.2f MB  %d allocs  %s\n",
                info.bytes / 1024^2, info.n, site)
    end
    return (y = y, alloc = alloc_path, total_bytes = total_bytes,
            n_allocs = n_allocs)
end

# ----------------------------------------------------------------------
# Mode :typestab — JET @report_opt on a single step!.
# ----------------------------------------------------------------------
function _mode_typestab(io, logdir; t_end, t_warmup, dt_outer, dt_method, stamp)
    @info "typestab mode" t_warmup dt_outer
    y, _ = _build_model(; backend=:yelmo, dt_method=dt_method)

    # Warmup so the analyzed call is the steady-state hot path.
    n_warmup = max(Int(round(t_warmup / dt_outer)), 1)
    for _ in 1:n_warmup
        step!(y, dt_outer)
    end

    @info "loading JET (this may take ~10 s)"
    @eval Main using JET

    out_path = joinpath(logdir, "typestab-$(stamp).txt")
    open(out_path, "w") do f
        println(f, "JET @report_opt step!(y, $(dt_outer))")
        println(f, "Reports type instabilities (Union/Any inferred, " *
                   "dynamic dispatch, runtime broadcast widening, ...)")
        println(f, "")
        # @report_opt returns a JETToplevelResult; show it.
        rep = Base.invokelatest(getfield(Main, :JET).report_opt,
                                step!, Tuple{typeof(y), Float64})
        show(f, MIME"text/plain"(), rep)
    end
    println(io, "Wrote JET type-stability report: $out_path")
    return (y = y, report = out_path)
end

# ----------------------------------------------------------------------
# Main dispatcher.
# ----------------------------------------------------------------------
function main_profile(; mode::Symbol = :bench,
                       backend::Symbol = :yelmo,
                       t_end::Real = 20.0,
                       t_warmup::Real = 1.0,
                       dt_outer::Real = 1.0,
                       dt_method::Union{Int,Nothing} = nothing,
                       logdir::AbstractString = LOGDIR_DEFAULT)
    mode in (:bench, :compare, :profile, :alloc, :typestab) ||
        error("mode must be one of :bench, :compare, :profile, :alloc, :typestab")
    cd(@__DIR__)
    mkpath(logdir)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    log_path = joinpath(logdir, "profile-$(mode)-$(stamp).log")

    open(log_path, "w") do io
        # Header
        @printf(io, "Yelmo.jl initmip-grl profile run\n")
        @printf(io, "  mode      : %s\n", mode)
        @printf(io, "  backend   : %s\n", backend)
        @printf(io, "  t_end     : %.1f yr\n", t_end)
        @printf(io, "  t_warmup  : %.1f yr\n", t_warmup)
        @printf(io, "  dt_outer  : %.2f yr\n", dt_outer)
        @printf(io, "  dt_method : %s\n", dt_method === nothing ? "(default)" : string(dt_method))
        @printf(io, "  timestamp : %s\n", stamp)
        @printf(io, "  Julia     : %s\n", VERSION)
        @printf(io, "  threads   : %d\n", Threads.nthreads())
        println(io, "")

        result = if mode === :bench
            _mode_bench(io; backend=backend, t_end=t_end, t_warmup=t_warmup,
                            dt_outer=dt_outer, dt_method=dt_method)
        elseif mode === :compare
            _mode_compare(io; t_end=t_end, t_warmup=t_warmup,
                              dt_outer=dt_outer, dt_method=dt_method)
        elseif mode === :profile
            _mode_profile(io, logdir; t_end=t_end, t_warmup=t_warmup,
                                       dt_outer=dt_outer, dt_method=dt_method,
                                       stamp=stamp)
        elseif mode === :alloc
            _mode_alloc(io, logdir; t_end=t_end, t_warmup=t_warmup,
                                     dt_outer=dt_outer, dt_method=dt_method,
                                     stamp=stamp)
        elseif mode === :typestab
            _mode_typestab(io, logdir; t_end=t_end, t_warmup=t_warmup,
                                        dt_outer=dt_outer, dt_method=dt_method,
                                        stamp=stamp)
        end

        @info "wrote log" log_path
        println(io, "")
        println(io, "── done ────────────────────────────────────────────────")
        return result
    end
end
