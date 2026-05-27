# Regenerate the EISMINT-moving Mirror fixture with Fortran's
# `log_timestep = .TRUE.` enabled, plus record the run wallclock as a
# NetCDF attribute. Output:
#
#   test/benchmarks/fixtures/eismint_moving_t25000.nc      (restart, with
#                                                           mirror_wallclock_seconds)
#   test/benchmarks/fixtures/eismint_moving_timesteps.nc   (Fortran's per-step log)
#
# Companion to `regen_trough_diva.jl`. Same TMPDIR-rooted approach to
# capture Fortran's `timesteps.nc` from the Mirror harness's internal
# rundir before it gets cleaned up at Julia exit.
cd(@__DIR__)
import Pkg; Pkg.activate("..")

using Yelmo
using Oceananigans: interior
using NCDatasets

include("helpers.jl")
using .YelmoBenchmarks

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))
const SPECS_DIR    = abspath(joinpath(@__DIR__, "specs"))
const EISMINT_NML  = joinpath(SPECS_DIR, "yelmo_EISMINT_moving.nml")
const FIX_RESTART  = joinpath(FIXTURES_DIR, "eismint_moving_t25000.nc")
const FIX_TS_LOG   = joinpath(FIXTURES_DIR, "eismint_moving_timesteps.nc")

const T_END = 25_000.0

# Patch log_timestep=True on the spec namelist (text substitution to
# avoid round-tripping through YelmoMirrorParameters).
function patch_namelist(src::AbstractString)
    txt = read(src, String)
    txt = replace(txt, r"log_timestep(\s*)=(\s*)False" => s"log_timestep\1=\2True")
    out = joinpath(@__DIR__, "..", "..", "logs", "regen_sia_eismint",
                   "yelmo_EISMINT_moving_patched.nml")
    mkpath(dirname(out))
    write(out, txt)
    @info "patched namelist" src dst=out
    return out
end

function run_mirror!(b::EISMINT1MovingBenchmark, namelist::AbstractString,
                     t_out::Float64)
    spec = BenchmarkSpec(
        name           = "eismint_moving_regen",
        namelist_path  = namelist,
        grid           = (xc = b.xc, yc = b.yc, grid_name = "EISMINT"),
        time_init      = 0.0,
        end_time       = t_out,
        output_times   = [t_out],
        dt             = 100.0,
        setup_initial_state! = (ymirror, t) ->
            YelmoBenchmarks._setup_eismint_moving_initial_state!(ymirror, b, t),
    )

    # Run inside a bounded TMPDIR so we can find Fortran's timesteps.nc
    # before it's auto-cleaned at Julia exit.
    tmp_root = mktempdir(; prefix = "eismint_regen_root_")
    old_tmpdir = get(ENV, "TMPDIR", nothing)
    ENV["TMPDIR"] = tmp_root
    out_dir = mktempdir(; prefix = "eismint_regen_")
    @info "running YelmoMirror" out_dir tmp_root t_out
    t0 = time()
    paths = try
        run_mirror_benchmark!(spec; fixtures_dir = out_dir, overwrite = true)
    finally
        if old_tmpdir === nothing
            delete!(ENV, "TMPDIR")
        else
            ENV["TMPDIR"] = old_tmpdir
        end
    end
    wallclock = time() - t0
    @info "YelmoMirror finished" wallclock_s=round(wallclock, digits=2)

    # Move restart into the canonical fixture dir.
    isfile(FIX_RESTART) && rm(FIX_RESTART)
    mv(paths[1], FIX_RESTART; force = true)
    @info "restart written" path=FIX_RESTART size_kb=round(filesize(FIX_RESTART)/1024, digits=1)

    # Find Fortran's `timesteps.nc` under the bounded tmp_root.
    src_log_candidates = String[]
    try
        for (root, _, files) in walkdir(tmp_root)
            "timesteps.nc" in files && push!(src_log_candidates,
                                              joinpath(root, "timesteps.nc"))
        end
    catch err
        @warn "walkdir failed; partial search" err
    end
    if !isempty(src_log_candidates)
        src_log = argmax(p -> stat(p).mtime, src_log_candidates)
        isfile(FIX_TS_LOG) && rm(FIX_TS_LOG)
        cp(src_log, FIX_TS_LOG; force = true)
        @info "Fortran timestep log copied" src=src_log dst=FIX_TS_LOG
    else
        @warn "No timesteps.nc found under $tmp_root."
    end

    # Provenance attrs on the restart.
    cpu_info = try; first(Sys.cpu_info()).model; catch; "unknown"; end
    NCDataset(FIX_RESTART, "a") do ds
        ds.attrib["mirror_wallclock_seconds"] = wallclock
        ds.attrib["mirror_t_init"]            = 0.0
        ds.attrib["mirror_t_end"]             = t_out
        ds.attrib["mirror_dt_outer_yr"]       = spec.dt
        ds.attrib["mirror_n_outer_steps"]     = Int(round(t_out / spec.dt))
        ds.attrib["mirror_julia_version"]     = string(VERSION)
        ds.attrib["mirror_cpu_model"]         = cpu_info
        ds.attrib["mirror_n_julia_threads"]   = Threads.nthreads()
        ds.attrib["mirror_pc_method"]         = "HEUN"
        ds.attrib["mirror_solver"]            = "sia"
    end
    @info "wallclock recorded" wallclock_s=round(wallclock, digits=2)
    return wallclock
end

b = EISMINT1MovingBenchmark()
namelist_patched = patch_namelist(EISMINT_NML)
wallclock = run_mirror!(b, namelist_patched, T_END)
println("\nDone. Mirror wallclock = $(round(wallclock, digits=2)) s")
