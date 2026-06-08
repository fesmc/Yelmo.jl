# Regenerate the TROUGH-F17 DIVA Mirror fixture with:
#   - log_timestep = .TRUE.   → Fortran writes timesteps.nc per accepted PC step
#   - pc_method = "HEUN"      → match Yelmo.jl's only available PC scheme
#   - mirror_wallclock_seconds attribute on the restart for the
#     speed comparison
#
# Output:
#   test/benchmarks/fixtures/trough_f17_t1000.nc        (restart, with
#                                                        mirror_wallclock_seconds)
#   test/benchmarks/fixtures/trough_f17_timesteps.nc    (Fortran's per-step log)
cd(@__DIR__)
import Pkg; Pkg.activate("..")

using Yelmo
using Oceananigans: interior
using NCDatasets

include("harness.jl")
using .YelmoBenchmarkHarness

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))
const SPECS_DIR    = abspath(joinpath(@__DIR__, "specs"))
const TROUGH_NML   = joinpath(SPECS_DIR, "yelmo_TROUGH.nml")
const FIX_RESTART  = joinpath(FIXTURES_DIR, "trough_f17_t1000.nc")
const FIX_TS_LOG   = joinpath(FIXTURES_DIR, "trough_f17_timesteps.nc")

# Build a patched namelist on disk: log_timestep=True, pc_method="HEUN".
# Text-substitution against the source namelist — keeps every other
# parameter exactly as committed.
function patch_namelist(src::AbstractString)
    txt = read(src, String)
    # `log_timestep    = False`  →  `log_timestep    = True`
    txt = replace(txt, r"log_timestep(\s*)=(\s*)False" => s"log_timestep\1=\2True")
    # `pc_method       = "AB-SAM"`  →  `pc_method       = "HEUN"`
    txt = replace(txt, r"pc_method(\s*)=(\s*)\"AB-SAM\"" => s"pc_method\1=\2\"HEUN\"")
    out = joinpath(@__DIR__, "..", "..", "logs", "regen_trough_diva",
                   "yelmo_TROUGH_patched.nml")
    mkpath(dirname(out))
    write(out, txt)
    @info "patched namelist" src dst=out
    return out
end

function run_trough_mirror!(b::TroughBenchmark, namelist::AbstractString,
                              t_out::Float64)
    # Single output time: the canonical t=1000 cold-start fixture
    # (Mirror's evolution from the F17 IC over [0, t_out]). Also the
    # reference end-state for `bench_diva_trough.jl`'s cold-start
    # comparison.
    spec = BenchmarkSpec(
        name           = "trough_f17_regen",
        namelist_path  = namelist,
        grid           = (xc = b.xc, yc = b.yc, grid_name = "TROUGH-F17"),
        time_init      = 0.0,
        end_time       = t_out,
        output_times   = [t_out],
        dt             = 5.0,
        setup_initial_state! = (ymirror, t) ->
            YelmoBenchmarkHarness._setup_trough_initial_state!(ymirror, b, t),
    )

    # Mirror's `generate_fixture!` creates its own internal
    # tempdir under TMPDIR (prefix `bench_<name>_`), `cd`s into it,
    # and gets it auto-cleaned at Julia exit. To capture Fortran's
    # `timesteps.nc` (which lands in that tempdir's CWD), we have to
    # find + copy it BEFORE Julia exits. We point TMPDIR at a
    # known location so the search is bounded.
    tmp_root = mktempdir(; prefix = "trough_regen_root_")
    old_tmpdir = get(ENV, "TMPDIR", nothing)
    ENV["TMPDIR"] = tmp_root
    out_dir = mktempdir(; prefix = "trough_regen_")
    @info "running YelmoMirror" out_dir t_out tmp_root
    t0 = time()
    paths = try
        generate_fixture!(spec;
                              fixtures_dir = out_dir,
                              overwrite    = true)
    finally
        # Restore TMPDIR for the rest of the Julia session.
        if old_tmpdir === nothing
            delete!(ENV, "TMPDIR")
        else
            ENV["TMPDIR"] = old_tmpdir
        end
    end
    wallclock = time() - t0
    @info "YelmoMirror finished" wallclock_s=round(wallclock, digits=2)

    # Move restart into the canonical fixture dir.
    @assert length(paths) == 1 "expected 1 restart fixture, got $(length(paths))"
    isfile(FIX_RESTART) && rm(FIX_RESTART)
    mv(paths[1], FIX_RESTART; force = true)
    @info "restart written (t=$(Int(t_out)))" path=FIX_RESTART size_kb=round(filesize(FIX_RESTART)/1024, digits=1)

    # Find Fortran's `timesteps.nc` under the now-bounded tmp_root.
    # `walkdir` may hit permission-denied entries (TemporaryItems on
    # macOS); guard with try/catch and skip those.
    src_log_candidates = String[]
    try
        for (root, _, files) in walkdir(tmp_root)
            if "timesteps.nc" in files
                push!(src_log_candidates, joinpath(root, "timesteps.nc"))
            end
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
        @warn "No timesteps.nc found under $tmp_root — Fortran's log_timestep wiring may not have triggered."
    end

    # Append wallclock + provenance attributes to the restart.
    cpu_info = try
        first(Sys.cpu_info()).model
    catch
        "unknown"
    end
    # Record wallclock + provenance on the restart fixture.
    NCDataset(FIX_RESTART, "a") do ds
        ds.attrib["mirror_wallclock_seconds_total"] = wallclock
        ds.attrib["mirror_t_init"]            = 0.0
        ds.attrib["mirror_t_end"]             = t_out
        ds.attrib["mirror_dt_outer_yr"]       = spec.dt
        ds.attrib["mirror_julia_version"]     = string(VERSION)
        ds.attrib["mirror_cpu_model"]         = cpu_info
        ds.attrib["mirror_n_julia_threads"]   = Threads.nthreads()
        ds.attrib["mirror_pc_method"]         = "HEUN"
        ds.attrib["mirror_solver"]            = "diva"
    end
    @info "wallclock recorded" wallclock_s=round(wallclock, digits=2)
    return wallclock
end

# Run.
b = TroughBenchmark(:F17; dx_km = 8.0)
namelist_patched = patch_namelist(TROUGH_NML)
wallclock = run_trough_mirror!(b, namelist_patched, 1000.0)
println("\nDone. Mirror wallclock = $(round(wallclock, digits=2)) s")
