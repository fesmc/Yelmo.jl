## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# Regenerate all benchmark fixtures.
#
# Two backends, dispatched on spec type:
#
#   - **AbstractBenchmark** (`BuelerBenchmark`, …): write the
#     closed-form solution directly to a NetCDF restart via
#     `write_fixture!(b, path; times)`. No YelmoMirror, no Fortran
#     library. Used for benchmarks with closed-form solutions
#     (BUELER-A, BUELER-B Halfar, etc.).
#   - **YelmoMirror** (`BenchmarkSpec`): drive YelmoMirror with the
#     spec's namelist and write a restart at each output time. Used
#     for benchmarks without analytical solutions (EISMINT, ISMIP-HOM,
#     slab, trough, MISMIP+, CalvingMIP). Requires libyelmo_c_api.so
#     to expose `yelmo_init_grid` and `yelmo_restart_write`.
#
# CI never runs this script — fixtures are pre-committed under
# `fixtures/`. Use `--overwrite` to clobber existing fixtures.
# Optionally pass spec names to regenerate just a subset.

using Yelmo

include("helpers.jl")
using .YelmoBenchmarks

# Spec registry. Heterogeneous: a Vector{Any} so we can hold both
# AbstractBenchmark (analytical) and BenchmarkSpec (YelmoMirror)
# entries side-by-side.
const SPECS = Any[
    BuelerBenchmark(:B; dx_km=50.0),
    # Future BenchmarkSpec entries get pushed here.
]

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

# Single output time for the BUELER-B Halfar smoke fixture. Multi-time
# regeneration is deferred to a future milestone alongside the
# multi-time `write_fixture!` extension.
const _BUELER_B_OUT_TIME = 1000.0

_spec_name(b::AbstractBenchmark) = YelmoBenchmarks._spec_name(b)
_spec_name(s::BenchmarkSpec)     = s.name

function _regenerate_one!(b::AbstractBenchmark; fixtures_dir, overwrite)
    name = _spec_name(b)
    t_out = _BUELER_B_OUT_TIME
    path = joinpath(fixtures_dir, "$(name)_t$(Int(round(t_out))).nc")

    if !overwrite && isfile(path)
        error("regenerate: $(path) exists; pass --overwrite to clobber.")
    end
    return write_fixture!(b, path; times=[t_out])
end

function _regenerate_one!(spec::BenchmarkSpec; fixtures_dir, overwrite)
    return run_mirror_benchmark!(spec; fixtures_dir, overwrite)
end

function main(args::Vector{String})
    overwrite = "--overwrite" in args
    only_names = filter(a -> !startswith(a, "--"), args)

    selected = isempty(only_names) ? SPECS :
               filter(s -> _spec_name(s) in only_names, SPECS)
    isempty(selected) && error(
        "regenerate.jl: no specs match $(only_names). " *
        "Available: $(join((_spec_name(s) for s in SPECS), ", ")).")

    println("Regenerating $(length(selected)) benchmark(s) → $FIXTURES_DIR")
    for spec in selected
        backend = spec isa AbstractBenchmark ? "analytical" : "mirror"
        println("\n=== $(_spec_name(spec))  ($backend) ===")
        paths = _regenerate_one!(spec; fixtures_dir=FIXTURES_DIR, overwrite=overwrite)
        for p in paths
            sz_kb = round(filesize(p) / 1024; digits=1)
            println("  wrote $(basename(p))  ($(sz_kb) KB)")
        end
    end
    println("\nDone.")
end

main(ARGS)
