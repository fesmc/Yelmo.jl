## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# Regenerate all benchmark fixtures via YelmoMirror.
#
# Run locally with `julia --project=test test/benchmarks/regenerate.jl`
# (from the repo root) or `julia --project=. regenerate.jl` (from
# `test/benchmarks/`). Requires `libyelmo_c_api.so` to be present at
# `yelmo/libyelmo/include/` AND to expose the `yelmo_init_grid` and
# `yelmo_restart_write` symbols.
#
# CI never runs this script — fixtures are pre-committed under
# `fixtures/`. Use `--overwrite` to clobber existing fixtures.

using Yelmo

include("helpers.jl")
using .YelmoBenchmarks

# Spec registry. To add a new benchmark, write a spec module in
# `specs/` and `push!` it here.
const SPECS = let specs = BenchmarkSpec[]
    include("specs/bueler_b_smoke.jl")
    push!(specs, BUELER_B_SMOKE_SPEC)
    specs
end

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

function main(args::Vector{String})
    overwrite = "--overwrite" in args
    only_names = filter(a -> !startswith(a, "--"), args)

    selected = isempty(only_names) ? SPECS :
               filter(s -> s.name in only_names, SPECS)
    isempty(selected) && error(
        "regenerate.jl: no specs match $(only_names). " *
        "Available: $(join((s.name for s in SPECS), ", ")).")

    println("Regenerating $(length(selected)) benchmark(s) → $FIXTURES_DIR")
    for spec in selected
        println("\n=== $(spec.name) ===")
        paths = run_mirror_benchmark!(spec;
                                      fixtures_dir = FIXTURES_DIR,
                                      overwrite    = overwrite)
        for p in paths
            sz_kb = round(filesize(p) / 1024; digits=1)
            println("  wrote $(basename(p))  ($(sz_kb) KB)")
        end
    end
    println("\nDone.")
end

main(ARGS)
