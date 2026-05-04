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
    TroughBenchmark(:F17; dx_km=8.0),
    HOMCBenchmark(:C; L_km=80.0, dx_km=2.0),
    MISMIP3DBenchmark(:Stnd; dx_km=16.0),
    EISMINT1MovingBenchmark(),    # 50 km, 25 kyr Mirror fixture (≈3-5 min)
    # Future BenchmarkSpec entries get pushed here.
]

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

# Default output time(s) per AbstractBenchmark type. BUELER-B uses the
# Halfar t=1000 snapshot; HOM-C is steady-state IC at t=0. MISMIP3D
# emits BOTH the analytical IC (t=0) and the YelmoMirror reference
# (t=500) so the lockstep test can compare against a Fortran-driven
# end-state while the standalone trajectory test still seeds from t=0.
_default_out_times(::AbstractBenchmark)         = [1000.0]
_default_out_times(::HOMCBenchmark)             = [0.0]
_default_out_times(::MISMIP3DBenchmark)         = [0.0, 500.0]
_default_out_times(::EISMINT1MovingBenchmark)   = [25000.0]

# MISMIP3D ATT-ramp fixture parameters. Compressed Pattyn-2017 protocol
# pivoted around the existing Stnd baseline `rf_const = 3.1536e-18`
# (Pa^-3 yr^-1) — see test/benchmarks/test_mismip3d_stnd_att.jl for
# discussion. Writes one Mirror fixture at t = ATT_TIMES[end].
const MISMIP3D_ATT_NAME  = "mismip3d_stnd_att"
const MISMIP3D_ATT_TIMES = [500.0, 1000.0, 1500.0]
const MISMIP3D_ATT_RF    = [3.1536e-18, 3.1536e-19, 3.1536e-18]

_spec_name(b::AbstractBenchmark) = YelmoBenchmarks._spec_name(b)
_spec_name(s::BenchmarkSpec)     = s.name

function _regenerate_one!(b::AbstractBenchmark; fixtures_dir, overwrite)
    name = _spec_name(b)
    paths = String[]
    for t_out in _default_out_times(b)
        path = joinpath(fixtures_dir, "$(name)_t$(Int(round(t_out))).nc")
        if !overwrite && isfile(path)
            error("regenerate: $(path) exists; pass --overwrite to clobber.")
        end
        append!(paths, write_fixture!(b, path; times=[t_out]))
    end
    return paths
end

function _regenerate_one!(spec::BenchmarkSpec; fixtures_dir, overwrite)
    return run_mirror_benchmark!(spec; fixtures_dir, overwrite)
end

# MISMIP3D ATT ramp: drives YelmoMirror through `MISMIP3D_ATT_TIMES`
# at the corresponding `MISMIP3D_ATT_RF` rate-factor values, writes
# one fixture at the final phase end.
function _regenerate_mismip3d_att!(; fixtures_dir, overwrite)
    b = MISMIP3DBenchmark(:Stnd; dx_km=16.0)
    t_end = Int(round(MISMIP3D_ATT_TIMES[end]))
    path = joinpath(fixtures_dir, "$(MISMIP3D_ATT_NAME)_t$(t_end).nc")
    if !overwrite && isfile(path)
        error("regenerate: $(path) exists; pass --overwrite to clobber.")
    end
    return YelmoBenchmarks._write_mismip3d_mirror_fixture_att!(
        b, path, MISMIP3D_ATT_TIMES, MISMIP3D_ATT_RF)
end

function main(args::Vector{String})
    overwrite = "--overwrite" in args
    only_names = filter(a -> !startswith(a, "--"), args)

    # Special-case dispatch for the MISMIP3D ATT ramp (not in SPECS
    # since it requires multi-phase orchestration that the generic
    # `run_mirror_benchmark!` does not support).
    do_att = isempty(only_names) || MISMIP3D_ATT_NAME in only_names
    only_att = !isempty(only_names) && all(==(MISMIP3D_ATT_NAME), only_names)

    if !only_att
        selected = isempty(only_names) ? SPECS :
                   filter(s -> _spec_name(s) in only_names, SPECS)
        if isempty(selected) && !do_att
            avail = join(vcat([_spec_name(s) for s in SPECS], MISMIP3D_ATT_NAME), ", ")
            error("regenerate.jl: no specs match $(only_names). Available: $(avail).")
        end

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
    end

    if do_att
        println("\n=== $(MISMIP3D_ATT_NAME)  (mirror, multi-phase ATT ramp) ===")
        paths = _regenerate_mismip3d_att!(fixtures_dir=FIXTURES_DIR,
                                          overwrite=overwrite)
        for p in paths
            sz_kb = round(filesize(p) / 1024; digits=1)
            println("  wrote $(basename(p))  ($(sz_kb) KB)")
        end
    end

    println("\nDone.")
end

main(ARGS)
