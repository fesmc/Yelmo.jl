## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# v0 integration test for the pure-Julia YelmoModel scaffolding.
#
# Round-trips the full no-physics path: construct a YelmoModel from a
# Yelmo Fortran restart, init_state!, open an output NetCDF via the
# AbstractYelmoModel-dispatched init_output, run a 5-step loop calling
# step!(y, dt) + write_output!, close, then assert that the output
# carries 5 time slices.
#
# The restart path is hard-coded for v0 — a smaller fixture committed to
# the repo will replace this once one is available.

using Test
using Yelmo
using Oceananigans: interior
using NCDatasets

const RESTART_PATH = "/Users/alrobi001/models/yelmox/output/16KM/test/restart-0.000-kyr/yelmo_restart.nc"

@testset "YelmoModel v0 scaffolding" begin
    @assert isfile(RESTART_PATH) "Restart fixture not found at $(RESTART_PATH)"

    rundir = mktempdir(; prefix="yelmo_model_v0_")
    out_path = joinpath(rundir, "yelmo_model.nc")

    # The Yelmo Fortran restart does not carry every variable in the
    # model variable tables (no `dta` group, missing `bnd::domain_mask`).
    # Skip `dta` and relax strict-mode for the remaining groups in v0.
    # No `p` is passed — the constructor should auto-build defaults and
    # emit a warning.
    y = @test_logs (:warn, r"No parameters supplied") match_mode=:any YelmoModel(
        RESTART_PATH, 0.0;
        rundir = rundir,
        alias  = "ymodel-v0",
        groups = (:bnd, :dyn, :mat, :thrm, :tpo),
        strict = false,
    )

    @test y isa AbstractYelmoModel
    @test y.alias == "ymodel-v0"
    @test y.time  == 0.0
    @test y.p isa YelmoModelParameters
    @test y.p.name == "ymodel-v0"

    # Sanity checks on a few loaded fields — values should be non-trivial,
    # i.e. came from the restart, not the default-initialised allocation.
    @test maximum(interior(y.tpo.H_ice))  > 0
    @test maximum(abs, interior(y.dyn.ux)) > 0
    @test minimum(interior(y.thrm.T_ice)) < 273.16  # ice ought to be cold somewhere

    init_state!(y, 0.0)
    @test y.time == 0.0

    out = init_output(y, out_path;
                      selection = OutputSelection(groups=[:tpo, :dyn, :thrm, :mat, :bnd]))

    # 5-iter loop: step + write. (No initial-state write — the v0 spec
    # says exactly 5 slices.)
    dt = 1.0
    for k in 1:5
        step!(y, dt)
        write_output!(out, y)
    end
    close(out)

    @test y.time == 5.0

    ds = NCDataset(out_path)
    try
        @test length(ds["time"]) == 5
        @test ds["time"][:] == [1.0, 2.0, 3.0, 4.0, 5.0]
        @test haskey(ds, "H_ice")

        # `tau_relax` exists in both `bnd` and `tpo` groups — they're
        # genuinely distinct fields (input forcing vs. realised
        # timescale). The IO layer disambiguates by group-prefixing
        # both occurrences. Plain `tau_relax` must not be present;
        # both prefixed names must be.
        @test !haskey(ds, "tau_relax")
        @test  haskey(ds, "bnd_tau_relax")
        @test  haskey(ds, "tpo_tau_relax")

        # Non-colliding names stay plain (regression guard).
        @test haskey(ds, "z_bed")        # bnd-only
        @test haskey(ds, "ux_bar")       # dyn-only
    finally
        close(ds)
    end
end
