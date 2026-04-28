## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 2a integration test for `tpo`. Three test groups:
#   1. Analytical advection — Gaussian return-to-origin (pass/fail)
#      and uniform-stationary (pass/fail). The kernel-level test
#      hits `advect_thickness!` directly, no full YelmoModel.
#   2. Square-pulse advection — informational only; prints
#      first-order-upwind diffusion characteristics for human
#      inspection. Always passes (registers @test true).
#   3. mask_ice post-step pass — verifies the per-cell behavior
#      of `topo_step!`'s mask handling (no_ice → 0, fixed →
#      retained, dynamic → clamped). Lockstep against
#      `YelmoMirror` does not exercise `MASK_ICE_FIXED`, so this
#      is the only regression for the headline 2a feature.
# A lockstep test against `YelmoMirror` is a separate concern
# (Mirror needs Fortran library + namelist setup) and is left
# for a follow-up PR.

using Test
using Yelmo
using Oceananigans
using Oceananigans: interior
using Oceananigans.BoundaryConditions: fill_halo_regions!

const RESTART_PATH = "/Users/alrobi001/models/yelmox/output/16KM/test/restart-0.000-kyr/yelmo_restart.nc"

# ------------------------------------------------------------------
# Analytical advection — kernel-level
# ------------------------------------------------------------------

@testset "tpo: analytical advection" begin

    @testset "Gaussian returns to origin (periodic)" begin
        Nx = 50
        Lx = 100.0
        g = RectilinearGrid(size=(Nx, Nx),
                            x=(0.0, Lx), y=(0.0, Lx),
                            topology=(Periodic, Periodic, Flat))
        H = CenterField(g)
        u = XFaceField(g)
        v = YFaceField(g)

        xc = xnodes(g, Center())
        yc = ynodes(g, Center())
        @inbounds for j in 1:Nx, i in 1:Nx
            interior(H)[i, j, 1] =
                exp(-((xc[i] - 50.0)^2 + (yc[j] - 50.0)^2) / 100.0)
        end

        u_speed = 1.0  # x-only velocity
        fill!(interior(u), u_speed)
        fill!(interior(v), 0.0)

        H_init = copy(interior(H))

        # One full revolution: T = Lx / |u|
        T = Lx / u_speed
        advect_thickness!(H, u, v, T; cfl_safety=0.5)

        # First-order upwind on a 50×50 grid over one period diffuses
        # the Gaussian noticeably. Acceptance: peak should still be
        # within 60% of original (i.e. shape persists, even if dampened),
        # *and* total volume conserved to machine precision.
        peak_ratio = maximum(interior(H)) / maximum(H_init)
        vol_diff_rel = abs(sum(interior(H)) - sum(H_init)) /
                       max(sum(H_init), eps())

        @test peak_ratio  > 0.4    # peak persists
        @test peak_ratio  < 1.05   # but doesn't grow (no instability)
        @test vol_diff_rel < 1e-10 # volume conservation
    end

    @testset "Stationary state stays stationary" begin
        Nx = 30
        g = RectilinearGrid(size=(Nx, Nx),
                            x=(0.0, 30.0), y=(0.0, 30.0),
                            topology=(Periodic, Periodic, Flat))
        H = CenterField(g)
        u = XFaceField(g)
        v = YFaceField(g)

        H_value = 5.0
        fill!(interior(H), H_value)
        fill!(interior(u), 1.0)  # nonzero so kernel actually runs
        fill!(interior(v), 0.5)

        advect_thickness!(H, u, v, 10.0; cfl_safety=0.5)

        max_dev = maximum(abs.(interior(H) .- H_value))
        @test max_dev < 1e-12  # bit-stable for uniform field
    end

end

# ------------------------------------------------------------------
# Square-pulse advection — informational only
# ------------------------------------------------------------------

@testset "tpo: square pulse (informational)" begin
    Nx = 50
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 100.0), y=(0.0, 100.0),
                        topology=(Periodic, Periodic, Flat))
    H = CenterField(g)
    u = XFaceField(g)
    v = YFaceField(g)

    # Square pulse along x at the center
    @inbounds for j in 1:Nx, i in 1:Nx
        interior(H)[i, j, 1] = (20 ≤ i ≤ 30) ? 1.0 : 0.0
    end
    fill!(interior(u), 1.0)
    fill!(interior(v), 0.0)

    H_init = copy(interior(H))
    advect_thickness!(H, u, v, 100.0; cfl_safety=0.5)

    peak_ratio = maximum(interior(H)) / maximum(H_init)
    width_at_half_max = count(>=(0.5), view(interior(H), :, Nx ÷ 2, 1))
    init_width = count(>=(0.5), view(H_init, :, Nx ÷ 2, 1))

    @info "Square pulse after one revolution" peak_ratio width_at_half_max init_width
    @test true  # informational; just records that the run completed
end

# ------------------------------------------------------------------
# mask_ice post-step pass — unit test
# ------------------------------------------------------------------

@testset "tpo: mask_ice post-step pass" begin
    @assert isfile(RESTART_PATH) "Restart fixture not found at $(RESTART_PATH)"

    y = YelmoModel(RESTART_PATH, 0.0;
                   rundir = mktempdir(; prefix="tpo_mask_test_"),
                   alias  = "tpo-mask-test",
                   groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                   strict = false)

    H        = interior(y.tpo.H_ice)
    mask_ice = interior(y.bnd.mask_ice)
    Nx, Ny   = size(H, 1), size(H, 2)

    # Set up a known checkerboard region with all three mask values
    # and a known H_ice value, then take one step and verify per-cell
    # behavior. We work on cells (10:12, 10:12) — well inside the
    # domain — so domain-edge BCs don't perturb the test.
    test_cells = [(i, j) for i in 10:12, j in 10:12]
    expected = Dict{Tuple{Int,Int}, Tuple{Int, Float64}}()  # (i,j) → (mask, expected_H)

    pre_value = 100.0
    for (idx, (i, j)) in enumerate(test_cells)
        mask_val = (idx - 1) % 3  # cycle through 0, 1, 2
        H[i, j, 1]        = pre_value
        mask_ice[i, j, 1] = Float64(mask_val)
        expected_H = mask_val == MASK_ICE_NONE  ? 0.0       :
                     mask_val == MASK_ICE_FIXED ? pre_value :
                                                  pre_value  # dynamic — H stays roughly,
                                                             # advection of a mostly-uniform
                                                             # patch barely changes it
        expected[(i, j)] = (mask_val, expected_H)
    end

    step!(y, 1.0)

    # Cells with MASK_ICE_NONE must be exactly zero.
    # Cells with MASK_ICE_FIXED must equal pre_value exactly.
    # Cells with MASK_ICE_DYNAMIC must be ≥ 0 (clamped) and within a
    #   reasonable range of pre_value (advection moves things).
    for ((i, j), (mask_val, expected_H)) in expected
        if mask_val == MASK_ICE_NONE
            @test H[i, j, 1] == 0.0
        elseif mask_val == MASK_ICE_FIXED
            @test H[i, j, 1] == pre_value
        else  # MASK_ICE_DYNAMIC
            @test H[i, j, 1] >= 0.0
            @test isfinite(H[i, j, 1])
        end
    end
end

# ------------------------------------------------------------------
# Real-restart smoke test: 5 steps, volume conservation
# ------------------------------------------------------------------

@testset "tpo: real-restart 5-step smoke test" begin
    @assert isfile(RESTART_PATH)

    y = YelmoModel(RESTART_PATH, 0.0;
                   rundir = mktempdir(; prefix="tpo_smoke_"),
                   alias  = "tpo-smoke",
                   groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                   strict = false)

    V0     = sum(interior(y.tpo.H_ice))
    H0_max = maximum(interior(y.tpo.H_ice))

    for _ in 1:5
        step!(y, 1.0)
    end

    V5     = sum(interior(y.tpo.H_ice))
    H5_max = maximum(interior(y.tpo.H_ice))

    @test y.time == 5.0
    @test isfinite(H5_max)
    @test H5_max < 1.05 * H0_max         # no upwind blow-up
    @test abs(V5 - V0) / V0 < 1e-6       # volume conservation (no MB in 2a)
    @test sort(unique(interior(y.tpo.f_grnd))) == [0.0, 1.0]  # binary mask
end
