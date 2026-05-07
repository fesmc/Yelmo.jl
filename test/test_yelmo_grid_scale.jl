## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Tests for the aligned-resolution conservative remapping helpers
# (`src/utils/grid_scale.jl`): construction, hi→lo block-mean,
# lo→hi replicate, round-trip invariants, and 3D per-layer paths.

using Test
using Yelmo
using Oceananigans

# ---------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------

@testset "grid_scale: GridScaleWeights — explicit (Nx_lo, Ny_lo, Nx_hi, Ny_hi)" begin
    w = GridScaleWeights(4, 3, 12, 9)
    @test w.s == 3
    @test (w.Nx_lo, w.Ny_lo) == (4, 3)
    @test (w.Nx_hi, w.Ny_hi) == (12, 9)
end

@testset "grid_scale: GridScaleWeights — derived from scale factor" begin
    w = GridScaleWeights(5, 4, 2)
    @test w.s == 2
    @test (w.Nx_hi, w.Ny_hi) == (10, 8)
end

@testset "grid_scale: GridScaleWeights — non-divisible / mismatched factor errors" begin
    @test_throws ErrorException GridScaleWeights(4, 3, 13, 9)        # not divisible
    @test_throws ErrorException GridScaleWeights(4, 3, 12, 6)        # x:3, y:2 (anisotropic)
    @test_throws ErrorException GridScaleWeights(4, 3, 0)            # s < 1
    @test_throws ErrorException GridScaleWeights(0, 3, 0, 6)         # zero size
end

@testset "grid_scale: GridScaleWeights — from RectilinearGrid pair" begin
    Lx, Ly = 100.0e3, 75.0e3
    Nx_lo, Ny_lo = 5, 3
    s = 4
    grid_lo = RectilinearGrid(CPU();
        size = (Nx_lo, Ny_lo, 1),
        x = (0, Lx), y = (0, Ly), z = (0, 1),
        topology = (Bounded, Bounded, Bounded))
    grid_hi = RectilinearGrid(CPU();
        size = (Nx_lo * s, Ny_lo * s, 1),
        x = (0, Lx), y = (0, Ly), z = (0, 1),
        topology = (Bounded, Bounded, Bounded))
    w = GridScaleWeights(grid_lo, grid_hi)
    @test w.s == s
    @test (w.Nx_lo, w.Ny_lo) == (Nx_lo, Ny_lo)
    @test (w.Nx_hi, w.Ny_hi) == (Nx_lo * s, Ny_lo * s)
end

@testset "grid_scale: GridScaleWeights — RectilinearGrid extent mismatch errors" begin
    grid_lo = RectilinearGrid(CPU();
        size = (4, 3, 1), x = (0, 40.0e3), y = (0, 30.0e3), z = (0, 1),
        topology = (Bounded, Bounded, Bounded))
    # Same Nx_hi/Ny_hi ratio but a DIFFERENT physical x extent.
    grid_hi_bad = RectilinearGrid(CPU();
        size = (8, 6, 1), x = (0, 50.0e3), y = (0, 30.0e3), z = (0, 1),
        topology = (Bounded, Bounded, Bounded))
    @test_throws ErrorException GridScaleWeights(grid_lo, grid_hi_bad)
end

# ---------------------------------------------------------------------
# hi → lo: block-mean coarsening
# ---------------------------------------------------------------------

@testset "grid_scale: map_field_to_lo — uniform field round-trips identically" begin
    w   = GridScaleWeights(3, 2, 4)
    src = fill(7.5, w.Nx_hi, w.Ny_hi)
    dst = map_field_to_lo(src, w)
    @test size(dst) == (w.Nx_lo, w.Ny_lo)
    @test all(dst .== 7.5)
end

@testset "grid_scale: map_field_to_lo — block-mean of known pattern" begin
    # 2:1 coarsening of a 4×4 src into a 2×2 dst.
    # Block (1,1) covers src[1:2, 1:2] = [1, 2; 5, 6]   → mean 3.5
    # Block (2,1) covers src[3:4, 1:2] = [3, 4; 7, 8]   → mean 5.5
    # Block (1,2) covers src[1:2, 3:4] = [9, 10; 13, 14] → mean 11.5
    # Block (2,2) covers src[3:4, 3:4] = [11, 12; 15, 16] → mean 13.5
    w   = GridScaleWeights(2, 2, 2)
    src = Float64.(reshape(1:16, 4, 4))
    dst = map_field_to_lo(src, w)
    @test dst[1, 1] ≈ 3.5
    @test dst[2, 1] ≈ 5.5
    @test dst[1, 2] ≈ 11.5
    @test dst[2, 2] ≈ 13.5
end

@testset "grid_scale: map_field_to_lo — shape mismatch errors" begin
    w = GridScaleWeights(2, 2, 2)
    @test_throws ErrorException map_field_to_lo(zeros(3, 4), w)        # wrong src
    dst_bad = zeros(3, 2)
    @test_throws ErrorException map_field_to_lo!(dst_bad, zeros(4, 4), w)
end

# ---------------------------------------------------------------------
# lo → hi: replicate refinement
# ---------------------------------------------------------------------

@testset "grid_scale: map_field_to_hi — replicate to s×s blocks" begin
    w   = GridScaleWeights(2, 2, 3)
    src = Float64[1.0  2.0;
                  3.0  4.0]
    dst = map_field_to_hi(src, w)
    @test size(dst) == (6, 6)
    # The (1,1) lo cell maps to dst[1:3, 1:3] — all 1.0.
    @test all(dst[1:3, 1:3] .== 1.0)
    @test all(dst[4:6, 1:3] .== 3.0)
    @test all(dst[1:3, 4:6] .== 2.0)
    @test all(dst[4:6, 4:6] .== 4.0)
end

@testset "grid_scale: map_field_to_hi — shape mismatch errors" begin
    w = GridScaleWeights(2, 2, 3)
    @test_throws ErrorException map_field_to_hi(zeros(3, 2), w)
    dst_bad = zeros(7, 6)
    @test_throws ErrorException map_field_to_hi!(dst_bad, zeros(2, 2), w)
end

# ---------------------------------------------------------------------
# Round-trip invariants (intensive field)
# ---------------------------------------------------------------------

@testset "grid_scale: lo → hi → lo is identity (intensive field)" begin
    # Replicating then averaging an s×s block of identical values
    # returns the original — exact round-trip for lo-originating
    # fields.
    w   = GridScaleWeights(5, 4, 3)
    src = Float64.(reshape(1:20, 5, 4))
    dst = map_field_to_lo(map_field_to_hi(src, w), w)
    @test maximum(abs.(dst .- src)) < 1e-12
end

@testset "grid_scale: integral of (value · area) is conservative for hi → lo" begin
    # Σ(hi cell value) = s² · Σ(lo cell value) since every hi cell
    # has area = lo_area / s² and intensive coarsening means the
    # value-weighted-area integral is preserved.
    w   = GridScaleWeights(2, 3, 4)
    src = randn(w.Nx_hi, w.Ny_hi)
    dst = map_field_to_lo(src, w)
    @test sum(src) ≈ (w.s * w.s) * sum(dst) atol = 1e-12
end

# ---------------------------------------------------------------------
# 3D overloads
# ---------------------------------------------------------------------

@testset "grid_scale: 3D coarsen + refine round-trip" begin
    w  = GridScaleWeights(3, 2, 2)
    Nz = 4
    src_lo = Float64.(reshape(1:(w.Nx_lo * w.Ny_lo * Nz),
                              w.Nx_lo, w.Ny_lo, Nz))
    src_hi = map_field_to_hi(src_lo, w)
    @test size(src_hi) == (w.Nx_hi, w.Ny_hi, Nz)
    back = map_field_to_lo(src_hi, w)
    @test maximum(abs.(back .- src_lo)) < 1e-12
end

@testset "grid_scale: 3D vertical-dim mismatch errors" begin
    w = GridScaleWeights(2, 2, 2)
    src_hi = zeros(4, 4, 5)
    dst_lo_bad = zeros(2, 2, 4)         # wrong Nz
    @test_throws ErrorException map_field_to_lo!(dst_lo_bad, src_hi, w)
end

# ---------------------------------------------------------------------
# Edge case: s = 1 (no remap)
# ---------------------------------------------------------------------

@testset "grid_scale: s = 1 is the identity remap" begin
    w   = GridScaleWeights(4, 3, 1)
    src = Float64.(reshape(1:12, 4, 3))
    @test map_field_to_hi(src, w) == src
    @test map_field_to_lo(src, w) == src
end
