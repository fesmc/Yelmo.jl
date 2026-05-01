## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 3d / PR-A.1 commit 3: periodic-y topology smoke tests.
#
# Three goals:
#
#   1. Confirm `RectilinearGrid` accepts `topology=(Bounded, Periodic, Flat)`
#      and that `fill_halo_regions!` wraps interior values into halo
#      cells across the y-axis automatically (Oceananigans).
#
#   2. Verify the SIA wrapper assertion has been relaxed to allow
#      periodic-y topology (the `calc_uxy_sia_3D!` kernel should accept
#      `(Bounded, Periodic, Bounded)` ice grids without an error).
#
#   3. Verify the topology-aware `_neighbor_jm1` / `_neighbor_im1`
#      helpers dispatch correctly on the topology singleton.
#
# Note: the SIA kernel's YFace-indexing convention `Uy[i, j+1, k]` was
# designed for `topology(grid, 2) === Bounded` (interior shape
# `(Nx, Ny+1, 1)`). Under Periodic-y, YFace interior shape is
# `(Nx, Ny, 1)` — the `j+1` write at `j=Ny` goes out of bounds. This
# is a follow-up scope item: the kernel runs through the fact_ab loop
# but can BoundsError at the velocity write. PR-A.1 lays the
# foundation (assertion relaxed, helpers in place, `dyn.scratch` prepped)
# but the full periodic-y SIA integration test arrives once the
# YFace-indexing convention is generalised.

using Test
using Yelmo
using Oceananigans
using Oceananigans: interior
using Oceananigans.Grids: topology, Bounded, Periodic
using Oceananigans.BoundaryConditions: fill_halo_regions!

# ======================================================================
# Test 1 — RectilinearGrid construction with periodic-y topology
# ======================================================================

@testset "grid: periodic-y RectilinearGrid construction" begin
    Nx, Ny = 4, 3
    g = RectilinearGrid(size=(Nx, Ny),
                        x=(0.0, Float64(Nx)), y=(0.0, Float64(Ny)),
                        topology=(Bounded, Periodic, Flat))
    @test topology(g, 1) === Bounded
    @test topology(g, 2) === Periodic
    @test topology(g, 3) === Flat
end

@testset "grid: yelmo_define_grids passes through periodic-y" begin
    # `resolve_boundaries(:periodic_y)` returns `(Bounded, Periodic)`.
    # Smoke: a small custom grid build via `yelmo_define_grids`.
    xc = collect(0.5:1.0:3.5)            # 4 cells, dx=1
    yc = collect(0.5:1.0:2.5)            # 3 cells, dy=1
    zeta_ac     = [0.0, 0.5, 1.0]
    zeta_r_ac   = [0.0, 1.0]
    g, gt, gr = Yelmo.YelmoCore.yelmo_define_grids(xc, yc, zeta_ac, zeta_r_ac;
                                                   boundaries = :periodic_y)
    @test topology(g,  2) === Periodic
    @test topology(gt, 2) === Periodic
    @test topology(gr, 2) === Periodic
end

# ======================================================================
# Test 2 — fill_halo_regions! wraps under periodic-y
# ======================================================================

@testset "grid: CenterField halo-fills wrap on periodic-y axis" begin
    Nx, Ny = 4, 3
    g = RectilinearGrid(size=(Nx, Ny),
                        x=(0.0, Float64(Nx)), y=(0.0, Float64(Ny)),
                        topology=(Bounded, Periodic, Flat))
    f = CenterField(g)
    # Fill interior with a y-pattern: column j has value j.
    for j in 1:Ny, i in 1:Nx
        interior(f)[i, j, 1] = Float64(j)
    end
    fill_halo_regions!(f)

    # `parent(f)` exposes the halo-padded array. Halo width is
    # symmetric on the periodic axis (Nh_each = (size(pa, 2) - Ny) ÷ 2).
    pa  = parent(f)
    Nh_each = (size(pa, 2) - Ny) ÷ 2
    @test Nh_each ≥ 1

    # In Oceananigans' parent-array layout, interior[i, j, 1] sits at
    # pa[Nh_each_x + i, Nh_each_y + j, 1] for 1-based interior indices.
    # We want to confirm the halo wraps: the cell immediately south of
    # the interior (parent index = Nh_each_y) should equal interior(j=Ny);
    # the cell immediately north (parent index = Nh_each_y + Ny + 1)
    # should equal interior(j=1).
    Nh_each_x = (size(pa, 1) - Nx) ÷ 2
    # First interior x-column: pa-row at Nh_each_x + 1.
    pa_col_x = Nh_each_x + 1
    @test pa[pa_col_x, Nh_each + 0,      1] == Float64(Ny)   # one below first interior
    @test pa[pa_col_x, Nh_each + Ny + 1, 1] == Float64(1)    # one above last interior
end

@testset "grid: XFaceField halo-fills under periodic-y" begin
    Nx, Ny = 4, 3
    g = RectilinearGrid(size=(Nx, Ny),
                        x=(0.0, Float64(Nx)), y=(0.0, Float64(Ny)),
                        topology=(Bounded, Periodic, Flat))
    xf = XFaceField(g)
    # XFaceField has interior shape (Nx+1, Ny, 1) under (Bounded, Periodic).
    @test size(interior(xf)) == (Nx + 1, Ny, 1)
    # Fill with column-pattern.
    for j in 1:Ny, i in 1:Nx+1
        interior(xf)[i, j, 1] = Float64(j)
    end
    fill_halo_regions!(xf)
    pa = parent(xf)
    # The y-halo wraps for an XFaceField too. Use the same Nh-symmetric
    # parent-index calculation as for CenterField.
    Nh_each_y = (size(pa, 2) - Ny) ÷ 2
    Nh_each_x = (size(pa, 1) - (Nx + 1)) ÷ 2
    pa_col_x  = Nh_each_x + 1
    @test pa[pa_col_x, Nh_each_y + 0,      1] == Float64(Ny)
    @test pa[pa_col_x, Nh_each_y + Ny + 1, 1] == Float64(1)
end

@testset "grid: YFaceField interior shape differs by topology" begin
    Nx, Ny = 4, 3
    g_bnd  = RectilinearGrid(size=(Nx, Ny),
                             x=(0.0, Float64(Nx)), y=(0.0, Float64(Ny)),
                             topology=(Bounded, Bounded, Flat))
    g_per  = RectilinearGrid(size=(Nx, Ny),
                             x=(0.0, Float64(Nx)), y=(0.0, Float64(Ny)),
                             topology=(Bounded, Periodic, Flat))
    @test size(interior(YFaceField(g_bnd))) == (Nx, Ny + 1, 1)
    @test size(interior(YFaceField(g_per))) == (Nx, Ny,     1)
end

# ======================================================================
# Test 3 — _neighbor_jm1 / _neighbor_im1 helpers dispatch correctly
# ======================================================================

@testset "grid: _neighbor helpers dispatch on topology" begin
    nbjm1 = Yelmo.YelmoModelDyn._neighbor_jm1
    nbim1 = Yelmo.YelmoModelDyn._neighbor_im1
    Ny = 5
    Nx = 4
    # Bounded: clamp to 1.
    @test nbjm1(1, Ny, Bounded)  == 1
    @test nbjm1(2, Ny, Bounded)  == 1
    @test nbjm1(Ny, Ny, Bounded) == Ny - 1
    # Periodic: wrap.
    @test nbjm1(1, Ny, Periodic) == Ny    # j=1 wraps to Ny
    @test nbjm1(2, Ny, Periodic) == 1
    @test nbjm1(Ny, Ny, Periodic) == Ny - 1
    # x-axis analog.
    @test nbim1(1, Nx, Bounded)  == 1
    @test nbim1(1, Nx, Periodic) == Nx
end

# ======================================================================
# Test 4 — calc_uxy_sia_3D! topology assertion was relaxed
# ======================================================================
#
# The assertion change should accept (Bounded, Periodic, *) without an
# error. The kernel uses `Uy[i, j+1, k]` writes on YFaceField, which
# under periodic-y has interior shape `(Nx, Ny, 1)` — so writing at
# j=Ny would be out of bounds. The full kernel run is therefore
# expected to BoundsError on YFace writes; the assertion-relaxation
# test only confirms we get past the topology-check error and into
# the kernel body. Once the YFace indexing convention is generalised
# in a follow-up commit, the kernel can run end-to-end on periodic-y.

@testset "calc_uxy_sia_3D!: (Bounded, Periodic, Bounded) gets past assertion" begin
    Nx, Ny, Nz = 4, 3, 2
    zeta_ac = collect(range(0.0, 1.0; length=Nz + 1))
    zeta_aa = [0.5 * (zeta_ac[k] + zeta_ac[k+1]) for k in 1:Nz]

    g3 = RectilinearGrid(size=(Nx, Ny, Nz),
                         x=(0.0, Float64(Nx)), y=(0.0, Float64(Ny)),
                         z=zeta_ac,
                         topology=(Bounded, Periodic, Bounded))

    taud_acx = XFaceField(g3); fill!(interior(taud_acx), 0.0)
    taud_acy = YFaceField(g3); fill!(interior(taud_acy), 0.0)
    f_ice    = CenterField(g3); fill!(interior(f_ice),   1.0)
    H_ice    = CenterField(g3); fill!(interior(H_ice),   1000.0)
    ATT      = CenterField(g3); fill!(interior(ATT),     1e-16)
    tau_xz   = XFaceField(g3); fill!(interior(tau_xz),   0.0)
    tau_yz   = YFaceField(g3); fill!(interior(tau_yz),   0.0)
    ux       = XFaceField(g3)
    uy       = YFaceField(g3)

    # With all-zero stress and zero taud, every velocity contribution
    # is zero — even the bed-segment closed-form `tau_xz_bed = -taud_acx`
    # is zero. The kernel should run without erroring on the topology
    # assertion AND without bounds-erroring (since j+1 writes only go
    # to indices that are in bounds: `Uy[i, j+1, 1]` for j ∈ 1:Ny would
    # be out-of-bounds at j=Ny under periodic-y).
    #
    # In the current unmodified code, periodic-y triggers the BoundsError.
    # Once the YFace indexing convention is generalised, the kernel
    # should run cleanly. For now, expect a `BoundsError` for the
    # periodic-y kernel run — this test pins the pre-fix behaviour so
    # the follow-up generalisation can be detected as a behavioural
    # change.
    @test_throws BoundsError calc_uxy_sia_3D!(ux, uy, tau_xz, tau_yz,
                                              taud_acx, taud_acy,
                                              H_ice, f_ice, ATT, 3.0,
                                              zeta_aa)
end

@testset "calc_uxy_sia_3D!: bounded-y still works (regression check)" begin
    # Spot-check that relaxing the assertion didn't accidentally break
    # bounded-y. (The full SIA suite covers this; here a minimal
    # smoke just to be paranoid.)
    Nx, Ny, Nz = 4, 4, 2
    zeta_ac = collect(range(0.0, 1.0; length=Nz + 1))
    zeta_aa = [0.5 * (zeta_ac[k] + zeta_ac[k+1]) for k in 1:Nz]

    g3 = RectilinearGrid(size=(Nx, Ny, Nz),
                         x=(0.0, Float64(Nx)), y=(0.0, Float64(Ny)),
                         z=zeta_ac,
                         topology=(Bounded, Bounded, Bounded))

    taud_acx = XFaceField(g3); fill!(interior(taud_acx), 0.0)
    taud_acy = YFaceField(g3); fill!(interior(taud_acy), 0.0)
    f_ice    = CenterField(g3); fill!(interior(f_ice),   1.0)
    H_ice    = CenterField(g3); fill!(interior(H_ice),   1000.0)
    ATT      = CenterField(g3); fill!(interior(ATT),     1e-16)
    tau_xz   = XFaceField(g3); fill!(interior(tau_xz),   0.0)
    tau_yz   = YFaceField(g3); fill!(interior(tau_yz),   0.0)
    ux       = XFaceField(g3)
    uy       = YFaceField(g3)

    calc_uxy_sia_3D!(ux, uy, tau_xz, tau_yz,
                     taud_acx, taud_acy,
                     H_ice, f_ice, ATT, 3.0, zeta_aa)
    @test all(isfinite, interior(ux))
    @test all(isfinite, interior(uy))
end
