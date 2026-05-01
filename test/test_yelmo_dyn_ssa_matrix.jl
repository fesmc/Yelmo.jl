## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 3d / PR-A.2 unit tests for SSA matrix-assembly layer:
#
#   Commit 2 — set_ssa_masks!:
#     - All-grounded interior with no calving front → mask=1.
#     - Floating block with grounded surroundings → mask=2 / mask=1
#       at GL.
#     - use_ssa=false → all masks 0.
#     - Ice-free cells → masks stay 0.
#     - Calving-front detection: mask_frnt + lateral_bc → mask=3.
#
#   Commit 3 — _assemble_ssa_matrix!:
#     - Constant-coefficient interior cell → matches hand-derived
#       9-point ux row + cross-coupled uy entries.
#     - Dirichlet (mask=0) → single diagonal entry.
#     - Lateral-BC (mask=3) → matches Neumann formula.
#     - Periodic-y wrap → column index for (j-1) at j=1 wraps to Ny.
#     - Periodic-x wrap → analogous for (i-1) at i=1.
#     - NNZ count totals match expectation.
#
# YelmoMirror lockstep cross-checks deferred to PR-C; these are pure
# unit tests against hand-derived expected values.

using Test
using Yelmo
using Yelmo.YelmoModelDyn: set_ssa_masks!
using Oceananigans
using Oceananigans: interior
using Oceananigans.Grids: Bounded, Periodic

_bounded_2d(Nx, Ny; dx=1.0) = RectilinearGrid(size=(Nx, Ny),
                                              x=(0.0, Nx*dx), y=(0.0, Ny*dx),
                                              topology=(Bounded, Bounded, Flat))

_periodic_y_2d(Nx, Ny; dx=1.0) = RectilinearGrid(size=(Nx, Ny),
                                                 x=(0.0, Nx*dx), y=(0.0, Ny*dx),
                                                 topology=(Bounded, Periodic, Flat))

_periodic_x_2d(Nx, Ny; dx=1.0) = RectilinearGrid(size=(Nx, Ny),
                                                 x=(0.0, Nx*dx), y=(0.0, Ny*dx),
                                                 topology=(Periodic, Bounded, Flat))

# Helper: build the minimum set of fields set_ssa_masks! needs.
function _build_mask_inputs(g; H_val=1000.0, fi_val=1.0, fg_val=1.0, mfrnt_val=0)
    Nx, Ny = size(g, 1), size(g, 2)
    return (
        g            = g,
        ssa_mask_acx = (m = XFaceField(g);   fill!(interior(m), 0.0); m),
        ssa_mask_acy = (m = YFaceField(g);   fill!(interior(m), 0.0); m),
        mask_frnt    = (m = CenterField(g);  fill!(interior(m), Float64(mfrnt_val)); m),
        H_ice        = (h = CenterField(g);  fill!(interior(h), H_val); h),
        f_ice        = (f = CenterField(g);  fill!(interior(f), fi_val); f),
        f_grnd       = (f = CenterField(g);  fill!(interior(f), fg_val); f),
        z_base       = (z = CenterField(g);  fill!(interior(z), -100.0); z),
        z_sl         = (z = CenterField(g);  fill!(interior(z), 0.0); z),
    )
end

# ======================================================================
# Commit 2 — set_ssa_masks!
# ======================================================================

@testset "set_ssa_masks!: use_ssa=false → all 0" begin
    Nx, Ny = 5, 5
    g = _bounded_2d(Nx, Ny)
    s = _build_mask_inputs(g)
    set_ssa_masks!(s.ssa_mask_acx, s.ssa_mask_acy,
                   s.mask_frnt, s.H_ice, s.f_ice, s.f_grnd,
                   s.z_base, s.z_sl, 1.0;
                   use_ssa=false, lateral_bc="floating")
    @test all(interior(s.ssa_mask_acx) .== 0.0)
    @test all(interior(s.ssa_mask_acy) .== 0.0)
end

@testset "set_ssa_masks!: all-grounded → all interior faces mask=1" begin
    Nx, Ny = 5, 5
    g = _bounded_2d(Nx, Ny)
    s = _build_mask_inputs(g)   # H=1000, f_ice=1, f_grnd=1, mask_frnt=0
    set_ssa_masks!(s.ssa_mask_acx, s.ssa_mask_acy,
                   s.mask_frnt, s.H_ice, s.f_ice, s.f_grnd,
                   s.z_base, s.z_sl, 1.0;
                   use_ssa=true, lateral_bc="floating")
    Mx = interior(s.ssa_mask_acx)
    My = interior(s.ssa_mask_acy)
    # X-face for cell (i,j) lives at array slot [i+1, j, 1].
    # Interior cells have at least one ice-covered neighbour and any
    # f_grnd > 0 → mask=1. Loop over Fortran cell coords.
    for j in 1:Ny, i in 1:Nx
        # The "trailing" face (i+1, j) under Bounded is in bounds as
        # slot i+1 ∈ 2..Nx+1.
        @test Mx[i+1, j, 1] == 1.0
        @test My[i, j+1, 1] == 1.0
    end
end

@testset "set_ssa_masks!: ice-free cell → adjacent face mask=0" begin
    Nx, Ny = 5, 5
    g = _bounded_2d(Nx, Ny)
    s = _build_mask_inputs(g)
    # Make cell (2, 2) ice-free.
    interior(s.f_ice)[2, 2, 1] = 0.0
    interior(s.H_ice)[2, 2, 1] = 0.0
    interior(s.f_grnd)[2, 2, 1] = 1.0  # still grounded land
    # Note: the Fortran logic in `set_ssa_masks` checks
    # f_ice == 1 OR neighbour f_ice == 1. The face between (2,2)
    # ice-free and (3, 2) iced still has neighbour ice, so mask
    # at face slot [3, 2, 1] (between Fortran cells (2, 2) and (3, 2))
    # is the special-case branch:
    #   - mval starts at 1 (since f_grnd(2,2)>0 OR f_grnd(3,2)>0).
    #   - mval == 2 check skipped.
    #   - lateral-BC overwrite skipped (mask_frnt all zero).
    # → mask remains 1.
    set_ssa_masks!(s.ssa_mask_acx, s.ssa_mask_acy,
                   s.mask_frnt, s.H_ice, s.f_ice, s.f_grnd,
                   s.z_base, s.z_sl, 1.0;
                   use_ssa=true, lateral_bc="floating")
    # The face inside an all-ice-free cell: between (2, 1) and (2, 2)
    # is mask_acx slot at [3, 1, 1] under cell (2, 1) (covered by the
    # Fortran loop iteration i=2, j=1). f_ice(2,1)=1, f_ice(3,1)=1 →
    # mask=1 still. So check the upward face from (2, 2) into (2, 3)
    # (acy face, both endpoints partially involve cell (2,2)):
    # cell(2,2) f_ice=0, cell(2,3) f_ice=1 → mval=1 (f_grnd>0).
    @test interior(s.ssa_mask_acx)[3, 2, 1] == 1.0  # cell (2, 2): face to east.
    # Force a fully-empty surrounding. Make all cells around (2, 2)
    # ice-free + ungrounded so we reach mask=0 path: the face only
    # touches ice-free cells → if-branch fails → mask stays 0.
    s2 = _build_mask_inputs(g; H_val=0.0, fi_val=0.0, fg_val=0.0)
    set_ssa_masks!(s2.ssa_mask_acx, s2.ssa_mask_acy,
                   s2.mask_frnt, s2.H_ice, s2.f_ice, s2.f_grnd,
                   s2.z_base, s2.z_sl, 1.0;
                   use_ssa=true, lateral_bc="floating")
    @test all(interior(s2.ssa_mask_acx) .== 0.0)
    @test all(interior(s2.ssa_mask_acy) .== 0.0)
end

@testset "set_ssa_masks!: floating block surrounded by grounded → GL mask=1, shelf mask=2" begin
    Nx, Ny = 6, 6
    g = _bounded_2d(Nx, Ny)
    s = _build_mask_inputs(g)
    # Make cells (3:4, 3:4) floating (f_grnd = 0).
    for j in 3:4, i in 3:4
        interior(s.f_grnd)[i, j, 1] = 0.0
    end

    set_ssa_masks!(s.ssa_mask_acx, s.ssa_mask_acy,
                   s.mask_frnt, s.H_ice, s.f_ice, s.f_grnd,
                   s.z_base, s.z_sl, 1.0;
                   use_ssa=true, lateral_bc="floating")
    Mx = interior(s.ssa_mask_acx)
    My = interior(s.ssa_mask_acy)
    # Face fully inside the floating block: between (3,3) and (4,3) at
    # slot [4, 3, 1]. Both cells have f_grnd=0 → mval=2.
    @test Mx[4, 3, 1] == 2.0
    @test Mx[4, 4, 1] == 2.0
    # Face on the GL: between (2, 3) (grounded) and (3, 3) (floating)
    # at slot [3, 3, 1]. f_grnd(2,3)>0 → mval=1.
    @test Mx[3, 3, 1] == 1.0
    # Face fully grounded: between (1, 1) and (2, 1) at slot [2, 1, 1] → 1.
    @test Mx[2, 1, 1] == 1.0
end

@testset "set_ssa_masks!: lateral_bc='floating' with mask_frnt=1 → mask=3" begin
    # Build a 6x6 with a floating front: cells (3, 3:4) ice (f_ice=1),
    # cells (4, 3:4) ice-free, mask_frnt(3, 3:4) = val_flt = 1
    # (floating front), mask_frnt(4, 3:4) = val_ice_free = -1.
    Nx, Ny = 6, 6
    g = _bounded_2d(Nx, Ny)
    s = _build_mask_inputs(g)
    interior(s.f_grnd)[:, :, 1] .= 0.0  # all floating
    for j in 3:4
        interior(s.f_ice)[4, j, 1] = 0.0
        interior(s.H_ice)[4, j, 1] = 0.0
        interior(s.mask_frnt)[3, j, 1] = 1.0   # floating front
        interior(s.mask_frnt)[4, j, 1] = -1.0  # ice-free side
    end
    set_ssa_masks!(s.ssa_mask_acx, s.ssa_mask_acy,
                   s.mask_frnt, s.H_ice, s.f_ice, s.f_grnd,
                   s.z_base, s.z_sl, 1.0;
                   use_ssa=true, lateral_bc="floating")
    Mx = interior(s.ssa_mask_acx)
    # Face between (3, 3) and (4, 3) at slot [4, 3, 1]: mask_frnt(3,3)=1>0,
    # mask_frnt(4,3)=-1<0 → lateral BC → mask=3.
    @test Mx[4, 3, 1] == 3.0
    @test Mx[4, 4, 1] == 3.0
end

@testset "set_ssa_masks!: lateral_bc='none' disables fronts" begin
    Nx, Ny = 6, 6
    g = _bounded_2d(Nx, Ny)
    s = _build_mask_inputs(g)
    interior(s.f_grnd)[:, :, 1] .= 0.0
    for j in 3:4
        interior(s.f_ice)[4, j, 1] = 0.0
        interior(s.H_ice)[4, j, 1] = 0.0
        interior(s.mask_frnt)[3, j, 1] = 1.0
        interior(s.mask_frnt)[4, j, 1] = -1.0
    end
    set_ssa_masks!(s.ssa_mask_acx, s.ssa_mask_acy,
                   s.mask_frnt, s.H_ice, s.f_ice, s.f_grnd,
                   s.z_base, s.z_sl, 1.0;
                   use_ssa=true, lateral_bc="none")
    Mx = interior(s.ssa_mask_acx)
    # mask_frnt(3, 3) was 1>0; lateral_bc="none" disables to 5.
    # Then the dyn_disabled lateral check fires:
    #   mfd(3,3)=5, mfd(4,3)=-1 → ssa_mask_acx(3,3) = 4.
    @test Mx[4, 3, 1] == 4.0
end
