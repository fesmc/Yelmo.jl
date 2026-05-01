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
using Yelmo.YelmoModelDyn: set_ssa_masks!, _assemble_ssa_matrix!
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

# ======================================================================
# Commit 3 — _assemble_ssa_matrix!
# ======================================================================

# Helper: construct the COO-buffer scratch and full input set for the
# matrix-assembly kernel. Returns a NamedTuple of all fields/buffers.
function _build_matrix_inputs(g; visc=1e10, beta=1e9, taud=1e5, mask_val=1.0)
    Nx, Ny = size(g, 1), size(g, 2)
    N_cells = Nx * Ny
    N_rows  = 2 * N_cells
    N_nz_max = 2 * 9 * N_cells

    return (
        g            = g,
        Nx           = Nx,
        Ny           = Ny,
        # Scratch buffers
        I_idx        = Vector{Int}(undef, N_nz_max),
        J_idx        = Vector{Int}(undef, N_nz_max),
        vals         = Vector{Float64}(undef, N_nz_max),
        b_vec        = Vector{Float64}(undef, N_rows),
        nnz_ref      = Ref{Int}(0),
        # Velocities (Picard input).
        ux_b         = (u = XFaceField(g);   fill!(interior(u), 0.0); u),
        uy_b         = (u = YFaceField(g);   fill!(interior(u), 0.0); u),
        # Coefficients.
        beta_acx     = (b = XFaceField(g);   fill!(interior(b), beta); b),
        beta_acy     = (b = YFaceField(g);   fill!(interior(b), beta); b),
        visc_eff_int = (v = CenterField(g);  fill!(interior(v), visc); v),
        visc_ab      = (v = Field((Face(), Face(), Center()), g);
                        fill!(interior(v), visc); v),
        # Masks.
        ssa_mask_acx = (m = XFaceField(g);   fill!(interior(m), mask_val); m),
        ssa_mask_acy = (m = YFaceField(g);   fill!(interior(m), mask_val); m),
        mask_frnt    = (m = CenterField(g);  fill!(interior(m), 0.0); m),
        # Geometry.
        H_ice        = (h = CenterField(g);  fill!(interior(h), 1000.0); h),
        f_ice        = (f = CenterField(g);  fill!(interior(f), 1.0);    f),
        # Forcing.
        taud_acx     = (t = XFaceField(g);   fill!(interior(t), taud); t),
        taud_acy     = (t = YFaceField(g);   fill!(interior(t), taud); t),
        taul_int_acx = (t = XFaceField(g);   fill!(interior(t), 0.0);  t),
        taul_int_acy = (t = YFaceField(g);   fill!(interior(t), 0.0);  t),
    )
end

# Lookup the matrix entry at (row, col) from COO buffers.
function _coo_get(I_idx, J_idx, vals, nnz, row, col)
    acc = 0.0
    found = false
    for k in 1:nnz
        if I_idx[k] == row && J_idx[k] == col
            acc += vals[k]
            found = true
        end
    end
    return found, acc
end

# Count COO entries with given row.
_coo_count_in_row(I_idx, nnz, row) = sum(I_idx[k] == row for k in 1:nnz)

@testset "_assemble_ssa_matrix!: 5x5 constant-coefficient interior ux row" begin
    Nx, Ny = 5, 5
    dx = 1000.0    # 1 km
    g  = _bounded_2d(Nx, Ny; dx=dx)
    visc = 1e10
    beta = 1e9
    taud = 1e5
    s = _build_matrix_inputs(g; visc=visc, beta=beta, taud=taud)

    # Use a free-slip / no-slip box. The Fortran "DEFAULT" boundary
    # branch is no-slip, but we want an interior cell (3, 3) that
    # is purely inner-SSA — that's any boundary type, since (3, 3)
    # is in the interior. Pick :bounded → all-no-slip.
    _assemble_ssa_matrix!(
        s.I_idx, s.J_idx, s.vals, s.b_vec, s.nnz_ref,
        s.ux_b, s.uy_b,
        s.beta_acx, s.beta_acy,
        s.visc_eff_int, s.visc_ab,
        s.ssa_mask_acx, s.ssa_mask_acy, s.mask_frnt,
        s.H_ice, s.f_ice,
        s.taud_acx, s.taud_acy,
        s.taul_int_acx, s.taul_int_acy,
        dx, dx, 0.0;
        boundaries=:bounded, lateral_bc="floating",
    )

    # Hand-derive the ux row at Fortran cell (3, 3).
    ij2n(i, j) = (j - 1) * Nx + i
    row_ux(i, j) = 2 * ij2n(i, j) - 1
    row_uy(i, j) = 2 * ij2n(i, j)

    nr = row_ux(3, 3)
    h2 = dx * dx

    # Inner-SSA stencil with constant viscosity + beta:
    expected = Dict{Int, Float64}()
    expected[row_ux(3, 3)] = -4.0 * 2 * visc / h2 - 2 * visc / h2 - beta
    expected[row_ux(4, 3)] =  4.0 * visc / h2
    expected[row_ux(2, 3)] =  4.0 * visc / h2
    expected[row_ux(3, 4)] =  visc / h2
    expected[row_ux(3, 2)] =  visc / h2
    expected[row_uy(3, 3)] = -3.0 * visc / h2
    expected[row_uy(4, 3)] =  3.0 * visc / h2
    expected[row_uy(4, 2)] = -3.0 * visc / h2
    expected[row_uy(3, 2)] =  3.0 * visc / h2

    for (col, expval) in expected
        found, val = _coo_get(s.I_idx, s.J_idx, s.vals, s.nnz_ref[], nr, col)
        @test found
        @test val ≈ expval atol=1e-10 * abs(expval) + 1e-15
    end

    # Row count: exactly 9 entries on this row.
    @test _coo_count_in_row(s.I_idx, s.nnz_ref[], nr) == 9

    # RHS = taud_acx(3, 3).
    @test s.b_vec[nr] ≈ taud
end

@testset "_assemble_ssa_matrix!: Dirichlet (mask=0) row" begin
    Nx, Ny = 5, 5
    dx = 1.0
    g  = _bounded_2d(Nx, Ny; dx=dx)
    s  = _build_matrix_inputs(g)

    # Override mask at face for cell (3, 3) (X-face slot [4, 3, 1]):
    # set to 0 → Dirichlet.
    interior(s.ssa_mask_acx)[4, 3, 1] = 0.0

    _assemble_ssa_matrix!(
        s.I_idx, s.J_idx, s.vals, s.b_vec, s.nnz_ref,
        s.ux_b, s.uy_b,
        s.beta_acx, s.beta_acy,
        s.visc_eff_int, s.visc_ab,
        s.ssa_mask_acx, s.ssa_mask_acy, s.mask_frnt,
        s.H_ice, s.f_ice,
        s.taud_acx, s.taud_acy,
        s.taul_int_acx, s.taul_int_acy,
        dx, dx, 0.0;
        boundaries=:bounded,
    )

    nr = 2 * ((3 - 1) * Nx + 3) - 1   # row_ux(3, 3)

    # Single diagonal entry, value 1.0.
    @test _coo_count_in_row(s.I_idx, s.nnz_ref[], nr) == 1
    found, val = _coo_get(s.I_idx, s.J_idx, s.vals, s.nnz_ref[], nr, nr)
    @test found && val == 1.0
    @test s.b_vec[nr] == 0.0
end

@testset "_assemble_ssa_matrix!: lateral BC (mask=3) ice-free to right" begin
    # Setup: cell (3, 3) is iced (f_ice = 1), cell (4, 3) is ice-free
    # (f_ice < 1). Set ssa_mask_acx at face [4, 3] to 3. Lateral BC
    # case 1 fires.
    Nx, Ny = 5, 5
    dx = 1000.0
    g  = _bounded_2d(Nx, Ny; dx=dx)
    visc = 1e10
    taul = 5e5
    s = _build_matrix_inputs(g; visc=visc)
    interior(s.ssa_mask_acx)[4, 3, 1] = 3.0
    interior(s.f_ice)[4, 3, 1] = 0.0
    interior(s.taul_int_acx)[4, 3, 1] = taul

    _assemble_ssa_matrix!(
        s.I_idx, s.J_idx, s.vals, s.b_vec, s.nnz_ref,
        s.ux_b, s.uy_b,
        s.beta_acx, s.beta_acy,
        s.visc_eff_int, s.visc_ab,
        s.ssa_mask_acx, s.ssa_mask_acy, s.mask_frnt,
        s.H_ice, s.f_ice,
        s.taud_acx, s.taud_acy,
        s.taul_int_acx, s.taul_int_acy,
        dx, dx, 0.0;
        boundaries=:bounded,
    )

    ij2n(i, j) = (j - 1) * Nx + i
    row_ux(i, j) = 2 * ij2n(i, j) - 1
    row_uy(i, j) = 2 * ij2n(i, j)

    nr = row_ux(3, 3)

    # Hand-derive Case 1 (ice-free to right): N_aa = visc.
    inv_dx = 1 / dx
    inv_dy = 1 / dx
    expected = Dict{Int, Float64}()
    expected[row_ux(2, 3)] = -4.0 * inv_dx * visc       # ux(im1, j)
    expected[row_uy(3, 2)] = -2.0 * inv_dy * visc       # uy(i, jm1)
    expected[row_ux(3, 3)] =  4.0 * inv_dx * visc       # ux(i, j)
    expected[row_uy(3, 3)] =  2.0 * inv_dy * visc       # uy(i, j)

    for (col, expval) in expected
        found, val = _coo_get(s.I_idx, s.J_idx, s.vals, s.nnz_ref[], nr, col)
        @test found
        @test val ≈ expval atol=1e-10 * abs(expval) + 1e-15
    end

    @test _coo_count_in_row(s.I_idx, s.nnz_ref[], nr) == 4
    @test s.b_vec[nr] ≈ taul
end

@testset "_assemble_ssa_matrix!: NNZ check (5x5, all-active no-slip box)" begin
    Nx, Ny = 5, 5
    dx = 1.0
    g  = _bounded_2d(Nx, Ny; dx=dx)
    s  = _build_matrix_inputs(g)

    _assemble_ssa_matrix!(
        s.I_idx, s.J_idx, s.vals, s.b_vec, s.nnz_ref,
        s.ux_b, s.uy_b,
        s.beta_acx, s.beta_acy,
        s.visc_eff_int, s.visc_ab,
        s.ssa_mask_acx, s.ssa_mask_acy, s.mask_frnt,
        s.H_ice, s.f_ice,
        s.taud_acx, s.taud_acy,
        s.taul_int_acx, s.taul_int_acy,
        dx, dx, 0.0;
        boundaries=:bounded,
    )

    # Boundary rows: with :bounded → all-no-slip, every boundary
    # row is a single-diagonal entry. The boundary cells are
    # i=1, i=Nx, j=1, j=Ny. For 5x5 that's (5-1)*4 + 4 corner
    # double-counted = 16 unique boundary cells, but we need to
    # be careful because the kernel writes one row for ux and one
    # for uy per cell, and the "boundary row" check happens
    # independently per-row. So:
    #
    # For each cell:
    #   - ux row: if i==1 or i==Nx (no-slip → single entry, since
    #     bcs[3]=bcs[1]=:no_slip) OR (j==1 or j==Ny under no-slip)
    #     → single entry. Else: 9 entries.
    #   - uy row: same predicate, single entry on boundary; 9 inside.
    #
    # Interior cells: i ∈ {2, 3, 4}, j ∈ {2, 3, 4} → 9 cells × 2 rows
    # × 9 entries = 162 entries.
    # Boundary cells: 25 - 9 = 16 cells × 2 rows × 1 entry = 32 entries.
    # Total = 194.
    @test s.nnz_ref[] == 194
end

@testset "_assemble_ssa_matrix!: periodic-y wraps j-1 column index" begin
    # 4x3 grid with periodic-y. For ux at cell (2, 1), the inner-SSA
    # stencil reads `Nab(i, jm1)` and `uy(i, jm1)`/`uy(ip1, jm1)`/
    # `ux(i, jm1)` columns. Under periodic-y at j == 1, jm1 wraps to
    # Ny == 3.
    Nx, Ny = 4, 3
    dx = 1.0
    g  = _periodic_y_2d(Nx, Ny; dx=dx)
    s  = _build_matrix_inputs(g)

    _assemble_ssa_matrix!(
        s.I_idx, s.J_idx, s.vals, s.b_vec, s.nnz_ref,
        s.ux_b, s.uy_b,
        s.beta_acx, s.beta_acy,
        s.visc_eff_int, s.visc_ab,
        s.ssa_mask_acx, s.ssa_mask_acy, s.mask_frnt,
        s.H_ice, s.f_ice,
        s.taud_acx, s.taud_acy,
        s.taul_int_acx, s.taul_int_acy,
        dx, dx, 0.0;
        boundaries=:periodic_y,
    )

    ij2n(i, j) = (j - 1) * Nx + i
    row_ux(i, j) = 2 * ij2n(i, j) - 1
    row_uy(i, j) = 2 * ij2n(i, j)

    # ux at (2, 1) — interior in x (no left/right edge), j == 1 →
    # under :periodic_y, bcs(4) is :periodic, so this is NOT a
    # boundary row. The stencil hits column row_ux(2, jm1) where
    # jm1 wraps to 3.
    nr = row_ux(2, 1)
    col_wrap = row_ux(2, 3)   # ux(i, jm1) with jm1 = Ny = 3
    found, val = _coo_get(s.I_idx, s.J_idx, s.vals, s.nnz_ref[], nr, col_wrap)
    @test found
    # Value: visc / h² with our defaults.
    @test val ≈ 1e10
    # Cross-coupling uy(ip1=3, jm1=3) too:
    col_uy_wrap = row_uy(3, 3)
    found2, val2 = _coo_get(s.I_idx, s.J_idx, s.vals, s.nnz_ref[], nr, col_uy_wrap)
    @test found2
    @test val2 ≈ -3.0 * 1e10
end

@testset "_assemble_ssa_matrix!: NNZ buffer size adequacy" begin
    # Even for the largest expected case (all interior cells, all
    # 9-point + lots of cross-coupling), our 2 * 9 * Nx * Ny estimate
    # must hold. Try 8x8 with periodic-y → wrap means the j boundary
    # rows are also 9-point. nnz <= max stays under buffer.
    Nx, Ny = 8, 8
    dx = 1.0
    g  = _periodic_y_2d(Nx, Ny; dx=dx)
    s  = _build_matrix_inputs(g)
    N_nz_max = 2 * 9 * Nx * Ny

    _assemble_ssa_matrix!(
        s.I_idx, s.J_idx, s.vals, s.b_vec, s.nnz_ref,
        s.ux_b, s.uy_b,
        s.beta_acx, s.beta_acy,
        s.visc_eff_int, s.visc_ab,
        s.ssa_mask_acx, s.ssa_mask_acy, s.mask_frnt,
        s.H_ice, s.f_ice,
        s.taud_acx, s.taud_acy,
        s.taul_int_acx, s.taul_int_acy,
        dx, dx, 0.0;
        boundaries=:periodic_y,
    )

    @test s.nnz_ref[] <= N_nz_max
end

