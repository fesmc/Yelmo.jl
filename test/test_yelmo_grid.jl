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
# Note: as of the PR-A.1 polish commit, the SIA kernel's face-indexing
# convention is generalised via `_ip1_modular` / `_jp1_modular` (see
# `src/dyn/topology_helpers.jl`). Under Periodic-y the `j+1` write
# wraps modularly to slot 1 instead of going out of bounds, so the
# kernel runs end-to-end on `(Bounded, Periodic, Bounded)`.

using Test
using Yelmo
using Yelmo.YelmoModelDyn: calc_velocity_sia!
using Oceananigans
using Oceananigans: interior
using Oceananigans.Grids: topology, Bounded, Periodic
using Oceananigans.BoundaryConditions: fill_halo_regions!
using NCDatasets

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

@testset "grid: _ip1_modular / _jp1_modular dispatch on topology" begin
    ip1m = Yelmo.YelmoModelDyn._ip1_modular
    jp1m = Yelmo.YelmoModelDyn._jp1_modular
    Nx = 4
    Ny = 5

    # x-axis Bounded: writes `i+1` into face slot 2..Nx+1.
    @test ip1m(1, Nx, Bounded) == 2
    @test ip1m(2, Nx, Bounded) == 3
    @test ip1m(Nx, Nx, Bounded) == Nx + 1   # 5

    # x-axis Periodic: wraps the eastern face. At i=Nx the eastern face
    # is the wrapped Nx-th-cell face = slot 1 (mod1(Nx+1, Nx) = 1).
    @test ip1m(1, Nx, Periodic) == 2
    @test ip1m(2, Nx, Periodic) == 3
    @test ip1m(Nx - 1, Nx, Periodic) == Nx
    @test ip1m(Nx, Nx, Periodic) == 1

    # y-axis: same shape.
    @test jp1m(1, Ny, Bounded) == 2
    @test jp1m(Ny, Ny, Bounded) == Ny + 1
    @test jp1m(1, Ny, Periodic) == 2
    @test jp1m(Ny, Ny, Periodic) == 1

    # Symmetry: Bounded vs Periodic agree on `i + 1` for i < Nx.
    for i in 1:(Nx - 1)
        @test ip1m(i, Nx, Bounded) == ip1m(i, Nx, Periodic)
    end
    for j in 1:(Ny - 1)
        @test jp1m(j, Ny, Bounded) == jp1m(j, Ny, Periodic)
    end

    # Periodic full-loop: applying `_ip1_modular` Nx times gets back to
    # the starting face (via the mod1 wrap).
    let i = 1
        for _ in 1:Nx
            i = ip1m(i, Nx, Periodic)
            # Stay in 1..Nx range under Periodic.
            @test 1 ≤ i ≤ Nx
        end
        @test i == 1   # back to start after Nx steps
    end
end

# ======================================================================
# Test 4 — calc_uxy_sia_3D! runs end-to-end under periodic-y
# ======================================================================
#
# As of the PR-A.1 polish commit, the SIA kernel's face-write
# convention is generalised via `_ip1_modular` / `_jp1_modular` (see
# `src/dyn/topology_helpers.jl`), so the kernel runs cleanly on
# `(Bounded, Periodic, Bounded)`. The previous version of this test
# used `@test_throws BoundsError` to pin the pre-polish behaviour; this
# version asserts the kernel completes without error and produces
# finite output that matches the bounded-y answer in the interior.

@testset "calc_uxy_sia_3D!: (Bounded, Periodic, Bounded) runs cleanly" begin
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

    # All-zero stress and zero taud → every velocity contribution is
    # zero. The kernel should now run without erroring AND produce a
    # finite (zero) result.
    calc_uxy_sia_3D!(ux, uy, tau_xz, tau_yz,
                     taud_acx, taud_acy,
                     H_ice, f_ice, ATT, 3.0,
                     zeta_aa)
    @test all(isfinite, interior(ux))
    @test all(isfinite, interior(uy))
    @test all(interior(ux) .== 0.0)
    @test all(interior(uy) .== 0.0)
end

# Periodic-y uniform-slab SIA integration. With a y-uniform setup
# (uniform thickness, surface slope only in x), the SIA velocity field
# should be y-uniform after the kernel runs. This is a much tighter
# regression than the all-zero test above: a non-trivial stress field
# is exercised, AND the periodic-y wrap is verified by checking that
# the Ny-th and 1st rows are bit-identical (since under periodic-y
# they share the same modular index).

@testset "calc_velocity_sia!: periodic-y uniform x-slope produces y-uniform output" begin
    Nx, Ny, Nz = 6, 4, 3
    H0     = 2000.0
    n_glen = 3.0
    ATT0   = 1.0e-16

    zeta_ac = collect(range(0.0, 1.0; length=Nz + 1))
    zeta_c  = [0.5 * (zeta_ac[k] + zeta_ac[k+1]) for k in 1:Nz]

    # Build twin grids: one Bounded-y, one Periodic-y. Same Nx/Ny.
    g_bnd = RectilinearGrid(size=(Nx, Ny, Nz),
                            x=(0.0, Float64(Nx)), y=(0.0, Float64(Ny)),
                            z=zeta_ac,
                            topology=(Bounded, Bounded, Bounded))
    g_per = RectilinearGrid(size=(Nx, Ny, Nz),
                            x=(0.0, Float64(Nx)), y=(0.0, Float64(Ny)),
                            z=zeta_ac,
                            topology=(Bounded, Periodic, Bounded))

    function _make_state(g)
        # Allocate fields.
        ux_i     = XFaceField(g)
        uy_i     = YFaceField(g)
        ux_i_bar = XFaceField(g)
        uy_i_bar = YFaceField(g)
        ux_i_s   = XFaceField(g)
        uy_i_s   = YFaceField(g)
        tau_xz   = XFaceField(g)
        tau_yz   = YFaceField(g)
        H_ice    = CenterField(g)
        f_ice    = CenterField(g)
        taud_acx = XFaceField(g)
        taud_acy = YFaceField(g)
        ATT      = CenterField(g)

        # Uniform thickness, fully iced.
        fill!(interior(H_ice), H0)
        fill!(interior(f_ice), 1.0)
        fill!(interior(ATT),   ATT0)

        # Pre-load taud_acx with a constant non-zero value (pure
        # x-slope SIA driving stress); taud_acy = 0 (no y-slope).
        # All slots get the same value for a y-uniform test, including
        # the leading slot (Bounded) or the wrapped slot (Periodic).
        fill!(interior(taud_acx), 50_000.0)   # 50 kPa, eastward drive
        fill!(interior(taud_acy), 0.0)

        return (; ux_i, uy_i, ux_i_bar, uy_i_bar, ux_i_s, uy_i_s,
                tau_xz, tau_yz, H_ice, f_ice, taud_acx, taud_acy, ATT)
    end

    s_bnd = _make_state(g_bnd)
    s_per = _make_state(g_per)

    calc_velocity_sia!(s_bnd.ux_i, s_bnd.uy_i, s_bnd.ux_i_bar, s_bnd.uy_i_bar,
                       s_bnd.ux_i_s, s_bnd.uy_i_s,
                       s_bnd.tau_xz, s_bnd.tau_yz,
                       s_bnd.H_ice, s_bnd.f_ice,
                       s_bnd.taud_acx, s_bnd.taud_acy, s_bnd.ATT,
                       zeta_c, n_glen)

    calc_velocity_sia!(s_per.ux_i, s_per.uy_i, s_per.ux_i_bar, s_per.uy_i_bar,
                       s_per.ux_i_s, s_per.uy_i_s,
                       s_per.tau_xz, s_per.tau_yz,
                       s_per.H_ice, s_per.f_ice,
                       s_per.taud_acx, s_per.taud_acy, s_per.ATT,
                       zeta_c, n_glen)

    Ux_per     = interior(s_per.ux_i)
    Ux_bar_per = interior(s_per.ux_i_bar)
    Ux_s_per   = interior(s_per.ux_i_s)

    # Finite output everywhere.
    @test all(isfinite, Ux_per)
    @test all(isfinite, Ux_bar_per)
    @test all(isfinite, Ux_s_per)

    # Y-uniform: every j-row of Ux equals the j=1 row (the SIA solver
    # should not introduce any y-dependence given uniform inputs).
    for j in 2:Ny
        @test interior(s_per.ux_i)[:, j, :]     ≈ interior(s_per.ux_i)[:, 1, :]
        @test interior(s_per.ux_i_bar)[:, j, :] ≈ interior(s_per.ux_i_bar)[:, 1, :]
    end

    # Match with bounded-y in the y-interior away from boundaries.
    # Under Bounded-y the default `FluxBoundaryCondition: Nothing` halo
    # leaves the j = Ny+1 / j = 0 halo at zero — different from the
    # Periodic-y wrap (which fills those halos with the wrapped value).
    # That difference shows up at j = 1 and j = Ny (where the bed-segment
    # 4-corner stencil reads from the halo). Compare only the interior
    # rows j ∈ 2..Ny-1 where the halo doesn't matter; this still
    # exercises the modular face-write path for every i.
    Ux_bnd = interior(s_bnd.ux_i)
    Ux_per = interior(s_per.ux_i)
    j_interior = 2:Ny-1
    @test !isempty(j_interior)
    for i in 1:Nx
        @test Ux_per[i+1, j_interior, :] ≈ Ux_bnd[i+1, j_interior, :]
    end
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

# ======================================================================
# Test 5 — `_load_into_field!` is topology-aware on face fields
# ======================================================================
#
# Regression for the Periodic-y / Periodic-x branch added to the
# face-field `_load_into_field!` methods in `src/YelmoCore.jl`.
# Fixture data is written at `(Nx, Ny)` Center resolution (Yelmo
# Fortran's "all variables on cell-centred grid" convention). The
# loader has to reshape that for Yelmo.jl's staggered Field interiors:
#
#   - Bounded-y YFaceField interior `(Nx, Ny+1, 1)` — write into
#     slots `2:end`, replicate slot `1` from slot `2`.
#   - Periodic-y YFaceField interior `(Nx, Ny, 1)` — direct copy,
#     no replicate slot.
#
# Pre-fix, the YFaceField loader assumed Bounded-y unconditionally,
# so under Periodic-y the broadcast `int[:, 2:end, 1] .= data` raised
# a DimensionMismatch.

@testset "loader: _load_into_field! face-field topology dispatch" begin
    Nx, Ny = 4, 3
    # Synthetic test pattern: data[i, j] = 10 * i + j.
    data_x = Float64[10*i + j for i in 1:Nx, j in 1:Ny]
    data_y = Float64[10*i + j for i in 1:Nx, j in 1:Ny]

    mktempdir() do dir
        fp = joinpath(dir, "synth_face_load.nc")
        NCDataset(fp, "c") do ds
            defDim(ds, "x", Nx)
            defDim(ds, "y", Ny)
            v = defVar(ds, "ux_face", Float64, ("x", "y"))
            v[:, :] = data_x
            v2 = defVar(ds, "uy_face", Float64, ("x", "y"))
            v2[:, :] = data_y
        end

        # ----- Bounded-y YFaceField: legacy path, regression guard -----
        g_bnd = RectilinearGrid(size=(Nx, Ny),
                                x=(0.0, Float64(Nx)), y=(0.0, Float64(Ny)),
                                topology=(Bounded, Bounded, Flat))
        yf_bnd = YFaceField(g_bnd)
        @test size(interior(yf_bnd)) == (Nx, Ny + 1, 1)
        NCDataset(fp, "r") do ds
            Yelmo.YelmoCore._load_into_field!(yf_bnd, ds["uy_face"])
        end
        intb = interior(yf_bnd)
        # Slots 2..Ny+1 carry data[:, j] for j = 1..Ny.
        for j in 1:Ny, i in 1:Nx
            @test intb[i, j + 1, 1] == data_y[i, j]
        end
        # Slot 1 replicated from slot 2 (= data[:, 1]).
        for i in 1:Nx
            @test intb[i, 1, 1] == data_y[i, 1]
        end

        # ----- Periodic-y YFaceField: the fix -----
        g_per_y = RectilinearGrid(size=(Nx, Ny),
                                  x=(0.0, Float64(Nx)), y=(0.0, Float64(Ny)),
                                  topology=(Bounded, Periodic, Flat))
        yf_per = YFaceField(g_per_y)
        @test size(interior(yf_per)) == (Nx, Ny, 1)
        NCDataset(fp, "r") do ds
            Yelmo.YelmoCore._load_into_field!(yf_per, ds["uy_face"])
        end
        intp = interior(yf_per)
        # Direct copy: data[i, j] writes to int[i, j, 1] for j = 1..Ny.
        for j in 1:Ny, i in 1:Nx
            @test intp[i, j, 1] == data_y[i, j]
        end

        # ----- Bounded-x XFaceField: legacy path, regression guard -----
        xf_bnd = XFaceField(g_bnd)
        @test size(interior(xf_bnd)) == (Nx + 1, Ny, 1)
        NCDataset(fp, "r") do ds
            Yelmo.YelmoCore._load_into_field!(xf_bnd, ds["ux_face"])
        end
        intxb = interior(xf_bnd)
        for j in 1:Ny, i in 1:Nx
            @test intxb[i + 1, j, 1] == data_x[i, j]
        end
        for j in 1:Ny
            @test intxb[1, j, 1] == data_x[1, j]
        end

        # ----- Periodic-x XFaceField: the symmetric fix -----
        g_per_x = RectilinearGrid(size=(Nx, Ny),
                                  x=(0.0, Float64(Nx)), y=(0.0, Float64(Ny)),
                                  topology=(Periodic, Bounded, Flat))
        xf_per = XFaceField(g_per_x)
        @test size(interior(xf_per)) == (Nx, Ny, 1)
        NCDataset(fp, "r") do ds
            Yelmo.YelmoCore._load_into_field!(xf_per, ds["ux_face"])
        end
        intxp = interior(xf_per)
        for j in 1:Ny, i in 1:Nx
            @test intxp[i, j, 1] == data_x[i, j]
        end
    end
end
