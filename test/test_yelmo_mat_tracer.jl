## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Kernel-level tests for the online age-tracer port (`mat`).
#
# Covers `calc_tracer_3D!` (driver) and the inactive-mask /
# zero-tendency invariants. Detailed end-to-end behaviour
# (advection of a column tracer field under non-trivial velocity
# fields) is deferred to a future regression test against
# YelmoMirror; this file is the unit-level safety net.

using Test
using Yelmo
using Oceananigans

# ----- Common fixture helper -----

function _tracer_grid(Nx::Int, Ny::Int, Nz_aa::Int; dx::Float64=1.0e3)
    # Bounded × Bounded × Bounded with Nz_aa interior centres on the
    # z axis. zeta_aa = `znodes(Center())` covers (0, 1) interior
    # midpoints, zeta_ac = `znodes(Face())` covers [0, 1] of length
    # Nz_aa + 1 — matches Path B convention.
    return RectilinearGrid(CPU();
        size = (Nx, Ny, Nz_aa),
        x = (0, Nx * dx),
        y = (0, Ny * dx),
        z = (0, 1),
        topology = (Bounded, Bounded, Bounded),
    )
end

@testset "mat: calc_tracer_3D! — inactive mask pins to X_srf" begin
    Nx, Ny, Nz = 7, 5, 4
    grid    = _tracer_grid(Nx, Ny, Nz)
    zeta_aa = znodes(grid, Center())
    zeta_ac = znodes(grid, Face())
    @assert length(zeta_aa) == Nz
    @assert length(zeta_ac) == Nz + 1

    X_ice = CenterField(grid)
    H_ice = CenterField(grid)
    bmb   = CenterField(grid)
    ux    = XFaceField(grid)
    uy    = YFaceField(grid)
    uz    = ZFaceField(grid)

    # Pre-fill with a non-zero "old" tracer value so we can verify
    # the inactive-mask path overwrites it.
    fill!(interior(X_ice), 42.0)
    # Sub-threshold ice thickness everywhere (`< 100 m`) →
    # `mask_tracers = false` for every column → every cell gets
    # pinned to `X_srf`.
    fill!(interior(H_ice), 50.0)

    X_srf = 1000.0
    Yelmo.calc_tracer_3D!(X_ice, X_srf, ux, uy, uz, H_ice, bmb,
                          collect(zeta_aa), collect(zeta_ac),
                          1.0e3, 1.0; kappa = 1.5)

    # Interior cells should all equal X_srf via the inactive-mask path.
    @test all(==(X_srf), interior(X_ice))
end

@testset "mat: calc_tracer_3D! — zero velocity + X_init=X_srf is fixed point" begin
    # With u = v = w = 0, bmb = 0, and X_ice already at X_srf
    # everywhere, a column at the steady state must remain at X_srf
    # (boundary value, no advection, no diffusion source).
    Nx, Ny, Nz = 9, 5, 6
    grid    = _tracer_grid(Nx, Ny, Nz)
    zeta_aa = collect(znodes(grid, Center()))
    zeta_ac = collect(znodes(grid, Face()))

    X_ice = CenterField(grid)
    H_ice = CenterField(grid)
    bmb   = CenterField(grid)
    ux    = XFaceField(grid)
    uy    = YFaceField(grid)
    uz    = ZFaceField(grid)

    X_srf = 2000.0
    fill!(interior(X_ice), X_srf)
    fill!(interior(H_ice), 1500.0)   # >> H_min, so cells are active
    # bmb = 0 → bmb_tot = -1e-3 (just the bmb_thinning term) → the
    # X_base extrapolation sees X[1] = X[2] = X_srf, so X_base = X_srf
    # too — fixed-point preserved.

    Yelmo.calc_tracer_3D!(X_ice, X_srf, ux, uy, uz, H_ice, bmb,
                          zeta_aa, zeta_ac, 1.0e3, 1.0; kappa = 1.5)

    # Interior columns should still be at X_srf to floating-point.
    @test maximum(abs.(interior(X_ice) .- X_srf)) < 1e-9
end

@testset "mat: calc_tracer_3D! — surface BC pulls column toward X_srf" begin
    # Initialize tracer at 0 everywhere, hit it with X_srf > 0.
    # The implicit solve must inject the surface value into the
    # interior — at minimum the topmost interior centre should
    # have moved measurably toward X_srf after one step (Crank-
    # Nicolson with strong diffusivity).
    Nx, Ny, Nz = 9, 5, 6
    grid    = _tracer_grid(Nx, Ny, Nz)
    zeta_aa = collect(znodes(grid, Center()))
    zeta_ac = collect(znodes(grid, Face()))

    X_ice = CenterField(grid)
    H_ice = CenterField(grid)
    bmb   = CenterField(grid)
    ux    = XFaceField(grid)
    uy    = YFaceField(grid)
    uz    = ZFaceField(grid)

    fill!(interior(X_ice), 0.0)
    fill!(interior(H_ice), 1500.0)

    X_srf = 1000.0
    # Use a very large kappa so the column moves visibly in one step.
    Yelmo.calc_tracer_3D!(X_ice, X_srf, ux, uy, uz, H_ice, bmb,
                          zeta_aa, zeta_ac, 1.0e3, 1.0; kappa = 1.0e8)

    # Interior columns: i ∈ 3..Nx-2, j ∈ 3..Ny-2 — pick the centre
    # column to verify directly. The topmost interior centre `Nz`
    # should have absorbed surface value via the boundary stencil.
    icenter = div(Nx, 2) + 1
    jcenter = div(Ny, 2) + 1
    column_top = interior(X_ice)[icenter, jcenter, Nz]
    @test column_top > 0.0
    @test column_top < X_srf
    # Border fill: the i=1 column should match the i=3 column.
    @test interior(X_ice)[1, jcenter, Nz] == interior(X_ice)[3, jcenter, Nz]
    @test interior(X_ice)[2, jcenter, Nz] == interior(X_ice)[3, jcenter, Nz]
end

@testset "mat: calc_tracer_3D! — Periodic topology errors clearly" begin
    # Periodic axes are not supported (the border-fill step would
    # clobber wraparound). The kernel should error explicitly.
    grid_periodic = RectilinearGrid(CPU();
        size = (8, 6, 4),
        x = (0, 8 * 1e3),
        y = (0, 6 * 1e3),
        z = (0, 1),
        topology = (Periodic, Bounded, Bounded),
    )
    X_ice = CenterField(grid_periodic)
    H_ice = CenterField(grid_periodic); fill!(interior(H_ice), 1500.0)
    bmb   = CenterField(grid_periodic)
    ux    = XFaceField(grid_periodic)
    uy    = YFaceField(grid_periodic)
    uz    = ZFaceField(grid_periodic)

    @test_throws ErrorException Yelmo.calc_tracer_3D!(
        X_ice, 1000.0, ux, uy, uz, H_ice, bmb,
        collect(znodes(grid_periodic, Center())),
        collect(znodes(grid_periodic, Face())),
        1.0e3, 1.0; kappa = 1.5)
end
