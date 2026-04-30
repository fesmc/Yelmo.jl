## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 3c (Commit 1) tests for the SIA kernels:
#
#   - `calc_shear_stress_3D!`  — per-layer x/y shear stress.
#   - `calc_uxy_sia_3D!`       — depth-recurrence horizontal velocity.
#
# These are kernel-level analytical unit tests. The wrapper
# `calc_velocity_sia!` and the `solver == "sia"` dispatch in
# `dyn_step!` land in Commit 2; integration / convergence tests
# against the Halfar dome land in Commit 4.

using Test
using Yelmo
using Oceananigans
using Oceananigans: interior

# 3D ice grid builder, matching the layout used by `yelmo_define_grids`
# (Bounded × Bounded × Bounded) and parametrised by an explicit
# `zeta_ac` vector of layer interfaces.
function _bounded_3d(Nx, Ny, zeta_ac; dx::Real = 1.0)
    return RectilinearGrid(size=(Nx, Ny, length(zeta_ac) - 1),
                           x=(0.0, Nx*dx), y=(0.0, Ny*dx),
                           z=zeta_ac,
                           topology=(Bounded, Bounded, Bounded))
end

# zeta_aa = layer-centre vector. For uniform layers between zeta_ac[1]
# and zeta_ac[end], zeta_aa[k] = 0.5 * (zeta_ac[k] + zeta_ac[k+1]).
_zeta_aa(zeta_ac) = [0.5 * (zeta_ac[k] + zeta_ac[k+1]) for k in 1:length(zeta_ac)-1]

# ======================================================================
# Test 1 — Zero-input → zero-output sanity.
# ======================================================================

@testset "sia: kernels — zero input gives zero output" begin
    Nx, Ny    = 3, 3
    zeta_ac   = collect(range(0.0, 1.0; length=5))   # Nz_aa = 4
    zeta_aa   = _zeta_aa(zeta_ac)
    Nz        = length(zeta_aa)
    g3        = _bounded_3d(Nx, Ny, zeta_ac; dx=1.0)

    taud_acx  = XFaceField(g3);  fill!(interior(taud_acx), 0.0)
    taud_acy  = YFaceField(g3);  fill!(interior(taud_acy), 0.0)
    f_ice     = CenterField(g3); fill!(interior(f_ice),    1.0)
    H_ice     = CenterField(g3); fill!(interior(H_ice),    1000.0)
    ATT       = CenterField(g3); fill!(interior(ATT),      1e-16)

    tau_xz    = XFaceField(g3)
    tau_yz    = YFaceField(g3)
    ux        = XFaceField(g3)
    uy        = YFaceField(g3)

    calc_shear_stress_3D!(tau_xz, tau_yz, taud_acx, taud_acy, f_ice, zeta_aa)
    @test maximum(abs.(interior(tau_xz))) == 0.0
    @test maximum(abs.(interior(tau_yz))) == 0.0

    calc_uxy_sia_3D!(ux, uy, tau_xz, tau_yz, taud_acx, taud_acy,
                     H_ice, f_ice, ATT, 3.0, zeta_aa)
    @test maximum(abs.(interior(ux))) == 0.0
    @test maximum(abs.(interior(uy))) == 0.0
end

# ======================================================================
# Test 2 — Linear-zeta tau_xz check (sign convention).
# ======================================================================
#
# With taud_acx = -1000 Pa uniformly and taud_acy = 0, we expect
#     tau_xz(i, j, k) = -(1 - zeta_aa[k]) * taud_acx
#                     =  1000 * (1 - zeta_aa[k])
# at every interior x-face, and tau_yz = 0 everywhere.

@testset "sia: calc_shear_stress_3D! — linear-zeta sign convention" begin
    Nx, Ny    = 3, 3
    zeta_ac   = collect(range(0.0, 1.0; length=5))   # Nz_aa = 4
    zeta_aa   = _zeta_aa(zeta_ac)
    Nz        = length(zeta_aa)
    g3        = _bounded_3d(Nx, Ny, zeta_ac; dx=1.0)

    taud_acx  = XFaceField(g3);  fill!(interior(taud_acx), -1000.0)
    taud_acy  = YFaceField(g3);  fill!(interior(taud_acy), 0.0)
    f_ice     = CenterField(g3); fill!(interior(f_ice),    1.0)

    tau_xz    = XFaceField(g3)
    tau_yz    = YFaceField(g3)

    calc_shear_stress_3D!(tau_xz, tau_yz, taud_acx, taud_acy, f_ice, zeta_aa)

    # Interior x-face slots [i+1, j, k] for i in 1:Nx, j in 1:Ny.
    Txz = interior(tau_xz)
    for k in 1:Nz
        expected = 1000.0 * (1.0 - zeta_aa[k])
        for j in 1:Ny, i in 1:Nx
            @test abs(Txz[i+1, j, k] - expected) < 1e-12
        end
    end

    # tau_yz must be exactly zero everywhere.
    @test maximum(abs.(interior(tau_yz))) == 0.0
end

# ======================================================================
# Test 3 — Boundary-replicate sanity on a 5×5 grid.
# ======================================================================

@testset "sia: calc_shear_stress_3D! — boundary replicate, no NaN/Inf" begin
    Nx, Ny    = 5, 5
    zeta_ac   = collect(range(0.0, 1.0; length=5))   # Nz_aa = 4
    zeta_aa   = _zeta_aa(zeta_ac)
    Nz        = length(zeta_aa)
    g3        = _bounded_3d(Nx, Ny, zeta_ac; dx=1.0)

    # Non-uniform taud_acx: linearly varying in i, uniform in j.
    taud_acx  = XFaceField(g3)
    Tx        = interior(taud_acx)
    for j in 1:Ny, i in 1:Nx
        Tx[i+1, j, 1] = -100.0 * i
    end
    taud_acy  = YFaceField(g3); fill!(interior(taud_acy), 0.0)
    f_ice     = CenterField(g3); fill!(interior(f_ice), 1.0)

    tau_xz    = XFaceField(g3)
    tau_yz    = YFaceField(g3)

    calc_shear_stress_3D!(tau_xz, tau_yz, taud_acx, taud_acy, f_ice, zeta_aa)

    Txz = interior(tau_xz)
    @test all(isfinite, Txz)
    @test all(isfinite, interior(tau_yz))

    # Per-layer scaling: tau_xz[i+1, j, k] = -(1 - zeta_aa[k]) * taud_acx[i+1, j, 1]
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        expected = -(1.0 - zeta_aa[k]) * Tx[i+1, j, 1]
        @test abs(Txz[i+1, j, k] - expected) < 1e-12
    end

    # Replicated leading slot: Txz[1, j, k] = Txz[2, j, k]
    for k in 1:Nz, j in 1:Ny
        @test Txz[1, j, k] == Txz[2, j, k]
    end
end

# ======================================================================
# Test 4 — calc_uxy_sia_3D! recurrence smoke (uniform layer-tau).
# ======================================================================
#
# Set tau_xz = -1000 Pa at every interior x-face / every layer (so the
# layer-mid average is also -1000), tau_yz = 0, H_ice = 1000 m,
# ATT = 1e-16 Pa^-3 yr^-1, n_glen = 3. Run kernel 2 and check that:
#   - ux is finite everywhere (no NaN / Inf);
#   - ux(:, :, 1) = 0 (boundary condition);
#   - |ux| is monotonically non-decreasing in k for k >= 2 (the recurrence
#     adds a positive contribution at each layer because fact_ac > 0
#     and the sign of (tau(k) + tau(k-1)) is fixed throughout the column);
#   - uy stays at zero (since tau_yz is zero).

@testset "sia: calc_uxy_sia_3D! — recurrence smoke" begin
    Nx, Ny    = 4, 4
    zeta_ac   = collect(range(0.0, 1.0; length=5))   # Nz_aa = 4
    zeta_aa   = _zeta_aa(zeta_ac)
    Nz        = length(zeta_aa)
    g3        = _bounded_3d(Nx, Ny, zeta_ac; dx=1.0)

    # Inputs that are unused in the kernel body but required by the
    # signature.
    taud_acx  = XFaceField(g3); fill!(interior(taud_acx), 0.0)
    taud_acy  = YFaceField(g3); fill!(interior(taud_acy), 0.0)
    f_ice     = CenterField(g3); fill!(interior(f_ice), 1.0)

    H_ice     = CenterField(g3); fill!(interior(H_ice), 1000.0)
    ATT       = CenterField(g3); fill!(interior(ATT),   1e-16)

    # Direct fill on tau_xz / tau_yz (skipping kernel 1 to isolate
    # kernel 2's recurrence behaviour). Layer-uniform tau_xz = -1000.
    tau_xz    = XFaceField(g3)
    tau_yz    = YFaceField(g3)
    Txz       = interior(tau_xz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Txz[i+1, j, k] = -1000.0
    end
    # Replicate leading slot to match what kernel 1 would produce.
    @views Txz[1, :, :] .= Txz[2, :, :]
    fill!(interior(tau_yz), 0.0)

    ux = XFaceField(g3)
    uy = YFaceField(g3)

    calc_uxy_sia_3D!(ux, uy, tau_xz, tau_yz, taud_acx, taud_acy,
                     H_ice, f_ice, ATT, 3.0, zeta_aa)

    Ux = interior(ux)
    Uy = interior(uy)

    @test all(isfinite, Ux)
    @test all(isfinite, Uy)

    # k = 1 boundary: ux is zero (kernel initialises and only writes
    # k = 2:Nz).
    for j in 1:Ny, i in 1:Nx
        @test Ux[i+1, j, 1] == 0.0
    end

    # Recurrence sign: Ux moves monotonically in one direction with k
    # at every interior face. We don't pin the magnitude — just that
    # each step keeps the same sign and grows in absolute value.
    for j in 2:Ny-1, i in 2:Nx-1   # interior facial cells only
        for k in 2:Nz
            @test abs(Ux[i+1, j, k]) >= abs(Ux[i+1, j, k-1])
        end
    end

    # tau_yz = 0 ⇒ uy stays at zero everywhere (recurrence accumulates
    # 0.5*(0 + 0) at every layer).
    @test maximum(abs.(Uy)) == 0.0
end
