## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# mat 1.2 unit tests for the Glen-law viscosity primitives:
#
#   - calc_viscosity_glen!:
#       * zero-strain limit reduces to 0.5·ATT^(-1/n)·eps_0^((1-n)/n)
#       * scaling with `de` for de ≫ eps_0: visc(α·de) = visc(de)·α^((1-n)/n)
#       * `f_ice = 0` zeros the column
#       * `visc_min` floor clamps small ATT or large de cases
#
#   - calc_visc_int!:
#       * uniform visc → ∫₀¹ visc dζ · H = visc · H (exact under
#         constant-extrapolation Center-stagger convention)
#       * linear-in-zeta visc → integral matches mean(visc) · H
#       * `f_ice = 0` zeros the cell
#
# Pure unit tests against hand-derived expected values, no fixtures.

using Test
using Yelmo
using Oceananigans
using Oceananigans.Fields: interior

# Helper: allocate the (visc, de, ATT, f_ice, H_ice) state on a small
# Nx × Ny × Nz Center grid and return a NamedTuple.
function _make_visc_state(Nx, Ny, Nz; H_const=2000.0, dx=1e3, dy=1e3)
    g_2d = RectilinearGrid(size=(Nx, Ny),
                           x=(0.0, Nx*dx), y=(0.0, Ny*dy),
                           topology=(Bounded, Bounded, Flat))

    # Uniform zeta_aa over (0, 1) — interior Center stagger, length Nz.
    zeta_aa = collect(range(0.5/Nz, 1.0 - 0.5/Nz, length=Nz))
    zeta_ac = vcat(0.0, 0.5*(zeta_aa[1:end-1] .+ zeta_aa[2:end]), 1.0)

    g_3d = RectilinearGrid(size=(Nx, Ny, Nz),
                           x=(0.0, Nx*dx), y=(0.0, Ny*dy), z=zeta_ac,
                           topology=(Bounded, Bounded, Bounded))

    visc  = CenterField(g_3d)
    de    = CenterField(g_3d)
    ATT   = CenterField(g_3d)
    H_ice = CenterField(g_2d); fill!(interior(H_ice), H_const)
    f_ice = CenterField(g_2d); fill!(interior(f_ice), 1.0)
    visc_int = CenterField(g_2d)

    return (; g_2d, g_3d, zeta_aa, Nx, Ny, Nz, H_const,
              visc, de, ATT, H_ice, f_ice, visc_int)
end


@testset "calc_viscosity_glen! — zero-strain regularised limit" begin
    # de = 0 → de_reg = eps_0 → visc = 0.5·ATT^(-1/n)·eps_0^((1-n)/n).
    s = _make_visc_state(3, 3, 4)
    n_glen   = 3.0
    visc_min = 1.0e-30   # effectively disable floor
    eps_0    = 1.0e-6
    ATT_val  = 1.0e-18

    fill!(interior(s.de), 0.0)
    fill!(interior(s.ATT), ATT_val)

    calc_viscosity_glen!(s.visc, s.de, s.ATT, s.f_ice;
                         n_glen=n_glen, visc_min=visc_min, eps_0=eps_0)

    expected = 0.5 * ATT_val^(-1/n_glen) * eps_0^((1.0 - n_glen)/n_glen)
    V = interior(s.visc)
    for k in 1:s.Nz, j in 1:s.Ny, i in 1:s.Nx
        @test isapprox(V[i, j, k], expected; rtol=1e-12)
    end
end

@testset "calc_viscosity_glen! — strain-rate scaling for de ≫ eps_0" begin
    # In the regularised regime de ≫ eps_0, doubling de scales visc
    # by 2^((1-n)/n). With n = 3, the factor is 2^(-2/3) ≈ 0.62996.
    s1 = _make_visc_state(2, 2, 3)
    s2 = _make_visc_state(2, 2, 3)
    n_glen   = 3.0
    visc_min = 1.0e-30
    eps_0    = 1.0e-12   # very small so de = 1 is clearly ≫ eps_0
    ATT_val  = 1.0e-18

    fill!(interior(s1.de),  1.0); fill!(interior(s1.ATT), ATT_val)
    fill!(interior(s2.de),  2.0); fill!(interior(s2.ATT), ATT_val)

    calc_viscosity_glen!(s1.visc, s1.de, s1.ATT, s1.f_ice;
                         n_glen=n_glen, visc_min=visc_min, eps_0=eps_0)
    calc_viscosity_glen!(s2.visc, s2.de, s2.ATT, s2.f_ice;
                         n_glen=n_glen, visc_min=visc_min, eps_0=eps_0)

    factor = 2.0^((1.0 - n_glen)/n_glen)
    V1 = interior(s1.visc); V2 = interior(s2.visc)
    for k in 1:size(V1, 3), j in 1:size(V1, 2), i in 1:size(V1, 1)
        @test isapprox(V2[i, j, k], V1[i, j, k] * factor; rtol=1e-10)
    end
end

@testset "calc_viscosity_glen! — f_ice = 0 zeros column" begin
    s = _make_visc_state(2, 2, 3)
    fill!(interior(s.f_ice), 0.0)
    fill!(interior(s.de),  0.5)
    fill!(interior(s.ATT), 1.0e-18)

    calc_viscosity_glen!(s.visc, s.de, s.ATT, s.f_ice;
                         n_glen=3.0, visc_min=1.0e3, eps_0=1.0e-6)

    @test all(interior(s.visc) .== 0.0)
end

@testset "calc_viscosity_glen! — visc_min floor" begin
    # Pick ATT and de so the analytical visc is below the chosen floor.
    s = _make_visc_state(2, 2, 3)
    fill!(interior(s.de),  1.0e6)   # huge strain-rate → tiny viscosity
    fill!(interior(s.ATT), 1.0)     # large ATT → tiny ATT^(-1/n)
    visc_min = 1.0e3

    calc_viscosity_glen!(s.visc, s.de, s.ATT, s.f_ice;
                         n_glen=3.0, visc_min=visc_min, eps_0=1.0e-6)

    V = interior(s.visc)
    @test all(V .≈ visc_min)
end


@testset "calc_visc_int! — uniform visc → visc · H" begin
    # Constant-extrap convention: ∫₀¹ const dζ = const exactly. Then
    # visc_int = const · H_ice.
    s = _make_visc_state(3, 3, 5)
    fill!(interior(s.f_ice), 1.0)
    visc_const = 7.5e9
    fill!(interior(s.visc), visc_const)

    calc_visc_int!(s.visc_int, s.visc, s.H_ice, s.f_ice, s.zeta_aa)

    Vi = interior(s.visc_int)
    expected = visc_const * s.H_const
    for j in 1:s.Ny, i in 1:s.Nx
        @test isapprox(Vi[i, j, 1], expected; rtol=1e-12)
    end
end

@testset "calc_visc_int! — linear-in-zeta → mean · H" begin
    # visc(ζ) = a + b·ζ on Center grid. Constant-extrap at boundaries
    # reproduces the depth-mean exactly when the per-layer values are
    # symmetric about ζ = 0.5 (uniform zeta_aa with odd Nz centred is
    # the cleanest case). Use Nz = 5 with zeta_aa = (0.1, 0.3, 0.5, 0.7, 0.9).
    s = _make_visc_state(2, 2, 5)
    fill!(interior(s.f_ice), 1.0)
    a = 1.0e9; b = 4.0e9
    V = interior(s.visc)
    for k in 1:s.Nz
        V[:, :, k] .= a + b * s.zeta_aa[k]
    end

    calc_visc_int!(s.visc_int, s.visc, s.H_ice, s.f_ice, s.zeta_aa)

    Vi = interior(s.visc_int)
    # Constant-extrap on a uniformly-centred zeta_aa column gives the
    # exact arithmetic mean of the per-layer visc (each layer carries
    # weight 1/Nz under the trapezoidal scheme with constant boundary
    # extrapolation). For visc(ζ) = a + b·ζ symmetric about 0.5, the
    # mean is a + b·0.5 = a + b/2.
    expected = (a + b * 0.5) * s.H_const
    for j in 1:size(Vi, 2), i in 1:size(Vi, 1)
        @test isapprox(Vi[i, j, 1], expected; rtol=1e-12)
    end
end

@testset "calc_visc_int! — f_ice = 0 zeros cell" begin
    s = _make_visc_state(2, 2, 3)
    fill!(interior(s.f_ice), 0.0)
    fill!(interior(s.visc),  1.0e9)

    calc_visc_int!(s.visc_int, s.visc, s.H_ice, s.f_ice, s.zeta_aa)

    @test all(interior(s.visc_int) .== 0.0)
end
