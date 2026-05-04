## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# mat 1.4 unit tests for the enhancement-factor primitives.
#
# Both functions implement the same blend:
#
#     enh_ssa = f_grnd · enh_stream + (1 - f_grnd) · enh_shlf
#     enh     = f_shear · enh_shear + (1 - f_shear) · enh_ssa
#
# The 2D variant takes scalar `f_shear`, the 3D variant takes
# layer-dependent `f_shear`. Tests cover the four "pure" regime
# corners of the (f_grnd, f_shear) unit square plus a mixed case to
# verify the bilinear formula.

using Test
using Yelmo
using Oceananigans
using Oceananigans.Fields: interior


function _make_enh_state_2D(Nx, Ny)
    g = RectilinearGrid(size=(Nx, Ny), x=(0.0, 1.0), y=(0.0, 1.0),
                        topology=(Bounded, Bounded, Flat))
    return (; g, Nx, Ny,
              enh     = CenterField(g),
              f_grnd  = CenterField(g),
              f_shear = CenterField(g))
end

function _make_enh_state_3D(Nx, Ny, Nz)
    g_2d = RectilinearGrid(size=(Nx, Ny), x=(0.0, 1.0), y=(0.0, 1.0),
                           topology=(Bounded, Bounded, Flat))
    g_3d = RectilinearGrid(size=(Nx, Ny, Nz), x=(0.0, 1.0), y=(0.0, 1.0),
                           z=(0.0, 1.0),
                           topology=(Bounded, Bounded, Bounded))
    return (; g_2d, g_3d, Nx, Ny, Nz,
              enh     = CenterField(g_3d),
              f_grnd  = CenterField(g_2d),
              f_shear = CenterField(g_3d))
end


@testset "define_enhancement_factor_2D! — pure-regime corners" begin
    s = _make_enh_state_2D(3, 3)
    e_shear  = 3.0
    e_stream = 1.0
    e_shlf   = 0.7

    # Grounded shear: f_grnd=1, f_shear=1 → enh = enh_shear.
    fill!(interior(s.f_grnd),  1.0); fill!(interior(s.f_shear), 1.0)
    define_enhancement_factor_2D!(s.enh, s.f_grnd, s.f_shear;
        enh_shear=e_shear, enh_stream=e_stream, enh_shlf=e_shlf)
    @test all(interior(s.enh) .≈ e_shear)

    # Grounded stream: f_grnd=1, f_shear=0 → enh = enh_stream.
    fill!(interior(s.f_grnd),  1.0); fill!(interior(s.f_shear), 0.0)
    define_enhancement_factor_2D!(s.enh, s.f_grnd, s.f_shear;
        enh_shear=e_shear, enh_stream=e_stream, enh_shlf=e_shlf)
    @test all(interior(s.enh) .≈ e_stream)

    # Floating, no shear: f_grnd=0, f_shear=0 → enh = enh_shlf.
    fill!(interior(s.f_grnd),  0.0); fill!(interior(s.f_shear), 0.0)
    define_enhancement_factor_2D!(s.enh, s.f_grnd, s.f_shear;
        enh_shear=e_shear, enh_stream=e_stream, enh_shlf=e_shlf)
    @test all(interior(s.enh) .≈ e_shlf)

    # Floating + shear (algebraic edge case): enh = enh_shear regardless.
    fill!(interior(s.f_grnd),  0.0); fill!(interior(s.f_shear), 1.0)
    define_enhancement_factor_2D!(s.enh, s.f_grnd, s.f_shear;
        enh_shear=e_shear, enh_stream=e_stream, enh_shlf=e_shlf)
    @test all(interior(s.enh) .≈ e_shear)
end

@testset "define_enhancement_factor_2D! — bilinear blend" begin
    # Mixed values: hand-derive enh from the formula and check.
    s = _make_enh_state_2D(2, 2)
    e_shear  = 3.0
    e_stream = 1.0
    e_shlf   = 0.7

    # Per-cell test values.
    Fg = interior(s.f_grnd);  Fs = interior(s.f_shear)
    Fg[1, 1, 1] = 0.0;  Fs[1, 1, 1] = 0.5     # half-shear, fully floating
    Fg[2, 1, 1] = 0.5;  Fs[2, 1, 1] = 0.0     # GL stream
    Fg[1, 2, 1] = 1.0;  Fs[1, 2, 1] = 0.5     # mixed grounded
    Fg[2, 2, 1] = 0.25; Fs[2, 2, 1] = 0.75    # mostly-shear, mostly-floating

    define_enhancement_factor_2D!(s.enh, s.f_grnd, s.f_shear;
        enh_shear=e_shear, enh_stream=e_stream, enh_shlf=e_shlf)

    function expect(fg, fs)
        ssa = fg * e_stream + (1.0 - fg) * e_shlf
        return fs * e_shear + (1.0 - fs) * ssa
    end

    E = interior(s.enh)
    @test E[1, 1, 1] ≈ expect(0.0,  0.5)
    @test E[2, 1, 1] ≈ expect(0.5,  0.0)
    @test E[1, 2, 1] ≈ expect(1.0,  0.5)
    @test E[2, 2, 1] ≈ expect(0.25, 0.75)
end


@testset "define_enhancement_factor_3D! — pure regimes per layer" begin
    s = _make_enh_state_3D(2, 2, 4)
    e_shear  = 3.0
    e_stream = 1.0
    e_shlf   = 0.7

    # All-grounded; f_shear varies in z: layer 1 = pure shear,
    # layer 2 = pure stream, layers 3-4 = mid blend (0.5).
    fill!(interior(s.f_grnd), 1.0)
    Fs = interior(s.f_shear)
    Fs[:, :, 1] .= 1.0
    Fs[:, :, 2] .= 0.0
    Fs[:, :, 3] .= 0.5
    Fs[:, :, 4] .= 0.5

    define_enhancement_factor_3D!(s.enh, s.f_grnd, s.f_shear;
        enh_shear=e_shear, enh_stream=e_stream, enh_shlf=e_shlf)

    E = interior(s.enh)
    @test all(E[:, :, 1] .≈ e_shear)                      # pure shear
    @test all(E[:, :, 2] .≈ e_stream)                     # pure stream
    @test all(E[:, :, 3] .≈ 0.5*e_shear + 0.5*e_stream)   # half-half
    @test all(E[:, :, 4] .≈ 0.5*e_shear + 0.5*e_stream)
end

@testset "define_enhancement_factor_3D! — floating column" begin
    # f_grnd=0 everywhere; layer-dependent f_shear.
    # enh_ssa = enh_shlf for the column, then enh blends with shear.
    s = _make_enh_state_3D(2, 2, 3)
    e_shear  = 3.0
    e_stream = 1.0
    e_shlf   = 0.7

    fill!(interior(s.f_grnd), 0.0)
    Fs = interior(s.f_shear)
    Fs[:, :, 1] .= 0.0     # pure floating shelf
    Fs[:, :, 2] .= 1.0     # spurious shear-on-shelf — algebra still holds
    Fs[:, :, 3] .= 0.25

    define_enhancement_factor_3D!(s.enh, s.f_grnd, s.f_shear;
        enh_shear=e_shear, enh_stream=e_stream, enh_shlf=e_shlf)

    E = interior(s.enh)
    @test all(E[:, :, 1] .≈ e_shlf)
    @test all(E[:, :, 2] .≈ e_shear)
    @test all(E[:, :, 3] .≈ 0.25*e_shear + 0.75*e_shlf)
end

@testset "define_enhancement_factor_3D! — vertical-uniform f_shear matches 2D" begin
    # When `f_shear` is constant in z, the 3D kernel must agree with
    # the 2D kernel layer-by-layer.
    s3 = _make_enh_state_3D(3, 3, 5)
    s2 = _make_enh_state_2D(3, 3)
    e_shear  = 3.0
    e_stream = 1.0
    e_shlf   = 0.7

    # f_grnd: graded 0 → 1 across the grid.
    Fg3 = interior(s3.f_grnd); Fg2 = interior(s2.f_grnd)
    for i in 1:3, j in 1:3
        Fg3[i, j, 1] = (i + j - 2) / 4.0
        Fg2[i, j, 1] = Fg3[i, j, 1]
    end

    # f_shear: spatially varying, z-uniform.
    Fs3 = interior(s3.f_shear); Fs2 = interior(s2.f_shear)
    for i in 1:3, j in 1:3
        Fs2[i, j, 1] = 0.3 + 0.1 * i
        for k in 1:5
            Fs3[i, j, k] = Fs2[i, j, 1]
        end
    end

    define_enhancement_factor_2D!(s2.enh, s2.f_grnd, s2.f_shear;
        enh_shear=e_shear, enh_stream=e_stream, enh_shlf=e_shlf)
    define_enhancement_factor_3D!(s3.enh, s3.f_grnd, s3.f_shear;
        enh_shear=e_shear, enh_stream=e_stream, enh_shlf=e_shlf)

    E3 = interior(s3.enh); E2 = interior(s2.enh)
    for k in 1:5
        @test all(isapprox.(E3[:, :, k], E2[:, :, 1]; rtol=1e-12))
    end
end
