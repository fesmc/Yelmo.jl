## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# mat 1.5 unit tests for the 2D stress tensor.
#
# `calc_stress_tensor_2D!` builds (txx, tyy, txy, te, eig_1, eig_2)
# from `visc_bar` and the 2D strain-rate components. `txz` / `tyz`
# enter only via `te` and are passed in as inputs (allocation default
# zero in the current scope).
#
#   - Pure tensile (dxx = a, dyy = 0, dxy = 0):
#       txx = 2·visc·a, tyy = 0, txy = 0
#       te  = txx
#       (eig_1, eig_2) = (txx, 0)
#
#   - Pure shear (dxx = 0, dyy = 0, dxy = a):
#       txx = 0, tyy = 0, txy = 2·visc·a
#       te  = txy
#       (eig_1, eig_2) = (txy, -txy)
#
#   - Symmetric isotropic (dxx = dyy = a, dxy = 0): eigenvalues
#     should mathematically be (txx, txx) but Fortran's strict
#     `disc > 0` check returns `(0, 0)` — the Julia port replicates
#     this (intentional Fortran-parity bug).
#
#   - txz / tyz contributions to te.

using Test
using Yelmo
using Oceananigans
using Oceananigans.Fields: interior


function _make_stress_state(Nx, Ny)
    g = RectilinearGrid(size=(Nx, Ny), x=(0.0, 1.0), y=(0.0, 1.0),
                        topology=(Bounded, Bounded, Flat))
    return (; g, Nx, Ny,
              strs2D_txx = CenterField(g),
              strs2D_tyy = CenterField(g),
              strs2D_txy = CenterField(g),
              strs2D_te  = CenterField(g),
              strs2D_tau_eig_1 = CenterField(g),
              strs2D_tau_eig_2 = CenterField(g),
              visc_bar   = CenterField(g),
              strn2D_dxx = CenterField(g),
              strn2D_dyy = CenterField(g),
              strn2D_dxy = CenterField(g),
              strs2D_txz = CenterField(g),
              strs2D_tyz = CenterField(g))
end


# Wrapper that takes a `_make_stress_state` NamedTuple and runs the
# full driver — keeps the per-test setup compact.
function _run_stress!(s)
    calc_stress_tensor_2D!(s.strs2D_txx, s.strs2D_tyy, s.strs2D_txy,
                           s.strs2D_te,
                           s.strs2D_tau_eig_1, s.strs2D_tau_eig_2,
                           s.visc_bar,
                           s.strn2D_dxx, s.strn2D_dyy, s.strn2D_dxy,
                           s.strs2D_txz, s.strs2D_tyz)
end


@testset "calc_stress_tensor_2D! — pure tensile" begin
    s = _make_stress_state(3, 3)
    visc = 1.0e9
    a    = 0.01
    fill!(interior(s.visc_bar),   visc)
    fill!(interior(s.strn2D_dxx), a)
    fill!(interior(s.strn2D_dyy), 0.0)
    fill!(interior(s.strn2D_dxy), 0.0)

    _run_stress!(s)

    txx_expected = 2.0 * visc * a
    @test all(interior(s.strs2D_txx) .≈ txx_expected)
    @test all(interior(s.strs2D_tyy) .≈ 0.0)
    @test all(interior(s.strs2D_txy) .≈ 0.0)
    # te = sqrt(txx² + 0 + 0 + 0 + 0 + 0) = |txx|
    @test all(interior(s.strs2D_te) .≈ abs(txx_expected))
    # Eigenvalues of [txx 0; 0 0] = (txx, 0).
    @test all(interior(s.strs2D_tau_eig_1) .≈ txx_expected)
    @test all(interior(s.strs2D_tau_eig_2) .≈ 0.0)
end

@testset "calc_stress_tensor_2D! — pure shear" begin
    s = _make_stress_state(3, 3)
    visc = 1.0e9
    a    = 0.005
    fill!(interior(s.visc_bar),   visc)
    fill!(interior(s.strn2D_dxx), 0.0)
    fill!(interior(s.strn2D_dyy), 0.0)
    fill!(interior(s.strn2D_dxy), a)

    _run_stress!(s)

    txy_expected = 2.0 * visc * a
    @test all(interior(s.strs2D_txx) .≈ 0.0)
    @test all(interior(s.strs2D_tyy) .≈ 0.0)
    @test all(interior(s.strs2D_txy) .≈ txy_expected)
    # te = sqrt(0 + 0 + 0 + txy² + 0 + 0) = |txy|
    @test all(interior(s.strs2D_te) .≈ abs(txy_expected))
    # Eigenvalues of [0 t; t 0] = (+t, -t).
    @test all(interior(s.strs2D_tau_eig_1) .≈  txy_expected)
    @test all(interior(s.strs2D_tau_eig_2) .≈ -txy_expected)
end

@testset "calc_stress_tensor_2D! — degenerate symmetric (Fortran-parity bug)" begin
    # txx = tyy, txy = 0 → discriminant exactly 0, Fortran returns
    # (0, 0) per the strict `> 0` check. Julia port replicates.
    s = _make_stress_state(2, 2)
    visc = 1.0e9
    a    = 0.01
    fill!(interior(s.visc_bar),   visc)
    fill!(interior(s.strn2D_dxx), a)
    fill!(interior(s.strn2D_dyy), a)
    fill!(interior(s.strn2D_dxy), 0.0)

    _run_stress!(s)

    txx_expected = 2.0 * visc * a
    @test all(interior(s.strs2D_txx) .≈ txx_expected)
    @test all(interior(s.strs2D_tyy) .≈ txx_expected)
    @test all(interior(s.strs2D_txy) .≈ 0.0)
    # te = sqrt(txx² + tyy² + txx·tyy + 0 + 0 + 0) = sqrt(3)·|txx|
    @test all(interior(s.strs2D_te) .≈ sqrt(3.0) * abs(txx_expected))
    # Fortran-parity bug: (0, 0) instead of (txx, txx).
    @test all(interior(s.strs2D_tau_eig_1) .== 0.0)
    @test all(interior(s.strs2D_tau_eig_2) .== 0.0)
end

@testset "calc_stress_tensor_2D! — txz/tyz contribute only to te" begin
    s = _make_stress_state(2, 2)
    visc = 1.0e9
    a    = 0.001
    fill!(interior(s.visc_bar),   visc)
    fill!(interior(s.strn2D_dxx), a);  fill!(interior(s.strn2D_dyy), 0.0)
    fill!(interior(s.strn2D_dxy), 0.0)
    txz_val = 1.0e5
    tyz_val = 2.0e5
    fill!(interior(s.strs2D_txz), txz_val)
    fill!(interior(s.strs2D_tyz), tyz_val)

    _run_stress!(s)

    txx = 2.0 * visc * a
    expected_te = sqrt(txx*txx + 0.0 + 0.0 + 0.0 + txz_val^2 + tyz_val^2)
    @test all(interior(s.strs2D_te) .≈ expected_te)
    # txx / tyy / txy / eigenvalues unchanged by txz/tyz — same as
    # the pure-tensile case.
    @test all(interior(s.strs2D_txx) .≈ txx)
    @test all(interior(s.strs2D_tau_eig_1) .≈ txx)
    @test all(interior(s.strs2D_tau_eig_2) .≈ 0.0)
end

@testset "calc_2D_eigen_values_pt — scalar" begin
    # Asymmetric tensile: txx = 4, tyy = 1, txy = 0.
    # Eigenvalues are 4 and 1; calc_2D_eigen_values_pt returns sorted
    # (eig_1, eig_2) = (4, 1).
    @test calc_2D_eigen_values_pt(4.0, 1.0, 0.0) == (4.0, 1.0)

    # Off-diagonal: txx = 1, tyy = 1, txy = 1.
    # Eigenvalues are 2 and 0; sorted (2, 0).
    e1, e2 = calc_2D_eigen_values_pt(1.0, 1.0, 1.0)
    @test e1 ≈ 2.0
    @test e2 ≈ 0.0

    # Degenerate: should return (0, 0) per Fortran semantics.
    @test calc_2D_eigen_values_pt(2.0, 2.0, 0.0) == (0.0, 0.0)
end
