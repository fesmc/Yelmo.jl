# ----------------------------------------------------------------------
# 2D deviatoric-stress tensor and its principal-stress eigenvalues.
# Ports of:
#
#   - `calc_stress_tensor_2D` (deformation.f90:1828) — builds the
#     2D stress components from the depth-averaged viscosity and the
#     2D strain-rate tensor.
#   - `calc_2D_eigen_values`  (deformation.f90:1861) — quadratic
#     formula for the two real eigenvalues of the 2x2 stress block.
#
# Stress formulas (Thoma et al. 2014 Eq. 7; Lipscomb et al. 2019
# Eq. 44):
#
#     txx = 2 · visc_bar · dxx
#     tyy = 2 · visc_bar · dyy
#     txy = 2 · visc_bar · dxy
#     te  = sqrt(txx² + tyy² + txx · tyy + txy² + txz² + tyz²)
#
# `txz` / `tyz` enter only via `te`. In the current `mat_step!` scope
# they stay at their allocation default (zero) — Fortran's
# `calc_ymat` path likewise never writes them — but they are included
# as explicit inputs for signature parity with the Fortran routine.
#
# Eigenvalue branch: Fortran requires the discriminant to be **strictly
# positive** before computing the two roots. On a doubly-degenerate
# tensor (`txx == tyy`, `txy == 0`) the discriminant is exactly zero
# and the eigenvalues are returned as `(0, 0)` rather than the
# correct `(txx, txx)`. This is a bug-shaped edge case in the Fortran
# code; the Julia port replicates it to keep parity with Fortran-
# generated benchmark fixtures.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calc_stress_tensor_2D!, calc_2D_eigen_values_pt


"""
    calc_2D_eigen_values_pt(txx, tyy, txy) -> (eig_1, eig_2)

Per-cell scalar eigenvalues of the symmetric 2x2 block

    [ txx  txy
      txy  tyy ]

via the quadratic formula on `λ² - (txx+tyy)·λ + (txx·tyy - txy²) = 0`.
Returns `(eig_1, eig_2)` with `eig_1 >= eig_2`.

Faithful port of `yelmo/src/physics/deformation.f90:1861
calc_2D_eigen_values` — including the `b² - 4·c > 0` strict check.
For a doubly-degenerate tensor the discriminant is exactly zero, so
the function returns `(0, 0)` per Fortran semantics rather than the
mathematically-correct repeated root.
"""
@inline function calc_2D_eigen_values_pt(txx::Real, tyy::Real, txy::Real)
    b = -(Float64(txx) + Float64(tyy))
    c = Float64(txx) * Float64(tyy) - Float64(txy) * Float64(txy)
    disc = b*b - 4.0*c
    if disc > 0.0
        root = sqrt(disc)
        l1 = (-b + root) * 0.5
        l2 = (-b - root) * 0.5
        return l1 > l2 ? (l1, l2) : (l2, l1)
    else
        return (0.0, 0.0)
    end
end


"""
    calc_stress_tensor_2D!(strs2D_txx, strs2D_tyy, strs2D_txy,
                           strs2D_te,
                           strs2D_tau_eig_1, strs2D_tau_eig_2,
                           visc_bar,
                           strn2D_dxx, strn2D_dyy, strn2D_dxy,
                           strs2D_txz, strs2D_tyz) -> nothing

Build the 2D deviatoric-stress tensor components, effective stress,
and principal eigenvalues. Port of `yelmo/src/physics/
deformation.f90:1828 calc_stress_tensor_2D`.

Per-cell:

    txx = 2 · visc_bar · strn2D_dxx
    tyy = 2 · visc_bar · strn2D_dyy
    txy = 2 · visc_bar · strn2D_dxy
    te  = sqrt(txx² + tyy² + txx · tyy + txy² + txz² + tyz²)
    (eig_1, eig_2) = sorted real roots of [txx txy; txy tyy]

`strs2D_txz` and `strs2D_tyz` are input-only (the Fortran routine
reads them through the same struct without writing them). They
contribute only to `te` and stay at their allocation default in
the current `mat_step!` scope.

All arguments are 2D `CenterField`s.
"""
function calc_stress_tensor_2D!(strs2D_txx, strs2D_tyy, strs2D_txy,
                                strs2D_te,
                                strs2D_tau_eig_1, strs2D_tau_eig_2,
                                visc_bar,
                                strn2D_dxx, strn2D_dyy, strn2D_dxy,
                                strs2D_txz, strs2D_tyz)
    Txx = interior(strs2D_txx);   Tyy = interior(strs2D_tyy)
    Txy = interior(strs2D_txy);   Te  = interior(strs2D_te)
    E1  = interior(strs2D_tau_eig_1)
    E2  = interior(strs2D_tau_eig_2)
    Vb  = interior(visc_bar)
    Dxx = interior(strn2D_dxx);   Dyy = interior(strn2D_dyy)
    Dxy = interior(strn2D_dxy)
    Txz = interior(strs2D_txz);   Tyz = interior(strs2D_tyz)

    Nx = size(Txx, 1)
    Ny = size(Txx, 2)
    @assert size(Tyy, 1) == Nx && size(Tyy, 2) == Ny
    @assert size(Txy, 1) == Nx && size(Txy, 2) == Ny
    @assert size(Vb,  1) == Nx && size(Vb,  2) == Ny
    @assert size(Dxx, 1) == Nx && size(Dxx, 2) == Ny
    @assert size(Dyy, 1) == Nx && size(Dyy, 2) == Ny
    @assert size(Dxy, 1) == Nx && size(Dxy, 2) == Ny

    @inbounds for j in 1:Ny, i in 1:Nx
        v2 = 2.0 * Vb[i, j, 1]
        txx = v2 * Dxx[i, j, 1]
        tyy = v2 * Dyy[i, j, 1]
        txy = v2 * Dxy[i, j, 1]
        Txx[i, j, 1] = txx
        Tyy[i, j, 1] = tyy
        Txy[i, j, 1] = txy

        # Effective stress (Lipscomb et al. 2019 Eq. 44 form).
        txz = Txz[i, j, 1]; tyz = Tyz[i, j, 1]
        Te[i, j, 1] = sqrt(txx*txx + tyy*tyy + txx*tyy +
                           txy*txy + txz*txz + tyz*tyz)

        e1, e2 = calc_2D_eigen_values_pt(txx, tyy, txy)
        E1[i, j, 1] = e1
        E2[i, j, 1] = e2
    end
    return nothing
end
