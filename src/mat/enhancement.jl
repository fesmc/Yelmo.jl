# ----------------------------------------------------------------------
# Spatially-varying enhancement-factor parameterisations. Ports of:
#
#   - `define_enhancement_factor_2D` (deformation.f90:146) — per-cell
#     2D enhancement factor blended from shear / stream / shelf regimes
#   - `define_enhancement_factor_3D` (deformation.f90:107) — same
#     blend per-layer, with f_shear varying in z
#
# Formula (both 2D and 3D, per cell):
#
#     enh_ssa = f_grnd * enh_stream + (1 - f_grnd) * enh_shlf
#     enh     = f_shear * enh_shear + (1 - f_shear) * enh_ssa
#
# `enh_ssa` is the non-shear baseline (grounded streams vs floating
# shelves), and `f_shear` mixes that with the SIA-shear regime
# (`enh_shear`).
#
# The Fortran routines also take `uxy_srf` and declare a constant
# `uxy_srf_lim = 10.0`, but neither value is used in the function
# bodies (legacy-API leftovers). The Julia ports drop them — the
# call site in `mat_step!` simply doesn't pass surface velocity.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export define_enhancement_factor_2D!, define_enhancement_factor_3D!


"""
    define_enhancement_factor_2D!(enh, f_grnd, f_shear;
                                  enh_shear, enh_stream, enh_shlf) -> enh

Per-cell 2D enhancement factor:

    enh = f_shear · enh_shear + (1 - f_shear) ·
          (f_grnd · enh_stream + (1 - f_grnd) · enh_shlf)

Port of `yelmo/src/physics/deformation.f90:146 define_enhancement_factor_2D`.

`enh`, `f_grnd`, `f_shear` are 2D `CenterField`s. The keyword args
are scalars from `y.p.ymat` (`enh_shear`, `enh_stream`, `enh_shlf`).
"""
function define_enhancement_factor_2D!(enh, f_grnd, f_shear;
                                       enh_shear::Real,
                                       enh_stream::Real,
                                       enh_shlf::Real)
    E   = interior(enh)
    Fg  = interior(f_grnd)
    Fs  = interior(f_shear)

    Nx = size(E, 1)
    Ny = size(E, 2)
    @assert size(Fg, 1) == Nx && size(Fg, 2) == Ny "define_enhancement_factor_2D!: f_grnd size mismatch"
    @assert size(Fs, 1) == Nx && size(Fs, 2) == Ny "define_enhancement_factor_2D!: f_shear size mismatch"

    e_shear  = Float64(enh_shear)
    e_stream = Float64(enh_stream)
    e_shlf   = Float64(enh_shlf)

    @inbounds for j in 1:Ny, i in 1:Nx
        enh_ssa  = Fg[i, j, 1] * e_stream + (1.0 - Fg[i, j, 1]) * e_shlf
        E[i, j, 1] = Fs[i, j, 1] * e_shear + (1.0 - Fs[i, j, 1]) * enh_ssa
    end
    return enh
end


"""
    define_enhancement_factor_3D!(enh, f_grnd, f_shear;
                                  enh_shear, enh_stream, enh_shlf) -> enh

Per-layer 3D enhancement factor with the same blend formula as the
2D variant; `f_shear` varies in z.

    enh_ssa(i, j) = f_grnd · enh_stream + (1 - f_grnd) · enh_shlf
    enh(i, j, k)  = f_shear(i, j, k) · enh_shear +
                    (1 - f_shear(i, j, k)) · enh_ssa(i, j)

Port of `yelmo/src/physics/deformation.f90:107 define_enhancement_factor_3D`.

`enh` and `f_shear` are 3D CenterFields; `f_grnd` is 2D. Keyword
args are scalars.
"""
function define_enhancement_factor_3D!(enh, f_grnd, f_shear;
                                       enh_shear::Real,
                                       enh_stream::Real,
                                       enh_shlf::Real)
    E   = interior(enh)
    Fg  = interior(f_grnd)
    Fs  = interior(f_shear)

    Nx, Ny, Nz = size(E)
    @assert size(Fs) == size(E) "define_enhancement_factor_3D!: f_shear size mismatch"
    @assert size(Fg, 1) == Nx && size(Fg, 2) == Ny "define_enhancement_factor_3D!: f_grnd size mismatch"

    e_shear  = Float64(enh_shear)
    e_stream = Float64(enh_stream)
    e_shlf   = Float64(enh_shlf)

    @inbounds for j in 1:Ny, i in 1:Nx
        enh_ssa = Fg[i, j, 1] * e_stream + (1.0 - Fg[i, j, 1]) * e_shlf
        for k in 1:Nz
            E[i, j, k] = Fs[i, j, k] * e_shear +
                         (1.0 - Fs[i, j, k]) * enh_ssa
        end
    end
    return enh
end
