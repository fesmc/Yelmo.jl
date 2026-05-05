# ----------------------------------------------------------------------
# Vertical-discretization weights for the implicit thermal column
# solver (Hoffmann et al. 2018).
#
# Direct port of Fortran `physics/ice_enthalpy.f90:1048-1086
# calc_dzeta_terms`. The Fortran version (re)allocates `dzeta_a` and
# `dzeta_b` inside the routine; the Julia port writes into
# caller-supplied vectors so init code can size them once and own
# the storage (the natural place is alongside the other thrm grid
# metadata, computed once from the Oceananigans grid's `znodes`).
#
# Endpoints (k=1, k=nz_aa) are zero by convention. They are never
# referenced by the column solver — the basal and surface boundary
# conditions are imposed directly on the tridiagonal matrix.
# ----------------------------------------------------------------------

"""
    calc_dzeta_terms!(dzeta_a, dzeta_b, zeta_aa, zeta_ac) -> (dzeta_a, dzeta_b)

Compute the Hoffmann-2018 finite-difference weights `ak`, `bk` used
by the implicit vertical-diffusion solver.

  - `zeta_aa` — cell-centre vertical coordinates (length `nz_aa`),
    including the basal endpoint `0` and surface endpoint `1`.
  - `zeta_ac` — cell-edge vertical coordinates (length `nz_aa + 1`).
  - `dzeta_a`, `dzeta_b` — output weight vectors, length `nz_aa`.
    Endpoints (k=1, k=nz_aa) are set to zero; only the interior
    (k = 2 .. nz_aa-1) is populated.

The vectors are filled in place; the caller owns the storage. For a
3D thrm column-solve loop, allocate once at init time (e.g. when the
thrm grid is built from `znodes(grid)`) and reuse across timesteps.

Faithful to Fortran `physics/ice_enthalpy.f90:1048-1086`.
"""
function calc_dzeta_terms!(dzeta_a::AbstractVector{Float64},
                           dzeta_b::AbstractVector{Float64},
                           zeta_aa::AbstractVector{Float64},
                           zeta_ac::AbstractVector{Float64})
    nz_aa = length(zeta_aa)
    @assert length(dzeta_a) == nz_aa "dzeta_a length must equal length(zeta_aa)"
    @assert length(dzeta_b) == nz_aa "dzeta_b length must equal length(zeta_aa)"
    @assert length(zeta_ac) == nz_aa + 1 "zeta_ac must have length(zeta_aa) + 1"

    fill!(dzeta_a, 0.0)
    fill!(dzeta_b, 0.0)

    @inbounds for k in 2:(nz_aa - 1)
        dzeta_a[k] = 1.0 / ((zeta_ac[k+1] - zeta_ac[k]) * (zeta_aa[k]   - zeta_aa[k-1]))
        dzeta_b[k] = 1.0 / ((zeta_ac[k+1] - zeta_ac[k]) * (zeta_aa[k+1] - zeta_aa[k]))
    end

    return dzeta_a, dzeta_b
end
