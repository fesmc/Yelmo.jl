# Constants API

Physical constants live in the `Yelmo.YelmoConst` module and are
re-exported at the package level. The container type is
[`YelmoConstants`](@ref). Its fields and named groups follow Fortran
`input/yelmo_phys_const.nml` (checked by `test/test_par_schema.jl`), and
the Fortran `yelmo.phys_const` switch is mirrored by
`YelmoConstants(phys_const)`. A `YelmoModel` built with parameters `p`
uses `YelmoConstants(p)`, the `p.yelmo.phys_const` group, unless `c` is
given.

See the [concepts page](../concepts.md) for the parameters-vs-constants
design rationale.

## Container type and groups

```@docs
YelmoConstants
yelmo_constants
PHYS_CONST_GROUPS
earth_constants
```

## Selecting a group

```julia
c = YelmoConstants(:EISMINT)                  # &EISMINT
c = YelmoConstants("MISMIP3D"; rho_sw=1027.5) # &MISMIP3D + override
c = YelmoConstants(p)                         # p.yelmo.phys_const
```

Accepted names (as in Fortran `ybound_define_physical_constants`):

| `phys_const` | group |
|---|---|
| `Earth` | `&Earth` |
| `EISMINT`, `EISMINT1`, `EISMINT2` | `&EISMINT` |
| `MISMIP`, `MISMIP3D` | `&MISMIP3D` |
| `MISMIP+`, `MISMIPplus` | `&MISMIPplus` |
| `ISMIPHOM`, `ISMIP-HOM` | `&ISMIPHOM` |
| `CALVINGMIP`, `CalvingMIP` | `&CALVINGMIP` |
| `TROUGH` | `&TROUGH` |

For a custom preset, follow the named-factory pattern:

```julia
mars_constants(; kwargs...) =
    YelmoConstants(; g=3.71, rho_ice=900.0, kwargs...)

c_mars = mars_constants()                  # all Mars defaults
c_mars2 = mars_constants(rho_sw=1010.0)    # Mars + override
```

## Mask enum values

`YelmoConst` also exports the bit-pattern enums for the per-cell ice-
evolution mask (`MASK_ICE_*`) and the diagnostic bed-state mask
(`MASK_BED_*`). They are documented on the
[core API page](core.md#mask-constants) where they are re-exported
from `YelmoCore` for back-compat.
