# Constants API

Physical constants live in the `Yelmo.YelmoConst` module and are
re-exported at the package level. The container type is
[`YelmoConstants`](@ref); the Fortran `phys_const` namelist switch is
mirrored on the Julia side by the symbol-dispatch shortcut
[`YelmoConstants(::Symbol)`](@ref) and a set of named factories.

See the [concepts page](../concepts.md) for the parameters-vs-constants
design rationale.

## Container type and presets

```@docs
YelmoConstants
yelmo_constants
earth_constants
eismint_constants
mismip3d_constants
trough_constants
```

## Symbol-dispatch shortcut

For users who want to write `YelmoConstants(:EISMINT)` instead of
calling the named factory directly. Mirrors Fortran's
`select case (phys_const)` dispatch in
`yelmo_boundaries.f90:ybound_define_physical_constants`.

```julia
c = YelmoConstants(:EISMINT)              # → eismint_constants()
c = YelmoConstants(:MISMIP3D, rho_sw=1027.5)
```

Aliases:

| symbol | factory |
|---|---|
| `:Earth` | `earth_constants` |
| `:EISMINT` / `:EISMINT1` / `:EISMINT2` | `eismint_constants` |
| `:MISMIP` / `:MISMIP3D` | `mismip3d_constants` |
| `:TROUGH` | `trough_constants` |

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
