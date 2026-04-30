"""
    YelmoConst

Physical and non-physical constants for `YelmoModel`. Mirrors the
parameter-side split: `YelmoPar`/`YelmoParameters` ↔
`YelmoConst`/`YelmoConstants`.

Two flavours of constants live here:

  - `YelmoConstants` (struct, instantiable) — physical constants
    sourced per model. Defaults reproduce the Yelmo Fortran
    `&phys` namelist. Construct via `YelmoConstants(; kwargs...)`
    or the planet-specific `earth_constants(; kwargs...)`. The
    struct is immutable, so the same instance can be safely
    shared across multiple `YelmoModel`s when the physics is
    identical (e.g. multi-domain runs).

  - `MASK_ICE_NONE`, `MASK_ICE_FIXED`, `MASK_ICE_DYNAMIC`
    (module-level `const`) — non-physical bit-pattern enums for
    the `bnd.mask_ice` field. Not configurable; collected here
    so they're easy to find rather than buried in `YelmoCore`.

  - `MASK_BED_OCEAN` … `MASK_BED_PARTIAL` (module-level `const`) —
    enum values for the multi-valued `tpo.mask_bed` field. Mirrors
    the integer parameters at the top of Fortran
    `physics/topography.f90`; consumers in `yelmo_data.f90` and
    diagnostic output expect these exact integer values.
"""
module YelmoConst

export YelmoConstants, yelmo_constants, earth_constants,
       eismint_constants, mismip3d_constants, trough_constants
export MASK_ICE_NONE, MASK_ICE_FIXED, MASK_ICE_DYNAMIC
export MASK_BED_OCEAN, MASK_BED_LAND, MASK_BED_FROZEN, MASK_BED_STREAM,
       MASK_BED_GRLINE, MASK_BED_FLOAT, MASK_BED_ISLAND, MASK_BED_PARTIAL

# ---------------------------------------------------------------------------
# Non-physical constants — bit-pattern enums for bnd.mask_ice cells.
# ---------------------------------------------------------------------------

const MASK_ICE_NONE    = 0  # H_ice forced to 0
const MASK_ICE_FIXED   = 1  # H_ice held at its current value
const MASK_ICE_DYNAMIC = 2  # H_ice evolves freely

# ---------------------------------------------------------------------------
# Non-physical constants — multi-valued bed mask (tpo.mask_bed).
# Integer values mirror Fortran physics/topography.f90:10-17 verbatim.
# ---------------------------------------------------------------------------

const MASK_BED_OCEAN   = 0  # ice-free ocean
const MASK_BED_LAND    = 1  # ice-free land
const MASK_BED_FROZEN  = 2  # fully ice-covered, grounded, frozen base
const MASK_BED_STREAM  = 3  # fully ice-covered, grounded, temperate base
const MASK_BED_GRLINE  = 4  # grounding line cell
const MASK_BED_FLOAT   = 5  # fully ice-covered, floating
const MASK_BED_ISLAND  = 6  # reserved (Fortran does not currently emit)
const MASK_BED_PARTIAL = 7  # partially ice-covered cell

# ---------------------------------------------------------------------------
# Physical constants — instantiable per model.
# ---------------------------------------------------------------------------

"""
    YelmoConstants(; kwargs...) -> YelmoConstants

Container for the per-model physical constants. Defaults match the
Yelmo Fortran `&phys` namelist (Earth ice-sheet defaults). All fields
are `Float64`.

Override any subset via keyword arguments:

```julia
c = YelmoConstants(rho_ice=917.0)
y1 = YelmoModel(restart_a, 0.0; c=c)
y2 = YelmoModel(restart_b, 0.0; c=c)   # share the same constants
```

| field      | unit       | default       | meaning                                |
|------------|------------|---------------|----------------------------------------|
| sec_year   | s/yr       | 31_536_000    | 365·24·3600                            |
| g          | m/s²       | 9.81          | gravitational acceleration             |
| T0         | K          | 273.15        | reference freezing temperature         |
| rho_ice    | kg/m³      | 910.0         | density of ice                         |
| rho_w      | kg/m³      | 1000.0        | density of fresh water                 |
| rho_sw     | kg/m³      | 1028.0        | density of seawater                    |
| rho_a      | kg/m³      | 3300.0        | density of asthenosphere               |
| rho_rock   | kg/m³      | 2000.0        | density of bedrock (mantle/lith.)      |
| L_ice      | J/kg       | 333_500       | latent heat of fusion (ice/water)      |
| T_pmp_beta | K/Pa       | 9.8e-8        | pressure-melting-point coeff (G&B 2009)|
"""
Base.@kwdef struct YelmoConstants
    sec_year   ::Float64 = 31536000.0
    g          ::Float64 = 9.81
    T0         ::Float64 = 273.15
    rho_ice    ::Float64 = 910.0
    rho_w      ::Float64 = 1000.0
    rho_sw     ::Float64 = 1028.0
    rho_a      ::Float64 = 3300.0
    rho_rock   ::Float64 = 2000.0
    L_ice      ::Float64 = 333500.0
    T_pmp_beta ::Float64 = 9.8e-8
end

"""
    yelmo_constants(; kwargs...) -> YelmoConstants

Convenience constructor mirroring the `*_params(...)` factories in
`YelmoModelPar`. Equivalent to `YelmoConstants(; kwargs...)`.
"""
yelmo_constants(; kwargs...) = YelmoConstants(; kwargs...)

"""
    earth_constants(; kwargs...) -> YelmoConstants

Earth-default constants, identical to `YelmoConstants(; kwargs...)`
but explicit — kept for symmetry with the named-experiment
constructors below and for documentation searchability.
"""
earth_constants(; kwargs...) = YelmoConstants(; kwargs...)

# ---------------------------------------------------------------------------
# Named experiment constants — match the per-group entries in the Fortran
# `yelmo/input/yelmo_phys_const.nml`. Each diverges from Earth in a small
# subset of fields; the rest fall through to the YelmoConstants defaults.
# Mirrors the `select case (phys_const)` dispatch in
# `yelmo_boundaries.f90:ybound_define_physical_constants`.
# ---------------------------------------------------------------------------

const _EISMINT_DEFAULTS = (
    sec_year   = 31556926.0,   # EISMINT-specific year length
    rho_ice    = 917.0,
    T_pmp_beta = 9.7e-8,       # EISMINT2 (β1 = 8.66e-4 K/m)
)

const _MISMIP3D_DEFAULTS = (
    sec_year   = 31556926.0,
    rho_ice    = 900.0,
    T_pmp_beta = 9.7e-8,
)

const _TROUGH_DEFAULTS = (
    sec_year   = 31556926.0,
    rho_ice    = 918.0,
    T_pmp_beta = 9.7e-8,
)

"""
    YelmoConstants(preset::Symbol; kwargs...) -> YelmoConstants

Symbol-dispatch shortcut over the named-experiment constructors.
Supported presets (the symbol matches the Fortran `phys_const` group
name; aliases follow the dispatch in
`yelmo_boundaries.f90:ybound_define_physical_constants`):

| symbol                         | constructor             |
|--------------------------------|-------------------------|
| `:Earth`                       | `earth_constants`       |
| `:EISMINT` / `:EISMINT1` / `:EISMINT2` | `eismint_constants`  |
| `:MISMIP` / `:MISMIP3D`        | `mismip3d_constants`    |
| `:TROUGH`                      | `trough_constants`      |

`kwargs...` override any field on top of the preset defaults.

```julia
c1 = YelmoConstants(:EISMINT)
c2 = YelmoConstants(:MISMIP3D, rho_sw=1027.5)
```

For users who want to define their own preset, follow the named-
constructor pattern:

```julia
mars_constants(; kwargs...) =
    YelmoConstants(; g=3.71, rho_ice=900.0, kwargs...)
```
"""
function YelmoConstants(preset::Symbol; kwargs...)
    if preset === :Earth
        return earth_constants(; kwargs...)
    elseif preset === :EISMINT || preset === :EISMINT1 || preset === :EISMINT2
        return eismint_constants(; kwargs...)
    elseif preset === :MISMIP || preset === :MISMIP3D
        return mismip3d_constants(; kwargs...)
    elseif preset === :TROUGH
        return trough_constants(; kwargs...)
    else
        error("YelmoConstants: unknown preset :$(preset). " *
              "Supported: :Earth, :EISMINT (also :EISMINT1, :EISMINT2), " *
              ":MISMIP3D (also :MISMIP), :TROUGH.")
    end
end

"""
    eismint_constants(; kwargs...) -> YelmoConstants

Constants for the EISMINT / EISMINT1 / EISMINT2 intercomparison
suite. Diverges from `earth_constants()` in `sec_year`
(31_556_926.0), `rho_ice` (917.0), and `T_pmp_beta` (9.7e-8); other
fields fall through to the `YelmoConstants` defaults. `kwargs...`
override any field on top of these.

Matches the `&EISMINT` group in the Fortran
`yelmo/input/yelmo_phys_const.nml`.
"""
eismint_constants(; kwargs...) =
    YelmoConstants(; _EISMINT_DEFAULTS..., kwargs...)

"""
    mismip3d_constants(; kwargs...) -> YelmoConstants

Constants for the MISMIP / MISMIP3D experiments. Same `sec_year`
and `T_pmp_beta` as EISMINT but `rho_ice = 900.0`.

Matches the `&MISMIP3D` group in the Fortran
`yelmo/input/yelmo_phys_const.nml`.
"""
mismip3d_constants(; kwargs...) =
    YelmoConstants(; _MISMIP3D_DEFAULTS..., kwargs...)

"""
    trough_constants(; kwargs...) -> YelmoConstants

Constants for the TROUGH experiment. Same `sec_year` and
`T_pmp_beta` as EISMINT but `rho_ice = 918.0`.

Matches the `&TROUGH` group in the Fortran
`yelmo/input/yelmo_phys_const.nml`.
"""
trough_constants(; kwargs...) =
    YelmoConstants(; _TROUGH_DEFAULTS..., kwargs...)

end # module YelmoConst
