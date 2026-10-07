"""
    YelmoConst

Physical and non-physical constants for `YelmoModel`. Mirrors the
parameter-side split: `YelmoMirrorPar`/`YelmoMirrorParameters` ↔
`YelmoConst`/`YelmoConstants`.

Two flavours of constants live here:

  - `YelmoConstants` (struct, instantiable) — physical constants
    sourced per model. Fields and named groups follow Fortran
    `input/yelmo_phys_const.nml`. Construct via
    `YelmoConstants(; kwargs...)` or `YelmoConstants(phys_const)`. The
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
       eismint_constants, mismip3d_constants, trough_constants,
       PHYS_CONST_GROUPS
export MASK_ICE_NONE, MASK_ICE_FIXED, MASK_ICE_DYNAMIC
export MASK_BED_OCEAN, MASK_BED_LAND, MASK_BED_FROZEN, MASK_BED_STREAM,
       MASK_BED_GRLINE, MASK_BED_FLOAT, MASK_BED_ISLAND, MASK_BED_PARTIAL

# ---------------------------------------------------------------------------
# Non-physical constants — bit-pattern enums for bnd.mask_ice cells.
# ---------------------------------------------------------------------------

const MASK_ICE_NONE    = 0  # H_ice forced to 0
const MASK_ICE_FIXED   = 1  # H_ice prescribed (held at bnd.H_ice_ref)
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

Container for the per-model physical constants. The fields are the keys
of a group in Fortran Yelmo's `input/yelmo_phys_const.nml`; the defaults
are its `&Earth` group. All fields are `Float64`.

Override any subset via keyword arguments:

```julia
c = YelmoConstants(rho_ice=917.0)
y1 = YelmoModel(restart_a, 0.0; c=c)
y2 = YelmoModel(restart_b, 0.0; c=c)   # share the same constants
```

| field        | unit         | Earth         | meaning                                 |
|--------------|--------------|---------------|-----------------------------------------|
| sec_year     | s/yr         | 31_556_926    | year length (365.2422 d, CF/UDUNITS)    |
| g            | m/s²         | 9.81          | gravitational acceleration              |
| T0           | K            | 273.15        | reference freezing temperature          |
| rho_ice      | kg/m³        | 910.0         | density of ice                          |
| rho_w        | kg/m³        | 1000.0        | density of fresh water                  |
| rho_sw       | kg/m³        | 1028.0        | density of seawater                     |
| rho_asth     | kg/m³        | 3300.0        | density of the asthenosphere            |
| L_ice        | J/kg         | 333_500       | latent heat of fusion (ice/water)       |
| cp_ice       | J/(kg K)     | 2110.0        | specific heat capacity of ice           |
| cp_w         | J/(kg K)     | 4187.0        | specific heat capacity of pure water    |
| cp_ocn       | J/(kg K)     | 3974.0        | specific heat capacity, ocean mixed layer |
| T_pmp_beta   | K/Pa         | 9.8e-8        | pressure-melting-point slope (G&B 2009) |
| area_seasurf | km²          | 3.618e8       | global sea-surface area                 |
"""
Base.@kwdef struct YelmoConstants
    sec_year     ::Float64 = 31556926.0
    g            ::Float64 = 9.81
    T0           ::Float64 = 273.15
    rho_ice      ::Float64 = 910.0
    rho_w        ::Float64 = 1000.0
    rho_sw       ::Float64 = 1028.0
    rho_asth     ::Float64 = 3300.0
    L_ice        ::Float64 = 333500.0
    cp_ice       ::Float64 = 2110.0
    cp_w         ::Float64 = 4187.0
    cp_ocn       ::Float64 = 3974.0
    T_pmp_beta   ::Float64 = 9.8e-8
    area_seasurf ::Float64 = 3.618e8
end

"""
    yelmo_constants(; kwargs...) -> YelmoConstants

Convenience constructor mirroring the `*_params(...)` factories in
`YelmoPar`. Equivalent to `YelmoConstants(; kwargs...)`.
"""
yelmo_constants(; kwargs...) = YelmoConstants(; kwargs...)

# ---------------------------------------------------------------------------
# Named groups of Fortran `input/yelmo_phys_const.nml` (yelmo dev). Fields
# not listed take the `&Earth` value. `test/test_par_schema.jl` checks every
# group and key against the Fortran file.
# ---------------------------------------------------------------------------

const _BENCH_COMMON = (sec_year = 31556926.0, T_pmp_beta = 9.7e-8)

"""
    PHYS_CONST_GROUPS

Constants of each group of Fortran `input/yelmo_phys_const.nml`, as the
fields that differ from `&Earth`.
"""
const PHYS_CONST_GROUPS = Dict{Symbol, NamedTuple}(
    :Earth      => NamedTuple(),
    :EISMINT    => _BENCH_COMMON,
    :MISMIP3D   => (sec_year = 31536000.0, g = 9.8, rho_ice = 900.0, rho_sw = 1000.0,
                    T_pmp_beta = 9.7e-8),
    :MISMIPplus => (_BENCH_COMMON..., rho_ice = 918.0),
    :TROUGH     => (_BENCH_COMMON..., rho_ice = 918.0),
    :ISMIPHOM   => _BENCH_COMMON,
    :CALVINGMIP => (_BENCH_COMMON..., rho_ice = 917.0),
)

# `yelmo.phys_const` values accepted by Fortran
# (`yelmo_boundaries.f90:ybound_define_physical_constants`) => group.
const _PHYS_CONST_ALIASES = Dict{String, Symbol}(
    "Earth"      => :Earth,
    "EISMINT"    => :EISMINT,    "EISMINT1"   => :EISMINT, "EISMINT2" => :EISMINT,
    "MISMIP"     => :MISMIP3D,   "MISMIP3D"   => :MISMIP3D,
    "MISMIP+"    => :MISMIPplus, "MISMIPplus" => :MISMIPplus,
    "ISMIPHOM"   => :ISMIPHOM,   "ISMIP-HOM"  => :ISMIPHOM,
    "CALVINGMIP" => :CALVINGMIP, "CalvingMIP" => :CALVINGMIP,
    "TROUGH"     => :TROUGH,
)

"""
    YelmoConstants(phys_const; kwargs...) -> YelmoConstants

Constants of a group of Fortran `input/yelmo_phys_const.nml`, selected by
the `yelmo.phys_const` name (`Symbol` or `String`) with the same aliases
as Fortran:

| `phys_const`                           | group         |
|----------------------------------------|---------------|
| `Earth`                                | `&Earth`      |
| `EISMINT`, `EISMINT1`, `EISMINT2`      | `&EISMINT`    |
| `MISMIP`, `MISMIP3D`                   | `&MISMIP3D`   |
| `MISMIP+`, `MISMIPplus`                | `&MISMIPplus` |
| `ISMIPHOM`, `ISMIP-HOM`                | `&ISMIPHOM`   |
| `CALVINGMIP`, `CalvingMIP`             | `&CALVINGMIP` |
| `TROUGH`                               | `&TROUGH`     |

`kwargs...` override any field on top of the group values.

```julia
c1 = YelmoConstants(:EISMINT)
c2 = YelmoConstants("MISMIP3D"; rho_sw=1027.5)
```

A `YelmoModel` built from a `YelmoParameters` `p` uses
`YelmoConstants(p.yelmo.phys_const)` unless `c` is given.
"""
function YelmoConstants(phys_const::Union{Symbol,AbstractString}; kwargs...)
    group = get(_PHYS_CONST_ALIASES, String(phys_const), nothing)
    group === nothing && error("YelmoConstants: unknown phys_const \"$(phys_const)\". " *
                               "Supported: $(join(sort(collect(keys(_PHYS_CONST_ALIASES))), ", ")).")
    return YelmoConstants(; PHYS_CONST_GROUPS[group]..., kwargs...)
end

"""
    earth_constants(; kwargs...)    = YelmoConstants(:Earth; kwargs...)
    eismint_constants(; kwargs...)  = YelmoConstants(:EISMINT; kwargs...)
    mismip3d_constants(; kwargs...) = YelmoConstants(:MISMIP3D; kwargs...)
    trough_constants(; kwargs...)   = YelmoConstants(:TROUGH; kwargs...)

Named shortcuts for the groups of Fortran `input/yelmo_phys_const.nml`.
"""
earth_constants(; kwargs...)    = YelmoConstants(:Earth; kwargs...)
eismint_constants(; kwargs...)  = YelmoConstants(:EISMINT; kwargs...)
mismip3d_constants(; kwargs...) = YelmoConstants(:MISMIP3D; kwargs...)
trough_constants(; kwargs...)   = YelmoConstants(:TROUGH; kwargs...)

end # module YelmoConst
