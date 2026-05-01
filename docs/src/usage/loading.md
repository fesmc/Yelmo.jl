# Loading a model from a restart

Both [`YelmoModel`](@ref) and [`YelmoMirror`](@ref) take a NetCDF
restart file as their primary input. The same file format works for
both backends — the constructors read coordinates, allocate fields
according to the variable tables, and populate field interiors from
matching variable names in the file.

## YelmoModel constructor

```julia
y = YelmoModel(restart_file::String, time::Float64;
               alias  = "ymodel1",
               rundir = "./",
               p      = nothing,           # YelmoModelParameters or nothing
               c      = YelmoConstants(),  # physical constants
               groups = (:bnd, :dta, :dyn, :mat, :thrm, :tpo),
               strict = true)
```

What it does, in order:

1. If `p === nothing`, build a default [`YelmoModelParameters`](@ref)
   keyed off `alias` and emit a warning (so you don't silently run
   with defaults).
2. Read coordinate arrays (`xc`, `yc`, `zeta_ac`, `zeta_rock_ac`)
   from the restart and build three Oceananigans `RectilinearGrid`s
   (2D, 3D ice, 3D rock). NetCDF coordinate units of `km` are
   converted to metres on load.
3. Read the variable tables under `src/variables/model/` and
   pre-allocate one Oceananigans `Field` per variable on the
   appropriate grid, keyed by group name (`bnd`, `dta`, …, `tpo`).
4. Replace `tpo.H_ice` with a `CenterField` carrying Dirichlet
   `H = 0` boundary conditions on every domain edge (so the upwind
   advection halo reads zero past the boundary without per-cell
   branches).
5. Initialise `bnd.mask_ice` to `MASK_ICE_DYNAMIC` everywhere.
6. Call [`load_state!`](@ref) to copy variable data from the restart
   into the pre-allocated fields, restricted to the groups in
   `groups`.
7. Infer `bnd.mask_ice` from the restart: if `mask_ice` is in the
   file, use it; otherwise fall back to `ice_allowed` (allowed →
   `DYNAMIC`, not-allowed → `NONE`); otherwise leave the all-dynamic
   default. See the
   [mask constants](../api/core.md#mask-constants) for the bit
   pattern.

The `groups` keyword is useful when a Yelmo Fortran restart doesn't
carry every group — e.g. the data group `dta` is rarely populated on
output. `strict=false` further loosens the loader to skip individual
missing variables within a loaded group, leaving them at their
default-allocated value (zeros).

## Building parameters

Three ways to construct [`YelmoModelParameters`](@ref):

```julia
# 1. All defaults — what the constructor warning suggests.
p = YelmoModelParameters("demo")

# 2. Override individual groups via keyword arguments.
p = YelmoModelParameters("demo";
    yelmo = yelmo_params(grid_path = "data/GRL/16km.nc"),
    ydyn  = ydyn_params(solver = "ssa"),
    ytopo = ytopo_params(use_bmb = false, topo_rel = 0),
)

# 3. Read from a Yelmo Fortran namelist file.
p = YelmoModelPar.read_nml("Yelmo_GRL.nml")
# (Note the explicit module qualifier: YelmoPar.read_nml returns
#  YelmoParameters for the Mirror; the Model variant has the same
#  signature but a different return type, so they cannot share a
#  single generic.)
```

A `YelmoModelParameters` is an immutable nested struct: the top level
holds one struct per namelist group (`yelmo`, `ytopo`, `ycalv`,
`ydyn`, `ytill`, `yneff`, `ymat`, `ytherm`, `yelmo_masks`,
`yelmo_init_topo`, `yelmo_data`). To "edit" a single field of an
already-built parameter set, build a fresh group with overrides:

```julia
p = YelmoModelParameters("demo";
    ydyn = ydyn_params(; solver = "ssa", visc_method = 1),
)
# Equivalent to typing the whole defaults table with these two changes.
```

Round-trip to a Fortran namelist with [`write_nml`](@ref) /
`Yelmo.YelmoModelPar.read_nml`.

## Building constants

Use the symbol-dispatch shortcut over a named-experiment preset, or
the named factory directly:

```julia
c1 = YelmoConstants()                                  # Earth, all defaults
c2 = YelmoConstants(rho_ice = 917.0)                   # Earth + override
c3 = YelmoConstants(:EISMINT)                          # preset
c4 = YelmoConstants(:MISMIP3D, rho_sw = 1027.5)        # preset + override
c5 = mismip3d_constants(rho_sw = 1027.5)               # equivalent
```

The presets diverge from `Earth` in `sec_year`, `rho_ice`, and
`T_pmp_beta`. All other fields fall through to the
`YelmoConstants` defaults. See [`YelmoConstants`](@ref) for the
full table.

## YelmoMirror constructor

```julia
ymf = YelmoMirror(p::YelmoParameters, time::Float64;
                  alias    = "ylmo1",
                  rundir   = "./",
                  overwrite = false)
```

Different in three ways from `YelmoModel`:

1. The Fortran solver is initialised first (the constructor writes a
   namelist file under `rundir`, then `ccall`s `yelmo_init`). The
   Julia side allocates field arrays and pulls the initial state back
   over the C interface.
2. The variable layout comes from `src/variables/mirror/`, which
   matches the Fortran NetCDF schema (e.g. `cmb_flt_x`/`cmb_flt_y`
   instead of the model's `cmb_flt_acx`/`cmb_flt_acy`).
3. Only [`YelmoParameters`](@ref) (the Mirror parameter type) is
   accepted — these read from / write to `read_nml` /
   `write_nml` in the `YelmoPar` module rather than `YelmoModelPar`.

The constructor accepts a filename shortcut that wraps
`read_nml`:

```julia
ymf = YelmoMirror("Yelmo_GRL.nml", 0.0; alias="grl", rundir="./output")
```

After construction, the Mirror exposes the same six component groups,
the same `init_state!` / `step!` interface, and the same NetCDF
output module as `YelmoModel`.
