# Concepts

This page introduces the four ideas you'll encounter on every page of
the rest of the documentation: the *backend* split, the *six-component
state*, the *staggered grid*, and the *parameter / constants* split.

## Two backends, one interface

`Yelmo.jl` exposes two concrete model types that both subtype
[`AbstractYelmoModel`](@ref):

- [`YelmoMirror`](api/mirror.md) — the **Fortran** Yelmo model
  accessed via `ccall` to `libyelmo`. Each `step!` is one Fortran
  predictor / corrector step; the Julia side maintains a mirror copy
  of every field for inspection, output, and lockstep validation.
  Configured by a Fortran namelist file ([`YelmoParameters`](@ref)).
- [`YelmoModel`](@ref) — a **pure Julia** ice-sheet solver under
  active development. Each `step!` runs the in-Julia per-component
  phase pipeline, with no Fortran dependency. The topography and
  dynamics components (the latter via the SIA solver) are ported;
  material and thermodynamics components remain as forward stubs.
  Configured by a Julia parameter struct
  ([`YelmoModelParameters`](@ref)) — a structural twin of
  `YelmoParameters` that will diverge as the Julia port grows
  parameters the mirror does not need.

Why both: the mirror is the production solver today; the pure-Julia
model is being developed alongside it, validated by a lockstep diff
against the mirror at each milestone (see [`compare_state`](@ref) and
the [comparing-states](usage/comparing.md) guide). Once the Julia
port reaches feature parity, the mirror remains as a regression
oracle.

The two backends share:

- the same six-component state layout (next section),
- the same NetCDF restart format (read by both constructors),
- the same output module (`init_output` / `write_output!`),
- the same abstract `step!(y, dt)` signature.

So a script that takes an `AbstractYelmoModel` and runs a time loop
works against either backend with no changes.

## Six-component state

Every Yelmo model carries six named field groups, all `NamedTuple`s of
Oceananigans `Field`s:

| Group | Meaning | Examples |
|---|---|---|
| `bnd`  | **Boundary** forcing — read-only inputs to the dynamics | `smb_ref`, `bmb_shlf`, `z_bed`, `z_sl`, `mask_ice` |
| `dta`  | **Data** layer — observational reference targets | `H_ice_obs`, `uxy_s_obs` |
| `dyn`  | **Dynamics** state — velocities and viscous diagnostics | `ux`, `uy`, `ux_bar`, `uy_bar`, `taud_acx` |
| `mat`  | **Material** state — strain rates, stresses, anisotropy | `eps_eff`, `tau_eff`, `enh` |
| `thrm` | **Thermodynamic** state — temperatures and basal melt | `T_ice`, `T_pmp`, `bmb_grnd` |
| `tpo`  | **Topographic** state — thickness, grounding, mass-balance bookkeeping | `H_ice`, `H_grnd`, `f_ice`, `f_grnd`, `smb`, `bmb`, `mb_net` |

The layout is canonical: variable names, units, dimensions and grid
staggering for each group are defined in the markdown tables under
`src/variables/model/` (and `src/variables/mirror/` for the mirror's
slightly different schema). The constructor reads those tables and
allocates one `Field` per row on the right grid. See the
[variables](variables.md) page for the full inventory.

When the documentation says "the topography step" or "writes
`tpo.smb`", these are component-group qualifiers — `y.tpo.smb`
addresses the surface-mass-balance field in the topography group of
model `y`.

## Staggered grids and Oceananigans `Field` locations

Yelmo uses **Arakawa C-staggered** discretisations: scalars live at
cell centres (aa-nodes), x-component velocities at the east/west
faces (acx-nodes, "x-face"), and y-component velocities at the
north/south faces (acy-nodes, "y-face"). The Julia side encodes the
same staggering through Oceananigans `Field` location parameters:

| Yelmo node | Oceananigans type | Naming convention |
|---|---|---|
| aa (cell centre) | `CenterField` | (default — most fields) |
| acx (x-face)     | `XFaceField`  | `*_acx`, `ux*` |
| acy (y-face)     | `YFaceField`  | `*_acy`, `uy*` |
| zeta-face (3D ice) | `ZFaceField` | `uz`, `uz_star`, `jvel_dz*` |

The dispatch is automatic: the constructor's `make_field` helper
reads each variable's name and matches it against the patterns
`XFACE_VARIABLES`, `YFACE_VARIABLES`, `ZFACE_VARIABLES` (all exported
from `YelmoCore`) to pick the right field type. This means calving-
velocity fields like `tpo.cr_acx` and `tpo.cmb_flt_acx` automatically
land on the correct face nodes without per-call boilerplate.

Boundary conditions: the ice thickness `tpo.H_ice` carries a
**Dirichlet `H = 0` halo** on every domain edge, so the upwind
advection kernel reads "no ice past the boundary" via Oceananigans'
standard halo machinery rather than via per-cell branches.

## Parameters vs constants

Two distinct concepts, two distinct types:

- [`YelmoModelParameters`](@ref) — **per-run configuration**: solver
  choices (`solver = "diva"`), thresholds (`H_min_grnd`,
  `H_min_flt`), parameterisation switches (`bmb_gl_method = "pmp"`,
  `topo_rel = 0`). These are read from a namelist or built from
  Julia keyword arguments. There is one parameter object per model.

- [`YelmoConstants`](@ref) — **physical constants**: ice density
  (`rho_ice = 910.0`), gravity (`g = 9.81`), seconds per year. These
  rarely change within a run, but **may differ across experimental
  setups** — e.g. EISMINT uses 917 kg/m³ for ice, MISMIP3D uses 900,
  the Earth default uses 910. Yelmo Fortran handles this via the
  `phys_const` namelist switch that selects a `&phys` group from
  `yelmo_phys_const.nml`. The Julia side mirrors this with named
  constructors:

  ```julia
  c1 = YelmoConstants()                       # Earth defaults
  c2 = YelmoConstants(:EISMINT)               # symbol-dispatched preset
  c3 = mismip3d_constants(rho_sw = 1027.5)    # named factory + override
  ```

  `YelmoConstants` is immutable, so the same instance can be shared
  across multi-domain runs without going through model parameters.
  Pass via `YelmoModel(restart, t; c=c2)`; read from `y.c.rho_ice` in
  user code.

The structural separation matters because the kernels read constants
from `y.c` directly — for example,
[`calc_H_grnd!`](@ref) accepts `rho_ice, rho_sw` rather than reading
them from the parameter struct, so a custom set of constants flows
through without parameter-table maintenance.

## Time and units

- Time is `Float64` years.
- Lengths are metres (Yelmo NetCDF restarts that store coordinates in
  km are converted on load — see [`load_grids_from_restart`](@ref)).
- Mass-balance fields are m/yr (ice-equivalent thickness rate).
- The reference year length is configurable via `c.sec_year`. The
  EISMINT / MISMIP3D / TROUGH presets use 31_556_926 s/yr; the Earth
  default uses 31_536_000 s/yr (365 × 86_400). Inconsistent year
  conventions are a frequent source of cross-model bias — pick a
  convention per project and stick to it.
