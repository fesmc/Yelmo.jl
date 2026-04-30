# Calving (`topo_step!` phase 7)

Yelmo.jl uses a single calving formulation: the **level-set
function (LSF) flux method**. A calving-front velocity field is
computed on staggered ac-nodes per a chosen physical law, then the
level-set function `lsf` is advected at that velocity and ice is
killed wherever the level set has flipped to ocean.

The Fortran reference has two formulations — a vertical mass-balance
form (`calving_aa.f90`) and the LSF flux form (`calving_ac.f90` +
`lsf_module.f90`). Yelmo.jl ports only the latter; the aa-form is
out of scope.

## Conventions

- **Sign**: `lsf < 0` ⇒ ice domain, `lsf > 0` ⇒ ocean. The zero level
  set is the calving front.
- **Velocities**: `u_bar`, `v_bar` (depth-averaged ice velocity) live
  on staggered XFace / YFace nodes. So do all calving-front
  velocities (`cr_acx`, `cr_acy`) and the per-direction calving rates
  (`cmb_flt_acx`, `cmb_flt_acy`, `cmb_grnd_acx`, `cmb_grnd_acy`).
  Sign convention follows Fortran: a positive `cr` represents
  retreat opposite to the ice flow, so the law output for an
  equilibrium front is `cr = -u_bar` (advection velocity becomes
  `w = u + cr = 0`).

## Pipeline

`calving_step!(y, dt)` runs as phase 7 of `topo_step!`, between DMB
and relaxation. The full sequence:

| # | Step | Helper | Output |
|---|---|---|---|
| 1 | Snapshot lsf      | — | `tpo.lsf_n` |
| 2 | Refresh grounding | `calc_H_grnd!`, `determine_grounded_fractions!` | `tpo.H_grnd`, `f_grnd`, `f_grnd_acx`, `f_grnd_acy` |
| 3 | Floating law      | dispatch on `ycalv.calv_flt_method` | `tpo.cmb_flt_acx`, `tpo.cmb_flt_acy` |
| 4 | Grounded law      | dispatch on `ycalv.calv_grnd_method` | `tpo.cmb_grnd_acx`, `tpo.cmb_grnd_acy` |
| 5 | Merge             | `merge_calving_rates!` | `tpo.cr_acx`, `tpo.cr_acy` |
| 6 | Advect lsf        | `lsf_update!` (uses `advect_tracer!`) | `tpo.lsf` |
| 7 | Above-SL pin      | inline | `tpo.lsf` (set to −1 over land) |
| 8 | Redistance        | `lsf_redistance!` (per `dt_lsf`) | `tpo.lsf` |
| 9 | Build kill cmb    | inline | `tpo.cmb`, `tpo.cmb_flt`, `tpo.cmb_grnd` |
| 10 | Apply tendency   | `apply_tendency!`, `calc_f_ice!` | `tpo.H_ice`, `tpo.f_ice`, `tpo.cmb` |
| 11 | Post-kill consistency | inline | `tpo.lsf` |
| 12 | dlsfdt diagnostic | inline | `tpo.dlsfdt` |

The phase is gated by `ycalv.use_lsf`. When false, `cmb`, `cmb_flt`,
`cmb_grnd`, and `dlsfdt` are zeroed on entry and the rest of the
pipeline is skipped.

## Calving-front velocity laws (step 3 / 4)

Three laws are wired through. They all write a horizontal calving
velocity in m/yr, signed opposite to ice flow.

### `equil`

```math
cr = -u_\text{bar}
```

Pins the front in place — `w = u + cr ≡ 0`, so the level set is
stationary. Useful for production runs where the front position is
imposed and only interior ice dynamics are evolving.

### `threshold`

```math
cr_x = -u_\text{bar} \cdot \max\!\left(0,\; 1 + \frac{H_c - H_\text{acx}}{H_c}\right)
```

The factor ramps linearly from zero (at `H ≥ 2·Hc`) to `1` (at `H =
Hc`) to `2` (at `H = 0`). `Hc` is `ycalv.Hc_ref_flt` or
`ycalv.Hc_ref_grnd`. At ice/ocean borders the ice thickness is
staggered to the ice-side cell so an open-ocean neighbour does not
collapse `H_acx` to zero.

### `vm-m16`

Morlighem et al. (2016) von-Mises calving:

```math
cr = -u_\text{bar} \cdot \max\!\left(0,\; \frac{\tau_1}{\tau_\text{ice}}\right)
```

Requires the 1st principal stress `tau_1` from `mat`. Not yet ported
in Yelmo.jl — calling this errors at step time.

### Merging (step 5)

`merge_calving_rates!` combines `cmb_flt_*` and `cmb_grnd_*` into a
single front velocity per face:

- **Floating face** (`f_grnd_ac == 0`): use `cmb_flt`.
- **Grounded face below SL** (mean `z_bed` across face below mean
  `z_sl`): use `cmb_grnd`.
- **Grounded face above SL**: pin to `cr = -u_bar`. Without this, a
  marine retreat law on a land-terminating face would be allowed to
  advance into above-SL cells via lsf advection.

## LSF advection (step 6)

`lsf_update!` advances `lsf` by `dt` years at the front velocity

```math
w = u_\text{bar} + cr
```

via `advect_tracer!` — the same first-order upwind kernel used for
`H_ice` advection in phase 1. Internally CFL-aware.

Before advection the front velocity is extrapolated outward into the
ocean along its natural axis (`extrapolate_ocn_acx!`,
`extrapolate_ocn_acy!`) using a single-pass forward + backward sweep
keyed off the raw ice velocity (`u_bar == 0` ⇒ ocean). This gives
upwind advection a sensible velocity in cells just outside the ice
where `lsf` still has gradients.

After advection `lsf` is saturated to `[-1, 1]` to keep the field
bounded under the upwind diffusion. Redistancing restores the slope.

## Redistancing (step 8)

Yelmo.jl uses **Sussman/Osher Hamilton-Jacobi redistancing** to
restore `|∇φ| ≈ 1` near the zero level set without moving it,
replacing Fortran's periodic `±1` re-flag and post-advection
neighbour-snap. The PDE is

```math
\frac{\partial \phi}{\partial \tau} + \operatorname{sgn}(\phi_0)\,(|\nabla\phi| - 1) = 0
```

with

- `φ₀` = the lsf field at the start of redistancing, held fixed for
  the duration. This is what prevents the zero level set from
  drifting.
- `sgn(φ₀) = φ₀ / √(φ₀² + ε²)` — smoothed sign with `ε = max(dx, dy)`.
- Godunov upwind discretisation for `|∇φ|`.
- Pseudo-timestep `dτ = 0.5 · min(dx, dy)`.
- Zero-gradient (Neumann) boundaries — clamped index reads.

The trigger is controlled by `ycalv.dt_lsf`:

| `dt_lsf` | Behaviour |
|---|---|
| `< 0`  | Redistance every step. |
| `== 0` | Never. |
| `> 0`  | Fire when `[t, t+dt]` crosses an integer multiple of `dt_lsf`. Robust to non-integer `dt`. |

The Fortran semantics (`dt_lsf > 0` ⇒ periodic ±1 re-flag) are
intentionally replaced — `dt_lsf` in Yelmo.jl now controls the
cadence of a real signed-distance restoration, not a destructive
flag.

## Kill (steps 9–11)

Where `lsf > 0` and the cell still holds ice, the kill rate

```math
cmb = -\frac{H}{dt}
```

is built into `tpo.cmb` and applied via `apply_tendency!` with
`adjust_mb=true`, so the realised tendency reflects whatever clipping
or under-melt protection `apply_tendency!` performs.

A separate aa-stagger magnitude diagnostic is filled per cell:

```math
\text{cmb\_flt}[i,j] = \sqrt{\Big(\tfrac{cmb\_flt\_acx[i,j] + cmb\_flt\_acx[i+1,j]}{2}\Big)^2 + \Big(\tfrac{cmb\_flt\_acy[i,j] + cmb\_flt\_acy[i,j+1]}{2}\Big)^2}
```

(and similarly for `cmb_grnd`). Useful for output / inspection but
not used in the mass balance.

After applying the kill and refreshing `f_ice`, a post-kill
consistency pass forces `lsf = +1` where `H ≤ 0` and the bed is
below SL — handles the case where ice was removed by external
forcing but lsf still says "ice". This pass is **skipped for
`calv_flt_method == "equil"`** so the front stays pinned even under
transient external thinning.

## Differences from Fortran

1. **No vertical mass-balance form.** Fortran's `calving_aa.f90` (the
   `mb-form` path) is not ported.
2. **No neighbour-based reset** of `lsf` after advection (Fortran
   `lsf_module.f90:808-818`). Redistancing handles the slope.
3. **Real redistancing** instead of periodic `±1` re-flag (Fortran
   `dt_lsf` block). Same parameter controls the cadence.
4. **Single-pass extrapolation** of the front velocity into the
   ocean. Fortran iterates a `do while` until convergence; Yelmo.jl
   does forward + backward sweep in O(N).
5. **`cmb_flt_x/y` renamed to `cmb_flt_acx/acy`** in the model-side
   variable table so the regex-based allocator stages them on
   XFace / YFace grids. The Yelmo Mirror keeps the Fortran
   `_x`/`_y` names since it snapshots the Fortran NetCDF schema.

## Tests

`test/test_yelmo_topo.jl` covers (nine testsets, 30 assertions):

- `lsf_init!` — sign assignment from `H_ice` and bed/SL.
- `extrapolate_ocn_ac{x,y}!` — fill in both directions, reference
  not mutated, empty rows untouched.
- `lsf_redistance!` — `|∇φ| = 1` near front within 1e-3, zero set
  preserved within 1e-3.
- `lsf_update!` — passive transport at uniform `u`, saturation holds.
- `calc_calving_equil_ac!` — `cr = -u_bar`.
- `calc_calving_threshold_ac!` — exact face values including the
  ice/ocean border ice-side stagger; `Hc = 0` errors.
- `merge_calving_rates!` — all three branches.
- `calving_step!` end-to-end on a synthetic shelf — kill registers,
  `cmb = -500/dt`, mass balance closes (`err_total < 1e-9`).
- `calving_step!` dispatch — `vm-m16` errors, unknown method errors.

## Configuration

Per [`YelmoModelPar.YcalvParams`](../src/YelmoModelPar.jl):

| Field | Default | Meaning |
|---|---|---|
| `use_lsf`          | `false`     | Master switch for the calving phase. |
| `calv_flt_method`  | `"vm-l19"`* | One of `"none"`/`"zero"`, `"equil"`, `"threshold"`, `"vm-m16"`. |
| `calv_grnd_method` | `"zero"`    | Same set as `calv_flt_method`. |
| `dt_lsf`           | `-1.0`      | Redistancing cadence (see table above). |
| `Hc_ref_flt`       | `200.0`     | Threshold thickness for floating, m. |
| `Hc_ref_grnd`      | `200.0`     | Threshold thickness for grounded, m. |
| `tau_ice`          | `250e3`     | Ice fracture strength for vm-m16, Pa. |

*The default carries over from Fortran's nameset, but `vm-l19` is an
aa-form law not ported in Yelmo.jl. Set explicitly when enabling
`use_lsf`.
