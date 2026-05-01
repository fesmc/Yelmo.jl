# Dynamics

The dynamics step `dyn_step!(y, dt)` advances the velocity-related
state in `y.dyn` by one outer time step. The current milestone wires
through the full pre- and post-solver chain plus a Shallow-Ice
Approximation (SIA) solver dispatch; SSA / hybrid / DIVA solvers
land in subsequent milestones.

This page documents:

1. The phase-by-phase pipeline inside `dyn_step!`.
2. The driving stress and lateral boundary stress at the ice front.
3. The effective basal pressure `N_eff` (six methods).
4. The bed-roughness chain `cb_tgt → cb_ref → c_bed`.
5. The SIA velocity solver under the **Option C** vertical-stagger
   convention.

## Phase pipeline

`dyn_step!` runs through nine phases per call. Phase order matches
Fortran's `calc_ydyn` body (`yelmo_dynamics.f90:48`):

| # | Phase | Helper | Output |
|---|---|---|---|
| 1 | Snapshot prev velocity                  | —                              | `dyn.ux_bar_prev`, `dyn.uy_bar_prev`, scratch `uxy_prev` |
| 2 | Driving stress                          | `calc_driving_stress!`         | `dyn.taud_acx`, `dyn.taud_acy` |
| 3 | Optional grounding-line refinement      | `calc_driving_stress_gl!`      | `dyn.taud_acx`, `dyn.taud_acy` |
| 4 | Lateral boundary stress at the ice front| `calc_lateral_bc_stress_2D!`   | `dyn.taul_int_acx`, `dyn.taul_int_acy` |
| 5 | Effective pressure + bed-roughness chain| `calc_ydyn_neff!`, `calc_cb_ref!`, `calc_c_bed!` | `dyn.N_eff`, `dyn.cb_tgt`, `dyn.cb_ref`, `dyn.c_bed` |
| 6 | Solver dispatch                         | `calc_velocity_sia!` (or no-op)| `dyn.ux_i`, `dyn.uy_i`, `dyn.ux_i_bar`, `dyn.uy_i_bar`, plus combined `dyn.ux`, `dyn.uy`, `dyn.ux_bar`, `dyn.uy_bar` |
| 7 | Underflow clip                          | inline                          | `dyn.ux/uy`, `dyn.ux_bar/uy_bar` |
| 8 | Velocity Jacobian + `uz` + strain rates | (deferred — milestone 3h)       | — |
| 9 | Diagnostics                             | `calc_ice_flux!`, `calc_magnitude_from_staggered!`, `calc_vel_ratio!` | `dyn.qq*`, `dyn.uxy*`, `dyn.taud`, `dyn.taub`, `dyn.f_vbvs`, `dyn.duxydt` |

`dyn_step!` does **not** advance `y.time` — that is owned by
`topo_step!`, which runs first.

## 1. Driving stress

The depth-integrated SIA driving stress on the ac-staggered faces
between cells `(i, j)` and `(i+1, j)` (analogous in y):

```math
\tau_d^x = \rho_i\,g\,H_\mathrm{mid}^x(i, j)\,\frac{\partial z_s}{\partial x}\bigg|_{i+\tfrac12, j}
\,,\qquad
\tau_d^y = \rho_i\,g\,H_\mathrm{mid}^y(i, j)\,\frac{\partial z_s}{\partial y}\bigg|_{i, j+\tfrac12}.
```

`H_mid` is a **margin-aware** face-average of `H_ice_dyn`: when one
side of the face is fully covered and the other is partially
covered, the kernel pairs the fully-covered cell's thickness with
zero rather than averaging through the margin. This avoids a
spurious `H_mid` halving at one-cell-wide ice fronts, which would
otherwise collapse the driving stress to half its physical value
exactly where it matters.

`H_ice_dyn` and `f_ice_dyn` are the dynamic-ice fields produced by
[`calc_dynamic_ice_fields!`](@ref) and
[`extend_floating_slab!`](@ref) — these extend a thin floating slab
under the ice front so the SIA pressure gradient does not collapse
to zero one cell upstream of the calving front. The surface
gradients `dzsdx`, `dzsdy` come from the `tpo` group and are
themselves computed by [`calc_gradient_acx!`](@ref) /
[`calc_gradient_acy!`](@ref) in the topography diagnostic phase.

The component is clipped per cell to `|τ_d| ≤ ydyn.taud_lim`.

Port of `velocity_general.f90:985 calc_driving_stress`.

### Optional grounding-line refinement

When `ydyn.taud_gl_method != 0`, [`calc_driving_stress_gl!`](@ref)
post-processes `taud_acx` / `taud_acy` at faces near the grounding
line. Methods `-1`, `1`, `2`, `3` mirror the Fortran reference; the
default `0` is a no-op. The orchestrator passes `beta_gl_stag = 1`
unconditionally (matching Fortran), which leaves only method `-1` as
an active branch in the current port — the others are dead code in
the Fortran reference under that staggering choice.

## 2. Lateral boundary stress

At a marine or floating ice front, the depth-integrated lateral
stress on the front face is

```math
\tau_\mathrm{bc}^\mathrm{int} =
  \tfrac12\,\rho_i\,g\,H_\mathrm{ice}^2 - \tfrac12\,\rho_w\,g\,H_\mathrm{ocn}^2
```

(Lipscomb et al. 2019, Eqs. 11–12; Winkelmann et al. 2011, Eq. 27),
where the submerged ocean column at the front cell is

```math
H_\mathrm{ocn} = H_\mathrm{ice}\,
  \biggl(1 - \min\!\Bigl(\tfrac{z_s - z_\mathit{sl}}{H_\mathrm{ice}},\,1\Bigr)\biggr).
```

For a fully-grounded front above sea level (`z_s > z_sl` ⇒
submerged fraction ≤ 0), `H_ocn = 0` and the second term collapses.

[`calc_lateral_bc_stress_2D!`](@ref) writes the value at each face
that straddles a front (one cell with `mask_frnt > 0` adjacent to
one with `mask_frnt < 0`); all other faces are zeroed. The
`mask_frnt` field comes from [`calc_ice_front!`](@ref) in the
topography diagnostic phase.

Port of `velocity_general.f90:1450 calc_lateral_bc_stress_2D`.

## 3. Effective basal pressure

[`calc_ydyn_neff!`](@ref) writes `dyn.N_eff` at aa-cell centres,
dispatching on `y.p.yneff.method`:

| `method` | Formula |
|---|---|
| `-1` | No-op (assume `N_eff` is set externally). |
| `0`  | Constant: `N_eff = neff_const`. |
| `1`  | Overburden: `N_eff = ρ_i · g · H_eff`. |
| `2`  | Marine connectivity (Leguy et al. 2014, Eq. 14). |
| `3`  | Till basal pressure (van Pelt & Bueler 2015, Eq. 23). |
| `4`  | As method 3 but with constant till saturation `H_w = H_w_max · s_const`. |
| `5`  | Two-valued blend `f_pmp · (δ P_0) + (1 - f_pmp) · P_0`. |

All methods scale `H_ice` to "effective" thickness (zero for
partially-covered cells, full thickness for fully-covered) before
computing the overburden, mirroring `calc_H_eff(set_frac_zero=true)`
in the Fortran. Floating cells (`f_grnd == 0`) get `N_eff = 0`
(except method 0).

Method 2 (marine connectivity) is the most commonly used for marine
ice sheets:

```math
N_\mathit{eff} = \rho_i g\,H_\mathrm{eff} - p_w,
\qquad
p_w = \rho_i g\,H_\mathrm{eff}\,
  \bigl[1 - (1 - x)^p\bigr],
\qquad
x = \min\!\bigl(1,\,H_\mathrm{float} / H_\mathrm{eff}\bigr).
```

`p` is `yneff.p` (typically 1 or 2). At `H_eff = H_float` the
formula collapses to the floating-pressure limit (`N_eff = 0`).

**Subgrid sampling is not yet ported** — `yneff.nxi > 0` errors with
a deferral pointer (the Fortran reference samples `H_w` over the
cell using either Gaussian quadrature or a uniform grid; the Julia
port plugs in `FastGaussQuadrature.jl` later).

Port of `yelmo_dynamics.f90:832 calc_ydyn_neff` plus the four
`calc_effective_pressure_*` helpers from `basal_dragging.f90`.

## 4. Bed-roughness chain

Two helpers, both writing 2D Center fields:

[`calc_cb_ref!`](@ref) computes the **reference till-friction
coefficient** `cb_tgt`. Two-stage:

1. **Elevation scaling** (`scale_zb`): per cell, Gaussian-sample
   `z_bed ± f_sd · z_bed_sd` over `n_sd` perturbation points (linear
   abscissae over `[-1, 1]`, N(0, 1) weights), evaluate
   `λ(z_bed)`, take a weighted mean → `cb_ref = cf_ref · λ̄`,
   clamped to `≥ cf_min`. The two `λ` flavours are:
   - **`scale_zb = 1`** (linear): `λ = clamp((z_bed - z0) / (z1 - z0), 0, 1)`.
   - **`scale_zb = 2`** (exponential): `λ = exp((z_bed - z1) / (z1 - z0))`,
     capped at 1.
2. **Sediment scaling** (`scale_sed`): adjusts the elevation result
   with `H_sed` cover. `scale_sed = 1` writes the min of the
   elevation result and a linear `cf_min ↔ cf_ref` blend over
   `H_sed ∈ [H_sed_min, H_sed_max]`. `2`/`3` apply a multiplicative
   factor `1 - (1 - f_sed) · λ_sed`; `2` reapplies the `cf_min`
   floor, `3` does not (Schannwell et al. 2023 method).

Then [`calc_c_bed!`](@ref) computes the **basal drag coefficient**

```math
c_\mathit{bed} = c \cdot N_\mathit{eff},
\qquad
c = \begin{cases}
  c_b^\mathit{ref}              & \text{if } \mathit{is\_angle} = \mathit{false}, \\
  \tan(c_b^\mathit{ref} \cdot \pi / 180^\circ) & \text{if } \mathit{is\_angle} = \mathit{true},
\end{cases}
```

(the `is_angle` branch is the Bueler & van Pelt 2015 till-strength
angle formulation). Optional thermal scaling (`scale_T = 1`) blends
toward a frozen-bed reference value as `T'_b` drops below `T_frz`
— this is what gives `c_bed` its temperature-dependent kick in
mixed-base regions.

If `ytill.method == 1`, the orchestrator copies `cb_tgt → cb_ref`
each step; other `till_method` values leave `cb_ref` at its restart-
loaded / externally-supplied value.

Port of `physics/basal_dragging.f90`:
`calc_cb_ref` (line 62), `calc_c_bed` (line 278),
`calc_lambda_bed_lin` (line 843), `calc_lambda_bed_exp` (line 870).

## 5. SIA velocity solver

The Shallow-Ice Approximation drops the longitudinal stress terms,
leaving the depth-integrated horizontal velocity at each
height-coordinate `ζ` as

```math
u(x, y, \zeta) = u_b(x, y) - 2\,(\rho_i g)^n\,
  \bigl|\nabla z_s\bigr|^{n-1}\,\frac{\partial z_s}{\partial x}\,
  \int_0^\zeta A(T)\,(z_s - z)^n\,dz,
```

with `n = ydyn.n_glen` Glen's-law exponent and `A(T)` the temperature-
dependent rate factor (read from `mat.ATT`). Yelmo splits this into
three kernels:

### Per-layer shear stress

[`calc_shear_stress_3D!`](@ref) writes the SIA shear stresses on
`(acx, acy)` × `ζ` at the **Center** vertical positions:

```math
\tau_{xz}(i, j, k) = -(1 - \zeta_k)\,\tau_d^x(i, j),
\qquad
\tau_{yz}(i, j, k) = -(1 - \zeta_k)\,\tau_d^y(i, j).
```

These are SIA-only scratch buffers, not part of the model state —
they live as `y.dyn.scratch.sia_tau_xz` / `…tau_yz` and are
recomputed every `dyn_step!`.

### Depth-recurrence

[`calc_uxy_sia_3D!`](@ref) integrates the velocity field upward from
the bed using the Pollard / DeConto recurrence

```math
u(k) = u(k-1) + \tfrac12\,\mathit{fact\_ac}(k)\,
  \bigl(\tau(k) + \tau(k-1)\bigr),
```

where `fact_ac(k)` bundles a 4-cell ATT/H average plus an effective-
stress factor `τ_\mathrm{eff}^{(n_\mathrm{glen}-1)/2}`. The bed
boundary segment is integrated explicitly with `u_b = 0` (no-slip)
and `τ_\mathrm{xz}^\mathit{bed} = -τ_d^x` (closed form).

### Wrapper and surface segment

`calc_velocity_sia!` is the user-facing entry point (in
`Yelmo.YelmoModelDyn`; see the
[dynamics API page](../api/dynamics.md#sia-velocity-solver)). It:

1. Fills halos on `H_ice`, `f_ice`, `ATT`, `taud_acx`/`acy` so the
   4-cell averages in the kernels see the Bounded-Neumann replica.
2. Calls `calc_shear_stress_3D!` for the per-layer stresses.
3. Calls `calc_uxy_sia_3D!` for the 3D Center-staggered velocity.
4. Adds the **surface segment**: integrates from `ζ_c[N_z]` to `1`
   starting at `ux_i[k=N_z]`, with `τ_xz^surf = 0` (closed form) and
   `ATT_surf ≈ ATT[k=N_z]`. This produces the 2D
   `ux_i_s` / `uy_i_s` surface boundary values.
5. Computes the **depth-average** `ux_i_bar` / `uy_i_bar` by
   trapezoidal integration over `ζ ∈ [0, 1]` with explicit bed
   (`u = 0`) and surface (`ux_i_s`) endpoints.

### Option C vertical convention

Yelmo.jl uses Oceananigans' `Center()` z-stagger for 3D fields:
`ux_i`, `uy_i`, `ATT`, `tau_xz`, `tau_yz` live at **interior layer
midpoints** (length `Nz_aa`), not at faces. The bed (`ζ = 0`) and
surface (`ζ = 1`) endpoints are NOT in `zeta_c`; their boundary
values are handled explicitly by the kernel (bed segment) and the
wrapper (surface segment).

This is "Option C" in the project's design notes — chosen because it
maps cleanly onto Oceananigans' standard Field machinery without
introducing a custom z-stagger. The trade-off is that the `solver ==
"fixed"` surface-velocity diagnostic reports the topmost Center
value rather than the actual surface; the SIA branch corrects for
this by overwriting `ux_s` / `uy_s` with the wrapper-produced
`ux_i_s + ux_b`. The `solver = "fixed"` limitation is a known issue
to revisit when temperature-dependent ATT lands.

Port of `physics/velocity_sia.f90` (`calc_shear_stress_3D` line 63,
`calc_uxy_sia_3D` line 110) plus the Option C extension for the
surface segment and the depth average.

## Tests

`test/test_yelmo_dyn.jl` covers:

- Kernel-level: `calc_driving_stress!`, `calc_lateral_bc_stress_2D!`,
  the six `calc_ydyn_neff!` methods, `calc_cb_ref!` (across the
  `scale_zb` / `scale_sed` cross-product), `calc_c_bed!` (linear
  and angle modes; thermal scaling).
- Restart-consistency: the chain `cb_ref → c_bed` reproduces a
  Fortran-Yelmo restart's `c_bed` field within tolerance.

`test/test_yelmo_sia.jl` and `test/benchmarks/test_sia.jl` cover the
SIA solver:

- Kernel-level: `calc_shear_stress_3D!`, `calc_uxy_sia_3D!`, and
  the wrapper `calc_velocity_sia!` against analytical references.
- BUELER-B Halfar dome convergence: at `t = 1000 yr` from the
  analytical Halfar solution, the SIA-driven thinning rate matches
  the analytical reference to within 10% across four refinement
  levels — verifying the full SIA + topography pipeline end-to-end.

The benchmarks live under `test/benchmarks/` with a fixture restart
file `bueler_b_t1000.nc` and a `regenerate.jl` script that rebuilds
the fixture from the analytical Halfar formula.
