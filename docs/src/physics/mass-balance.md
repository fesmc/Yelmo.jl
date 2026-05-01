# Mass balance

The topography step accumulates ice thickness changes from five
mass-balance contributions, applied in fixed order:

```math
H^{n+1} = H^{n}_{\mathrm{adv}} + \Delta t \cdot
  \bigl(\mathit{smb} + \mathit{bmb} + \mathit{fmb} +
        \mathit{dmb} + \mathit{cmb} + \mathit{mb}_\mathrm{relax}
        + \mathit{mb}_\mathrm{resid}\bigr).
```

`H^{n}_{\mathrm{adv}}` is the post-advection thickness from phase 2
(see the [advection page](advection.md)). Each tendency is realised
by [`apply_tendency!`](@ref) with `adjust_mb=true`, so the field
written to `tpo.{smb,bmb,fmb,dmb,cmb,mb_relax,mb_resid}` is the
**realised** rate (after non-negativity clamping), not the raw
forcing. Their sum is `tpo.mb_net`.

Per-step the closure

```math
\frac{\partial H}{\partial t}\bigg|_\mathrm{step}
  = \mathit{dHidt} = \mathit{dHidt}_\mathrm{dyn} + \mathit{mb}_\mathrm{net}
```

holds to within `apply_tendency!`'s clipping tolerance; the
slab-conservation tests in `test_yelmo_topo.jl` verify this for
SMB / BMB / relaxation in isolation.

## `apply_tendency!` — the realisation step

[`apply_tendency!`](@ref) is the per-cell update used by every mass-
balance phase. Pseudocode:

```julia
for each (i, j):
    g  = clamp(G_mb[i,j], -mb_lim, +mb_lim)         # 1. clip
    H_new = H[i,j] + dt * g                         # 2. apply
    H_new = max(H_new, 0)                           # 3. clamp
    H_new = abs(H_new) < 1e-9 ? 0 : H_new           # 4. denoise
    H[i,j] = H_new
    if adjust_mb:
        G_mb[i,j] = (H_new - H_prev) / dt           # 5. record realised
end
```

The `mb_lim = 9999.0 m/yr` clip is a safety rail against runaway
forcing. The `1e-9 m` denoising matches Fortran's `TOL` parameter and
prevents micro-residual ice from confusing margin detection.

`adjust_mb=true` is used for every phase except the final residual
cleanup — it makes the per-phase output reflect what was actually
realised after clamping. Without this, `mb_net` would drift from
`dHidt` by the cumulative clamp residual.

Port of `apply_tendency` in
`yelmo/src/physics/mass_conservation.f90:112`.

## `mbal_tendency!` — the pre-clip

[`mbal_tendency!`](@ref) takes a raw forcing field (e.g.
`bnd.smb_ref`) and produces a realisable tendency by zeroing
unphysical configurations:

1. **No melt without ice**: `G[i,j] = 0` if `G < 0` and `H == 0`.
2. **No accumulation in open ocean**: `G[i,j] = 0` if `f_grnd == 0`
   and `H == 0`.
3. **Don't over-melt available ice**: `G[i,j] = -H/dt` if
   `H + dt·G < 0`.

Called by SMB, BMB, FMB, DMB before `apply_tendency!` so that the
per-phase recorded tendency starts in a realisable regime.

Port of `calc_G_mbal` in `yelmo/src/physics/mass_conservation.f90:486`.

## Surface mass balance (SMB)

Phase 6. The simplest pipeline:

```julia
mbal_tendency!(tpo.smb, tpo.H_ice, tpo.f_grnd, bnd.smb_ref, dt)
apply_tendency!(tpo.H_ice, tpo.smb, dt; adjust_mb=true)
```

`bnd.smb_ref` is the boundary-supplied surface mass balance forcing
(m/yr ice equivalent) — Yelmo doesn't compute SMB internally; it
trusts whatever climate forcing was loaded into `bnd.smb_ref`.

## Basal mass balance (BMB)

Phase 8. Combines a grounded melt field (`thrm.bmb_grnd`, from the
thermodynamics solver — currently zero in the Julia port) and a sub-
shelf melt field (`bnd.bmb_shlf`, boundary forcing) into a single per-
cell rate, with the grounding-line treatment selected by
`ytopo.bmb_gl_method`:

| `bmb_gl_method` | Rule |
|---|---|
| `"fcmp"` (flotation criterion) | `bmb_shlf` where `H_grnd ≤ 0`, else `bmb_grnd`. |
| `"fmp"`  (full melt)           | `bmb_shlf` where `f_grnd < 1`, else `bmb_grnd`. |
| `"pmp"`  (partial melt)        | Linear blend `f_grnd · bmb_grnd + (1-f_grnd) · bmb_shlf` for `f_grnd < 1`, else `bmb_grnd`. |
| `"nmp"`  (no melt)             | `bmb_shlf` where `f_grnd == 0`, else `bmb_grnd`. |
| `"pmpt"` (partial + tidal)     | Subgrid tidal-zone parameterisation. **Not yet ported** — requires `calc_subgrid_array`. |

After the switch, cells with positive `H_grnd` (grounded) and zero
`H_ice` (bare grounded land) are forced to `bmb = 0` so dry land
doesn't melt.

Then the standard pipeline:

```julia
calc_bmb_total!(tpo.bmb_ref, thrm.bmb_grnd, bnd.bmb_shlf,
                tpo.H_ice, tpo.H_grnd, tpo.f_grnd_bmb,
                ytopo.bmb_gl_method)
mbal_tendency!(tpo.bmb, tpo.H_ice, tpo.f_grnd, tpo.bmb_ref, dt)
apply_tendency!(tpo.H_ice, tpo.bmb, dt; adjust_mb=true)
```

The whole BMB phase is gated on `ytopo.use_bmb`; setting `false`
zeros `tpo.bmb_ref` and `tpo.bmb` and skips the apply.

Port of `calc_bmb_total` in
`yelmo/src/physics/topography.f90:1555`.

## Frontal mass balance (FMB)

Phase 10. Frontal melt at the **lateral** boundary between marine ice
and open ocean. A cell is a marine front if it has ice (`H > 0`),
sits below flotation (`H_grnd < H_ice`), and borders at least one
ice-free cell.

For such cells, define the submerged-front depth

```math
\mathit{dz} = \begin{cases}
  H_\mathrm{eff} \cdot \rho_i / \rho_\mathit{sw}            & \text{floating } (H_\mathit{grnd} < 0), \\
  \max\!\bigl((H_\mathrm{eff} - H_\mathit{grnd})\,\rho_i/\rho_\mathit{sw},\, 0\bigr) & \text{grounded marine},
\end{cases}
```

with `H_eff = H_ice / f_ice` for partially covered cells. The
**area-of-front fraction** is

```math
\mathit{frac} = \frac{n_\mathrm{margin}\,\mathit{dz}\,\Delta x}{\Delta x^2}
              = \frac{n_\mathrm{margin}\,\mathit{dz}}{\Delta x},
```

where `n_margin ∈ {1, 2, 3, 4}` counts ice-free orthogonal
neighbours. Three methods (`ytopo.fmb_method`):

| `fmb_method` | Rule |
|---|---|
| `0` | Pass-through: `fmb = bnd.fmb_shlf`. |
| `1` | Mean of `bnd.bmb_shlf` over ice-free neighbours, scaled by `frac · ytopo.fmb_scale`. |
| `2` | Local `bnd.fmb_shlf` scaled by `frac` (no `fmb_scale`). |

Method 1 is the physics path: a melt rate prescribed in the *open*
ocean is rebalanced onto the *ice* face that's exposed to it,
weighted by the submerged-front area. Method 2 is a simplified
variant.

Out-of-domain neighbour reads of `H_ice` resolve to 0, so domain-edge
cells correctly count the boundary as an ice-free neighbour.

Same `use_bmb` gate as the BMB phase; same realisation pipeline.

Port of `calc_fmb_total` in
`yelmo/src/physics/topography.f90:1658`.

## Discharge mass balance (DMB)

Phase 12. Subgrid solid-discharge calving (Calov et al. 2015). The
full kernel needs `dist_grline` and `dist_margin` distance-to-feature
fields, which are not yet computed in the Julia port. Only the
no-op branch (`dmb_method = 0`, the default) is wired through;
calling with any other method raises an explicit deferred-
implementation error.

The [`calc_mb_discharge!`](@ref) signature already mirrors the
Fortran for forward compatibility — once the distance fields land,
filling in the body is a self-contained follow-up.

## Calving mass balance (CMB)

Phase 14. The level-set flux method, described in detail on the
[calving page](calving.md). At a high level: a calving-front velocity
field is computed on staggered ac-nodes, the level-set function is
advected at that velocity, and ice is killed wherever the level set
flips to ocean.

## Relaxation tendency

Phase 15 (optional). Relaxes `H_ice` toward a reference state on a
prescribed timescale. Described on the
[relaxation page](relaxation.md).

## Residual cleanup

Phase 17. The final pass before diagnostics: removes ice in cells
where the dynamic + source updates produced an unphysical
configuration. Four sub-passes:

1. **Disallowed cells**: `H = 0` where `bnd.ice_allowed == 0`.
2. **Margin too thin**: at margins, zero cells with effective
   thickness below `ytopo.H_min_flt` (floating) or
   `ytopo.H_min_grnd` (grounded). Sub-tolerance ice
   (`H < 1e-6 m`) is also zeroed.
3. **Islands**: cells with `H > 0` and *every* orthogonal neighbour
   `H = 0` are zeroed.
4. **Margin cap**: margin cells thicker than the max of their
   ice-covered neighbours are clamped down to that max. Prevents
   margin runaway under explicit upwind.

The realised cleanup is converted to a tendency

```math
\mathit{mb}_\mathrm{resid} = 1.1 \cdot \frac{H_\mathrm{new} - H_\mathrm{old}}{\Delta t}
```

with the `1.1` factor providing a small overshoot so that
[`apply_tendency!`](@ref)'s non-negativity clamp lands on exactly
`H_new`. (The downstream `apply_tendency!` clamps the realised delta
back to `H_new`, so the tendency overshoot is purely arithmetic
slack.)

Two simplifications versus the Fortran:

- The EISMINT-summit averaging block (Fortran lines 651–658) is
  dropped.
- The trailing per-BC border-zeroing switch (Fortran lines 761–805)
  is dropped — Oceananigans' Dirichlet halo on `H_ice` already
  enforces `H = 0` outside the domain. Out-of-domain neighbour reads
  resolve to 0 via the `_h_or_zero` helper, keeping margin / island
  detection consistent with the BC.

Port of `calc_G_boundaries` in
`yelmo/src/physics/mass_conservation.f90:609`.

## Tests

The mass-balance pipeline is covered by the slab conservation tests
in `test/test_yelmo_topo.jl`:

- One slab per phase (SMB, BMB, relaxation), prescribing one
  tendency and verifying both the realised thinning and the closure
  `dHidt = dHidt_dyn + mb_net` to `1e-9`.
- Kernel-level unit tests for `apply_tendency!`, `mbal_tendency!`,
  `resid_tendency!`, `calc_bmb_total!`, `calc_fmb_total!`,
  `calc_mb_discharge!`.
