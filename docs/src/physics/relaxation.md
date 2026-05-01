# Relaxation

Optional phase (15) of `topo_step!`. Pulls `H_ice` toward a reference
thickness `H_ref` on a per-cell timescale `tau_relax`. Useful for
imposing a reference geometry in spinup runs, for nudging ice shelves
toward observations, or for the standard "relax-toward-previous"
mode used by some adaptive-timestep schemes.

The phase is gated on `ytopo.topo_rel`. When `topo_rel == 0` the
phase is skipped entirely and `tpo.mb_relax` is zeroed.

## Continuous problem

Per cell, the relaxation tendency is

```math
\frac{\partial H}{\partial t}\bigg|_\mathrm{relax} =
  \frac{H_\mathrm{ref} - H}{\tau_\mathrm{relax}}.
```

with two special cases governed by the value of `tau_relax[i,j]`:

| `tau_relax[i,j]` | Behaviour |
|---|---|
| `> 0` | Standard exponential relaxation at this rate. |
| `== 0` | "Impose ice thickness" in a single step: `dHdt = (H_ref - H) / max(dt, 1)`. |
| `< 0`  | Sentinel: no relaxation in this cell. `dHdt = 0`. |

Combined with `apply_tendency!`'s non-negativity clamp, this lets a
caller suppress relaxation in any subset of cells without paying for
a separate mask.

## Two helpers, two phases

[`set_tau_relax!`](@ref) builds the per-cell timescale field given a
spatial-mask selector; [`calc_G_relaxation!`](@ref) converts the
timescale into a `dHdt` field. The orchestration is:

```julia
# 1. Build tau_relax. Skipped (and bnd.tau_relax used directly) when
#    topo_rel == -1 — the user supplies the timescale field.
if y.p.ytopo.topo_rel == -1
    interior(y.tpo.tau_relax) .= interior(y.bnd.tau_relax)
else
    set_tau_relax!(y.tpo.tau_relax, y.tpo.H_ice, y.tpo.f_grnd,
                   y.tpo.mask_grz, y.bnd.H_ice_ref,
                   y.p.ytopo.topo_rel, y.p.ytopo.topo_rel_tau)
end

# 2. Pick the reference state.
H_ref = y.p.ytopo.topo_rel_field == "H_ref"   ? y.bnd.H_ice_ref :
        y.p.ytopo.topo_rel_field == "H_ice_n" ? y.tpo.H_ice_n   :
        error("unknown topo_rel_field")

# 3. Build the tendency and apply it.
calc_G_relaxation!(y.tpo.mb_relax, y.tpo.H_ice, H_ref,
                   y.tpo.tau_relax, dt)
apply_tendency!(y.tpo.H_ice, y.tpo.mb_relax, dt; adjust_mb=true)
```

## Spatial-mask modes (`topo_rel`)

[`set_tau_relax!`](@ref) supports five modes, selected by the integer
`ytopo.topo_rel`:

| `topo_rel` | Where `tau_relax` is set to `tau` |
|---|---|
| `-1` | (Use `bnd.tau_relax` directly — `set_tau_relax!` is bypassed.) |
| `0`  | (Skipped — relaxation phase disabled.) |
| `1`  | Floating ice and ice-free points (i.e. wherever `f_grnd == 0` or `H_ref == 0`). |
| `2`  | Floating + grounding-line ice (a grounded cell with at least one floating orthogonal neighbour). |
| `3`  | Every cell. |
| `4`  | Grounding-zone points keyed off `mask_grz`. **Not yet ported** — the Julia port has not yet computed `mask_grz`. |

All other cells are set to `_TAU_OFF = -1.0`, the no-relaxation
sentinel that `calc_G_relaxation!` recognises.

For mode 2 the grounding-line "edge" detection reads neighbours via
a saturated lookup that treats out-of-domain reads as **grounded**
(`f_grnd = 1`), so domain edges don't spuriously trigger the
grounding-zone rule. This matches Yelmo Fortran's
`get_neighbor_indices_bc_codes` default.

Port of `set_tau_relax` in
`yelmo/src/physics/mass_conservation.f90:819`.

## Choosing a reference field (`topo_rel_field`)

Two reference targets are wired through:

| `topo_rel_field` | Target |
|---|---|
| `"H_ref"`   | `bnd.H_ice_ref` — typically observations, or a fixed initial state. |
| `"H_ice_n"` | The start-of-step thickness snapshot from phase 1. Use this for "relax toward the previous step" — useful in adaptive predictor / corrector schemes that want a soft restoring force. |

Any other string raises an error at step time.

## Numerical notes

The single-step "impose" mode (`tau_relax == 0`) deserves a note:
when `dt > 0` the formula `(H_ref - H) / dt` produces a tendency
that, when applied with `apply_tendency!`, lands `H` exactly on
`H_ref` (modulo the non-negativity clamp). When `dt ≤ 0` the formula
falls back to `(H_ref - H) / 1.0` so the call is well-defined; this
branch isn't normally exercised by `topo_step!` since the caller
checks `dt > 0` upstream.

The output `dHdt` is fully overwritten by [`calc_G_relaxation!`](@ref),
so the caller does not need to zero `tpo.mb_relax` beforehand. When
the relaxation phase is disabled (`topo_rel == 0`), `tpo.mb_relax` is
zeroed at phase entry instead.

Port of `calc_G_relaxation` in
`yelmo/src/physics/mass_conservation.f90:916`.

## Tests

The slab-conservation testset in `test/test_yelmo_topo.jl` includes
a relaxation case: a uniform slab is initialised away from a
prescribed `H_ref`, relaxation is the only enabled tendency, and the
test verifies both the realised thinning trajectory and the closure
`dHidt = dHidt_dyn + mb_net = mb_relax` to `1e-9`.

The kernel-level testset for `set_tau_relax!` covers each of modes
1–3 across a synthetic mask configuration; mode 4 is asserted to
error explicitly.
