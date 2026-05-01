# Comparing model states

The two-backend design hinges on being able to verify that the pure-
Julia [`YelmoModel`](@ref) reproduces the Fortran-backed
[`YelmoMirror`](@ref) field-for-field. The
[`compare_state`](@ref) function performs that lockstep diff.

## Quickstart

```julia
using Yelmo

y_jl = YelmoModel("yelmo_restart.nc", 0.0; alias="model", strict=false)
y_fr = YelmoMirror("yelmo.nml", 0.0;       alias="mirror")

# After matching a few time steps...
init_state!(y_jl, 0.0); init_state!(y_fr, 0.0)
for _ in 1:5
    step!(y_jl, 1.0)
    step!(y_fr, 1.0)
end

result = compare_state(y_jl, y_fr; tol=1e-3)
println(result)
```

`compare_state` returns a [`StateComparison`](@ref) with `passes ::
Bool`, `n_compared`, `n_skipped`, and `failures :: Vector` of per-
field violations. The `show` method prints a short summary and the
first five offenders if any.

## What it compares

The comparison iterates over the six component groups (`bnd`, `dta`,
`dyn`, `mat`, `thrm`, `tpo`). For each field present in **both**
models with the **same shape**, it computes

```math
\mathrm{rel\_linf} =
  \frac{\max\bigl|a_{ij} - b_{ij}\bigr|}{\max\bigl|b_{ij}\bigr|}
```

(or the absolute max difference when the reference field is
identically zero). A field passes if `rel_linf ≤ tol`.

Fields present on only one side, or with mismatched shapes, are
skipped — not failures. This matters because the model and mirror
schemas occasionally diverge:

- Calving velocities are named `cmb_flt_acx`/`acy` in the model and
  `cmb_flt_x`/`y` in the mirror (the mirror keeps Fortran's NetCDF
  naming; the model uses the staggered-grid suffix that drives the
  allocator).
- The model's `tpo.dist_grline` and `dist_margin` are not yet
  computed; the mirror has them populated by the Fortran solver.

These show up in `n_skipped` rather than as failures.

## Tuning the tolerance

A reasonable default is `tol = 1e-3` (0.1% relative L∞). Tighter
tolerances are appropriate for individual kernels under controlled
conditions:

- Linear-grounding-line `determine_grounded_fractions!` benchmarks
  pass at `< 1e-12` (see the
  [grounded-fraction page](../physics/grounded-fraction.md)).
- The slab-conservation tests for SMB / BMB / relaxation close at
  `1e-9` for the residual `dHidt - dHidt_dyn - mb_net`.

For full-model lockstep against the Fortran mirror, `1e-3` to `1e-2`
is realistic — the Julia and Fortran kernels share algorithms but
not the order of summation, so machine-epsilon agreement is not
expected.

## When agreement breaks

The most common causes of cross-backend disagreement are, in rough
order of likelihood:

1. **Schema mismatch**: a field exists in one schema but not the
   other. Fix in `src/variables/{model,mirror}/yelmo-variables-*.md`.
2. **Different reference constants**: e.g. the model uses
   `YelmoConstants(:Earth)` while the mirror's namelist sets
   `phys_const = "EISMINT"`. Match them with the corresponding preset.
3. **Accumulation order differences in physics kernels** — typically
   8–10 ULP differences after a few hundred steps. Tolerate at
   `tol = 1e-2` for long-time-horizon comparisons.
4. **Genuine port bug**. The `failures` list points to the offending
   `(group, name)` pair; bisect by stepping one component at a time
   to localise the divergence.

A useful pattern when bisecting is to run with `step!` replaced by
the per-component function (`topo_step!(y, dt)` only) so each
component's port can be validated in isolation.
