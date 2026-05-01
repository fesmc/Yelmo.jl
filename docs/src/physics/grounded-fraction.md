# Subgrid grounded fraction — algorithm and stabilisation

The grounded fraction `f_grnd[i, j] ∈ [0, 1]` measures the
sub-grid-resolved area of cell `(i, j)` that is anchored to the bed
(as opposed to floating). It is read by basal mass balance,
calving, and grounding-line stress treatments. Yelmo.jl uses the
**CISM bilinear-interpolation scheme** of Leguy et al. (2021), via
IMAU-ICE v2.0, ported from `physics/topography.f90:1896` (the
linked Yelmo Fortran tree). The Julia implementation
[`src/topo/grounded.jl`](https://github.com/fesmc/Yelmo.jl/blob/main/src/topo/grounded.jl)
introduces two analytical-limit fixes that improve numerical
accuracy without changing the algorithm structure.

This document describes:

1. The flotation diagnostic `H_grnd` and the bilinear sub-cell
   reconstruction.
2. The four canonical sign-pattern scenarios and their analytical
   area formulas.
3. The two numerical fragilities of the original Fortran formula
   and our analytical-limit replacements.
4. Convergence on linear and smooth-nonlinear grounding lines.

## 1. Flotation diagnostic

Define the per-cell flotation function

```math
H_\mathrm{grnd}(x, y) = H_\mathrm{ice}(x, y) - \max\bigl(z_\mathrm{sl}(x, y) - z_\mathrm{bed}(x, y),\, 0\bigr) \cdot \frac{\rho_\mathrm{sw}}{\rho_\mathrm{ice}}
```

so that `H_grnd > 0` at grounded points and `H_grnd < 0` at
floating points. The per-cell sub-grid grounded fraction is

```math
f_\mathrm{grnd}(i, j) = \frac{1}{|\Omega_{ij}|} \int_{\Omega_{ij}} \mathbb{1}[H_\mathrm{grnd}(x, y) \ge 0] \, dx\, dy.
```

## 2. CISM sub-cell reconstruction

The cell `Ω_{ij}` is split into four equal quadrants
(`NW`, `NE`, `SW`, `SE`). On each quadrant, `H_grnd` is approximated
by the bilinear interpolant of its values at the four quadrant
corners. With `f_flt = -H_grnd` (positive ⇒ floating, negative ⇒
grounded), the corner values are constructed from a 9-point stencil
of cell-centred `f_flt` samples:

```
       j+1     j       j-1
i-1   ─•─────•───────•─
        ╲   ╱ ╲     ╱
         f_NW   f_N   f_NE
        ╲   ╱     ╲ ╱
i      ─•───•───────•─
            f_W  f_m  f_E
        ╱   ╲     ╱ ╲
         f_SW   f_S   f_SE
        ╱   ╲ ╱     ╲
i+1   ─•─────•───────•─
```

Each value is a 4-cell mean of `f_flt` at the surrounding cell
centres. For example,

```math
f_{NW}(i, j) = \tfrac{1}{4}\bigl(f_\mathrm{flt}(i{-}1, j{+}1) + f_\mathrm{flt}(i, j{+}1) + f_\mathrm{flt}(i{-}1, j) + f_\mathrm{flt}(i, j)\bigr).
```

For any *bilinear* `f_flt(x, y) = a + b x + c y + d x y`, this
4-cell mean equals the value of `f_flt` at the corner exactly —
the corner stencil is consistent to second order in `dx`. For
smooth nonlinear `f_flt` the truncation error is `O(dx²)`.

Each quadrant's 4-corner bilinear interpolant is then
analytically integrated to give the quadrant's grounded fraction;
the cell's `f_grnd` is the mean of the four quadrant fractions.

## 3. Analytical area kernel

Map each quadrant onto the unit square `[0, 1]²` with corner values
`f_NW`, `f_NE`, `f_SW`, `f_SE`. Define the bilinear interpolant in
terms of the SW corner:

```math
f(u, v) = \mathit{aa} + \mathit{bb}\,u + \mathit{cc}\,v + \mathit{dd}\,u\,v
```

where

```math
\begin{aligned}
\mathit{aa} &= f_{SW},\\
\mathit{bb} &= f_{SE} - f_{SW},\\
\mathit{cc} &= f_{NW} - f_{SW},\\
\mathit{dd} &= f_{NE} + f_{SW} - f_{NW} - f_{SE}.
\end{aligned}
```

Note that `dd = 0` exactly when the bilinear interpolant degenerates
to a linear function. We compute

```math
\varphi(\mathit{aa}, \mathit{bb}, \mathit{cc}, \mathit{dd}) = \int_0^1 \int_0^1 \mathbb{1}[f(u, v) \le 0] \, du\, dv.
```

By convention ``f \le 0`` is grounded and ``f > 0`` is floating;
corners exactly equal to zero sit on the level set, contribute
measure zero to the integral, and are bookkept as grounded.

### 3.1 Trivial cases

If all four corner values are ``\le 0``, return ``\varphi = 1``. If
all four are ``> 0``, return ``\varphi = 0``.

### 3.2 Canonical sign patterns

The remaining 14 mixed sign patterns reduce by 90° rotation to four
canonical scenarios:

| Scenario | Sign pattern (NW, NE, SW, SE) | Geometry |
|---|---|---|
| 1 | (+, +, ≤0, +) | only SW grounded — small triangle near SW |
| 2 | (≤0, ≤0, >0, ≤0) | only SW floating — complement of scenario 1 |
| 3 | (+, +, ≤0, ≤0) | south grounded, north floating — horizontal strip |
| 4 | (+, ≤0, ≤0, +) | saddle: SW & NE grounded, SE & NW floating |

For each scenario the level-set curve ``f = 0`` has a known geometry
within the unit square, and the area ``\varphi`` is computable in
closed form by integrating along `u`. The level-set curve is

```math
v(u) = -\frac{\mathit{aa} + \mathit{bb}\,u}{\mathit{cc} + \mathit{dd}\,u},
```

which is a hyperbola for ``dd \ne 0`` and a line for `dd = 0`.

### 3.3 Scenario 1 (SW corner grounded)

The grounded region is the curvilinear triangle bounded by the
bottom edge from `(0, 0)` to `(u_x, 0)` with `u_x = -aa/bb`, the
level-set arc from `(u_x, 0)` to `(0, v_y)` with `v_y = -aa/cc`,
and the left edge back to `(0, 0)`. Its area is

```math
\varphi_1 = \frac{(\mathit{bb}\,\mathit{cc} - \mathit{aa}\,\mathit{dd}) \log\bigl|1 - \frac{\mathit{aa}\,\mathit{dd}}{\mathit{bb}\,\mathit{cc}}\bigr| + \mathit{aa}\,\mathit{dd}}{\mathit{dd}^2}.
```

This is the standard Fortran/CISM formula. It has two numerical
fragilities, addressed in §4.

### 3.4 Scenario 2 (SW corner floating)

By antisymmetry, the floating fraction equals
``\varphi_1(-aa, -bb, -cc, -dd)``, so

```math
\varphi_2 = 1 - \varphi_1(-\mathit{aa}, -\mathit{bb}, -\mathit{cc}, -\mathit{dd}).
```

### 3.5 Scenario 3 (south grounded, north floating)

The level set crosses from `(0, v_W)` on the left edge to `(1, v_E)`
on the right edge. The grounded region lies below the curve.
Integrating `v(u)` on `[0, 1]`:

```math
\varphi_3 = F(1) - F(0), \quad F(u) = \frac{(\mathit{bb}\,\mathit{cc} - \mathit{aa}\,\mathit{dd}) \log|\mathit{cc} + \mathit{dd}\,u| - \mathit{bb}\,\mathit{dd}\,u}{\mathit{dd}^2}.
```

Within scenario 3 both `cc > 0` and `cc + dd > 0`, so the log
arguments are strictly positive and the formula is well-conditioned.

### 3.6 Scenario 4 (saddle)

The grounded region splits into two opposite-corner triangles. The
SW triangle has area ``\varphi_1``. The NE triangle, after a 180°
relabelling, has area ``\varphi_1`` evaluated on the rotated corner
values. The total is the sum.

## 4. Numerical fragilities and analytical limits

The Fortran reference patches two issues with finite-ε shifts (the
`_FRAC_FTOL` snap on near-zero corners, and the ±0.1 perturbation
when ``|dd| < 10^{-4}``). Both are removable singularities — we
replace them with the corresponding analytical limits.

### 4.1 The ``dd \to 0`` limit (linear branch)

When ``|dd| < \varepsilon_\mathrm{lin}`` (we use ``10^{-12}``), the
interpolant ``f(u, v) = aa + bb\,u + cc\,v + dd\,uv`` is effectively
a plane and the level set ``f = 0`` is a straight line. The formula
for ``\varphi_1`` has a ``1/dd^2`` singularity that is in fact
removable: as ``dd \to 0``, the numerator vanishes at the same rate.
Using ``\log(1 - x) = -x - x^2/2 + O(x^3)`` with
``x = aa\,dd/(bb\,cc)``, expanding the numerator, and dividing by
``dd^2``, we obtain

```math
\lim_{dd \to 0} \varphi_1 = \frac{\mathit{aa}^2}{2\,\mathit{bb}\,\mathit{cc}}.
```

Rather than evaluate this Taylor limit, we observe that for `dd = 0`
the grounded set is *exactly* a half-plane intersected with the
unit square. The grounded area is a polygon (≤ 5 vertices) whose
area is given exactly by the **Sutherland–Hodgman clip + shoelace**
construction. We compute that polygon directly:

```julia
_grounded_area_linear_unit_square(aa, bb, cc)
```

This is exact in floating-point arithmetic for any linear `f_flt`,
and avoids the ill-conditioned division by ``dd^2``. The Fortran
±0.1 perturbation is replaced by this branch.

### 4.2 The ``\delta \to 0`` limit (saddle)

In scenarios 1, 2, and 4 the formula has another removable
singularity at

```math
\delta \equiv \mathit{bb}\,\mathit{cc} - \mathit{aa}\,\mathit{dd} \to 0,
```

which corresponds geometrically to the level-set curve passing
through the corner being integrated to (e.g. the SW corner
sitting exactly on the curve). At the singularity,
``\log|1 - aa\,dd/(bb\,cc)| = \log|0|``, and we have the
indeterminate form ``0 \cdot \infty``. Using
``\delta\,\log|\delta| \to 0`` as ``\delta \to 0``:

```math
\delta \cdot \log\bigl|\tfrac{\delta}{\mathit{bb}\,\mathit{cc}}\bigr|
= \delta(\log|\delta| - \log|\mathit{bb}\,\mathit{cc}|) \to 0,
```

so the surviving term is ``aa\,dd / dd^2 = aa/dd``. We thus take

```math
\lim_{\delta \to 0} \varphi_1 = \frac{\mathit{aa}}{\mathit{dd}}.
```

In the Julia kernel this is the helper

```julia
_phi_corner_grounded(aa, bb, cc, dd)
```

which switches to ``aa/dd`` when
``|\delta| < \varepsilon_\mathrm{sad} \cdot \max(|bb\,cc|,\,|aa\,dd|,\,1)``,
with ``\varepsilon_\mathrm{sad} = 10^{-12}`` (relative; scale-
invariant under uniform rescaling of the corner values).

The Fortran `±_FRAC_FTOL` snap is no longer needed because the
formula evaluates the correct limit as a corner approaches zero,
without shifting the input values.

### 4.3 Sign-assignment convention

With the limits in place, we use the strict convention
"``f \le 0`` is grounded, ``f > 0`` is floating". A corner with
``f_\bullet = 0`` is grounded by convention; the chosen scenario
formula then evaluates the correct limit as if ``f_\bullet \to 0^-``.
The opposite convention "``f \ge 0`` is floating" gives the same
analytical area (symmetric measure-zero level set), so the
convention choice only affects scenario routing.

## 5. Convergence

The `test/test_yelmo_topo.jl` benchmark suite measures the kernel's
accuracy on two analytical references.

### 5.1 Linear GL (Tier 1)

For any linear ``H_\mathrm{grnd}(x, y) = a x + b y + c``, the
analytical grounded fraction in cell `(i, j)` is the area of
``\{a x + b y + c \ge 0\} \cap \mathrm{cell}_{ij}``, computable by
half-plane / unit-square clipping. Eight parameterised `(a, b, c)`
cases — vertical, horizontal, two diagonals, four oblique — all
reproduce to **less than ``10^{-12}`` per cell**, since the linear
branch dispatches on ``|dd| < 10^{-12}`` and computes the clipped
polygon area exactly in floating-point.

### 5.2 Circular GL (Tier 2)

For ``H_\mathrm{grnd}(x, y) = R^2 - (x - x_0)^2 - (y - y_0)^2``, the
grounded set is a disk of area ``\pi R^2``. Sweeping
``N_x \in \{20, 40, 80, 160\}`` on ``L = 10``, ``R = 4``, centred,
and reporting the relative L¹ error
``\varepsilon = \bigl|A_\mathrm{est} - \pi R^2\bigr| / (\pi R^2)``:

| ``N_x`` | ``\Delta x`` | Relative error ``\varepsilon`` | Rate |
|---|---|---|---|
| 20  | 0.500   | 5.4 × 10⁻³ | — |
| 40  | 0.250   | 1.3 × 10⁻³ | 2.02 |
| 80  | 0.125   | 3.3 × 10⁻⁴ | 2.01 |
| 160 | 0.0625  | 8.3 × 10⁻⁵ | 2.01 |

Clean **second-order convergence** at every refinement, matching
the bilinear truncation order of the corner stencil. Before the
stabilisation, the same test produced rates 2.85, 0.97, −0.67 —
the negative final rate was the perturbation noise floor
(``\sim 5 \times 10^{-4}``) overtaking the truncation error.

## 6. Reference

- Leguy, G. R., Lipscomb, W. H., & Asay-Davis, X. S. (2021).
  *Marine ice sheet experiments with the Community Ice Sheet Model.*
  The Cryosphere, 15(7), 3229–3253.
- Yelmo Fortran: `physics/topography.f90:determine_grounded_fractions`
  (line 1896 onwards in the linked Yelmo source tree).
- Julia port:
  [`src/topo/grounded.jl`](https://github.com/fesmc/Yelmo.jl/blob/main/src/topo/grounded.jl).
- Tests:
  [`test/test_yelmo_topo.jl`](https://github.com/fesmc/Yelmo.jl/blob/main/test/test_yelmo_topo.jl)
  — search for `determine_grounded_fractions!` testsets.
