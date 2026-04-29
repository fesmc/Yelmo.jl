# Overview — Milestone 2: port `tpo` to Julia

The `tpo` (topography) component evolves ice-sheet thickness `H_ice` via mass conservation, then derives surface elevation, ice/grounding masks, and gradients from it. It is the first physics component we port after the v0 scaffolding.

`YelmoModelTopo` lives at `src/topo/YelmoModelTopo.jl` (sibling to `YelmoCore`, with a subfolder for helpers). Its public surface is one orchestration call, `topo_step!(y::YelmoModel, dt)`, that `step!(::YelmoModel, dt)` invokes in fixed order ahead of the (still-stub) `dyn`, `mat`, `thrm` calls.

# What `tpo` does (per Yelmo Fortran)

Reference: `yelmo/src/yelmo_topography.f90:calc_ytopo_pc` and `yelmo/src/physics/{topography,mass_conservation,solver_advection*}.f90`.

Per timestep, `calc_ytopo_pc` runs in this order:

1. **Mass-conservation advection** of `H_ice` using vertically-averaged velocity `(ux_bar, uy_bar)` from `ydyn`. Two solver choices:
   - `impl-lis`: implicit, first-order upwind, sparse linear solve (LIS in Fortran). **Primary** for the Julia port.
   - `expl-upwind`: explicit second-order upwind, CFL-limited. **Secondary**, useful for comparisons.
2. **Ice-fraction mask** update (`f_ice`, `f_grnd`) from `H_ice + z_bed + z_sl` flotation logic.
3. **SMB tendency** applied from `bnd.smb`.
4. **BMB tendency** applied (combined `thrm.bmb_grnd` for grounded + `bnd.bmb_shlf` for floating).
5. **FMB tendency** (frontal melt) applied.
6. **DMB tendency** (subgrid discharge near grounding line, Calov+ 2015).
7. **Calving** — multiple methods (threshold, von Mises Lipscomb+ 2019, eigencalving Levermann+ 2012, kill). Floating + grounded handled separately. Inputs: stresses from `mat`, strains from `dyn`.
8. **Optional relaxation** toward `bnd.H_ice_ref` or previous state.
9. **Residual boundary adjustment** — clamp `H_ice ≥ 0`, enforce `bnd.ice_allowed`.
10. **Diagnostic update** — `dHidt`, `mb_err`, `z_srf`, `z_base`, surface/thickness gradients on staggered nodes (`dzsdx`, `dzsdy`, `dHidx`, `dHidy`).

Predictor-corrector wrapping (`pred`/`corr` substructs) averages two passes per step in the Fortran. Whether to port that or use a simpler one-pass scheme in v1 is open.

# Inputs to `tpo`

- `ydyn`: `ux_bar`, `uy_bar` (advection); `uxy_b`, `strn2D` (calving, residual)
- `ybnd`: `smb`, `bmb_shlf`, `fmb_shlf`, `z_bed`, `z_sl`, `z_bed_sd`, `H_ice_ref`, `ice_allowed`
- `ytherm`: `bmb_grnd`
- `ymat`: `strs2D`

For milestone 2, dyn/therm/mat fields are read straight from the restart-loaded state and held fixed across steps (no physics evolving them yet).

# Open design questions for the grill

- **Scope of v1**: which of phases 1–10 are in milestone 2 vs. deferred (e.g. calving, DMB, predictor-corrector).
- **Solver implementation**: `impl-lis` via Julia sparse-matrix assembly + `LinearSolve.jl` / `Krylov.jl`, vs. building on Oceananigans' advection operators where possible.
- **Time-stepping**: predictor-corrector port vs. single-pass for v1.
- **Calving design**: pluggable backend (struct dispatch) vs. config-string switch.
- **Boundary conditions**: how to map Yelmo's boundary-code system onto Oceananigans' grid topologies.
- **Validation**: lockstep against `YelmoMirror` with fixed velocity, plus an analytical advection benchmark.

# Constraints (already agreed in v0)

- Julia-native, Oceananigans-first numerics — not bit-matching the Fortran.
- Sibling module pattern: `src/topo/YelmoModelTopo.jl` plus helpers in `src/topo/`.
- Hard-coded ordered orchestration in `step!(::YelmoModel, dt)`; `topo_step!` is the entry.
- Develop in a worktree; tests live alongside the v0 integration test.
