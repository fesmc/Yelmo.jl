# ----------------------------------------------------------------------
# Adaptive predictor-corrector timestepping for `step!(YelmoModel, dt)`.
#
# Selection is driven by the `&yelmo` namelist parameters that mirror
# Fortran Yelmo (already declared in `YelmoParams`):
#
#   - `dt_method`     : 0 = fixed forward Euler (default),
#                       2 = adaptive predictor-corrector.
#   - `pc_method`     : "AB-SAM" (default), "HEUN", "FE-SBE".
#   - `pc_controller` : "PI42" (this PR), other Söderlind variants future.
#   - `pc_tol`        : rejection threshold on truncation-error proxy
#                       `eta` (m/yr).
#   - `pc_eps`        : minimum-eta floor used by the controller to
#                       prevent dt oscillation.
#   - `pc_n_redo`     : max rejection-retries per outer step.
#   - `dt_min`        : minimum dt (yr); dt is never reduced below this.
#   - `cfl_max`       : CFL safety factor (already used by
#                       `advect_tracer!`'s internal substepping).
#
# Architecture:
#
#   - `PCScheme`      : abstract type. Concrete: `AB_SAM` (default),
#                       `HEUN`, `FE_SBE`.
#   - `PIController`  : abstract type. Concrete: `PI42`; further
#                       controllers (H312b, H321PID, …) plug in via
#                       methods on `_dt_ratio`.
#   - `PCSnapshot`    : stores the prognostic state needed to roll back
#                       a rejected attempt — `H_ice`, the velocity
#                       fields (so the next SSA Picard solve has a good
#                       warm start), and `time`. All other model fields
#                       are diagnostics, recomputed by
#                       `update_diagnostics!` on restore.
#   - `PCScratch`     : per-model persistent scratch (predictor-state
#                       buffer, error/dt history). Lazily allocated on
#                       first adaptive call and cached in
#                       `y.dyn.scratch.pc_scratch`.
#
# Heun specifics (this PR):
#
#   The simplest 2nd-order PC that works without refactoring
#   `topo_step!` to take an explicit velocity argument: run the
#   existing FE step twice, then average.
#
#     y_pred = step_FE(y_n)              ! topo + dyn at y_n's state
#     y_**   = step_FE(y_pred)           ! topo + dyn at y_pred's state
#     y_corr = (y_n + y_**) / 2          ! Heun corrector identity
#
#   For H_ice this gives the standard Heun update
#   `H_corr = H_n + dt/2 * (k1 + k2)` with `k1 = (H_pred-H_n)/dt`,
#   `k2 = (H_** - H_pred)/dt`. Cost per attempt: 2 SSA solves (vs FE's
#   1) — Heun's allowed dt grows ~2-5× for matched accuracy, so net
#   wallclock is favourable for adaptive use.
#
# FE-SBE / AB-SAM (Step 2 PR):
#
#   Both use Fortran's "predictor + 1 corrector" pattern with 1 SSA
#   solve per attempt, requiring `topo_step!` to accept an explicit
#   velocity field for advection. That refactor is intentionally
#   deferred so this PR is reviewable on its own terms; the
#   `PCScheme` abstraction here is the integration point.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

# Extend `YelmoCore._select_step!` (declared as a stub there) with the
# adaptive-PC method. YelmoCore.step! calls into this dispatcher when
# `dt_method != 0`. Importing the symbol here means the method we
# add at the bottom of this file lands on the same generic function
# YelmoCore is referencing. Same trick for `_step_fe!`: forward-declared
# in YelmoCore so `step!` (FE branch) can refer to it; body added below.
import .YelmoCore: _select_step!, _step_fe!

# Pull in the timing macro so the per-phase wraps below land on the
# same generic that YelmoCore.step! sees.
using .YelmoTiming: @timed_section

export PCScheme, HEUN, FE_SBE, AB_SAM
export PIController, PI42

# ===== Scheme types =====

abstract type PCScheme end

"""
    HEUN

Heun's method (improved Euler). 2-stage explicit RK, order 2. The
truncation-error proxy is `tau = 1/(6·dt) · (H_corr − H_pred)`
matching Fortran `yelmo_timesteps.f90:381`.

**Implementation note (limitation).** The Yelmo.jl `HEUN`
implementation realises Heun via two consecutive `_step_fe!` calls
(predictor + virtual-lookahead) and averages: `H_corr = (H_n + H_**)/2`.
Stage 2 of this pattern runs the **full** `topo_step!` cascade —
calving, `resid_tendency!` cleanup, mass-balance kernels — on the
predictor's intermediate `H_pred` geometry. For the nonlinear
cascade (calving thresholds, `H_min_*` cleanup, level-set front
events) this means the corrector's inputs were biased by stage-2's
calving-on-`H_pred`, and `H_corr = (H_n + H_**)/2` no longer
reproduces the trapezoidal-rule answer for those nonlinear
contributions. The default Yelmo.jl benchmarks exercise none of
those kernels (see `test_mismip3d_stnd*.jl` etc.), so this is benign
for the current test suite.

For configurations with active LSF calving, finite `H_min_flt` /
`H_min_grnd`, or `topo_rel != 0`, consider `FE_SBE`, which uses the
"both stages start from `H_n`" pattern where the topo cascade only
sees `H_n`'s geometry.

`AB_SAM` (Fortran's native default) is identical to `HEUN` here
except its predictor uses an Adams-Bashforth
extrapolation that pulls in the previous outer step's ΔH; HEUN is
the bootstrap form (`β2 = 0` — no history term).
"""
struct HEUN <: PCScheme end

"""
    FE_SBE

Forward Euler predictor + Semi-implicit Backward Euler corrector.

Two-stage scheme structurally similar to `HEUN` (both run two FE
cycles per attempt), but with the architectural distinction that
the corrector's full topo cascade starts from `H_n`'s geometry, not
`H_pred`'s:

  - Predictor: `_step_fe!(y, dt)` from `y_n` → `(H_pred, u_pred, …)`.
  - `restore_H_only!`: roll `H_ice` and `time` back to `y_n`; keep
    velocity (= `u_pred`) and other state untouched.
  - Corrector: `_step_fe!(y, dt)` from `(H_n, u_pred, …)`. The
    embedded `topo_step!` advects `H_n` by `u_pred`; the embedded
    `dyn_step!` resolves SSA at `H_corr`.

Cost: **2 SSA solves per attempt** (same as `HEUN`). The benefit
over `HEUN` is correctness for nonlinear cascade kernels (calving
thresholds, `H_min_*` cleanup, level-set front events) — these only
ever see `H_n`'s geometry under FE-SBE.

**Performance note.** Empirically `FE_SBE` is significantly slower
wall-clock than `HEUN` or `AB_SAM` on smooth-flow benchmarks at the
same `pc_tol`. Two compounding effects: (a) the truncation-factor is
`1/2` vs `HEUN`'s `1/6` (and `AB_SAM`'s ζ/(3·(ζ+1)) ≈ 1/6), so the
controller picks ~3× smaller dt for the same `|H_corr − H_pred|`;
(b) `FE_SBE`'s `|H_corr − H_pred|` is structurally larger because
it lacks the Heun-style `k1 ≈ k2` cancellation. The package default
is `HEUN`; reach for `FE_SBE` when nonlinear cascade kernels are
active and the geometric correctness matters more than wall time.

Velocity carried forward is `u_corr` (the post-corrector SSA
solution).

Truncation-error proxy: `tau = 1/(2·dt) · (H_corr − H_pred)`
matching Fortran `yelmo_timesteps.f90:329`.
"""
struct FE_SBE <: PCScheme end

"""
    AB_SAM

Adams-Bashforth predictor + Semi-implicit Adams-Moulton corrector
(Fortran's native `pc_method = "AB-SAM"` default; Yelmo.jl's
package default since 2026-05-12 is `"HEUN"` for margin-heavy
performance reasons — see `YelmoPar.jl` comment).

Yelmo.jl-style implementation: structurally identical to `HEUN` (2×
`_step_fe!` with corrector identity `H_corr = (H_n + H_**)/2`), but
the predictor's `H_pred` is replaced by an Adams-Bashforth
extrapolation using the previous accepted outer step's full ΔH:

    ΔH_AB = (1 + ζ/2)·ΔH_FE − (ζ²/2)·ΔH_prev
    H_pred = H_n + ΔH_AB

where `ΔH_FE = H_n + dt·k_n` (the result of stage-1 `_step_fe!`),
`ΔH_prev = H_corr_{n−1} − H_n_{n−1}` (saved in `PCScratch.ΔH_prev`),
and `ζ = dt_n / dt_{n−1}`. Bootstrap (first outer step, no history):
fall back to HEUN behaviour (β2 = 0).

Truncation-error proxy is dt-history-dependent
(`yelmo_timesteps.f90:357`):

    eta = ζ / (3·(ζ+1)·dt) · |H_corr − H_pred|

**Caveat.** This mixes the *full-cascade* tendency (advection +
mass balance + calving + relaxation) with the previous step's
full-cascade ΔH, not just the advective component as Fortran does.
For configurations without active mass balance / calving the schemes
are mathematically equivalent. With those kernels active, the
results will differ — same architectural caveat that applies to
Yelmo.jl's `HEUN`. Faithful Fortran-AB-SAM equivalence requires the
deferred `topo_step!` refactor that exposes `dHidt_dyn` directly.
"""
struct AB_SAM <: PCScheme end

# Order of the underlying method (used by the controller's gain
# normalisation `k = k_m / pc_k`). All currently supported schemes
# are 2nd order.
pc_order(::PCScheme) = 2

# Truncation-error coefficient for `tau = factor · (H_corr − H_pred) / dt`.
# Matches Fortran `yelmo_timesteps.f90:329 / 357 / 381`.
pc_error_factor(::HEUN)   = 1.0 / 6.0
pc_error_factor(::FE_SBE) = 1.0 / 2.0
# AB-SAM's error factor is dt-history-dependent (`yelmo_timesteps.f90:357`):
#   tau = ζ / (3·(ζ+1)·dt) · |H_corr − H_pred|, ζ = dt_n / dt_{n-1}.
# Computed inline inside `pc_step!(::AB_SAM, …)` since the constant-factor
# signature can't carry ζ. Kept as a method so callers asking for a default
# (constant-dt) factor still get a sensible number.
pc_error_factor(::AB_SAM) = 1.0 / 6.0   # ζ → 1 limit

# Number of past `dt` / `eta` samples the scheme needs in addition to
# the current step. Heun is self-starting (no history); AB-SAM needs 1.
pc_history_required(::PCScheme) = 0
pc_history_required(::AB_SAM)   = 1


# ===== Controllers =====

abstract type PIController end

"""
    PI42

Söderlind & Wang (2006) PI-controller. Default in Fortran Yelmo.
Computes the dt-ratio `rho = dt_{n+1} / dt_n` from the truncation-
error history with proportional + integral gain.
"""
struct PI42 <: PIController end

# Resolve a string from the namelist to a concrete scheme instance.
function _resolve_pc_scheme(name::AbstractString)
    name == "HEUN"   && return HEUN()
    name == "FE-SBE" && return FE_SBE()
    name == "AB-SAM" && return AB_SAM()
    error("Unknown pc_method=\"$name\". " *
          "Supported: \"AB-SAM\" (default), \"HEUN\", \"FE-SBE\".")
end

function _resolve_pc_controller(name::AbstractString)
    name == "PI42" && return PI42()
    error("Unknown pc_controller=\"$name\". Supported: \"PI42\".")
end


# ===== Snapshot / restore =====

"""
    PCSnapshot

Prognostic state captured at the start of each PC attempt. On a
rejected attempt, calling `restore!(y, snap)` rolls the model back
to exactly this state and refreshes diagnostics via
`update_diagnostics!`.

Snapshotted fields:

  - `H_ice`  (Center) — the actual prognostic.
  - `ux_b`, `uy_b`     (XFace, YFace) — basal velocities; warm-start
                                        for the next SSA Picard solve.
  - `ux_bar`, `uy_bar` (XFace, YFace) — depth-averaged velocities;
                                        used by `advect_tracer!`.
  - `time`                              — model time.

All other fields (`f_grnd`, `H_grnd`, `z_srf`, mass-balance terms,
…) are diagnostics computed from `H_ice` + boundary inputs by
`update_diagnostics!` after restore.
"""
mutable struct PCSnapshot
    H_ice::Array{Float64,3}
    ux_b::Array{Float64,3}
    uy_b::Array{Float64,3}
    ux_bar::Array{Float64,3}
    uy_bar::Array{Float64,3}
    # Advective-PC fields: dHidt_dyn evolves across substeps (each
    # predictor saves it as dHidt_dyn_n then overwrites with the β-mix
    # result), so a rejected attempt has to restore the pre-attempt
    # value. lsf likewise. Snapshotted unconditionally so the legacy
    # path also covers them (cheap; ~Nx*Ny floats each).
    dHidt_dyn::Array{Float64,3}
    lsf::Array{Float64,3}
    time::Float64
end

function _alloc_pc_snapshot(y)
    return PCSnapshot(
        copy(interior(y.tpo.H_ice)),
        copy(interior(y.dyn.ux_b)),
        copy(interior(y.dyn.uy_b)),
        copy(interior(y.dyn.ux_bar)),
        copy(interior(y.dyn.uy_bar)),
        copy(interior(y.tpo.dHidt_dyn)),
        copy(interior(y.tpo.lsf)),
        y.time,
    )
end

"""
    snapshot!(snap::PCSnapshot, y) -> snap

Copy the current prognostic state of `y` into `snap`, in place.
"""
function snapshot!(snap::PCSnapshot, y)
    copyto!(snap.H_ice,     interior(y.tpo.H_ice))
    copyto!(snap.ux_b,      interior(y.dyn.ux_b))
    copyto!(snap.uy_b,      interior(y.dyn.uy_b))
    copyto!(snap.ux_bar,    interior(y.dyn.ux_bar))
    copyto!(snap.uy_bar,    interior(y.dyn.uy_bar))
    copyto!(snap.dHidt_dyn, interior(y.tpo.dHidt_dyn))
    copyto!(snap.lsf,       interior(y.tpo.lsf))
    snap.time = y.time
    return snap
end

"""
    restore!(y, snap::PCSnapshot) -> y

Roll the model back to the state captured by `snap`, then refresh
all diagnostics so derived fields (`f_grnd`, `H_grnd`, `z_srf`, …)
are consistent with the restored `H_ice`.
"""
function restore!(y, snap::PCSnapshot)
    copyto!(interior(y.tpo.H_ice),     snap.H_ice)
    copyto!(interior(y.dyn.ux_b),      snap.ux_b)
    copyto!(interior(y.dyn.uy_b),      snap.uy_b)
    copyto!(interior(y.dyn.ux_bar),    snap.ux_bar)
    copyto!(interior(y.dyn.uy_bar),    snap.uy_bar)
    copyto!(interior(y.tpo.dHidt_dyn), snap.dHidt_dyn)
    copyto!(interior(y.tpo.lsf),       snap.lsf)
    y.time = snap.time
    Yelmo.update_diagnostics!(y)
    return y
end

"""
    restore_H_only!(y, snap::PCSnapshot) -> y

Roll back `H_ice` and `time` to the snapshotted values, but **leave
the velocity Fields untouched**. Used by `FE_SBE` (and future
`AB_SAM`) to reset the prognostic state between predictor and
corrector stages while preserving the freshly-solved post-predictor
velocity `u_pred` for use as the corrector's advection velocity.

Calls `update_diagnostics!` to refresh `f_grnd`, `H_grnd`, `z_srf`,
mask fields, etc. from the restored `H_ice`.
"""
function restore_H_only!(y, snap::PCSnapshot)
    copyto!(interior(y.tpo.H_ice), snap.H_ice)
    y.time = snap.time
    Yelmo.update_diagnostics!(y)
    return y
end


# ===== Per-model persistent scratch =====

"""
    PCScratch

Per-model storage for the adaptive driver: the rollback snapshot,
predictor / corrector buffers for `H_ice`, and short error / dt
history rings used by the PI controller. Allocated lazily on the
first adaptive call; cached in `y.dyn.scratch.pc_scratch[]`.
"""
mutable struct PCScratch
    snap::PCSnapshot
    H_pred::Array{Float64,3}
    H_corr::Array{Float64,3}
    # Accepted-step ΔH = H_corr − H_n from the previous outer accepted
    # step. Used by AB-SAM's predictor for the Adams-Bashforth
    # extrapolation; ignored by HEUN / FE-SBE. `is_bootstrapped` is
    # `false` until the first accept; AB-SAM falls back to HEUN-style
    # predictor while `false`.
    ΔH_prev::Array{Float64,3}
    # Predictor-state diagnostics, snapshotted by
    # `_snapshot_pred_diagnostics!` immediately after each scheme
    # records `H_pred`. Consumed by `_compute_pc_eta` to evaluate the
    # Fortran `set_pc_mask` exclusions on the predictor side. The
    # corrector-side diagnostics are read live from `y.tpo.f_ice` /
    # `y.tpo.H_grnd` (refreshed by `update_diagnostics!` at the end of
    # the corrector stage).
    f_ice_pred::Array{Float64,3}
    H_grnd_pred::Array{Float64,3}
    # Per-cell `|tau|` field, populated only when `pc_eta_masked = true`.
    # Used by the isolated-outlier check inside `_compute_pc_eta`.
    pc_tau::Array{Float64,3}
    # Advective-PC buffers (`pc_advective = true`). Predictor / corrector
    # stage outputs (Fortran `tpo%now%pred` / `tpo%now%corr`) and the
    # `H_scratch` buffer used by the snapshot-diff advection-tendency
    # helper. Lazily allocated; unused under `pc_advective = false`
    # except for the (small) struct overhead.
    pred_buf::YelmoModelTopo.PCStageBuf
    corr_buf::YelmoModelTopo.PCStageBuf
    H_scratch::Array{Float64,3}
    is_bootstrapped::Bool
    eta_history::Vector{Float64}   # newest at end; capped to history_max
    dt_history::Vector{Float64}    # likewise
    n_steps_taken::Int
    n_rejections::Int
end

const PC_HISTORY_MAX = 3

function _alloc_pc_scratch(y)
    H = interior(y.tpo.H_ice)
    return PCScratch(
        _alloc_pc_snapshot(y),
        zeros(Float64, size(H)),    # H_pred
        zeros(Float64, size(H)),    # H_corr
        zeros(Float64, size(H)),    # ΔH_prev
        zeros(Float64, size(H)),    # f_ice_pred
        zeros(Float64, size(H)),    # H_grnd_pred
        zeros(Float64, size(H)),    # pc_tau
        YelmoModelTopo._alloc_pc_stage_buf(y),  # pred_buf
        YelmoModelTopo._alloc_pc_stage_buf(y),  # corr_buf
        zeros(Float64, size(H)),       # H_scratch (advection snapshot)
        false,
        Float64[],
        Float64[],
        0, 0,
    )
end

# Snapshot the predictor-state ice-fraction and flotation diagnostics
# from `y.tpo` into the PC scratch. Called by every `pc_step!` method
# at the moment when `y` holds the predictor state (i.e., immediately
# after `copyto!(scratch.H_pred, ...)`). The snapshots are consumed by
# `_compute_pc_eta` when evaluating the Fortran-style `set_pc_mask`
# exclusions on the predictor side; the corrector-side equivalents are
# read live from `y.tpo` after `update_diagnostics!` runs at the end
# of the corrector stage.
#
# Cheap copies of two `Nx × Ny` slices — kept unconditionally (i.e.
# regardless of the `pc_eta_masked` setting) so toggling masking on
# at runtime never sees stale buffers.
function _snapshot_pred_diagnostics!(scratch::PCScratch, y)
    copyto!(scratch.f_ice_pred,  interior(y.tpo.f_ice))
    copyto!(scratch.H_grnd_pred, interior(y.tpo.H_grnd))
    return scratch
end

# ----------------------------------------------------------------------
# Truncation-error proxy `eta` from a finished PC attempt.
#
# `factor` is the scheme-specific multiplier on `|H_corr − H_pred| / dt`
# (1/6 for HEUN, 1/2 for FE-SBE, ζ/(3·(ζ+1)) for AB-SAM — the caller
# supplies it).
#
# `y.p.yelmo.pc_eta_masked` selects between two evaluations:
#
#   - `false` (legacy / unmasked): the original Yelmo.jl behaviour —
#     a global `max(|H_corr − H_pred|)` across every cell. Margin /
#     calving-front / grounding-line cells dominate the result.
#
#   - `true` (default, Fortran-equivalent): port of Fortran's
#     `set_pc_mask` (yelmo_timesteps.f90:166) + `calc_pc_eta`
#     (yelmo_timesteps.f90:275). The maxval is taken only over cells
#     that pass:
#
#       * H_pred ≥ 10 m AND H_corr ≥ 10 m (skip thin / transitioning).
#       * No 9-neighbour `f_ice < 1` in either pred or corr (skip the
#         ice-margin band).
#       * H_grnd_pred > 0 AND H_grnd_corr > 0 (skip floating + GL).
#       * Not an isolated outlier — `|tau| > 2·pc_eps` with no
#         4-neighbour `|tau| > pc_eps` is also masked out.
#
#     Predictor-side `f_ice` / `H_grnd` come from `scratch.f_ice_pred`
#     / `scratch.H_grnd_pred` (snapshotted by each scheme right after
#     it records `H_pred`). Corrector-side comes live from `y.tpo`.
# ----------------------------------------------------------------------
function _compute_pc_eta(factor::Float64, scratch::PCScratch, y, dt::Float64)
    H_pred = scratch.H_pred
    H_corr = scratch.H_corr

    if !y.p.yelmo.pc_eta_masked
        diff_max = 0.0
        @inbounds @simd for i in eachindex(H_corr)
            d = abs(H_corr[i] - H_pred[i])
            diff_max = max(diff_max, d)
        end
        return factor * diff_max / dt
    end

    # Build the per-cell |tau| field once so the isolated-outlier
    # check below has neighbour values to compare against.
    pc_tau = scratch.pc_tau
    inv_dt_factor = factor / dt
    @inbounds @simd for i in eachindex(pc_tau)
        pc_tau[i] = abs(H_corr[i] - H_pred[i]) * inv_dt_factor
    end

    pc_eps     = Float64(y.p.yelmo.pc_eps)
    fice_pred  = scratch.f_ice_pred
    Hgrnd_pred = scratch.H_grnd_pred
    fice_corr  = interior(y.tpo.f_ice)
    Hgrnd_corr = interior(y.tpo.H_grnd)

    H_lim = 10.0    # [m] — Fortran `set_pc_mask` constant.
    nx = size(pc_tau, 1)
    ny = size(pc_tau, 2)

    eta_max = 0.0
    @inbounds for j in 1:ny, i in 1:nx
        # Thin / transitioning cell.
        if H_pred[i, j, 1] < H_lim || H_corr[i, j, 1] < H_lim
            continue
        end

        im1 = max(i - 1, 1)
        ip1 = min(i + 1, nx)
        jm1 = max(j - 1, 1)
        jp1 = min(j + 1, ny)

        # 9-neighbour margin check (any partial-ice neighbour disqualifies).
        is_margin = false
        for jj in jm1:jp1, ii in im1:ip1
            if fice_pred[ii, jj, 1] < 1.0 || fice_corr[ii, jj, 1] < 1.0
                is_margin = true
                break
            end
        end
        is_margin && continue

        # Floating / grounding-line cell.
        if Hgrnd_pred[i, j, 1] <= 0.0 || Hgrnd_corr[i, j, 1] <= 0.0
            continue
        end

        # Isolated outlier: large local |tau| with no 4-neighbour above
        # `pc_eps`. Fortran applies this regardless of the geometric
        # exclusions above.
        tau_ij = pc_tau[i, j, 1]
        if tau_ij > 2.0 * pc_eps
            n_above = 0
            pc_tau[im1, j,   1] > pc_eps && (n_above += 1)
            pc_tau[ip1, j,   1] > pc_eps && (n_above += 1)
            pc_tau[i,   jm1, 1] > pc_eps && (n_above += 1)
            pc_tau[i,   jp1, 1] > pc_eps && (n_above += 1)
            n_above == 0 && continue
        end

        if tau_ij > eta_max
            eta_max = tau_ij
        end
    end

    return eta_max
end


# Push (eta, dt) onto the history rings, trimming to PC_HISTORY_MAX.
function _push_history!(scratch::PCScratch, eta::Float64, dt::Float64)
    push!(scratch.eta_history, eta)
    push!(scratch.dt_history,  dt)
    while length(scratch.eta_history) > PC_HISTORY_MAX
        popfirst!(scratch.eta_history)
    end
    while length(scratch.dt_history) > PC_HISTORY_MAX
        popfirst!(scratch.dt_history)
    end
    return scratch
end


# ===== PC step (per scheme) =====

# Plain forward-Euler step: one (topo, dyn, mat, thrm) chain. Used by
# Heun as its stage primitive and by the `dt_method = 0` branch of the
# dispatcher. Phase order matches the Fortran per-step loop
# (`yelmo_ice.f90:268-286`): `calc_ydyn` reads the previous step's
# `mat.ATT`, then `calc_ymat` computes a fresh `ATT` from the
# just-solved velocity field for the next step, then `calc_ytherm`
# updates ice/bed temperatures using the just-solved velocity and
# stress fields.
function _step_fe!(y, dt::Float64)
    @timed_section y :topo Yelmo.topo_step!(y, dt)
    @timed_section y :dyn  Yelmo.dyn_step!(y, dt)
    @timed_section y :mat  Yelmo.mat_step!(y, dt)
    @timed_section y :thrm Yelmo.therm_step!(y, dt)
    return y
end

"""
    pc_step!(scheme, y, dt, scratch) -> eta::Float64

Run one predictor-corrector attempt: assumes the caller has already
captured `y_n` into `scratch.snap`, and that `y` currently holds that
same `y_n` state. Returns the truncation-error proxy `eta` (m/yr).
The model state on return is the corrector result.

Dispatches on `PCScheme`. Concrete methods: `HEUN`, `FE_SBE`, `AB_SAM`.

Branches internally on `y.p.yelmo.pc_advective`:

  - `true` (default, Fortran-equivalent) — runs Fortran's advective
    predictor / corrector via `topo_pc_step!`. ONE `dyn_step!` per
    attempt. β-mixing applies to the advective tendency only; the full
    MB cascade runs in both the predictor and corrector blocks.
  - `false` (legacy) — two full `_step_fe!` cascades per attempt,
    mixing the full-cascade tendency. Preserved for A/B comparison;
    see memory `pc_refactor_design.md`.
"""
function pc_step!(scheme::PCScheme, y, dt::Float64, scratch::PCScratch)
    if y.p !== nothing && y.p.yelmo.pc_advective
        return _pc_step_advective!(scheme, y, dt, scratch)
    end
    return _pc_step_legacy!(scheme, y, dt, scratch)
end


# ===== Advective PC (Fortran-equivalent, `pc_advective = true`) =====

# β-coefficients for the advective predictor/corrector (Fortran
# `set_pc_beta_coefficients`, yelmo_timesteps.f90:38-163).
# Returns a 4-tuple `(β1, β2, β3, β4)`:
#   predictor: dHidt_dyn = β1·dHidt_now + β2·dHidt_dyn_n
#   corrector: dHidt_dyn = β3·dHidt_now + β4·dHidt_dyn_n
_pc_beta(::HEUN,   ::PCScratch, ::Float64) = (1.0, 0.0, 0.5, 0.5)
_pc_beta(::FE_SBE, ::PCScratch, ::Float64) = (1.0, 0.0, 1.0, 0.0)
function _pc_beta(::AB_SAM, scratch::PCScratch, dt::Float64)
    # AB-SAM bootstrap: no previous-step dHidt_dyn_n history → fall back
    # to HEUN-style predictor (β2 = 0). After the first accepted substep
    # the dt_history is populated and ζ is well-defined.
    if !scratch.is_bootstrapped || isempty(scratch.dt_history)
        return (1.0, 0.0, 0.5, 0.5)
    end
    dt_prev = scratch.dt_history[end]
    ζ = dt / dt_prev
    return (1.0 + 0.5 * ζ, -0.5 * ζ, 0.5, 0.5)
end

# Truncation-error multiplier for `tau = factor · |H_corr − H_pred| / dt`
# (Fortran yelmo_timesteps.f90:329 / 357 / 381).
_pc_factor(::HEUN,   ::PCScratch, ::Float64) = 1.0 / 6.0
_pc_factor(::FE_SBE, ::PCScratch, ::Float64) = 1.0 / 2.0
function _pc_factor(::AB_SAM, scratch::PCScratch, dt::Float64)
    if !scratch.is_bootstrapped || isempty(scratch.dt_history)
        return 1.0 / 6.0
    end
    dt_prev = scratch.dt_history[end]
    ζ = dt / dt_prev
    return ζ / (3.0 * (ζ + 1.0))
end

# Single advective PC attempt for all three schemes. On exit:
#   - `y.tpo.H_ice` holds H_corr with diagnostics consistent (the PC
#     driver subsequently restores to `H_ice_n` for mat/therm and runs
#     `topo_pc_step!(:advance)` to commit the chosen stage).
#   - `scratch.pred_buf` / `scratch.corr_buf` populated.
#   - `scratch.H_pred` / `scratch.H_corr` mirror the buffer thicknesses
#     so `_compute_pc_eta` can read them.
function _pc_step_advective!(scheme::PCScheme, y, dt::Float64, scratch::PCScratch)
    (β1, β2, β3, β4) = _pc_beta(scheme, scratch, dt)

    @timed_section y :pc_predictor begin
        Yelmo.topo_pc_step!(y, dt;
                            mode = :predictor,
                            β1 = β1, β2 = β2,
                            pred_buf  = scratch.pred_buf,
                            corr_buf  = scratch.corr_buf,
                            H_scratch = scratch.H_scratch)
    end
    copyto!(scratch.H_pred, scratch.pred_buf.H_ice)
    # Snapshot predictor-state f_ice / H_grnd for the masked-eta path,
    # while live `y.tpo` still reflects H_pred.
    _snapshot_pred_diagnostics!(scratch, y)

    # ONE dyn solve at H_pred → u_pred.
    @timed_section y :dyn Yelmo.dyn_step!(y, dt)

    @timed_section y :pc_corrector begin
        Yelmo.topo_pc_step!(y, dt;
                            mode = :corrector,
                            β3 = β3, β4 = β4,
                            pred_buf  = scratch.pred_buf,
                            corr_buf  = scratch.corr_buf,
                            H_scratch = scratch.H_scratch)
    end
    copyto!(scratch.H_corr, scratch.corr_buf.H_ice)

    # Eta-masking reads corrector-state f_ice / H_grnd live from `y.tpo`
    # (set by topo_pc_step!(:corrector)'s closing update_diagnostics).
    factor = _pc_factor(scheme, scratch, dt)
    return _compute_pc_eta(factor, scratch, y, dt)
end


# ===== Legacy PC (full-cascade twice, `pc_advective = false`) =====

function _pc_step_legacy!(::HEUN, y, dt::Float64, scratch::PCScratch)
    snap = scratch.snap

    # Stage 1: full FE step  y_n  →  y_pred
    @timed_section y :pc_predictor _step_fe!(y, dt)
    copyto!(scratch.H_pred, interior(y.tpo.H_ice))
    # Snapshot predictor-state f_ice / H_grnd before stage 2 overwrites
    # them. Consumed by `_compute_pc_eta` for the masked-eta path.
    _snapshot_pred_diagnostics!(scratch, y)

    # Stage 2: full FE step  y_pred  →  y_**
    @timed_section y :pc_corrector _step_fe!(y, dt)

    # Heun corrector identity: H_corr = (H_n + H_**) / 2.
    # (Derivation: with k1 = (H_pred − H_n)/dt and k2 = (H_** − H_pred)/dt,
    # the Heun update H_n + dt/2·(k1 + k2) = (H_n + H_**)/2.)
    H_now = interior(y.tpo.H_ice)
    @inbounds @simd for i in eachindex(scratch.H_corr)
        scratch.H_corr[i] = 0.5 * (snap.H_ice[i] + H_now[i])
    end

    # Set state to corrector. Time advances by dt (not 2·dt) — the
    # second FE stage was a virtual lookahead, not a real step.
    copyto!(H_now, scratch.H_corr)

    # Re-apply the mask_ice pass to the corrected H_ice. The Heun average
    # H_corr = (H_n + H_**)/2 can violate mask invariants: if a
    # MASK_ICE_NONE cell had H_n > 0 before the step, the average gives
    # H_corr > 0 even though H_** = 0 (both FE stages apply the mask).
    # MASK_ICE_FIXED cells are restored to bnd.H_ice_ref, matching
    # Fortran's mask-pass-at-end-of-PC-corrector convention.
    apply_mask_ice_pass!(y)

    y.time = snap.time + dt
    Yelmo.update_diagnostics!(y)

    # Truncation-error proxy. Velocity carried forward into the next
    # outer step is `u_**` (from stage 2's dyn solve), matching
    # Fortran's "carry the predictor-state velocity" behaviour
    # (yelmo_ice.f90 outer loop — never re-solves dyn at the corrector
    # state). The Picard loop on the next step's predictor will tug
    # the velocity back to consistency with the new H_n. The maxval
    # is masked when `pc_eta_masked = true` (default) to mirror
    # Fortran's `set_pc_mask` filter on margin / GL / floating /
    # thin-ice / isolated-outlier cells.
    return _compute_pc_eta(pc_error_factor(HEUN()), scratch, y, dt)
end

"""
    pc_step!(::FE_SBE, y, dt, scratch) -> eta::Float64

FE-SBE predictor-corrector — both topo cascades operate on `H_n`'s
geometry; only the advection velocity differs.

Sequence:

  1. **Predictor**: `_step_fe!(y, dt)` — full FE cycle (topo + dyn +
     mat + therm) from `y_n`. Result: `(H_pred, u_pred, mat_pred,
     therm_pred)`. Save `H_pred`.
  2. **Restore H, keep state**: `restore_H_only!(y, snap)` rolls
     `H_ice` and `time` back to `y_n`, but leaves the velocity, mat,
     and therm fields untouched (so the corrector advects with
     `u_pred` and uses `mat_pred` for SSA viscosity).
  3. **Corrector**: `_step_fe!(y, dt)` — full FE cycle from
     `(H_n, u_pred, mat_pred, …)`. Inside, `topo_step!` advects
     `H_n` by `u_pred`, then `dyn_step!` resolves SSA at `H_corr`
     giving `u_corr`, then mat/therm refresh.

Cost: **2 SSA solves per attempt** (same as `HEUN`). The
architectural distinction vs `HEUN` is that the corrector's full
topo cascade — calving, `resid_tendency!`, mass-balance kernels —
operates on `H_n`'s geometry, not `H_pred`'s. This matters for
configurations with active LSF calving, finite `H_min_*`, or
`topo_rel != 0`.

Velocity carried forward to the next outer step is `u_corr` (the
post-corrector SSA solution at `H_corr`).

Truncation-error proxy: `tau = 1/(2·dt) · |H_corr − H_pred|`,
matching Fortran `yelmo_timesteps.f90:329`.
"""
function _pc_step_legacy!(::FE_SBE, y, dt::Float64, scratch::PCScratch)
    snap = scratch.snap

    # Stage 1 — predictor: full FE cycle from y_n.
    # y is now at (H_pred, u_pred, mat_pred, therm_pred).
    @timed_section y :pc_predictor _step_fe!(y, dt)
    copyto!(scratch.H_pred, interior(y.tpo.H_ice))
    # Snapshot predictor-state f_ice / H_grnd before `restore_H_only!`
    # overwrites them via its embedded `update_diagnostics!`. Consumed
    # by `_compute_pc_eta` for the masked-eta path.
    _snapshot_pred_diagnostics!(scratch, y)

    # Stage 2 — restore H_ice + time to y_n; keep u_pred (and
    # mat_pred, therm_pred) in place. `update_diagnostics!` (called
    # inside `restore_H_only!`) refreshes f_grnd, H_grnd, z_srf, mask
    # fields, etc. from H_n.
    restore_H_only!(y, snap)

    # Stage 3 — corrector: full FE cycle from (H_n, u_pred, …).
    # `topo_step!` reads `y.dyn.ux_bar` / `y.dyn.uy_bar` by default
    # (which now hold `u_pred`), so it advects `H_n` by `u_pred`.
    # The full cascade (calving, mass balance, resid_tendency!, …)
    # sees H_n's geometry — the key architectural difference vs
    # HEUN's stage-2-on-H_pred pattern. dyn_step! then resolves SSA
    # at H_corr; mat/therm refresh once more.
    @timed_section y :pc_corrector _step_fe!(y, dt)
    # y is now at (H_corr, u_corr, ...) with time = t_n + dt.

    # FE_SBE leaves `y.tpo.H_ice` at the corrector geometry but
    # `scratch.H_corr` was last touched in a previous attempt — copy
    # so `_compute_pc_eta` reads `H_corr` from the canonical place.
    copyto!(scratch.H_corr, interior(y.tpo.H_ice))

    # Truncation-error proxy. Masked maxval when
    # `pc_eta_masked = true` (default), matching Fortran's
    # `set_pc_mask` exclusions.
    return _compute_pc_eta(pc_error_factor(FE_SBE()), scratch, y, dt)
end

"""
    pc_step!(::AB_SAM, y, dt, scratch) -> eta::Float64

Adams-Bashforth predictor + Semi-implicit Adams-Moulton corrector,
implemented in the Yelmo.jl HEUN style (full-cascade tendency,
Heun corrector identity). See `AB_SAM` docstring for the caveat.

Bootstrap: when `!scratch.is_bootstrapped` (no previous accepted
outer step yet), fall back to a HEUN step. After the first accept
the bootstrap flag is flipped by `_adaptive_step!`.
"""
function _pc_step_legacy!(::AB_SAM, y, dt::Float64, scratch::PCScratch)
    # Bootstrap path: no previous accepted ΔH yet → run HEUN.
    if !scratch.is_bootstrapped || isempty(scratch.dt_history)
        return _pc_step_legacy!(HEUN(), y, dt, scratch)
    end

    snap = scratch.snap
    dt_prev = scratch.dt_history[end]
    ζ = dt / dt_prev
    half_ζ  = 0.5 * ζ
    half_ζ2 = 0.5 * ζ^2

    # Stage 1a: full FE step y_n → y_pred_FE.
    @timed_section y :pc_predictor _step_fe!(y, dt)

    # Stage 1b: replace H_pred_FE with the AB-extrapolated predictor.
    # Algebra: ΔH_AB = (1 + ζ/2)·ΔH_FE − (ζ²/2)·ΔH_prev,
    # where ΔH_FE = H_pred_FE − H_n, ΔH_prev = H_corr_{n-1} − H_n_{n-1}.
    H_now = interior(y.tpo.H_ice)
    @inbounds @simd for i in eachindex(H_now)
        ΔH_FE = H_now[i] - snap.H_ice[i]
        H_now[i] = snap.H_ice[i] +
                   (1.0 + half_ζ) * ΔH_FE -
                   half_ζ2 * scratch.ΔH_prev[i]
    end
    # Diagnostics (f_grnd, z_srf, …) computed by stage-1 `_step_fe!`
    # reflect H_pred_FE; refresh from the AB-extrapolated H_pred.
    Yelmo.update_diagnostics!(y)
    copyto!(scratch.H_pred, H_now)
    # Snapshot AB-predictor-state f_ice / H_grnd (post `update_diagnostics!`)
    # before stage 2 overwrites them. Consumed by `_compute_pc_eta`.
    _snapshot_pred_diagnostics!(scratch, y)

    # Stage 2: full FE step y_pred_AB → y_** (k_** at AB predictor state).
    @timed_section y :pc_corrector _step_fe!(y, dt)

    # Heun corrector identity (Yelmo.jl-style; matches Fortran AB-SAM's
    # corrector β3=β4=0.5 only in the linear-cascade case).
    H_now = interior(y.tpo.H_ice)
    @inbounds @simd for i in eachindex(scratch.H_corr)
        scratch.H_corr[i] = 0.5 * (snap.H_ice[i] + H_now[i])
    end
    copyto!(H_now, scratch.H_corr)
    apply_mask_ice_pass!(y)
    y.time = snap.time + dt
    Yelmo.update_diagnostics!(y)

    # Truncation-error proxy: factor = ζ / (3·(ζ+1)). Masked maxval
    # when `pc_eta_masked = true` (default), matching Fortran's
    # `set_pc_mask` exclusions.
    factor = ζ / (3.0 * (ζ + 1.0))
    return _compute_pc_eta(factor, scratch, y, dt)
end


# ===== Step controller =====

"""
    _dt_ratio(controller, eta_n, eta_nm1, dt_n, dt_nm1, eps, k_method)
        -> rho

Compute the multiplicative dt-ratio `rho = dt_{n+1} / dt_n` from
the truncation-error history. `eps` is the eta-floor (`pc_eps`);
`k_method` is the PC scheme order (`pc_order(scheme)`).

PI42 from Söderlind & Wang (2006) Eq. 3.12 (also Fortran
`yelmo_timesteps.f90` PI42 case). Gains:

  - `k_i = 2 / (k_method · 5)` — integral
  - `k_p = 1 / (k_method · 5)` — proportional

Formula:

    rho = (eps / eta_n)^(k_i + k_p) · (eps / eta_nm1)^(-k_p)

With no history (`eta_nm1 ≤ 0`), the proportional term is dropped
(falls back to integral-only ratio).
"""
function _dt_ratio(::PI42, eta_n::Real, eta_nm1::Real,
                   dt_n::Real, dt_nm1::Real,
                   eps::Real, k_method::Int)
    k_i = 2.0 / (k_method * 5.0)
    k_p = 1.0 / (k_method * 5.0)
    # Floor eta at a tiny constant (Fortran `eta_tol = 1e-8`,
    # yelmo_timesteps.f90:294) to avoid `(eps/0)^x` blow-ups.
    # IMPORTANT: do NOT floor at `eps` itself — that collapses
    # `(eps/eta)` to 1 when actual eta < eps, freezing dt at
    # whatever it was after the first transient.
    eta_floor = 1.0e-8
    eta_n_safe   = max(eta_n,   eta_floor)
    eta_nm1_safe = max(eta_nm1, eta_floor)
    if eta_nm1 <= 0
        # First step: integral term only.
        return (eps / eta_n_safe) ^ (k_i + k_p)
    end
    return (eps / eta_n_safe) ^ (k_i + k_p) *
           (eps / eta_nm1_safe) ^ (-k_p)
end

# Per-step clamp on the dt growth/shrink ratio. The Söderlind & Wang
# (2006) recommendation is [0.5, 2.0], but Fortran Yelmo applies no
# such clamp after the PI42 controller (`yelmo_timesteps.f90:506` —
# `rhohat_n = rho_n` directly, the smoothing is commented out). We use
# a wider [0.2, 10.0] bracket so dt can recover from transient
# rejections faster without being capped at 2× per step. This keeps
# Yelmo.jl's response curve closer to Fortran's while still limiting
# pathological per-step swings.
_clamp_dt_ratio(rho) = clamp(rho, 0.2, 10.0)


# ===== Outer-step boundary limiter =====

# Avoid a lopsided (1 big + 1 tiny) sub-step pair at the end of an
# outer interval. Mirrors Fortran `limit_adaptive_timestep`
# (yelmo_timesteps.f90:735): if the controller's `dt_now` is more than
# half of `remaining` but less than `remaining`, halve `remaining` so
# we'll take exactly two equal sub-steps. Otherwise leave `dt_now`
# alone (clamped to `remaining`).
#
# Examples (`remaining = 1.0`):
#   dt_now = 0.95 → 0.5     (halved: > 0.5 of remaining)
#   dt_now = 0.5  → 0.5     (unchanged)
#   dt_now = 0.4  → 0.4     (unchanged: ≤ 0.5 of remaining)
#   dt_now = 1.0  → 1.0     (unchanged: equals remaining)
#   dt_now = 1.5  → 1.0     (clamped to remaining)
#
# This is asymmetric — only large fractions of `remaining` get cut.
# An earlier port used equal-N-tiling
# (`dt = remaining / ceil(remaining/dt_now)`) which was strictly more
# aggressive than Fortran and caused a ratchet-down when combined with
# seeding `dt_now` from `dt_history[end]` across outer calls.
function _limit_step(dt_now::Float64, remaining::Float64)
    remaining > 0 || return 0.0
    dt = min(dt_now, remaining)
    if dt / remaining > 0.5 && dt < remaining
        return 0.5 * remaining
    end
    return dt
end


# ===== Outer adaptive driver =====

"""
    _adaptive_step!(y, dt_outer, scheme, controller, scratch, p_yelmo) -> y

Advance `y` by exactly `dt_outer` years using the adaptive PC
machinery. Internally the driver may take many sub-steps with
controller-chosen `dt`, and may reject and retry attempts up to
`pc_n_redo` times when the truncation error exceeds `pc_tol`.

The controller (`_dt_ratio`) is recomputed fresh at the top of every
sub-step from the persistent `eta_history` / `dt_history` rings,
mirroring Fortran's `set_adaptive_timestep_pc` call at the head of
the inner `do n = 1, nstep` loop in `yelmo_ice.f90`. On the very
first call ever (empty history) we seed conservatively at
`min(dt_outer, 1.0)`.

Before each sub-step `_limit_step` clamps the controller's `dt_now`
to avoid a lopsided big-then-tiny pair at the outer-step boundary.
"""
function _adaptive_step!(y, dt_outer::Float64,
                         scheme::PCScheme, controller::PIController,
                         scratch::PCScratch, p_yelmo)
    target_time = y.time + dt_outer
    pc_tol    = Float64(p_yelmo.pc_tol)
    pc_eps    = Float64(p_yelmo.pc_eps)
    pc_n_redo = Int(p_yelmo.pc_n_redo)
    dt_min    = Float64(p_yelmo.dt_min)
    dt_ceil   = dt_outer    # never exceed the requested outer step

    while y.time < target_time - 1e-9
        remaining = target_time - y.time

        # Run the controller fresh from history at the top of every
        # sub-step. Empty history (first call ever) → conservative seed.
        dt_now = if isempty(scratch.dt_history)
            min(dt_ceil, 1.0)
        else
            eta_n   = scratch.eta_history[end]
            eta_nm1 = length(scratch.eta_history) >= 2 ? scratch.eta_history[end-1] : 0.0
            dt_n    = scratch.dt_history[end]
            dt_nm1  = length(scratch.dt_history)  >= 2 ? scratch.dt_history[end-1]  : 0.0
            rho = _dt_ratio(controller, eta_n, eta_nm1, dt_n, dt_nm1,
                            pc_eps, pc_order(scheme))
            clamp(dt_n * _clamp_dt_ratio(rho), dt_min, dt_ceil)
        end

        # Clip to `remaining` so the outer step lands exactly on
        # `target_time`. The `dt_min` floor is honoured only while we
        # still have at least `dt_min` of remaining; once `remaining`
        # falls below `dt_min` (the final fragment of the outer step),
        # we take the small remainder rather than overshoot. Without
        # this, output cadence drifts by up to `dt_min` per outer call
        # — visible as restart times like 10.08 yr instead of 10.0 yr.
        dt_attempt = min(max(_limit_step(dt_now, remaining), dt_min),
                         remaining)
        eta = NaN
        accepted_iter = 0
        wallclock_s = 0.0

        accepted = false
        for iter_redo in 1:pc_n_redo
            snapshot!(scratch.snap, y)
            t0 = time()
            eta = pc_step!(scheme, y, dt_attempt, scratch)
            wallclock_s += time() - t0

            # Accept if error within tol, or last redo, or hit dt_min.
            if eta <= pc_tol || iter_redo == pc_n_redo || dt_attempt <= dt_min
                accepted = true
                accepted_iter = iter_redo
                break
            end

            # Reject: roll back and shrink dt. Fortran reduction
            # factor (yelmo_ice.f90:361):
            #   rho = 0.7 · (1 + (eta − tol)/10)^(-1)
            scratch.n_rejections += 1
            restore!(y, scratch.snap)
            rho_reject = 0.7 / (1.0 + (eta - pc_tol) / 10.0)
            dt_attempt = max(dt_attempt * rho_reject, dt_min)
        end

        scratch.n_steps_taken += 1
        # Save the accepted-step ΔH for legacy AB-SAM's predictor (and
        # flip the bootstrap flag). At this point `scratch.snap.H_ice`
        # is H_n for *this* attempt and `y.tpo.H_ice` is H_corr — exactly
        # what legacy AB-SAM's next predictor needs as `ΔH_prev`. Cheap
        # copy whether or not the active scheme uses it.
        let H_corr = interior(y.tpo.H_ice)
            @inbounds @simd for i in eachindex(scratch.ΔH_prev)
                scratch.ΔH_prev[i] = H_corr[i] - scratch.snap.H_ice[i]
            end
        end
        scratch.is_bootstrapped = true

        # Finalise the accepted substep. The legacy path leaves
        # `y.tpo.H_ice` at H_corr with mat/therm already evaluated
        # (inside the second `_step_fe!`), so nothing more to do here.
        # The advective path leaves `y` at H_corr with mat/therm NOT
        # yet evaluated; finalise per Fortran's outer loop.
        _finalize_accepted_step!(y, dt_attempt, scratch, p_yelmo)

        # Store actual eta (NOT floored to pc_eps) so the controller
        # sees the true history. `_dt_ratio` does its own floor at
        # 1e-8 to keep the divisor finite.
        _push_history!(scratch, eta, dt_attempt)
        _maybe_log_timestep!(y, dt_attempt, eta, accepted_iter, wallclock_s)
    end

    return y
end

# Finalise an accepted advective substep: restore live `H_ice ← H_ice_n`
# (Fortran's `update_others_pc = false` evaluates mat/therm at the
# start-of-substep geometry, yelmo_ice.f90:382-389), run mat + therm,
# then `topo_pc_step!(:advance)` to commit `pred_buf` (or `corr_buf`).
# No-op for the legacy path, where `pc_step!` already advanced state to
# the corrector result and ran mat/therm inside the second `_step_fe!`.
function _finalize_accepted_step!(y, dt::Float64, scratch::PCScratch, p_yelmo)
    p_yelmo.pc_advective || return y

    # Restore H_ice ← H_ice_n for the mat/therm steps that follow.
    # Fortran's outer loop runs `calc_ymat` / `calc_ytherm` with
    # `H_ice = H_ice_n` when `update_others_pc = false` (the default).
    copyto!(interior(y.tpo.H_ice), interior(y.tpo.H_ice_n))
    copyto!(interior(y.tpo.lsf),   interior(y.tpo.lsf_n))
    Yelmo.update_diagnostics!(y)

    @timed_section y :mat  Yelmo.mat_step!(y, dt)
    @timed_section y :thrm Yelmo.therm_step!(y, dt)

    # Commit pred or corr stage outputs to live state.
    @timed_section y :pc_advance begin
        Yelmo.topo_pc_step!(y, dt;
                            mode = :advance,
                            pred_buf  = scratch.pred_buf,
                            corr_buf  = scratch.corr_buf,
                            H_scratch = scratch.H_scratch,
                            use_H_pred = p_yelmo.pc_use_H_pred,
                            advance_time = false)
    end
    # `topo_pc_step!(:advance)` keeps `advance_time = false` here; the
    # caller (`_adaptive_step!`) owns `y.time`. Bump it now.
    y.time += dt
    return y
end


# ===== `dt_method = 3` (frozen-velocity sub-cycling) — DEFERRED =====
#
# Originally planned as: 1 dyn solve at start-of-step → adaptive topo+MB
# sub-cycling with Richardson extrapolation (PI42 controller) → 1 mat at
# end. Operator-splitting model: amortise the expensive dyn solve over
# the outer dt while letting the cheap topo+MB step adapt freely.
#
# Status: implementation tried and reverted. The dyn-first ordering
# triggers a positive-feedback runaway on EISMINT-1 moving with
# dt_outer = 100 yr:
#
#   step 51 (t=5100):  H[:, 16] still smooth, max uxy ≈ 50 m/yr
#   step 52 (t=5200):  H profile becomes non-monotonic at i=9
#                       (1535 → 1935 → 1943 along the radial),
#                       SIA velocity at the kink jumps from -19 → -123
#   step 53 (t=5300):  amplification: max uxy → 320 m/yr,
#                       max H → 5366 m (vs steady state ≈ 3000)
#   step 54+:          full runaway, max H > 9000.
#
# Root cause sketch: `dyn_step!` at outer step k+1 reads `y.tpo.dzsdx`
# (and other surface-slope diagnostics) computed at the END of step k's
# last topo. The frozen-velocity sub-cycle of step k advances H by
# 100 yr with the velocity solved at H_n, leaving H slightly mis-aligned
# with the velocity that produced it. Step k+1's dyn solve at this
# mis-aligned state amplifies the inconsistency, the SIA `u ∝ H_face^5`
# scaling propagates the kink to neighbouring cells, and the next sub-
# cycle deposits even more ice unevenly. Cascade.
#
# Possible future directions:
#   - **Velocity-PC variant**: re-solve dyn after the sub-cycle and
#     average with the start-of-step velocity. This is essentially Heun
#     on the velocity-topo coupling and adds 1 extra dyn solve per
#     outer step. Restores stability at the cost of ~1.5× cost vs the
#     original "1 dyn per outer" target.
#   - **Smaller `dt_outer`**: at `dt_outer ≤ 10 yr` the frozen-velocity
#     assumption holds tighter and the runaway likely doesn't fire. But
#     the current `dt_method = 2` (adaptive Heun + PI42 + the Step-A
#     `H_ice_dyn`/`f_ice_dyn` fix) is already at 1.19× Fortran wall-
#     clock for `dt_outer = 100 yr`, so the motivation to add a
#     smaller-dt-only mode is weak.
#   - **Reorder the dyn-first chain to dyn-then-update_diagnostics-
#     then-topo**: explicit refresh of `dzsdx` etc. between dyn and
#     topo. Untested.
#
# A failed working prototype (with the Richardson + PI42 sub-cycler)
# is preserved on the branch `dt-method3-frozen-vel-failed` for future
# revisit (also see the trace logs under `logs/trace_dt3_*.log`).
#
# To re-enable, restore the `_frozen_vel_step!` body, the
# `FrozenVelScratch` struct, and the dispatch case in `_select_step!`.


# ===== Fixed-dt driver =====

"""
    _fixed_step!(y, dt_outer, scheme, scratch, p_yelmo) -> y

`dt_method = 0` driver: take a single PC attempt at the requested
`dt_outer`, with no controller and no rejection. The truncation-
error proxy `eta` is computed and pushed to history as a diagnostic
(observable via `scratch.eta_history`), but the model always
accepts the result.

This is intrinsically more robust than a plain forward-Euler step
(the Heun PC averages two RK stages so the local error is `O(dt²)`
rather than `O(dt)`) at the cost of running the dyn solve twice
per outer step. Use when you want a deterministic outer dt with
the safety net of the corrector identity.
"""
function _fixed_step!(y, dt_outer::Float64,
                      scheme::PCScheme, scratch::PCScratch, p_yelmo)
    snapshot!(scratch.snap, y)
    t0 = time()
    eta = pc_step!(scheme, y, dt_outer, scratch)
    # Advective path leaves mat/therm and the `:advance` commit to the
    # caller; legacy path runs them inline. Single shared finaliser.
    _finalize_accepted_step!(y, dt_outer, scratch, p_yelmo)
    wallclock_s = time() - t0
    scratch.n_steps_taken += 1
    scratch.is_bootstrapped = true
    _push_history!(scratch, eta, dt_outer)
    _maybe_log_timestep!(y, dt_outer, eta, 1, wallclock_s)
    return y
end


# ===== Lazy timestep-log accessor =====

# When `y.p.yelmo.log_timestep == true`, lazily create a `TimestepLog`
# at `<rundir>/yelmo_timesteps.nc` and append one row per accepted PC
# step. No-op otherwise. The log is cached on
# `y.dyn.scratch.timestep_log[]` (allocated as `Ref{Any}(nothing)` by
# `_alloc_yelmo_groups`, mirroring the `pc_scratch` lazy pattern).
function _maybe_log_timestep!(y, dt_now::Real, eta::Real,
                              iter_redo::Integer, wallclock_s::Real)
    (y.p === nothing || !y.p.yelmo.log_timestep) && return nothing
    cached = y.dyn.scratch.timestep_log[]
    log = if cached === nothing
        new_log = init_timestep_log!(y)
        y.dyn.scratch.timestep_log[] = new_log
        new_log
    else
        cached::TimestepLog
    end
    write_timestep_row!(log, y;
                        dt_now      = dt_now,
                        eta         = eta,
                        iter_redo   = iter_redo,
                        wallclock_s = wallclock_s)
    return nothing
end


# ===== Lazy scratch accessor =====

# Pull (or initialise on first call) the cached PCScratch from
# `y.dyn.scratch.pc_scratch`. The Ref{Any} field is allocated by
# `_alloc_yelmo_groups` in YelmoCore.jl alongside the SSA scratch.
function _ensure_pc_scratch!(y)
    cached = y.dyn.scratch.pc_scratch[]
    if cached !== nothing
        return cached::PCScratch
    end
    s = _alloc_pc_scratch(y)
    y.dyn.scratch.pc_scratch[] = s
    return s
end


# ===== Public entry: dispatch on dt_method =====

"""
    _select_step!(y, dt) -> y

Backend for `step!(YelmoModel, dt)`. Dispatches on
`y.p.yelmo.dt_method`:

  - `0` : fixed `dt_outer` Heun PC (one attempt, no controller, no
          rejection). `eta` is computed as a diagnostic.
  - `2` : adaptive Heun PC with PI42 controller and reject/retry.

Other values error explicitly. `dt_method = 3` (frozen-velocity sub-
cycling) was prototyped but reverted — see the comment block above
the `_fixed_step!` definition for what failed and where to revisit.

`y.p === nothing` (parameter-less benchmark constructions) falls
through to a plain forward-Euler step `_step_fe!` so simple
in-memory test setups keep working without a `YelmoParameters`.
"""
function _select_step!(y, dt::Float64)
    if y.p === nothing
        return _step_fe!(y, dt)
    end
    method  = Int(y.p.yelmo.dt_method)
    scratch = _ensure_pc_scratch!(y)
    if method == 0
        # Fixed-FE branch always uses HEUN as the diagnostic-eta scheme,
        # regardless of `pc_method`. The fixed-dt path doesn't actually
        # adapt, so the only role of the PC scheme here is computing
        # `eta` as a diagnostic. HEUN's bootstrap-form predictor is
        # history-free and avoids the AB-SAM `dt_zeta` bookkeeping that
        # would otherwise need a meaningful previous-step dt.
        return _fixed_step!(y, dt, HEUN(), scratch, y.p.yelmo)
    elseif method == 2
        scheme     = _resolve_pc_scheme(y.p.yelmo.pc_method)
        controller = _resolve_pc_controller(y.p.yelmo.pc_controller)
        return _adaptive_step!(y, dt, scheme, controller, scratch, y.p.yelmo)
    else
        error("step!: unsupported dt_method=$method (use 0=fixed-dt Heun " *
              "or 2=adaptive Heun).")
    end
end
