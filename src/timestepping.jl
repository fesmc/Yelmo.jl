# ----------------------------------------------------------------------
# Adaptive predictor-corrector timestepping for `step!(YelmoModel, dt)`.
#
# Selection is driven by the `&yelmo` namelist parameters that mirror
# Fortran Yelmo (already declared in `YelmoParams`):
#
#   - `dt_method`     : 0 = fixed forward Euler (default),
#                       2 = adaptive predictor-corrector.
#   - `pc_method`     : "FE-SBE" (default), "HEUN", "AB-SAM" (stub).
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
#   - `PCScheme`      : abstract type. Concrete: `HEUN`, `FE_SBE`
#                       (default); `AB_SAM` is a stub.
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
`H_min_grnd`, or `topo_rel != 0`, prefer `FE_SBE` (the default),
which uses the Fortran-style "both stages start from `H_n`" pattern
where the topo cascade only sees `H_n`'s geometry.
"""
struct HEUN <: PCScheme end

"""
    FE_SBE

Forward Euler predictor + Semi-implicit Backward Euler corrector.
Default `pc_method`. Faithful Fortran-style PC pattern matching
`yelmo_ice.f90:138-377`:

  - Predictor: full topo cascade from `H_n` advected by `u_n`,
    yielding `H_pred`.
  - SSA solve at `H_pred`, yielding `u_pred`.
  - Corrector: full topo cascade **from `H_n` again**, but advected
    by `u_pred`. Yields `H_corr`.
  - `mat_step!` + `therm_step!` run once afterward at the corrector
    state (matching Fortran's outer-loop ordering — these are not
    part of the PC iteration).

Both topo passes start from `H_n`, so calving / `resid_tendency!` /
mass-balance kernels only ever see `H_n`'s geometry — the corrector
result is consistent with the cascade's nonlinear behaviour at
`H_n`. Cost: **1 SSA solve per attempt** (vs `HEUN`'s 2). Velocity
carried forward is `u_pred` (the post-predictor SSA solution),
matching Fortran's pattern.

Truncation-error proxy: `tau = 1/(2·dt) · (H_corr − H_pred)`
matching Fortran `yelmo_timesteps.f90:329`.
"""
struct FE_SBE <: PCScheme end

"""
    AB_SAM

Adams-Bashforth predictor + Semi-implicit Adams-Moulton corrector
(Fortran's `pc_method = "AB-SAM"` default). Stub — implementation
deferred until we want to plug in dt-history-dependent error
formulas. The PI42 controller already maintains the dt-history ring.
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
pc_error_factor(::AB_SAM) =
    error("AB-SAM requires dt-history-dependent factor; Step 2 PR.")

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
    name == "AB-SAM" && error(
        "pc_method=\"AB-SAM\" is declared but not implemented. " *
        "Supported: \"HEUN\", \"FE-SBE\" (default).")
    error("Unknown pc_method=\"$name\". Supported: \"HEUN\", \"FE-SBE\" (default).")
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
    time::Float64
end

function _alloc_pc_snapshot(y)
    return PCSnapshot(
        copy(interior(y.tpo.H_ice)),
        copy(interior(y.dyn.ux_b)),
        copy(interior(y.dyn.uy_b)),
        copy(interior(y.dyn.ux_bar)),
        copy(interior(y.dyn.uy_bar)),
        y.time,
    )
end

"""
    snapshot!(snap::PCSnapshot, y) -> snap

Copy the current prognostic state of `y` into `snap`, in place.
"""
function snapshot!(snap::PCSnapshot, y)
    copyto!(snap.H_ice,  interior(y.tpo.H_ice))
    copyto!(snap.ux_b,   interior(y.dyn.ux_b))
    copyto!(snap.uy_b,   interior(y.dyn.uy_b))
    copyto!(snap.ux_bar, interior(y.dyn.ux_bar))
    copyto!(snap.uy_bar, interior(y.dyn.uy_bar))
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
    copyto!(interior(y.tpo.H_ice),  snap.H_ice)
    copyto!(interior(y.dyn.ux_b),   snap.ux_b)
    copyto!(interior(y.dyn.uy_b),   snap.uy_b)
    copyto!(interior(y.dyn.ux_bar), snap.ux_bar)
    copyto!(interior(y.dyn.uy_bar), snap.uy_bar)
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
        zeros(Float64, size(H)),
        zeros(Float64, size(H)),
        Float64[],
        Float64[],
        0, 0,
    )
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

Dispatches on `PCScheme`. Concrete methods: `HEUN`, `FE_SBE`.
"""
function pc_step!(::HEUN, y, dt::Float64, scratch::PCScratch)
    snap = scratch.snap

    # Stage 1: full FE step  y_n  →  y_pred
    @timed_section y :pc_predictor _step_fe!(y, dt)
    copyto!(scratch.H_pred, interior(y.tpo.H_ice))

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
    # Using snap.H_ice (= H_n) as the MASK_ICE_FIXED reference restores
    # the start-of-step thickness for fixed cells, matching Fortran's
    # mask-pass-at-end-of-PC-corrector convention.
    apply_mask_ice_pass!(y, snap.H_ice)

    y.time = snap.time + dt
    Yelmo.update_diagnostics!(y)

    # Truncation-error proxy. Velocity carried forward into the next
    # outer step is `u_**` (from stage 2's dyn solve), matching
    # Fortran's "carry the predictor-state velocity" behaviour
    # (yelmo_ice.f90 outer loop — never re-solves dyn at the corrector
    # state). The Picard loop on the next step's predictor will tug
    # the velocity back to consistency with the new H_n.
    factor = pc_error_factor(HEUN())
    diff_max = 0.0
    @inbounds @simd for i in eachindex(scratch.H_corr)
        d = abs(scratch.H_corr[i] - scratch.H_pred[i])
        diff_max = max(diff_max, d)
    end
    return factor * diff_max / dt
end

"""
    pc_step!(::FE_SBE, y, dt, scratch) -> eta::Float64

Faithful Fortran-style FE-SBE predictor-corrector. Both predictor
and corrector topo cascades operate on `H_n`'s geometry; only the
advection velocity differs.

Sequence (matches `yelmo_ice.f90:138-377` outer-loop body):

  1. **Predictor**: `topo_step!(y, dt)` — full topo cascade from
     `H_n` advected by `u_n` (the velocity present in `y.dyn` at
     attempt start, snapshotted in `scratch.snap`). Result: `H_pred`.
  2. **Velocity update**: `dyn_step!(y, dt)` — solve SSA at `H_pred`,
     yielding `u_pred` (1 SSA solve per attempt vs `HEUN`'s 2).
  3. **Restore H, keep velocity**: `restore_H_only!(y, snap)` puts
     `H_ice` back to `H_n` and resets `time`, but leaves
     `y.dyn.ux_bar = u_pred`.
  4. **Corrector**: `topo_step!(y, dt)` — same topo cascade run again
     from `H_n`, now advected by `u_pred`. Result: `H_corr`.
  5. **Material + thermal**: `mat_step!` + `therm_step!` once at the
     corrector state. These are *not* part of the PC iteration in
     Fortran — they run after the topo+dyn outer-loop body.

Velocity carried forward to the next outer step is `u_pred`, matching
Fortran's "carry the predictor-state velocity" behaviour.

Truncation-error proxy: `tau = 1/(2·dt) · |H_corr − H_pred|`,
matching Fortran `yelmo_timesteps.f90:329`.
"""
function pc_step!(::FE_SBE, y, dt::Float64, scratch::PCScratch)
    snap = scratch.snap

    # Stage 1 — predictor topo cascade with the snapshotted velocity.
    # `y.dyn.ux_bar` already holds `u_n` at this point because the
    # outer driver snapshotted into `scratch.snap` immediately before
    # calling us, and nothing has touched `y.dyn` since.
    @timed_section y :pc_predictor Yelmo.topo_step!(y, dt)
    copyto!(scratch.H_pred, interior(y.tpo.H_ice))

    # Stage 2 — solve SSA at `H_pred` to get `u_pred`. Updates
    # `y.dyn.ux_bar` / `y.dyn.uy_bar` (and `_b` versions) in place.
    @timed_section y :dyn Yelmo.dyn_step!(y, dt)

    # Stage 3 — restore `H_ice` and `time` to `y_n`, keeping the
    # post-Stage-2 velocity `u_pred` in `y.dyn`. `update_diagnostics!`
    # (called inside `restore_H_only!`) refreshes `f_grnd`, `H_grnd`,
    # `z_srf`, mask fields, etc. from `H_n`.
    restore_H_only!(y, snap)

    # Stage 4 — corrector topo cascade from `H_n` advected by `u_pred`.
    # `topo_step!` reads `y.dyn.ux_bar` / `y.dyn.uy_bar` by default,
    # which now hold `u_pred`. The full cascade (calving, mass balance,
    # `resid_tendency!`, …) sees `H_n`'s geometry — the key difference
    # vs `HEUN`-via-2-FE.
    @timed_section y :pc_corrector Yelmo.topo_step!(y, dt)
    # `y` is now at `(H_corr, u_pred, t_n + dt)`.

    # Stage 5 — material + thermal once at the corrector state. Not
    # part of the PC iteration in Fortran (yelmo_ice.f90 outer loop
    # ordering: topo+dyn iterate, mat/therm follow once).
    @timed_section y :mat  Yelmo.mat_step!(y, dt)
    @timed_section y :thrm Yelmo.therm_step!(y, dt)

    # Truncation-error proxy.
    factor = pc_error_factor(FE_SBE())
    H_corr = interior(y.tpo.H_ice)
    diff_max = 0.0
    @inbounds @simd for i in eachindex(H_corr)
        d = abs(H_corr[i] - scratch.H_pred[i])
        diff_max = max(diff_max, d)
    end
    return factor * diff_max / dt
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

# Per-step clamp on the dt growth/shrink ratio so the controller can
# only step in modest increments, smoothing transients. Söderlind &
# Wang (2006) recommend [0.5, 2.0]; we use the same bracket Fortran
# Yelmo applies after the controller (`dt_now = clamp(dt_n*rho, ...)`).
_clamp_dt_ratio(rho) = clamp(rho, 0.5, 2.0)


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

        dt_attempt = max(_limit_step(dt_now, remaining), dt_min)
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
        # Store actual eta (NOT floored to pc_eps) so the controller
        # sees the true history. `_dt_ratio` does its own floor at
        # 1e-8 to keep the divisor finite.
        _push_history!(scratch, eta, dt_attempt)
        _maybe_log_timestep!(y, dt_attempt, eta, accepted_iter, wallclock_s)
    end

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
    wallclock_s = time() - t0
    scratch.n_steps_taken += 1
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
in-memory test setups keep working without a `YelmoModelParameters`.
"""
function _select_step!(y, dt::Float64)
    if y.p === nothing
        return _step_fe!(y, dt)
    end
    method  = Int(y.p.yelmo.dt_method)
    scratch = _ensure_pc_scratch!(y)
    if method == 0
        # Heun is the only PC scheme implemented. Hardcode it here so a
        # default `pc_method = "AB-SAM"` (currently a stub) still gives
        # a usable diagnostic eta. `pc_method` is read but only honoured
        # when it matches an implemented scheme.
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
