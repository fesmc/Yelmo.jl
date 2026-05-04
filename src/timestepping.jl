# ----------------------------------------------------------------------
# Adaptive predictor-corrector timestepping for `step!(YelmoModel, dt)`.
#
# Selection is driven by the `&yelmo` namelist parameters that mirror
# Fortran Yelmo (already declared in `YelmoParams`):
#
#   - `dt_method`     : 0 = fixed forward Euler (default),
#                       2 = adaptive predictor-corrector.
#   - `pc_method`     : "HEUN" (this PR), "FE-SBE" / "AB-SAM" (future).
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
# Architecture (designed for future extension to FE-SBE / AB-SAM):
#
#   - `PCScheme`      : abstract type. Concrete: `HEUN` here; `FE_SBE`
#                       and `AB_SAM` are stubs that error pending Step 2.
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
"""
struct HEUN <: PCScheme end

"""
    FE_SBE

Forward Euler predictor + Semi-implicit Backward Euler corrector.
Stub — implementation lands in the Step 2 PR alongside the
explicit-velocity `topo_step!` refactor.
"""
struct FE_SBE <: PCScheme end

"""
    AB_SAM

Adams-Bashforth predictor + Semi-implicit Adams-Moulton corrector
(Fortran default). Stub — Step 2 PR.
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
    name == "FE-SBE" && error(
        "pc_method=\"FE-SBE\" is declared but not implemented in this PR. " *
        "Use \"HEUN\" for now; FE-SBE lands in the Step 2 PR.")
    name == "AB-SAM" && error(
        "pc_method=\"AB-SAM\" is declared but not implemented in this PR. " *
        "Use \"HEUN\" for now; AB-SAM lands in the Step 2 PR.")
    error("Unknown pc_method=\"$name\". Supported: \"HEUN\".")
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

# Plain forward-Euler step: one (topo, dyn, mat) chain. Used by Heun as
# its stage primitive and by the `dt_method = 0` branch of the
# dispatcher. Phase order matches the Fortran per-step loop
# (`yelmo_ice.f90:268-286`): `calc_ydyn` reads the previous step's
# `mat.ATT`, then `calc_ymat` computes a fresh `ATT` from the
# just-solved velocity field for the next step.
function _step_fe!(y, dt::Float64)
    @timed_section y :topo Yelmo.topo_step!(y, dt)
    @timed_section y :dyn  Yelmo.dyn_step!(y, dt)
    @timed_section y :mat  Yelmo.mat_step!(y, dt)
    return y
end

"""
    pc_step!(scheme, y, dt, scratch) -> eta::Float64

Run one predictor-corrector attempt: assumes the caller has already
captured `y_n` into `scratch.snap`, and that `y` currently holds that
same `y_n` state. Returns the truncation-error proxy `eta` (m/yr).
The model state on return is the corrector result.

Implemented for `HEUN` only in this PR; `FE_SBE` and `AB_SAM` error
out via the `pc_error_factor` chain.
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


# ===== Outer adaptive driver =====

"""
    _adaptive_step!(y, dt_outer, scheme, controller, scratch, p_yelmo) -> y

Advance `y` by exactly `dt_outer` years using the adaptive PC
machinery. Internally the driver may take many sub-steps with
controller-chosen `dt`, and may reject and retry attempts up to
`pc_n_redo` times when the truncation error exceeds `pc_tol`.

The first sub-step uses a small `dt = min(dt_outer, 1.0)` to avoid
any first-step transient on cold-started runs (the SSA `vel_max`
clamp can spike velocity at IC); subsequent sub-steps use the
controller's recommendation.
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

    # Initial dt for this outer call: re-use the last accepted sub-dt
    # if we have one, else seed conservatively.
    dt_now = if isempty(scratch.dt_history)
        min(dt_ceil, 1.0)
    else
        clamp(scratch.dt_history[end], dt_min, dt_ceil)
    end

    while y.time < target_time - 1e-9
        dt_attempt = min(dt_now, target_time - y.time)
        eta = NaN

        accepted = false
        for iter_redo in 1:pc_n_redo
            snapshot!(scratch.snap, y)
            eta = pc_step!(scheme, y, dt_attempt, scratch)

            # Accept if error within tol, or last redo, or hit dt_min.
            if eta <= pc_tol || iter_redo == pc_n_redo || dt_attempt <= dt_min
                accepted = true
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

        # Pick next sub-dt via the controller, clamped per-step
        # to [0.5, 2.0] of the previous dt and overall to [dt_min, dt_ceil].
        eta_n   = scratch.eta_history[end]
        eta_nm1 = length(scratch.eta_history) >= 2 ? scratch.eta_history[end-1] : 0.0
        dt_nm1  = length(scratch.dt_history)  >= 2 ? scratch.dt_history[end-1]  : 0.0
        rho = _dt_ratio(controller, eta_n, eta_nm1, dt_attempt, dt_nm1,
                        pc_eps, pc_order(scheme))
        dt_now = clamp(dt_attempt * _clamp_dt_ratio(rho), dt_min, dt_ceil)
    end

    return y
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

  - `0` : fixed forward Euler (one topo + one dyn pass).
  - `2` : adaptive predictor-corrector (this module's machinery).

Other values error explicitly. Default is `0` to preserve the
existing fixed-dt behaviour for all current tests.
"""
function _select_step!(y, dt::Float64)
    method = y.p === nothing ? 0 : Int(y.p.yelmo.dt_method)
    if method == 0
        return _step_fe!(y, dt)
    elseif method == 2
        scheme     = _resolve_pc_scheme(y.p.yelmo.pc_method)
        controller = _resolve_pc_controller(y.p.yelmo.pc_controller)
        scratch    = _ensure_pc_scratch!(y)
        return _adaptive_step!(y, dt, scheme, controller, scratch, y.p.yelmo)
    else
        error("step!: unsupported dt_method=$method (use 0=fixed FE or " *
              "2=adaptive PC).")
    end
end
