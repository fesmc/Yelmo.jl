# ----------------------------------------------------------------------
# Predictor-corrector time loop of `step!(YelmoModel, dt)` — port of
# Fortran `yelmo_update` (yelmo/src/yelmo_ice.f90:41-560) and the pc
# helpers of `yelmo_timesteps.f90`.
#
# Each internal step (Cheng et al., 2017):
#
#   1. choose dt: the remaining interval (`dt_method = 0`) or the
#      PI-controller step (`dt_method = 2`); `dt_min` on a cold start;
#   2. predictor topography (β1, β2), velocity solve at H_pred,
#      corrector topography (β3, β4) — see `topo_step!(y, dt, ::PCStage)`;
#   3. truncation error `tau` from H_corr − H_pred and its norm `eta`;
#      if `eta > pc_tol`, restore topography and dynamics and redo the
#      step with a smaller dt (up to `pc_n_redo` attempts);
#   4. `mat_step!`, `therm_step!` at H_n, then advance the topography
#      to H_pred (`pc_use_H_pred`) or H_corr.
#
# Parameters (`&yelmo`): `dt_method`, `dt_min`, `pc_method`
# ("AB-SAM", "HEUN", "FE-SBE"), `pc_controller`, `pc_use_H_pred`,
# `pc_filter_vel`, `pc_n_redo`, `pc_tol`, `pc_eps`.
#
# Not yet as in Fortran (yelmo dev): the error norm and mask
# (`_compute_pc_eta`), the controller limits (`_controller_dt`),
# `dt_method = 1`, the kill checks, and the pc state in restarts.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior, AbstractField

# Extend `YelmoCore._select_step!` (declared as a stub there) with the
# time loop below.
import .YelmoCore: _select_step!

using .YelmoTiming: @timed_section
using .YelmoModelTopo: PCPredictor, PCCorrector, PCAdvance, H_grnd_point

export PCScheme, HEUN, FE_SBE, AB_SAM
export PIController, PI42

# ===== Schemes =====

abstract type PCScheme end

"""
    FE_SBE

Forward Euler predictor, semi-implicit backward Euler corrector
(`pc_method = "FE-SBE"`): β = (1, 0, 1, 0), order 1, truncation error
`tau = (H_corr − H_pred)/(2·dt)`.
"""
struct FE_SBE <: PCScheme end

"""
    AB_SAM

Adams-Bashforth predictor, semi-implicit Adams-Moulton corrector
(`pc_method = "AB-SAM"`, the Fortran default): β = (1 + ζ/2, −ζ/2, ½, ½)
with ζ = dt/dt_prev, order 2, `tau = ζ·(H_corr − H_pred)/((3ζ + 3)·dt)`.
Until the first step is done (cold start) it runs as `FE_SBE`.
"""
struct AB_SAM <: PCScheme end

"""
    HEUN

Heun's method (`pc_method = "HEUN"`): β = (1, 0, ½, ½), order 2,
`tau = (H_corr − H_pred)/(6·dt)`.
"""
struct HEUN <: PCScheme end

# β coefficients `(β1, β2, β3, β4)` and order `pc_k` of a scheme
# (Fortran `set_pc_beta_coefficients`, yelmo_timesteps.f90:63-174):
#   predictor: dHidt_dyn = β1·f(H_n) + β2·f_{n-1}
#   corrector: dHidt_dyn = β3·f(H_pred) + β4·f(H_n)
pc_beta(::FE_SBE, ζ::Float64) = ((1.0, 0.0, 1.0, 0.0), 1)
pc_beta(::AB_SAM, ζ::Float64) = ((1.0 + 0.5 * ζ, -0.5 * ζ, 0.5, 0.5), 2)
pc_beta(::HEUN,   ζ::Float64) = ((1.0, 0.0, 0.5, 0.5), 2)

# Order of the scheme, used by the controller gains.
pc_order(s::PCScheme) = pc_beta(s, 1.0)[2]

# `tau = pc_tau_factor · (H_corr − H_pred)/dt` (Fortran
# `calc_pc_tau_fe_sbe/_ab_sam/_heun`, yelmo_timesteps.f90:383-460).
pc_tau_factor(::FE_SBE, ζ::Float64) = 1.0 / 2.0
pc_tau_factor(::AB_SAM, ζ::Float64) = ζ / (3.0 * ζ + 3.0)
pc_tau_factor(::HEUN,   ζ::Float64) = 1.0 / 6.0

# The scheme used for a step: AB-SAM needs the previous step, so on a
# cold start it runs as FE-SBE (Fortran yelmo_ice.f90:272-286, 339-349).
_effective_scheme(s::PCScheme, pc_active::Bool) = s
_effective_scheme(::AB_SAM, pc_active::Bool) = pc_active ? AB_SAM() : FE_SBE()

function _resolve_pc_scheme(name::AbstractString)
    name == "HEUN"   && return HEUN()
    name == "FE-SBE" && return FE_SBE()
    name == "AB-SAM" && return AB_SAM()
    error("Unknown pc_method=\"$name\". Supported: \"AB-SAM\", \"HEUN\", \"FE-SBE\".")
end

# ===== Controllers =====

abstract type PIController end

"""
    PI42

Söderlind & Wang (2006) PI controller (`pc_controller = "PI42"`, the
Fortran default): `rho = (eps/eta_n)^(k_i + k_p) · (eps/eta_nm1)^(−k_p)`
with `k_i = 2/(5·pc_k)`, `k_p = 1/(5·pc_k)`.
"""
struct PI42 <: PIController end

function _resolve_pc_controller(name::AbstractString)
    name == "PI42" && return PI42()
    error("Unknown pc_controller=\"$name\". Supported: \"PI42\".")
end

"""
    _dt_ratio(controller, eta_n, eta_nm1, eps, pc_k) -> rho

Ratio `dt_{n+1}/dt_n` from the error history (Fortran
`calc_pi_rho_pi42`). `eta` is floored at 1e-8.
"""
function _dt_ratio(::PI42, eta_n::Real, eta_nm1::Real, eps::Real, pc_k::Int)
    k_i = 2.0 / (pc_k * 5.0)
    k_p = 1.0 / (pc_k * 5.0)
    eta_n_safe   = max(eta_n,   1.0e-8)
    eta_nm1_safe = max(eta_nm1, 1.0e-8)
    return (eps / eta_n_safe)^(k_i + k_p) * (eps / eta_nm1_safe)^(-k_p)
end

# Per-step clamp on the dt ratio. Yelmo.jl-specific (Fortran dev caps
# the ratio at `pc_rho_max` only); replaced in the controller port.
_clamp_dt_ratio(rho) = clamp(rho, 0.2, 10.0)

# Avoid one big and one tiny step at the end of the interval: if dt is
# more than half of `remaining` (and less than it), take two equal steps
# (Fortran `limit_adaptive_timestep`, without its rounding of dt).
function _limit_step(dt_now::Float64, remaining::Float64)
    remaining > 0 || return 0.0
    dt = min(dt_now, remaining)
    if dt / remaining > 0.5 && dt < remaining
        return 0.5 * remaining
    end
    return dt
end

# PI-controller timestep for the next step, from the history of the last
# steps (`pc_dt[1]`, `pc_eta[1]` the latest).
function _controller_dt(controller::PIController, scratch, pc_eps::Float64,
                        dt_min::Float64, remaining::Float64)
    rho = _dt_ratio(controller, scratch.pc_eta[1], scratch.pc_eta[2], pc_eps, scratch.pc_k)
    dt  = clamp(scratch.pc_dt[1] * _clamp_dt_ratio(rho), dt_min, remaining)
    return min(max(_limit_step(dt, remaining), dt_min), remaining)
end

# ===== Redo reference =====

"""
    RedoRef

Copy of all topography and dynamics fields at the start of a step,
restored when the step is redone (Fortran `tpo_ref`, `dyn_ref`). The
other components are only updated after a step is accepted.
"""
struct RedoRef
    fields::Vector{Tuple{Array{Float64,3}, Array{Float64,3}}}   # (field data, copy)
    time::Base.RefValue{Float64}
end

function RedoRef(y)
    data = Array{Float64,3}[]
    for g in (y.tpo, y.dyn), v in values(g)
        v isa AbstractField && push!(data, parent(v.data))
    end
    return RedoRef([(d, similar(d)) for d in data], Ref(y.time))
end

function save!(ref::RedoRef, y)
    for (d, c) in ref.fields
        copyto!(c, d)
    end
    ref.time[] = y.time
    return ref
end

function restore!(y, ref::RedoRef)
    for (d, c) in ref.fields
        copyto!(d, c)
    end
    y.time = ref.time[]
    return y
end

# ===== Per-model state of the time loop =====

const PC_HISTORY = 3

"""
    PCScratch

State of the time loop kept between calls (Fortran `ytime`): the dt and
`eta` of the last steps (`pc_dt[1]`, `pc_eta[1]` the latest; initially
`dt_min` and `pc_eps`), whether the pc history is valid (`pc_active`,
`false` until the first step), the order used by the controller, the
redo reference, the truncation error field and step counters. Created on
the first `step!` and kept in `y.dyn.scratch.pc_scratch[]`.
"""
mutable struct PCScratch
    pc_dt::Vector{Float64}
    pc_eta::Vector{Float64}
    pc_active::Bool
    pc_k::Int
    ref::RedoRef
    pc_tau::Array{Float64,3}
    n_steps_taken::Int
    n_rejections::Int
end

function _alloc_pc_scratch(y)
    p = y.p.yelmo
    return PCScratch(fill(Float64(p.dt_min), PC_HISTORY),
                     fill(Float64(p.pc_eps), PC_HISTORY),
                     false,
                     pc_order(_resolve_pc_scheme(p.pc_method)),
                     RedoRef(y),
                     zeros(Float64, size(interior(y.tpo.H_ice))),
                     0, 0)
end

function _ensure_pc_scratch!(y)
    cached = y.dyn.scratch.pc_scratch[]
    cached !== nothing && return cached::PCScratch
    s = _alloc_pc_scratch(y)
    y.dyn.scratch.pc_scratch[] = s
    return s
end

# Latest dt and eta first (Fortran `cshift(…, shift=-1)`).
function _push_history!(scratch::PCScratch, dt::Float64, eta::Float64)
    for k in PC_HISTORY:-1:2
        scratch.pc_dt[k]  = scratch.pc_dt[k-1]
        scratch.pc_eta[k] = scratch.pc_eta[k-1]
    end
    scratch.pc_dt[1]  = dt
    scratch.pc_eta[1] = eta
    return scratch
end

# ===== Transport velocity =====

# Velocity that transports ice in the topography stages (Fortran
# `calc_transport_velocity`, yelmo_topography.f90:2179): the
# depth-averaged solution, or with `filter_vel` the mean of the current
# and previous solutions, with faces into ice-free cells closed
# (`set_inactive_margins!`, from the current f_ice). The level-set front
# variant (`front_subgrid != "none"`) is not ported (`check_ported`).
function _transport_velocity!(y, filter_vel::Bool)
    ux_t, uy_t = y.tpo.scratch.pc.ux_t, y.tpo.scratch.pc.uy_t
    if filter_vel
        interior(ux_t) .= 0.5 .* (interior(y.dyn.ux_bar) .+ interior(y.dyn.ux_bar_prev))
        interior(uy_t) .= 0.5 .* (interior(y.dyn.uy_bar) .+ interior(y.dyn.uy_bar_prev))
    else
        copyto!(interior(ux_t), interior(y.dyn.ux_bar))
        copyto!(interior(uy_t), interior(y.dyn.uy_bar))
    end
    Yelmo.set_inactive_margins!(ux_t, uy_t, y.tpo.f_ice)
    return nothing
end

# ===== Error norm =====

# Norm `eta` [m/yr] of the truncation error `tau = factor·|H_corr − H_pred|/dt`.
#
# With `yelmo.pc_eta_masked` (default), the maximum over the cells of the
# Fortran pc mask (`set_pc_mask`): H_pred and H_corr ≥ 10 m, no partly or
# not ice-covered cell in the 3×3 neighbourhood and H_grnd > 0 in both
# states (f_ice and H_grnd computed from H_pred and H_corr), and not an
# isolated outlier (|tau| > 2·pc_eps with no 4-neighbour above pc_eps).
# Without it, the maximum over all cells.
#
# Not yet as in Fortran dev: the norm (RMS of tau/(1 m + 0.01 H)),
# `pc_eta_H_min`, `pc_eta_u_min`, `pc_eta_trim`, periodic neighbours.
function _compute_pc_eta(factor::Float64, scratch::PCScratch, y, dt::Float64)
    H_pred = y.tpo.scratch.pc.pred.H_ice
    H_corr = y.tpo.scratch.pc.corr.H_ice
    pc_tau = scratch.pc_tau
    c = factor / dt
    @inbounds @simd for i in eachindex(pc_tau)
        pc_tau[i] = abs(H_corr[i] - H_pred[i]) * c
    end
    y.p.yelmo.pc_eta_masked || return maximum(pc_tau)

    pc_eps = Float64(y.p.yelmo.pc_eps)
    Zb  = interior(y.bnd.z_bed)
    Zsl = interior(y.bnd.z_sl)
    rho_sw_ice = y.c.rho_sw / y.c.rho_ice
    H_lim = 10.0
    nx, ny = size(pc_tau, 1), size(pc_tau, 2)

    eta_max = 0.0
    @inbounds for j in 1:ny, i in 1:nx
        (H_pred[i, j, 1] < H_lim || H_corr[i, j, 1] < H_lim) && continue

        im1 = max(i - 1, 1); ip1 = min(i + 1, nx)
        jm1 = max(j - 1, 1); jp1 = min(j + 1, ny)

        # Binary f_ice (front_subgrid = "none"): partly or not covered = H ≤ 0.
        is_margin = false
        for jj in jm1:jp1, ii in im1:ip1
            if H_pred[ii, jj, 1] <= 0.0 || H_corr[ii, jj, 1] <= 0.0
                is_margin = true
                break
            end
        end
        is_margin && continue

        H_grnd_pred = H_grnd_point(H_pred[i, j, 1], Zb[i, j, 1], Zsl[i, j, 1], rho_sw_ice)
        H_grnd_corr = H_grnd_point(H_corr[i, j, 1], Zb[i, j, 1], Zsl[i, j, 1], rho_sw_ice)
        (H_grnd_pred <= 0.0 || H_grnd_corr <= 0.0) && continue

        tau_ij = pc_tau[i, j, 1]
        if tau_ij > 2.0 * pc_eps
            n_above = (pc_tau[im1, j, 1] > pc_eps) + (pc_tau[ip1, j, 1] > pc_eps) +
                      (pc_tau[i, jm1, 1] > pc_eps) + (pc_tau[i, jp1, 1] > pc_eps)
            n_above == 0 && continue
        end
        eta_max = max(eta_max, tau_ij)
    end
    return eta_max
end

# ===== Time loop =====

const _TIME_TOL = 1e-5     # [yr] Fortran `time_tol`

"""
    _pc_loop!(y, dt_outer, scheme, controller, scratch) -> y

Advance `y` by `dt_outer` years with the predictor-corrector time loop
of Fortran `yelmo_update` (see the file header). `dt_method = 0` takes
the whole interval as one step (more if a step is redone); `dt_method =
2` takes PI-controller steps. The first step of a model (cold start) is
`dt_min`.
"""
function _pc_loop!(y, dt_outer::Float64, scheme::PCScheme,
                   controller::PIController, scratch::PCScratch)
    p = y.p.yelmo
    target    = y.time + dt_outer
    dt_min    = Float64(p.dt_min)
    pc_tol    = Float64(p.pc_tol)
    pc_eps    = Float64(p.pc_eps)
    pc_n_redo = Int(p.pc_n_redo)
    adaptive  = Int(p.dt_method) == 2

    # The controller uses the order of the scheme (Fortran sets it at the
    # start of each `yelmo_update`; a cold-start AB-SAM step leaves 1).
    scratch.pc_k = pc_order(scheme)

    while true
        dt_max = max(target - y.time, 0.0)
        dt_now = adaptive ? _controller_dt(controller, scratch, pc_eps, dt_min, dt_max) : dt_max
        scratch.pc_active || (dt_now = min(dt_min, dt_max))
        # Nothing to do on an already-active trajectory.
        (dt_now == 0.0 && scratch.pc_active) && break

        pc_n_redo > 1 && save!(scratch.ref, y)
        t_n = y.time

        eta = 0.0
        iter_redo = 0
        wallclock_s = 0.0
        for iter in 1:pc_n_redo
            iter_redo = iter
            t0 = time()
            time_now = t_n + dt_now
            abs(target - time_now) < _TIME_TOL && (time_now = target)
            y.time = time_now

            ζ = dt_now / scratch.pc_dt[1]
            s = _effective_scheme(scheme, scratch.pc_active)
            β, k = pc_beta(s, ζ)
            β1, β2, β3, β4 = β
            scratch.pc_k = k

            @timed_section y :topo @timed_section y :topo_pred begin
                _transport_velocity!(y, p.pc_filter_vel)
                Yelmo.topo_step!(y, dt_now, PCPredictor(); β1 = β1, β2 = β2)
            end
            @timed_section y :dyn Yelmo.dyn_step!(y, dt_now)
            @timed_section y :topo @timed_section y :topo_corr begin
                _transport_velocity!(y, p.pc_filter_vel)
                Yelmo.topo_step!(y, dt_now, PCCorrector(); β3 = β3, β4 = β4)
            end

            eta = _compute_pc_eta(pc_tau_factor(s, ζ), scratch, y, dt_now)
            wallclock_s += time() - t0

            if iter < pc_n_redo && dt_now > dt_min && eta > pc_tol
                # Redo with a smaller step (Fortran yelmo_ice.f90:385-397).
                scratch.n_rejections += 1
                restore!(y, scratch.ref)
                dt_now = max(dt_now * 0.7 / (1.0 + (eta - pc_tol) / 10.0), dt_min)
            else
                break
            end
        end

        # Accepted: the other components at H_n, then advance the topography.
        t0 = time()
        @timed_section y :mat  Yelmo.mat_step!(y, dt_now)
        @timed_section y :thrm Yelmo.therm_step!(y, dt_now)
        @timed_section y :topo @timed_section y :topo_adv Yelmo.topo_step!(y, dt_now, PCAdvance();
                                                                           use_H_pred = p.pc_use_H_pred)
        wallclock_s += time() - t0

        _push_history!(scratch, max(dt_now, dt_min), eta)
        scratch.pc_active = true
        scratch.n_steps_taken += 1
        _maybe_log_timestep!(y, dt_now, eta, iter_redo, wallclock_s)

        y.time >= target - _TIME_TOL && break
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

# ===== Timestep log =====

# When `y.p.yelmo.log_timestep == true`, lazily create a `TimestepLog`
# at `<rundir>/yelmo_timesteps.nc` and append one row per step. The log
# is cached on `y.dyn.scratch.timestep_log[]`.
function _maybe_log_timestep!(y, dt_now::Real, eta::Real,
                              iter_redo::Integer, wallclock_s::Real)
    y.p.yelmo.log_timestep || return nothing
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

# ===== Entry: dispatch on dt_method =====

"""
    _select_step!(y, dt) -> y

Backend of `step!(YelmoModel, dt)`: the predictor-corrector time loop
for `yelmo.dt_method` 0 (one step per call) or 2 (adaptive). Other
values are not ported.
"""
function _select_step!(y, dt::Float64)
    method = Int(y.p.yelmo.dt_method)
    method in (0, 2) ||
        error("step!: dt_method = $method is not ported (supported: 0, 2).")
    scheme     = _resolve_pc_scheme(y.p.yelmo.pc_method)
    controller = _resolve_pc_controller(y.p.yelmo.pc_controller)
    return _pc_loop!(y, dt, scheme, controller, _ensure_pc_scratch!(y))
end
