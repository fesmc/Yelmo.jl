# ----------------------------------------------------------------------
# Predictor-corrector time loop of `step!(YelmoModel, dt)` — port of
# Fortran `yelmo_update` (yelmo/src/yelmo_ice.f90:41-560) and the pc
# helpers of `yelmo_timesteps.f90`.
#
# Each internal step (Cheng et al., 2017):
#
#   1. choose dt: the remaining interval (`dt_method = 0`), the Courant
#      step (`dt_method = 1`) or the controller step (`dt_method = 2`);
#      `dt_min` on a cold start;
#   2. predictor topography (β1, β2), velocity solve at H_pred,
#      corrector topography (β3, β4) — see `topo_step!(y, dt, ::PCStage)`;
#   3. truncation error `tau` from H_corr − H_pred and its norm `eta`;
#      if `eta > pc_tol`, restore topography and dynamics and redo the
#      step with a smaller dt (up to `pc_n_redo` attempts);
#   4. `mat_step!`, `therm_step!` at H_n, then advance the topography
#      to H_pred (`pc_use_H_pred`) or H_corr.
#
#   5. kill checks (`yelmo_check_kill`).
#
# Parameters (`&yelmo`): `dt_method`, `dt_min`, `cfl_max`, `pc_method`
# ("AB-SAM", "HEUN", "FE-SBE"), `pc_controller`, `pc_use_H_pred`,
# `pc_filter_vel`, `pc_n_redo`, `pc_tol`, `pc_eps`, `pc_cfl_max`,
# `pc_rho_max`, `pc_eta_H_min`, `pc_eta_u_min`, `pc_eta_trim`,
# `disable_kill`.
#
# Not yet as in Fortran (yelmo dev): the pc state in restarts.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior, AbstractField
using Oceananigans.Grids: topology, Periodic

# Extend `YelmoCore._select_step!` (declared as a stub there) with the
# time loop below.
import .YelmoCore: _select_step!, pc_history, set_pc_history!
using .YelmoCore: PC_HISTORY

using .YelmoTiming: @timed_section
using .YelmoModelTopo: PCPredictor, PCCorrector, PCAdvance, H_grnd_point

export PCScheme, HEUN, FE_SBE, AB_SAM
export PIController, PI42, H312b, H312PID, H321PID, PID1

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
    PI42, H312b, H312PID, H321PID, PID1

Timestep controllers (`pc_controller`, Fortran `set_adaptive_timestep_pc`):
the ratio `rho = dt_{n+1}/dt_n` from the error norms `eta` of the last
three steps, aiming at `eta = pc_eps`. `PI42` (the default) is the
Söderlind & Wang (2006) PI controller,
`rho = (eps/eta_n)^(k_i + k_p) · (eps/eta_nm1)^(−k_p)` with
`k_i = 2/(5·pc_k)`, `k_p = 1/(5·pc_k)`; the others are Söderlind (2003)
H312b, H312PID, H321PID and a PID controller.
"""
struct PI42    <: PIController end
struct H312b   <: PIController end
struct H312PID <: PIController end
struct H321PID <: PIController end
struct PID1    <: PIController end

function _resolve_pc_controller(name::AbstractString)
    name == "PI42"    && return PI42()
    name == "H312b"   && return H312b()
    name == "H312PID" && return H312PID()
    name == "H321PID" && return H321PID()
    name == "PID1"    && return PID1()
    error("Unknown pc_controller=\"$name\". Supported: PI42, H312b, H312PID, H321PID, PID1.")
end

"""
    _dt_ratio(controller, eta, dt, eps, pc_k) -> rho

Ratio `dt_{n+1}/dt_n` from the error norms `eta = (eta_n, eta_nm1,
eta_nm2)` and timesteps `dt = (dt_n, dt_nm1, dt_nm2)` (each ≥ dt_min) of the
last steps (Fortran `calc_pi_rho_*`, yelmo_timesteps.f90:610-735).
"""
function _dt_ratio(::PI42, eta, dt, eps::Float64, pc_k::Int)
    k_i = 2.0 / (pc_k * 5.0)
    k_p = 1.0 / (pc_k * 5.0)
    return (eps / eta[1])^(k_i + k_p) * (eps / eta[2])^(-k_p)   # alpha_2 = 0
end

function _dt_ratio(::H312b, eta, dt, eps::Float64, pc_k::Int)
    k, b = Float64(pc_k), 8.0
    rho_nm1 = dt[1] / dt[2]
    rho_nm2 = dt[2] / dt[3]
    return (eps / eta[1])^(1.0 / (k * b)) * (eps / eta[2])^(2.0 / (k * b)) *
           (eps / eta[3])^(1.0 / (k * b)) * rho_nm1^(-3.0 / b) * rho_nm2^(-1.0 / b)
end

function _dt_ratio(::H312PID, eta, dt, eps::Float64, pc_k::Int)
    k_i = 0.08 / pc_k
    return (eps / eta[1])^(k_i / 4) * (eps / eta[2])^(k_i / 2) * (eps / eta[3])^(k_i / 4)
end

function _dt_ratio(::H321PID, eta, dt, eps::Float64, pc_k::Int)
    k_i = 0.1 / pc_k
    k_p = 0.45 / pc_k
    return (eps / eta[1])^(0.75 * k_i + 0.5 * k_p) * (eps / eta[2])^(0.5 * k_i) *
           (eps / eta[3])^(-(0.25 * k_i + 0.5 * k_p)) * (dt[1] / dt[2])
end

function _dt_ratio(::PID1, eta, dt, eps::Float64, pc_k::Int)
    k_i, k_p, k_d = 0.175, 0.075, 0.01
    return (eps / eta[1])^k_i * (eta[2] / eta[1])^k_p * (eta[2]^2 / (eta[1] * eta[3]))^k_d
end

"""
    _limit_adaptive_timestep(dt, dt_min, dt_max) -> dt

Fit `dt` to `[dt_min, dt_max]` (`dt_max` the time left in the call), avoid a
big and a tiny step at its end (more than half of `dt_max` → half of it),
and round smaller steps down to 4 decimals (at least 1e-4) (Fortran
`limit_adaptive_timestep`).
"""
function _limit_adaptive_timestep(dt::Float64, dt_min::Float64, dt_max::Float64)
    dt_max > 0 || return dt_max
    dt = min(max(dt, dt_min), dt_max)
    if dt / dt_max > 0.5 && dt < dt_max
        return 0.5 * dt_max
    elseif dt / dt_max < 0.5
        return max(1, floor(Int64, dt * 1e4)) * 1e-4
    end
    return dt
end

# Courant-limited timestep of the transport velocity (Fortran
# `calc_adv2D_timestep1`): per cell `C/(max|u| over its x faces/dx +
# max|v| over its y faces/dy + 0.1/dx)`, the minimum over the cells (the
# domain border is left out in non-periodic directions).
function _adv_timestep_min(ux, uy, dx::Float64, dy::Float64, cfl::Float64)
    Ux = interior(ux); Uy = interior(uy)
    Tx = topology(ux.grid, 1); Ty = topology(uy.grid, 2)
    nx, ny = size(Ux, 1), size(Uy, 2)
    per_x, per_y = Tx === Periodic, Ty === Periodic
    i1, i2 = per_x ? (1, nx) : (2, nx - 1)
    j1, j2 = per_y ? (1, ny) : (2, ny - 1)
    dt_min = Inf
    @inbounds for j in j1:j2, i in i1:i2
        ie = per_x ? mod1(i + 1, nx) : i + 1     # east face of cell i
        jn = per_y ? mod1(j + 1, ny) : j + 1     # north face of cell j
        u = max(abs(Ux[i, j, 1]), abs(Ux[ie, j, 1]))
        v = max(abs(Uy[i, j, 1]), abs(Uy[i, jn, 1]))
        u < 1e-15 && (u = 0.0)
        v < 1e-15 && (v = 0.0)
        dt_min = min(dt_min, cfl / (u / dx + v / dy + 0.1 / dx))
    end
    return dt_min
end

# Is there a checkerboard pattern in `var` (|var| ≥ lim with opposite signs
# on both sides in x or y)? Fortran `check_checkerboard`.
function _has_checkerboard(var, lim::Float64)
    V = interior(var)
    nx, ny = size(V, 1), size(V, 2)
    per_x = topology(var.grid, 1) === Periodic
    per_y = topology(var.grid, 2) === Periodic
    i1, i2 = per_x ? (1, nx) : (2, nx - 1)
    j1, j2 = per_y ? (1, ny) : (2, ny - 1)
    @inbounds for j in j1:j2, i in i1:i2
        v = V[i, j, 1]
        abs(v) >= lim || continue
        im1, ip1, jm1, jp1 = _neighbors(i, j, nx, ny, per_x, per_y)
        if (v * V[im1, j, 1] < 0 && v * V[ip1, j, 1] < 0) ||
           (v * V[i, jm1, 1] < 0 && v * V[i, jp1, 1] < 0)
            return true
        end
    end
    return false
end

# Neighbour indices: wrap in periodic directions, clamp at the border
# otherwise (Fortran `get_neighbor_indices_bc_codes`).
@inline function _neighbors(i, j, nx, ny, per_x::Bool, per_y::Bool)
    im1 = i > 1  ? i - 1 : (per_x ? nx : 1)
    ip1 = i < nx ? i + 1 : (per_x ? 1  : nx)
    jm1 = j > 1  ? j - 1 : (per_y ? ny : 1)
    jp1 = j < ny ? j + 1 : (per_y ? 1  : ny)
    return im1, ip1, jm1, jp1
end

# Timestep of the controller (Fortran `set_adaptive_timestep_pc`): ratio
# from the history, at most `pc_rho_max`, Courant cap `pc_cfl_max` on the
# transport velocity, then `_limit_adaptive_timestep`.
function _pc_timestep(controller::PIController, scratch, p, dt_min::Float64,
                      dt_max::Float64, ux_t, uy_t, dx::Float64, dy::Float64)
    dt  = (max(scratch.pc_dt[1], dt_min), max(scratch.pc_dt[2], dt_min),
           max(scratch.pc_dt[3], dt_min))
    eta = (scratch.pc_eta[1], scratch.pc_eta[2], scratch.pc_eta[3])
    rho = min(_dt_ratio(controller, eta, dt, Float64(p.pc_eps), scratch.pc_k),
              Float64(p.pc_rho_max))
    dt_new = min(rho * dt[1], _adv_timestep_min(ux_t, uy_t, dx, dy, Float64(p.pc_cfl_max)))
    return _limit_adaptive_timestep(dt_new, dt_min, dt_max)
end

# Courant timestep of `dt_method = 1` (Fortran `set_adaptive_timestep`):
# Courant number `cfl_max` on the transport velocity, ×0.05 if `dHidt`
# has a checkerboard pattern, then `_limit_adaptive_timestep`.
function _cfl_timestep(y, p, dt_min::Float64, dt_max::Float64, ux_t, uy_t,
                       dx::Float64, dy::Float64)
    dt = _adv_timestep_min(ux_t, uy_t, dx, dy, Float64(p.cfl_max))
    _has_checkerboard(y.tpo.dHidt, 1.0) && (dt *= 0.05)
    return _limit_adaptive_timestep(dt, dt_min, dt_max)
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

"""
    PCScratch

State of the time loop kept between calls (Fortran `ytime`): the dt and
`eta` of the last steps (`pc_dt[1]`, `pc_eta[1]` the latest; initially
`dt_min` and `pc_eps`), whether the pc history is valid (`pc_active`,
`false` until the first step), the order used by the controller, the
redo reference, the truncation error `pc_tau` [m/yr] and the mask of the
points in its norm, the Courant (`dt_adv`) and controller (`dt_pi`)
timesteps of the last step, and step counters. Created on the first
`step!` and kept in `y.dyn.scratch.pc_scratch[]`.
"""
mutable struct PCScratch
    pc_dt::Vector{Float64}
    pc_eta::Vector{Float64}
    pc_active::Bool
    pc_k::Int
    ref::RedoRef
    pc_tau::Array{Float64,3}
    pc_mask::Array{Bool,3}
    e2::Vector{Float64}          # squared scaled errors of the norm
    dt_adv::Float64
    dt_pi::Float64
    n_steps_taken::Int
    n_rejections::Int
    n_dtmin_run::Int             # consecutive steps at dt_min
end

function _alloc_pc_scratch(y)
    p = y.p.yelmo
    return PCScratch(fill(Float64(p.dt_min), PC_HISTORY),
                     fill(Float64(p.pc_eps), PC_HISTORY),
                     false,
                     pc_order(_resolve_pc_scheme(p.pc_method)),
                     RedoRef(y),
                     zeros(Float64, size(interior(y.tpo.H_ice))),
                     zeros(Bool, size(interior(y.tpo.H_ice))),
                     zeros(Float64, length(interior(y.tpo.H_ice))),
                     0.0, 0.0, 0, 0, 0)
end

function _ensure_pc_scratch!(y)
    cached = y.dyn.scratch.pc_scratch[]
    cached !== nothing && return cached::PCScratch
    s = _alloc_pc_scratch(y)
    y.dyn.scratch.pc_scratch[] = s
    return s
end

function pc_history(y::YelmoModel)
    s = y.dyn.scratch.pc_scratch[]
    (s === nothing || !s.pc_active) && return nothing
    return (copy(s.pc_dt), copy(s.pc_eta))
end

function set_pc_history!(y::YelmoModel, pc_dt::AbstractVector, pc_eta::AbstractVector)
    s = _ensure_pc_scratch!(y)
    copyto!(s.pc_dt, pc_dt)
    copyto!(s.pc_eta, pc_eta)
    s.pc_active = true
    return y
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

"""
    _pc_eta!(scratch, y, factor, dt) -> eta

Truncation error `pc_tau = factor·(H_corr − H_pred)/dt` [m/yr] and its norm
`eta` [1/yr] (Fortran `set_pc_mask` + `calc_pc_eta`): the RMS of the scaled
errors `|tau|/(1 m + 0.01·H_corr)` over the points of the pc mask, without
the `pc_eta_trim` fraction of largest ones; at least 1e-8.

The mask leaves out points with H_pred or H_corr < `pc_eta_H_min`, speed
`uxy_bar` < `pc_eta_u_min`, a partly or not ice-covered cell in the 3×3
neighbourhood in either state, H_grnd ≤ 0 in either state (f_ice and
H_grnd from H_pred and H_corr; f_ice binary, `front_subgrid = "none"`),
and isolated outliers (|tau| > 2·pc_eps with no 4-neighbour above
pc_eps). Neighbours wrap in periodic directions.
"""
function _pc_eta!(scratch::PCScratch, y, factor::Float64, dt::Float64)
    p = y.p.yelmo
    H_pred = y.tpo.scratch.pc.pred.H_ice
    H_corr = y.tpo.scratch.pc.corr.H_ice
    tau  = scratch.pc_tau
    mask = scratch.pc_mask
    c = factor / dt
    @inbounds @simd for i in eachindex(tau)
        tau[i] = (H_corr[i] - H_pred[i]) * c
    end

    pc_eps = Float64(p.pc_eps)
    H_min  = Float64(p.pc_eta_H_min)
    u_min  = Float64(p.pc_eta_u_min)
    U   = interior(y.dyn.uxy_bar)
    Zb  = interior(y.bnd.z_bed)
    Zsl = interior(y.bnd.z_sl)
    rho_sw_ice = y.c.rho_sw / y.c.rho_ice
    nx, ny = size(tau, 1), size(tau, 2)
    per_x = topology(y.tpo.H_ice.grid, 1) === Periodic
    per_y = topology(y.tpo.H_ice.grid, 2) === Periodic

    npts = 0
    e2 = scratch.e2
    @inbounds for j in 1:ny, i in 1:nx
        im1, ip1, jm1, jp1 = _neighbors(i, j, nx, ny, per_x, per_y)
        ok = true
        if H_pred[i, j, 1] < H_min || H_corr[i, j, 1] < H_min
            ok = false
        elseif U[i, j, 1] < u_min
            ok = false
        else
            for jj in (jm1, j, jp1), ii in (im1, i, ip1)
                if H_pred[ii, jj, 1] <= 0.0 || H_corr[ii, jj, 1] <= 0.0
                    ok = false
                end
            end
            if ok && (H_grnd_point(H_pred[i, j, 1], Zb[i, j, 1], Zsl[i, j, 1], rho_sw_ice) <= 0.0 ||
                      H_grnd_point(H_corr[i, j, 1], Zb[i, j, 1], Zsl[i, j, 1], rho_sw_ice) <= 0.0)
                ok = false
            end
        end
        a = abs(tau[i, j, 1])
        if a > 2.0 * pc_eps && abs(tau[im1, j, 1]) <= pc_eps && abs(tau[ip1, j, 1]) <= pc_eps &&
           abs(tau[i, jm1, 1]) <= pc_eps && abs(tau[i, jp1, 1]) <= pc_eps
            ok = false
        end
        mask[i, j, 1] = ok
        if ok
            npts += 1
            e2[npts] = (a / (1.0 + 0.01 * H_corr[i, j, 1]))^2
        end
    end

    npts == 0 && return 1e-8
    n_trim = min(floor(Int, Float64(p.pc_eta_trim) * npts), npts - 1)
    e = view(e2, 1:npts)
    total = sum(e)
    if n_trim > 0
        total -= sum(partialsort!(e, 1:n_trim; rev = true))
    end
    return max(sqrt(max(total, 0.0) / (npts - n_trim)), 1e-8)
end

# ===== Kill checks =====

# Stop the run if the state is invalid (Fortran `yelmo_check_kill`): ice
# thicker than 1e4 m, depth-averaged speed ≥ 2·ssa_vel_max, non-finite
# H_ice / uxy_bar / T_ice, mean of the last pc_eta > 10·pc_tol, or a
# `request`. Writes `<rundir>/yelmo_killed.nc` and throws, unless
# `yelmo.disable_kill`.
function _check_kill(y, scratch::PCScratch; request::Union{Nothing,String} = nothing)
    p = y.p.yelmo
    H = interior(y.tpo.H_ice)
    U = interior(y.dyn.uxy_bar)
    msg = if request !== nothing
        request
    elseif !all(isfinite, H)
        "Non-finite value (NaN or Inf) in H_ice."
    elseif !all(isfinite, U)
        "Non-finite value (NaN or Inf) in uxy_bar."
    elseif !all(isfinite, interior(y.thrm.T_ice))
        "Non-finite value (NaN or Inf) in T_ice."
    elseif maximum(abs, H) >= 1e4
        "Ice thickness too high."
    elseif maximum(abs, U) >= 2.0 * y.p.ydyn.ssa_vel_max
        "Depth-averaged velocity too fast."
    elseif sum(scratch.pc_eta) / length(scratch.pc_eta) > 10.0 * p.pc_tol
        "mean[pc_eta] > [10*pc_tol]: pc_eta = $(scratch.pc_eta)"
    else
        nothing
    end
    (msg === nothing || p.disable_kill) && return nothing
    path = joinpath(y.rundir, "yelmo_killed.nc")
    out = init_output(y, path)
    write_output!(out, y)
    close(out.ds)
    error("Yelmo killed at time = $(y.time): $msg  (state written to $path)")
end

# ===== Time loop =====

const _TIME_TOL = 1e-5     # [yr] Fortran `time_tol`

"""
    _pc_loop!(y, dt_outer, scheme, controller, scratch) -> y

Advance `y` by `dt_outer` years with the predictor-corrector time loop
of Fortran `yelmo_update` (see the file header). `dt_method = 0` takes
the whole interval as one step (more if a step is redone); `dt_method = 1`
takes Courant-limited steps (`cfl_max`), `dt_method = 2` controller steps
(`pc_controller`). The first step of a model (cold start) is `dt_min`.
"""
function _pc_loop!(y, dt_outer::Float64, scheme::PCScheme,
                   controller::PIController, scratch::PCScratch)
    p = y.p.yelmo
    target    = y.time + dt_outer
    dt_min    = Float64(p.dt_min)
    pc_tol    = Float64(p.pc_tol)
    pc_n_redo = Int(p.pc_n_redo)
    dt_method = Int(p.dt_method)
    dx, dy    = Yelmo.YelmoModelTopo._dx(y.g), Yelmo.YelmoModelTopo._dy(y.g)
    ux_t, uy_t = y.tpo.scratch.pc.ux_t, y.tpo.scratch.pc.uy_t
    # Steps this call would need at dt_min (for the dt_min stall check).
    nstep_dtmin = max(ceil(Int, (target - y.time) / dt_min), 1)

    # The controller uses the order of the scheme (Fortran sets it at the
    # start of each `yelmo_update`; a cold-start AB-SAM step leaves 1).
    scratch.pc_k = pc_order(scheme)
    log = _timestep_log!(y, scratch)

    while true
        dt_max = max(target - y.time, 0.0)
        # Courant and controller timesteps, both from the transport velocity.
        _transport_velocity!(y, p.pc_filter_vel)
        scratch.dt_adv = _cfl_timestep(y, p, dt_min, dt_max, ux_t, uy_t, dx, dy)
        scratch.dt_pi  = _pc_timestep(controller, scratch, p, dt_min, dt_max, ux_t, uy_t, dx, dy)
        dt_now = dt_method == 0 ? dt_max : dt_method == 1 ? scratch.dt_adv : scratch.dt_pi
        # Cold start: a step of dt_min (not beyond the end of the call).
        scratch.pc_active || (dt_now = min(dt_min, dt_max))
        # Nothing to do on an already-active trajectory.
        (dt_now == 0.0 && scratch.pc_active) && break

        ns_step0 = time_ns()
        pc_n_redo > 1 && save!(scratch.ref, y)
        t_n = y.time

        eta = 0.0
        iter_redo = 0
        ns_tpo = ns_dyn = UInt64(0)     # topography / dynamics wall time of the last attempt
        for iter in 1:pc_n_redo
            iter_redo = iter
            time_now = t_n + dt_now
            abs(target - time_now) < _TIME_TOL && (time_now = target)
            y.time = time_now

            ζ = dt_now / scratch.pc_dt[1]
            s = _effective_scheme(scheme, scratch.pc_active)
            β, k = pc_beta(s, ζ)
            β1, β2, β3, β4 = β
            scratch.pc_k = k

            ns0 = time_ns()
            @timed_section y :topo @timed_section y :topo_pred begin
                _transport_velocity!(y, p.pc_filter_vel)
                Yelmo.topo_step!(y, dt_now, PCPredictor(); β1 = β1, β2 = β2)
            end
            ns1 = time_ns()
            @timed_section y :dyn Yelmo.dyn_step!(y, dt_now)
            ns2 = time_ns()
            @timed_section y :topo @timed_section y :topo_corr begin
                _transport_velocity!(y, p.pc_filter_vel)
                Yelmo.topo_step!(y, dt_now, PCCorrector(); β3 = β3, β4 = β4)
            end
            ns3 = time_ns()
            ns_tpo = (ns1 - ns0) + (ns3 - ns2)
            ns_dyn = ns2 - ns1

            eta = _pc_eta!(scratch, y, pc_tau_factor(s, ζ), dt_now)

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
        @timed_section y :mat  Yelmo.mat_step!(y, dt_now)
        @timed_section y :thrm Yelmo.therm_step!(y, dt_now)
        ns4 = time_ns()
        @timed_section y :topo @timed_section y :topo_adv Yelmo.topo_step!(y, dt_now, PCAdvance();
                                                                           use_H_pred = p.pc_use_H_pred)
        ns5 = time_ns()
        ns_tpo += ns5 - ns4

        _push_history!(scratch, max(dt_now, dt_min), eta)
        scratch.pc_active = true
        scratch.n_steps_taken += 1
        scratch.n_dtmin_run = abs(dt_now - dt_min) < dt_min * 1e-3 ? scratch.n_dtmin_run + 1 : 0
        log === nothing || _write_timestep_row!(log, y, scratch;
                                                dt_now    = dt_now,
                                                pc_eta    = eta,
                                                speed     = _model_speed(dt_now, ns5 - ns_step0),
                                                speed_tpo = _model_speed(dt_now, ns_tpo),
                                                speed_dyn = _model_speed(dt_now, ns_dyn),
                                                iter_redo = iter_redo - 1)

        _check_kill(y, scratch)
        # Stuck at a small dt_min (Fortran yelmo_ice.f90:521-539).
        if scratch.n_dtmin_run >= min(50, nstep_dtmin) && dt_min <= 1e-2
            _check_kill(y, scratch; request = "Too many consecutive steps at dt_min " *
                        "($(scratch.n_dtmin_run)).")
        end

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

# Model speed [kyr/hr] of `dt` model years in `ns` nanoseconds of wall
# time; 0 if no time elapsed (Fortran `yelmo_calc_speed`).
_model_speed(dt::Float64, ns::Integer) = ns > 0 ? (dt * 1e-3) / (ns * 1e-9 / 3600.0) : 0.0

# The timestep log of `y` (`yelmo.log_timestep`), else `nothing`. Created
# on the first call, cached in `y.dyn.scratch.timestep_log[]`, with a
# first row of the controller state at the current time (Fortran
# `yelmo_init`).
function _timestep_log!(y, scratch::PCScratch)
    y.p.yelmo.log_timestep || return nothing
    cached = y.dyn.scratch.timestep_log[]
    cached === nothing || return cached::TimestepLog
    log = init_timestep_log!(y)
    y.dyn.scratch.timestep_log[] = log
    _write_timestep_row!(log, y, scratch; dt_now = 0.0, dt_adv = 0.0, dt_pi = scratch.pc_dt[1],
                         pc_eta = scratch.pc_eta[1], speed = 0.0, speed_tpo = 0.0,
                         speed_dyn = 0.0, iter_redo = 0, ssa_lin_iter = 0, ssa_lin_fail = 0,
                         adv_lin_iter = 0, adv_lin_fail = 0)
    return log
end

# A row of the log at `y.time`: the Courant and controller timesteps of
# the step and the solver counters of the model, unless given.
function _write_timestep_row!(log::TimestepLog, y, scratch::PCScratch; kwargs...)
    sd, st = y.dyn.scratch, y.tpo.scratch
    write_timestep_row!(log, y.time;
                        dt_adv       = scratch.dt_adv,
                        dt_pi        = scratch.dt_pi,
                        ssa_iter     = sd.ssa_iter_now[],
                        ssa_lin_iter = sd.ssa_lin_iter[],
                        ssa_lin_fail = sd.ssa_lin_fail[],
                        ssa_lim_n    = sd.ssa_lim_n[],
                        adv_lin_iter = st.adv_lin_iter[],
                        adv_lin_fail = st.adv_lin_fail[],
                        kwargs...)
    return log
end

# ===== Entry: dispatch on dt_method =====

"""
    _select_step!(y, dt) -> y

Backend of `step!(YelmoModel, dt)`: the predictor-corrector time loop
for `yelmo.dt_method` 0 (one step per call), 1 (Courant steps) or 2
(controller steps).
"""
function _select_step!(y, dt::Float64)
    method = Int(y.p.yelmo.dt_method)
    method in (0, 1, 2) ||
        error("step!: dt_method = $method is not recognised (supported: 0, 1, 2).")
    scheme     = _resolve_pc_scheme(y.p.yelmo.pc_method)
    controller = _resolve_pc_controller(y.p.yelmo.pc_controller)
    return _pc_loop!(y, dt, scheme, controller, _ensure_pc_scratch!(y))
end
