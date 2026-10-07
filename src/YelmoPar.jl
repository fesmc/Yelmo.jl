"""
    YelmoPar

Primary parameter module for the pure-Julia `YelmoModel`. This module owns the
generic `write_nml`, `read_nml`, and `compare` functions and the `*_params`
constructors; the Fortran-backed Mirror (`YelmoMirrorPar`) imports and extends
them for its own `YelmoMirrorParameters`, so users see a single generic
function across both backends.

The parameter schema follows Fortran Yelmo's `input/yelmo_defaults.nml`
(yelmo `dev`, `eda5462f`): every Fortran key is a field here, with the Fortran
default. Julia-only fields are listed in `JULIA_ONLY_KEYS`, each with the reason
it has no Fortran counterpart. `write_nml` always writes the complete namelist,
so `write_defaults_nml` produces a defaults file from Julia alone.

Options that exist in Fortran but are not yet ported are rejected by
`check_ported` when a `YelmoModel` is built (see `PORT_STATUS`).

Usage:
    using .YelmoPar
    p = YelmoParameters("experiment1";
        ydyn = YelmoPar.ydyn_params(solver="ssa"),
    )
    write_nml("run.nml", p)
"""
module YelmoPar

using ..YelmoSolvers: Solver, SSASolver

# Note on read_nml: YelmoPar.read_nml and YelmoMirrorPar.read_nml share the
# same signature `read_nml(::AbstractString)` but return different types, so
# they cannot be a single generic function. They live as separate functions in
# their respective module namespaces — call `YelmoMirrorPar.read_nml(...)` to
# get a `YelmoMirrorParameters`.

export YelmoParameters
export yelmo_params, ytopo_params, ycalv_params, ydyn_params,
       ytill_params, yhyd_params, ymat_params, ytrc_params, ytherm_params,
       yelmo_masks_params, yelmo_init_topo_params, yelmo_data_params
export write_nml, write_defaults_nml
export read_nml
export compare
export check_ported, with_ported_options

# ---------------------------------------------------------------------------
# &yelmo  (top-level Yelmo group)
# ---------------------------------------------------------------------------
Base.@kwdef struct YelmoParams
    domain              ::String  = "None"      # User par file must override this
    grid_name           ::String  = "None"      # User par file must override this
    grid_path           ::String  = "ice_data/{domain}/{grid_name}/{grid_name}_REGIONS.nc"
    phys_const          ::String  = "Earth"
    experiment          ::String  = "None"      # "None"/"EISMINT", "MISMIP3D", "MISMIP+", "TROUGH-F17", "SLAB", "ISMIPHOM", "periodic", ...
    mask_border         ::String  = "auto"      # Ice mask on the domain border: "auto", "none", "fixed", "dynamic"
    nml_ytopo           ::String  = "ytopo"
    nml_ycalv           ::String  = "ycalv"
    nml_ydyn            ::String  = "ydyn"
    nml_ytill           ::String  = "ytill"
    nml_ymat            ::String  = "ymat"
    nml_ytrc            ::String  = "ytrc"
    nml_ytherm          ::String  = "ytherm"
    nml_yhyd            ::String  = "yhyd"
    nml_masks           ::String  = "yelmo_masks"
    nml_init_topo       ::String  = "yelmo_init_topo"
    nml_data            ::String  = "yelmo_data"
    restart             ::String  = "None"
    restart_z_bed       ::Bool    = false       # Take z_bed (and z_bed_sd) from restart file
    restart_H_ice       ::Bool    = false       # Take H_ice from restart file
    restart_relax       ::Float64 = 1000.0      # [yrs] Years to relax from restart=>input topography
    log_timestep        ::Bool    = false
    log_mb_check        ::Bool    = false       # Print global mass-budget check (residual) every timestep
    disable_kill        ::Bool    = false       # Disable automatic kill if unstable
    zeta_scale          ::String  = "exp"       # "linear", "exp", "tanh"
    zeta_exp            ::Float64 = 2.0
    nz_aa               ::Int     = 10          # Vertical resolution in ice
    dt_method           ::Int     = 2           # 0: no internal timestep, 1: adaptive, cfl, 2: adaptive, pc
    dt_min              ::Float64 = 0.1         # [a] Minimum timestep
    cfl_max             ::Float64 = 0.1         # Maximum value is 1.0, lower will be more stable
    pc_method           ::String  = "AB-SAM"    # "FE-SBE", "AB-SAM", "HEUN"
    pc_controller       ::String  = "PI42"      # PI42, H312b, H312PID, H321PID, PID1
    pc_use_H_pred       ::Bool    = true        # Use predicted H_ice instead of corrected H_ice
    pc_filter_vel       ::Bool    = true        # Advect H_ice with mean of current and previous vel. solutions
    pc_n_redo           ::Int     = 5           # How many times can the same iteration be repeated (when high error exists)
    pc_tol              ::Float64 = 1.0         # [1/a] Redo the timestep when pc_eta > pc_tol
    pc_eps              ::Float64 = 0.02        # [1/a] Target pc_eta of the adaptive timestep (dt_method=2), <= pc_tol
    pc_cfl_max          ::Float64 = 0.5         # Courant-number cap on the pc adaptive timestep (dt_method=2)
    pc_rho_max          ::Float64 = 2.0         # Maximum growth factor dt_new/dt per step of the pc adaptive timestep
    pc_eta_H_min        ::Float64 = 10.0        # [m] Thinner ice is not included in the pc error norm
    pc_eta_u_min        ::Float64 = 0.0         # [m/yr] Slower ice is not included in the pc error norm
    pc_eta_trim         ::Float64 = 0.0         # [--] Fraction of points with the largest errors left out of the pc error norm
    write_metrics       ::Bool    = false       # Write numerics/speed metrics to yelmo_metrics.nc
    write_metrics_dt    ::Float64 = 100.0       # [yr] Output cadence for yelmo_metrics.nc
    # --- Julia-only (see JULIA_ONLY_KEYS) ---
    # Fortran's advective-only predictor-corrector (one `dyn_step!` per
    # substep, β-mixing on `dHidt_dyn`) when `true`; the legacy Yelmo.jl
    # path (two full `_step_fe!` cascades per substep) when `false`. Kept
    # until the reject-path symmetry regression at large `dt_outer` is
    # understood. See `src/timestepping.jl`.
    pc_advective        ::Bool    = false
    # Mask ice-margin / grounding-line / floating / thin-ice cells out of
    # the pc truncation error `eta` (Fortran `set_pc_mask` + `calc_pc_eta`).
    # `false` gives the unmasked global error (pre-2026-05-10 Yelmo.jl).
    pc_eta_masked       ::Bool    = true
    # Per-section wall-clock timing (`y.timer`, `src/timing.jl`).
    timing              ::Bool    = false
end
yelmo_params(; kwargs...) = YelmoParams(; kwargs...)
# ---------------------------------------------------------------------------
# &ytopo
# ---------------------------------------------------------------------------
Base.@kwdef struct YtopoParams
    solver              ::String  = "impl-lis"  # "none","expl","expl-upwind","impl-upwind","impl-lis", ...
    grad_lim            ::Float64 = 0.5         # [m/m] Maximum allowed slope in gradient calculations (dz/dx,dH/dx)
    grad_lim_zb         ::Float64 = 0.5         # [m/m] Maximum allowed slope in bed gradient (dzb/dx)
    slope_bg_x          ::Float64 = 0.0         # [m/m] Uniform background slope in x added to dzs/dx and dzb/dx
    slope_bg_y          ::Float64 = 0.0         # [m/m] Uniform background slope in y added to dzs/dy and dzb/dy
    front_subgrid       ::String  = "marine"    # Subgrid ice fronts: "none" (binary f_ice), "floating", "marine"
    front_H_eff_min     ::Float64 = 50.0        # [m] Minimum effective thickness of front cells (front_subgrid)
    front_dHdx          ::Float64 = 0.0         # [m/m] Thickness gradient assumed at a full front (front_subgrid)
    use_bmb             ::Bool    = true        # Use basal mass balance in mass conservation equation
    topo_fixed          ::Bool    = false       # Keep ice thickness fixed, perform other ytopo calculations
    topo_rel            ::Int     = 0           # 0: No relaxation; 1: relax shelf; 2: relax shelf + gl; 3: all points
    topo_rel_tau        ::Float64 = 10.0        # [a] Time scale for relaxation
    topo_rel_field      ::String  = "H_ref"     # "H_ref" or "H_ice_n"
    bmb_gl_method       ::String  = "pmp"       # "fcmp", "fmp", "pmp", "pmpt", "nmp"
    gl_sep              ::Int     = 1           # 1: Linear f_grnd_acx/acy and binary f_grnd, 2: area f_grnd
    gz_nx               ::Int     = 15          # [-] Number of interpolation points (nx*nx) for grounded area at the gl
    dist_grz            ::Float64 = 200.0       # [km] Radius of the "grounding-line zone" (grz)
    gz_Hg0              ::Float64 = 0.0         # Grounding zone, limit of penetration of bmb_grnd
    gz_Hg1              ::Float64 = 0.0         # Grounding zone, limit of penetration of bmb_shlf
    dmb_method          ::Int     = 0           # 0: no subgrid discharge, 1: subgrid discharge on
    dmb_alpha_max       ::Float64 = 60.0        # [deg] Maximum angle of slope from coast at which to allow discharge
    dmb_tau             ::Float64 = 100.0       # [yr]  Discharge timescale
    dmb_sigma_ref       ::Float64 = 300.0       # [m]   Reference bed roughness
    dmb_m_d             ::Float64 = 3.0         # [-]   Discharge distance scaling exponent
    dmb_m_r             ::Float64 = 1.0         # [-]   Discharge resolution scaling exponent
    fmb_method          ::Int     = 0           # 0: fmb_shlf; 1: fmb~bmb_shlf; 2: scaled by submerged front area; 3: Rignot et al. (2016)
    fmb_scale           ::Float64 = 1.0         # Scaling of fmb ~ scale*bmb
    fmb_lambda          ::Float64 = 1.0         # fmb_method=3: scaling of the Rignot et al. (2016) frontal melt
    # --- Julia-only (see JULIA_ONLY_KEYS) ---
    # Signed surface change `Δz_srf` across one periodic image in +x / +y,
    # added at the wrap face of the surface-gradient kernels, for periodic
    # benchmarks whose z_srf contains a uniform tilt (HOM-C:
    # `-tan(α) · Lx_m`). Config-time constant. Fortran dev instead keeps the
    # tilt out of z_srf/z_bed and adds `slope_bg_x/y` to the gradients;
    # this offset goes once the benchmarks are moved to slope_bg.
    dzsdx_periodic_offset ::Float64 = 0.0
    dzsdy_periodic_offset ::Float64 = 0.0
end
ytopo_params(; kwargs...) = YtopoParams(; kwargs...)
# ---------------------------------------------------------------------------
# &ycalv
# ---------------------------------------------------------------------------
"""
Calving methods that are supported by Yelmo.jl's `_dispatch_calving!`
(see `src/topo/calving.jl`). Used by `ycalv_params` to fail fast at
parameter construction when `use_lsf = true` and the requested
method is not implemented in the Julia port.
"""
const SUPPORTED_CALV_METHODS = ("none", "zero", "equil", "threshold", "vm-m16", "custom")

"""
Calving methods that exist in Fortran Yelmo (`yelmo/src/yelmo_topography.f90`)
but are not yet ported to Yelmo.jl. Listed separately so the validator
can produce a "known but unported" error message — distinct from the
"unrecognised method" error, which signals a typo or namelist drift.
"""
const KNOWN_UNPORTED_CALV_METHODS = ("vm-l19", "eigen", "simple", "flux", "kill", "kill-pos",
                                     "stress-b12", "ismip7", "exp1", "exp2", "exp3", "exp4", "exp5")

Base.@kwdef struct YcalvParams
    use_lsf             ::Bool    = true        # use level-set method
    lsf_method          ::String  = "snap"      # "snap" (neighbour-snap) or "redist" (Sussman/Osher)
    dt_lsf              ::Float64 = -1.0        # [yr] periodic LSF reflag interval; <= 0 disables ("snap" only)
    lsf_redist_n_iter   ::Int     = 5           # Sussman/Osher LSF redistancing iterations ("redist" only)
    calv_flt_method     ::String  = "vm-m16"    # use_lsf=T: "zero"/"none","equil","threshold","vm-m16","exp1"-"exp5"
    calv_grnd_method    ::String  = "vm-m16"    # use_lsf=T: "zero"/"none","equil","threshold","vm-m16","ismip7"
    H_min_grnd          ::Float64 = 5.0         # [m] Minimum ice thickness at grounded margin (thinner ice is ablated)
    H_min_flt           ::Float64 = 10.0        # [m] Minimum ice thickness at floating margin (thinner ice is ablated)
    H_min_tau           ::Float64 = 10.0        # [yr] Timescale for removing margin ice thinner than H_min_* and isolated partial cells
    sd_min              ::Float64 = 100.0       # [m] calv_grnd(z_bed_sd <= sd_min) = 0.0
    sd_max              ::Float64 = 500.0       # [m] calv_grnd(z_bed_sd >= sd_max) = calv_max
    calv_grnd_max       ::Float64 = 0.0         # [m/a] Maximum grounded calving rate from high stdev(z_bed)
    calv_tau            ::Float64 = 1.0         # [a] Characteristic calving time
    calv_thin           ::Float64 = 30.0        # [m/yr] Calving rate for very thin ice (use_lsf=False, vm-l19/eigen)
    k2                  ::Float64 = 3.2e9       # [m yr] eigen calving scaling factor
    w2                  ::Float64 = 0.0         # Weighting coefficient of 2nd principal strain/stress
    kt_ref              ::Float64 = 0.0025      # [m yr-1 Pa-1] vm-l19 calving scaling parameter
    kt_deep             ::Float64 = 0.1         # [m yr-1 Pa-1] vm-l19 calving scaling parameter for deep ocean
    tau_ice_flt         ::Float64 = 250.0e3     # [Pa] Ice strength failure, floating fronts (vm-m16)
    tau_ice_grnd        ::Float64 = 1.0e6       # [Pa] Ice strength failure, marine-grounded fronts (vm-m16)
    Hc_ref_flt          ::Float64 = 200.0       # [m] Calving limit in ice thickness - floating
    Hc_ref_grnd         ::Float64 = 200.0       # [m] Calving limit in ice thickness - marine-terminating
    Hc_ref_thin         ::Float64 = 50.0        # [m] Reference ice thickness for thin ice calving
    Hc_deep             ::Float64 = 500.0       # [m] Calving limit in ice thickness (thinner ice calves)
    zb_deep_0           ::Float64 = -1000.0     # [m] Bedrock elevation to begin transition to deep ocean
    zb_deep_1           ::Float64 = -1500.0     # [m] Bedrock elevation to end transition to deep ocean
    zb_sigma            ::Float64 = 0.0         # [m] Gaussian filtering of bedrock for calving transition to deep ocean
end
"""
    _validate_calv_method(method, label)

Throw a descriptive error if `method` is not in
`SUPPORTED_CALV_METHODS`. The two failure modes are reported
separately:

  - `method ∈ KNOWN_UNPORTED_CALV_METHODS` — a real Fortran-Yelmo
    method that has not yet been ported to Yelmo.jl. The error
    points at the unported list.
  - Otherwise — likely a typo or namelist drift; the error lists
    the supported set.
"""
function _validate_calv_method(method::AbstractString, label::AbstractString)
    method in SUPPORTED_CALV_METHODS && return nothing
    if method in KNOWN_UNPORTED_CALV_METHODS
        error("ycalv_params: $label = \"$method\" is a Fortran-Yelmo " *
              "calving method that has not been ported to Yelmo.jl. " *
              "Supported here: $(SUPPORTED_CALV_METHODS).")
    else
        error("ycalv_params: $label = \"$method\" is not recognised. " *
              "Supported: $(SUPPORTED_CALV_METHODS); " *
              "known-but-unported (Fortran-only): $(KNOWN_UNPORTED_CALV_METHODS).")
    end
end

# Factory function: validates calving method names before returning the
# struct. Validation only fires when `use_lsf = true` — otherwise the
# calving methods are dormant.
function ycalv_params(; kwargs...)
    p = YcalvParams(; kwargs...)
    if p.use_lsf
        _validate_calv_method(p.calv_flt_method,  "calv_flt_method")
        _validate_calv_method(p.calv_grnd_method, "calv_grnd_method")
    end
    return p
end
# ---------------------------------------------------------------------------
# &ydyn
# ---------------------------------------------------------------------------
Base.@kwdef struct YdynParams
    solver              ::String  = "diva"      # "fixed", "sia", "ssa", "hybrid", "diva", "diva-noslip"
    uz_method           ::Int     = 3           # 1: aa-staggering, 2: strain-rate-on-nodes, 3: jacobian, 4: flux-consistent
    visc_method         ::Int     = 1           # 0: constant visc=visc_const, 1: dynamic (quadrature points), 2: dynamic (aa-nodes)
    visc_const          ::Float64 = 1e7         # [Pa a] Constant value for viscosity (if visc_method=0)
    beta_method         ::Int     = 1           # -1: external; 0: constant; 1: linear; 2: pseudo-plastic; 3: reg. Coulomb; 4, 5: as 2, 3 at cell centre
    beta_const          ::Float64 = 1e3         # [Pa a m-1] Constant value of basal friction coefficient
    beta_q              ::Float64 = 1.0         # Dragging law exponent
    beta_u0             ::Float64 = 100.0       # [m/a] Speed scale of the friction law (beta_method=1-5)
    beta_gl_scale       ::Int     = 0           # 0: beta*beta_gl_f, 1: H_grnd linear, 2: Zstar, 3: beta*f_grnd on aa-nodes
    beta_gl_stag        ::Int     = 1           # -1: external; 0: simple; 1: upstream; 2: downstream; 3: f_grnd_ac; 4: Gladstone B2
    beta_gl_f           ::Float64 = 1.0         # [-] Scaling of beta at the grounding line (for beta_gl_scale=0)
    taud_gl_method      ::Int     = 0           # 0: binary; 1: f_grnd-weighted; 2: one-sided (Feldmann 2014); 3: linear (Gladstone 2010)
    H_grnd_lim          ::Float64 = 500.0       # [m] For beta_gl_scale=1, reduce beta linearly between H_grnd=0 and H_grnd_lim
    beta_min            ::Float64 = 100.0       # [Pa a m-1] Minimum value of beta allowed for grounded ice
    eps_0               ::Float64 = 1e-6        # [1/a] Regularization term for effective viscosity - minimum strain rate
    frz_scale           ::Bool    = true        # Reduce sliding where the bed is below the pmp: beta*f**(-q)
    frz_efold           ::Float64 = 3.0         # [K] e-folding temperature of the sliding speed
    frz_min             ::Float64 = 1e-3        # [-] Minimum sliding-speed factor for frozen beds
    # Fortran key `ssa_solver` ("residual" | "energy"), held as the
    # Julia-native `SSASolver`: `method` is the Fortran choice ("energy" ↔
    # `:energy_quadratic`), the other fields configure the Krylov/AMG
    # linear solve that replaces Lis. Written to the namelist as
    # `ssa_solver` plus the Julia-only `ssa_solver_*` keys (see
    # `_nml_entries(::SSASolver)`). The Picard iteration is set by the
    # Fortran keys `ssa_iter_*` below.
    ssa_solver          ::SSASolver = SSASolver()
    ssa_lis_opt_residual::String  = "-i minres -p jacobi -maxiter 100 -tol 1.0e-2 -initx_zeros false"  # Lis options (Fortran only)
    ssa_lis_opt_energy  ::String  = "-i cg -p jacobi -maxiter 200 -tol 1.0e-2 -initx_zeros false"      # Lis options (Fortran only)
    ssa_lat_bc          ::String  = "marine"    # "all","marine","floating","float","none"
    ssa_vel_lim_method  ::String  = "drag"      # "clip": clip each component at ssa_vel_max; "drag": smooth speed-limit drag
    ssa_vel_max         ::Float64 = 1e4         # [m a-1] Velocity limit
    ssa_vel_lim_tau     ::Float64 = 1e5         # [Pa] Speed-limit drag at ssa_vel_max (ssa_vel_lim_method="drag")
    ssa_iter_max        ::Int     = 20          # Maximum Picard iterations of the velocity solution
    ssa_iter_rel        ::Float64 = 0.7         # [--] Picard relaxation fraction [0:1]
    ssa_iter_conv       ::Float64 = 1e-2        # [--] L2 relative error convergence limit of the Picard iteration
    taud_lim            ::Float64 = 2e5         # [Pa] Maximum allowed driving stress
    neff_nxi            ::Int     = 0           # Subgrid interpolation of hyd%N onto dyn%N_eff. 0: none, 1: Gaussian quadrature, >1: nxi x nxi
end
ydyn_params(; kwargs...) = YdynParams(; kwargs...)
# ---------------------------------------------------------------------------
# &ytill
# ---------------------------------------------------------------------------
Base.@kwdef struct YtillParams
    method              ::Int     = 1           # -1: set externally; 1: calculate cb_ref online
    scale_zb            ::Int     = 1           # 0: none, 1: lin, 2: exp : scaling with elevation
    scale_sed           ::Int     = 0           # 0: none, 1: min(cb_zb,cb_sed), 2: cb_zb*(lambda_sed*f_sed), 3: 2 but no lower limit
    is_angle            ::Bool    = false       # cf_ref/cf_min are till strength angle?
    n_sd                ::Int     = 10          # Number of samples over z_bed_sd field
    f_sed               ::Float64 = 0.01        # Scaling reduction for thick sediments
    sed_min             ::Float64 = 5.0         # [m] Sediment thickness for no reduction in friction
    sed_max             ::Float64 = 15.0        # [m] Sediment thickness for maximum reduction in friction
    z0                  ::Float64 = -300.0      # [m] Bedrock rel. to sea level, lower limit
    z1                  ::Float64 = 200.0       # [m] Bedrock rel. to sea level, upper limit
    cf_min              ::Float64 = 0.1         # [-- or deg] Minimum value of cf
    cf_ref              ::Float64 = 0.8         # [-- or deg] Reference/const/max value of cf
end
ytill_params(; kwargs...) = YtillParams(; kwargs...)
# ---------------------------------------------------------------------------
# &yhyd  (basal hydrology; Fortran: FastHydrology)
#
# YelmoModel runs the till-water bucket (`method_til = 1`) with the N
# closures below; the K24 transport model (`method_transport = 1`,
# `k24_*`) is not ported (FastHydrology.jl will provide it).
# ---------------------------------------------------------------------------
Base.@kwdef struct YhydParams
    method_til          ::Int     = 1           # 0=NONE 1=BUCKET
    method_transport    ::Int     = 0           # 0=NONE 1=K24
    W_til_max           ::Float64 = 2.0         # [m] Maximum till water thickness (bucket capacity)
    mask_bc             ::Int     = 2           # 0=ZERO 1=IMPOSED 2=MIRROR
    W_til_bc            ::Float64 = 0.0         # [m]
    bkt_N_closure       ::Int     = 3           # -1=EXTERNAL 0=CONST 1=OVERBURDEN 2=MARINE 3=TILL 4=TWO_VALUE
    bkt_till_rate       ::Float64 = 1e-3        # [m/a] Till water drainage rate
    bkt_floating_mode   ::Int     = 0           # 0=ZERO 1=MARGIN_FILL (saturate floating + margin ring)
    const_N             ::Float64 = 1e7         # [Pa] N for bkt_N_closure=0
    marine_p            ::Float64 = 1.0         # Marine closure (Leguy 2014) exponent p
    marine_rho_sw       ::Float64 = 1028.0      # [kg/m3] ignored: the closure uses the model's rho_sw (YelmoConstants)
    till_N0             ::Float64 = 1000.0      # [Pa] Till closure (van Pelt & Bueler 2015) reference N
    till_delta          ::Float64 = 0.04        # [-] Till closure minimum N fraction of overburden
    till_e0             ::Float64 = 0.69        # [-] Till closure reference void ratio
    till_Cc             ::Float64 = 0.12        # [-] Till closure compressibility
    two_value_delta     ::Float64 = 0.02        # [-] Two-value closure: N fraction of overburden at temperate beds
    k24_substrate_type  ::Int     = 2           # 0=HARD 1=SOFT 2=MIXED
    k24_flux_solver     ::Int     = 3           # 0=RECURSIVE 1=ITERATIVE 2=TOPOSORT 3=TAPED
    k24_routing_scheme  ::Int     = 0           # 0=WARNER 1=GDS_WARNER 2=QUINN 3=TARBOTON 4=MODIFIED_TARBOTON 5=GDS_TARBOTON
    k24_quinn_original  ::Bool    = false       # QUINN only: Quinn et al.'s own slope x contour-length weights
    k24_fill_algorithm  ::Int     = -1          # -1=AUTO 0=JACOBI 1=LOWEST_NEIGHBOUR 2=PRIORITY_FLOOD
    k24_priority_flood_epsilon ::Float64 = 1.0  # [Pa] potential step across a filled pit/flat
    k24_q_conversion    ::Int     = -1          # -1=AUTO 0=OUTFLOW 1=FACE_AVERAGE
    k24_dissipation_discretization ::Int = -1   # -1=AUTO 0=CELL 1=FACE
    k24_friction_discretization ::Int = 0       # 0=CELL 1=STAGGERED
    k24_friction_quadrature ::Bool = false      # STAGGERED only: Gauss-point heat
    k24_friction_u_floor ::Float64 = 3.168808781402895e-11  # [m/s] STAGGERED beta floor
    k24_toposort_allow_cycles ::Bool = false
    k24_ub_hook         ::Bool    = true        # DIVA velocity iteration re-evaluates K24 N from u_b
    k24_drainage_mode   ::Int     = 0           # 0=BOTH 1=EFFICIENT_ONLY 2=INEFFICIENT_ONLY
    k24_water_thickness_algorithm ::Int = 0     # 0=DARCY_WEISBACH 1=LAMINAR 2=AREAL_CONDUIT
    k24_gradient_convention ::Int = 0           # 0=MEAN 1=LOCAL
    k24_sliding_law     ::Int     = 0           # 0=NO_FRICTION 1=WEERTMAN 2=POWER_PLASTIC 3=REG_COULOMB 4=PRESCRIBED_FIELD 5=REG_COULOMB_FIELD 6=SHAKTI_REG_COULOMB
    k24_manning_exponent ::Float64 = 3.0
    k24_latent_heat_water ::Float64 = 3.335e5   # [J/kg]
    k24_bed_thickness   ::Float64 = 0.1         # [m]
    k24_manning_coefficient_exponent ::Float64 = 1.25
    k24_bed_friction_exponent ::Float64 = 1.5
    k24_friction_factor ::Float64 = 0.1
    k24_till_factor     ::Float64 = 1.1
    k24_critical_discharge ::Float64 = 1.0      # [m3/s]
    k24_initial_cavity_height ::Float64 = 0.1   # [m]
    k24_coupling_length ::Float64 = 1e4         # [m]
    k24_coupling_length_kamb86 ::Float64 = 10.0
    k24_eta_w           ::Float64 = 5.703855806525211e-11  # [Pa s]
    k24_min_pressure_fraction ::Float64 = 0.0
    k24_W_min           ::Float64 = 0.0         # [m]
    k24_W_max           ::Float64 = 1e30        # [m]
    k24_q_min           ::Float64 = 0.0         # [m2/s]
    k24_q_max           ::Float64 = 1e30        # [m2/s]
    k24_fill_iters      ::Int     = 10
    k24_max_psi_out_calls ::Int   = 100000
    k24_dissipation_melt ::Bool   = true
    k24_max_dissipation_iters ::Int = 20
    k24_dissipation_rtol ::Float64 = 1e-12
    k24_dissipation_verbose ::Bool = true
    k24_max_coupling_iters ::Int = 20
    k24_coupling_rtol   ::Float64 = 1e-8
    k24_coupling_verbose ::Bool   = true
    k24_weertman_C      ::Float64 = 0.0
    k24_weertman_q      ::Float64 = 1/3
    k24_power_plastic_c_till ::Float64 = 0.0
    k24_power_plastic_q ::Float64 = 1.0
    k24_power_plastic_u0 ::Float64 = 3.168808781402895e-6  # [m/s]
    k24_reg_coulomb_c_till ::Float64 = 0.0
    k24_reg_coulomb_q   ::Float64 = 1/3
    k24_reg_coulomb_u0  ::Float64 = 3.168808781402895e-6  # [m/s]
    k24_shakti_C        ::Float64 = 0.0
    k24_shakti_n        ::Float64 = 3.0
    k24_shakti_lambda_coeff ::Float64 = 1.5
end
yhyd_params(; kwargs...) = YhydParams(; kwargs...)
# ---------------------------------------------------------------------------
# &ymat
# ---------------------------------------------------------------------------
Base.@kwdef struct YmatParams
    flow_law            ::String  = "glen"      # Only "glen" is possible right now
    rf_method           ::Int     = 1           # -1: set externally; 0: rf_const everywhere; 1: standard function
    rf_const            ::Float64 = 1e-18       # [Pa^-3 a^-1]
    rf_use_eismint2     ::Bool    = false       # Only applied for rf_method=1
    rf_with_water       ::Bool    = false       # Only applied for rf_method=1, scale rf by water content?
    n_glen              ::Float64 = 3.0         # Glen flow law exponent
    visc_min            ::Float64 = 1e3         # [Pa a] Minimum allowed viscosity
    de_max              ::Float64 = 100.0       # [a-1] Maximum allowed effective strain rate (strain heating, mat viscosity)
    enh_method          ::String  = "shear3D"   # "simple","shear2D","shear3D" (+ "-tracer" variants)
    enh_shear           ::Float64 = 3.0
    enh_stream          ::Float64 = 3.0
    enh_shlf            ::Float64 = 0.7
    enh_umin            ::Float64 = 50.0        # [m/yr] Minimum transition velocity to enh_stream ('*-tracer' enh methods)
    enh_umax            ::Float64 = 500.0       # [m/yr] Maximum transition velocity to enh_stream ('*-tracer' enh methods)
    tracer_method       ::String  = "expl"      # "expl", "impl": Eulerian solver for the '*-tracer' enh_bnd field
    tracer_impl_kappa   ::Float64 = 1.5         # [m2 a-1] Artificial diffusion for implicit enh_bnd solving
end
ymat_params(; kwargs...) = YmatParams(; kwargs...)
# ---------------------------------------------------------------------------
# &ytrc  (age / deposition-time tracers)
# ---------------------------------------------------------------------------
Base.@kwdef struct YtrcParams
    use_euler           ::Bool    = false       # Run the in-tree Eulerian age tracer?
    use_tracer          ::Bool    = false       # Run the Lagrangian particle backend (tracer)?
    use_elsa            ::Bool    = false       # Run the Lagrangian layer backend (elsa)?
    elsa_restart        ::Bool    = true        # On a restart, restore elsa's layers
    t_dep_source        ::String  = "euler"     # "euler", "trc", "elsa": authoritative deposition-time source
    time_end            ::Float64 = 0.0         # [yr] simulation end time for use_elsa
    calc_age            ::Bool    = false       # Calculate the Eulerian age tracer field?
    time_iso            ::Vector{Float64} = [-11.7, -29.0, -57.0, -115.0]  # [ka] isochrone deposition times
    tracer_method       ::String  = "expl"      # "expl", "impl": Eulerian age solver
    tracer_impl_kappa   ::Float64 = 1.5         # [m2 a-1] Artificial diffusion for implicit age solving
    elsa_nml            ::String  = "None"      # Path to elsa namelist file
    elsa_group          ::String  = "None"      # Group name within elsa_nml
    tracer_nml          ::String  = "None"      # Path to tracer namelist file (group "trc")
end
ytrc_params(; kwargs...) = YtrcParams(; kwargs...)
# ---------------------------------------------------------------------------
# &ytherm
# ---------------------------------------------------------------------------
Base.@kwdef struct YthermParams
    method              ::String  = "enth"      # "enth","temp","robin","robin-cold","linear","fixed"
    qb_method           ::Int     = 2           # 1: faces, 2: faces to quadrature nodes, 3: simple-stagger, 4: quadrature
    dt_method           ::String  = "FE"        # "FE", "AB", "SAM"
    solver_advec        ::String  = "impl-upwind"  # "expl", "impl-upwind"
    advecxy_order       ::Int     = 2           # Horizontal advection order: 1=upwind, 2=flux-limited 2nd-order upwind
    advecxy_cfl         ::Float64 = 0.5         # [--] Target Courant number per horizontal-advection sub-step
    advecxy_nmax        ::Int     = 10          # [--] Max horizontal-advection sub-steps
    gamma               ::Float64 = 1.0         # [K] Scalar for the pressure melting point decay function
    strain_heating      ::String  = "full"      # "full": 3D viscosity and strain rate, "sia": SIA approx., "none"
    use_const_cp        ::Bool    = false       # Use specified constant value of heat capacity?
    const_cp            ::Float64 = 2009.0      # [J kg-1 K-1] Specific heat capacity
    use_const_kt        ::Bool    = false       # Use specified constant value of heat conductivity?
    const_kt            ::Float64 = 6.62e7      # [J a-1 m-1 K-1] Thermal conductivity
    enth_cr             ::Float64 = 1e-3        # [--] Conductivity ratio for temperate ice
    omega_max           ::Float64 = 0.01        # [--] Maximum allowed water content fraction
    H_ice_thin          ::Float64 = 10.0        # [m] Skip column solver below this thickness (linear profile imposed)
    enth_cp_method      ::String  = "integral"  # Enthalpy heat capacity: "const" (cp_ref) or "integral" (int cp(T) dT)
    basal_bc_method     ::String  = "capacity"  # Grounded basal BC: "capacity" or "wtil" (till-water predictor, deprecated)
    cap_source          ::String  = "auto"      # [capacity] C from: "auto", "hyd", "till", "water", "none"
    cap_W_floor         ::Float64 = 0.0         # [m] [capacity, "till"/"water"] floor subtracted from the water thickness
    cap_eps             ::Float64 = 1e-4        # [m/a ice equiv.] [capacity] C below this counts as a dry bed
    gl_temperate        ::Bool    = true        # Hold grounded bases next to floating ice or open ocean at the pmp
    rock_method         ::String  = "equil"     # "equil" (not active bedrock), "active", or "fixed"
    nzr_aa              ::Int     = 5           # Number of vertical points in bedrock
    zeta_scale_rock     ::String  = "exp-inv"   # "linear", "exp-inv"
    zeta_exp_rock       ::Float64 = 2.0
    H_rock              ::Float64 = 2000.0      # [m] Lithosphere thickness
    rhoc_rock           ::Float64 = 2.0e6       # [J m-3 K-1] Volumetric heat capacity of bedrock (rho*cp)
    kt_rock             ::Float64 = 6.3e7       # [J a-1 m-1 K-1] Thermal conductivity of bedrock
end
ytherm_params(; kwargs...) = YthermParams(; kwargs...)
# ---------------------------------------------------------------------------
# &yelmo_masks
# ---------------------------------------------------------------------------
Base.@kwdef struct YelmoMasksParams
    basins_load         ::Bool    = true
    basins_path         ::String  = "ice_data/{domain}/{grid_name}/{grid_name}_BASINS-nasa.nc"
    basins_nms          ::Vector{String} = ["basin", "basin_mask"]
    regions_load        ::Bool    = true
    regions_path        ::String  = "ice_data/{domain}/{grid_name}/{grid_name}_REGIONS.nc"
    regions_nms         ::Vector{String} = ["mask", "None"]
end
yelmo_masks_params(; kwargs...) = YelmoMasksParams(; kwargs...)
# ---------------------------------------------------------------------------
# &yelmo_init_topo
# ---------------------------------------------------------------------------
Base.@kwdef struct YelmoInitTopoParams
    init_topo_load      ::Bool    = true
    init_topo_path      ::String  = "ice_data/{domain}/{grid_name}/{grid_name}_TOPO-M17.nc"
    init_topo_names     ::Vector{String} = ["H_ice", "z_bed", "z_bed_sd", "z_srf"]
    init_topo_state     ::Int     = 0           # 0: from file, 1: ice-free, 2: ice-free, rebounded
    z_bed_f_sd          ::Float64 = -1.0        # Scaling fraction to modify z_bed = z_bed + f_sd*z_bed_sd
    smooth_H_ice        ::Float64 = 0.0         # Smooth ice thickness field at loading time, with sigma=N*dx
    smooth_z_bed        ::Float64 = 0.0         # Smooth bedrock field at loading time, with sigma=N*dx
end
yelmo_init_topo_params(; kwargs...) = YelmoInitTopoParams(; kwargs...)
# ---------------------------------------------------------------------------
# &yelmo_data
# ---------------------------------------------------------------------------
Base.@kwdef struct YelmoDataParams
    pd_topo_load        ::Bool    = true
    pd_topo_path        ::String  = "ice_data/{domain}/{grid_name}/{grid_name}_TOPO-M17.nc"
    pd_topo_names       ::Vector{String} = ["H_ice", "z_bed", "z_bed_sd", "z_srf"]
    pd_tsrf_load        ::Bool    = true
    pd_tsrf_path        ::String  = "ice_data/{domain}/{grid_name}/{grid_name}_MARv3.11-ERA_annmean_1961-1990.nc"
    pd_tsrf_name        ::String  = "T_srf"     # Surface temperature (or near-surface temperature)
    pd_tsrf_monthly     ::Bool    = false
    pd_smb_load         ::Bool    = true
    pd_smb_path         ::String  = "ice_data/{domain}/{grid_name}/{grid_name}_MARv3.11-ERA_annmean_1961-1990.nc"
    pd_smb_name         ::String  = "smb"       # Surface mass balance
    pd_smb_monthly      ::Bool    = false
    pd_vel_load         ::Bool    = true
    pd_vel_path         ::String  = "ice_data/{domain}/{grid_name}/{grid_name}_VEL-J18.nc"
    pd_vel_names        ::Vector{String} = ["ux_srf", "uy_srf"]
    pd_age_load         ::Bool    = false
    pd_age_path         ::String  = "ice_data/{domain}/{grid_name}/{grid_name}_STRAT-M15.nc"
    pd_age_names        ::Vector{String} = ["age_iso", "depth_iso"]
    pd_age_to_time      ::Bool    = true        # Convert loaded isochrone ages [ka] to deposition times (t_dep = -age)
end
yelmo_data_params(; kwargs...) = YelmoDataParams(; kwargs...)
# Note: physical constants (rho_ice, rho_sw, g, ...) live in the `YelmoConst`
# module (`YelmoConstants`), shared across multi-domain runs. Read them from
# `y.c` on a constructed `YelmoModel`.

# ---------------------------------------------------------------------------
# Julia-only keys
# ---------------------------------------------------------------------------
"""
    JULIA_ONLY_KEYS

Parameters of `YelmoParameters` that have no key in Fortran's
`yelmo_defaults.nml`, by group, with the reason. `ssa_solver_*` are the
Julia-native linear-solver settings of `ydyn.ssa_solver` (written as extra
keys next to the Fortran `ssa_solver` choice).
"""
const JULIA_ONLY_KEYS = Dict(
    "yelmo" => Dict(
        "pc_advective"  => "Fortran's advective-only PC vs. the legacy two-cascade Yelmo.jl PC (until the reject-path regression is understood)",
        "pc_eta_masked" => "switch off the Fortran pc error mask (diagnostics)",
        "timing"        => "per-section wall-clock timing of YelmoModel"),
    "ytopo" => Dict(
        "dzsdx_periodic_offset" => "periodic-wrap correction for tilted z_srf; replaced by slope_bg_x once the benchmarks move to it",
        "dzsdy_periodic_offset" => "periodic-wrap correction for tilted z_srf; replaced by slope_bg_y once the benchmarks move to it"),
    "ydyn" => Dict(
        "ssa_solver_linear_method" => "Krylov method of the SSA linear solve (Fortran: Lis options)",
        "ssa_solver_precond"       => "preconditioner of the SSA linear solve (Fortran: Lis options)",
        "ssa_solver_smoother"      => "AMG smoother of the SSA linear solve",
        "ssa_solver_rtol"          => "relative tolerance of the SSA linear solve (Fortran: Lis options)",
        "ssa_solver_itmax"         => "maximum iterations of the SSA linear solve (Fortran: Lis options)"),
)

# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------
"""
    YelmoParameters
Top-level container holding one struct per namelist group. Construct via
`YelmoParameters(name; kwargs...)` to override individual groups.
"""
struct YelmoParameters
    name            ::String
    yelmo           ::YelmoParams
    ytopo           ::YtopoParams
    ycalv           ::YcalvParams
    ydyn            ::YdynParams
    ytill           ::YtillParams
    yhyd            ::YhydParams
    ymat            ::YmatParams
    ytrc            ::YtrcParams
    ytherm          ::YthermParams
    yelmo_masks     ::YelmoMasksParams
    yelmo_init_topo ::YelmoInitTopoParams
    yelmo_data      ::YelmoDataParams
end

# Namelist group order (= Fortran yelmo_defaults.nml).
const GROUPS = (:yelmo, :ytopo, :ycalv, :ydyn, :ytill, :yhyd, :ymat, :ytrc, :ytherm,
                :yelmo_masks, :yelmo_init_topo, :yelmo_data)

"""
    YelmoParameters(name; yelmo, ytopo, ...) -> YelmoParameters
Construct a `YelmoParameters` object. Any group can be supplied as a keyword
argument; omitted groups are filled with defaults. `name` is a label for the
parameter set and the stem of the output filename.
# Example
```julia
p = YelmoParameters("experiment1";
    ydyn = ydyn_params(solver="ssa"),
)
write_nml("run.nml", p)
```
"""
function YelmoParameters(name;
    yelmo           = yelmo_params(),
    ytopo           = ytopo_params(),
    ycalv           = ycalv_params(),
    ydyn            = ydyn_params(),
    ytill           = ytill_params(),
    yhyd            = yhyd_params(),
    ymat            = ymat_params(),
    ytrc            = ytrc_params(),
    ytherm          = ytherm_params(),
    yelmo_masks     = yelmo_masks_params(),
    yelmo_init_topo = yelmo_init_topo_params(),
    yelmo_data      = yelmo_data_params(),
)
    return YelmoParameters(
        name, yelmo, ytopo, ycalv, ydyn, ytill, yhyd, ymat, ytrc, ytherm,
        yelmo_masks, yelmo_init_topo, yelmo_data,
    )
end

function YelmoParameters(filename, name)
    p = read_nml(filename)
    return YelmoParameters(name, (getfield(p, g) for g in GROUPS)...)
end

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
"""
    format_value(v) -> String
Format a Julia value for Fortran namelist syntax.
"""
format_value(v::Bool)              = v ? "True" : "False"
format_value(v::AbstractString)    = "\"$(v)\""
format_value(v::Symbol)            = "\"$(v)\""
format_value(v::Int)               = string(v)
format_value(v::Float64)           = _fmt_float(v)
format_value(v::AbstractVector{<:AbstractString}) = join(["\"$s\"" for s in v], " ")
format_value(v::AbstractVector{<:Real})            = join(format_value.(v), ", ")
"""
    _fmt_float(x) -> String
Shortest representation that reads back to exactly `x` (Julia's `repr`,
e.g. `0.1`, `1.0e7`, `3.168808781402895e-11`); Fortran reads all of these.
"""
_fmt_float(x::Float64) = repr(x)

# ---------------------------------------------------------------------------
# Namelist entries of a field: `name => value` pairs. Plain values give one
# entry; Julia-native config objects (`SSASolver`) give the Fortran key plus
# their Julia-only `<name>_<field>` keys.
# ---------------------------------------------------------------------------

# Fortran `ssa_solver` choice ↔ `SSASolver.method`.
const SSA_METHOD_NML = Dict(:residual => "residual", :energy_quadratic => "energy",
                            :energy_nonlinear => "energy_nonlinear")
const SSA_METHOD_JL  = Dict(v => k for (k, v) in SSA_METHOD_NML)
const SSA_NML_FIELDS = (:linear_method, :precond, :smoother, :rtol, :itmax)

_nml_entries(name::Symbol, v) = (string(name) => v,)
_nml_entries(name::Symbol, v::SSASolver) =
    (string(name) => SSA_METHOD_NML[v.method],
     (string(name, "_", f) => getfield(v, f) for f in SSA_NML_FIELDS)...)

_from_nml(::Type{T}, name::Symbol, d::Dict{String,String}, default) where {T} =
    haskey(d, string(name)) ? parse_nml_value(T, d[string(name)]) : default

function _from_nml(::Type{SSASolver}, name::Symbol, d::Dict{String,String}, default::SSASolver)
    kw = Dict{Symbol,Any}()
    key = string(name)
    if haskey(d, key)
        s = parse_nml_value(String, d[key])
        haskey(SSA_METHOD_JL, s) || error("read_nml: ydyn.$key = \"$s\"; expected one of " *
                                          "$(sort(collect(keys(SSA_METHOD_JL)))).")
        kw[:method] = SSA_METHOD_JL[s]
    end
    for f in SSA_NML_FIELDS
        k = string(name, "_", f)
        haskey(d, k) || continue
        T = fieldtype(SSASolver, f)
        kw[f] = T === Symbol ? Symbol(parse_nml_value(String, d[k])) : parse_nml_value(T, d[k])
    end
    return SSASolver(; (f => getfield(default, f) for f in fieldnames(SSASolver))..., kw...)
end

"""
    write_group(io, group_name, s)
Write one namelist group to `io` from struct `s`.
"""
function write_group(io::IO, group_name::AbstractString, s)
    println(io, "&$(group_name)")
    for fname in fieldnames(typeof(s))
        for (k, v) in _nml_entries(fname, getfield(s, fname))
            println(io, "    $(rpad(k, 24)) = $(format_value(v))")
        end
    end
    println(io, "/\n")
end
"""
    write_nml(filename, p::YelmoParameters)
Write the complete Yelmo namelist of `p` (every Fortran key and the Julia-only
keys of `JULIA_ONLY_KEYS`).
"""
function write_nml(filename::AbstractString, p::YelmoParameters; overwrite::Bool=false)
    if isfile(filename) && !overwrite
        error("File already exists: $(filename). Use overwrite=true to overwrite.")
    end
    open(filename, "w") do io
        for g in GROUPS
            write_group(io, string(g), getfield(p, g))
        end
    end
    @info "Namelist written to $(filename)"
    return nothing
end
function write_nml(p::YelmoParameters; rundir::String="", overwrite::Bool=false)
    filename = joinpath(rundir, p.name * ".nml")
    write_nml(filename, p; overwrite)
    return nothing
end

"""
    write_defaults_nml(filename; overwrite=false)

Write the default `YelmoParameters` as a complete namelist: the Julia
counterpart of Fortran's `input/yelmo_defaults.nml`.
"""
write_defaults_nml(filename::AbstractString; overwrite::Bool=false) =
    write_nml(filename, YelmoParameters("yelmo_defaults"); overwrite)

### READING NML FILES ###

# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------

"""
    parse_nml_file(filename) -> Dict{String, Dict{String, String}}

Low-level parser. Returns a two-level dict:
    group_name => (field_name => raw_value_string)
Handles line continuation, inline comments, and multi-line values.
"""
function parse_nml_file(filename::AbstractString)
    groups = Dict{String, Dict{String, String}}()
    current_group = nothing
    current_key   = nothing
    current_val   = nothing

    for raw_line in eachline(filename)
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, '!') && continue          # comment line

        # Strip inline comments (outside of quoted strings)
        line = _strip_inline_comment(line)
        isempty(line) && continue

        # &group_name
        if startswith(line, '&')
            # Flush any pending continuation
            if current_group !== nothing && current_key !== nothing
                groups[current_group][current_key] = strip(current_val)
                current_key = current_val = nothing
            end
            current_group = lowercase(strip(line[2:end]))
            groups[current_group] = Dict{String, String}()
            continue
        end

        # End-of-group marker
        if line == "/" || line == "&end" || startswith(line, "/")
            if current_group !== nothing && current_key !== nothing
                groups[current_group][current_key] = strip(current_val)
                current_key = current_val = nothing
            end
            current_group = nothing
            continue
        end

        current_group === nothing && continue

        # key = value  (possibly continued on next line via trailing comma)
        if occursin('=', line)
            # Flush previous key if any
            if current_key !== nothing
                groups[current_group][current_key] = strip(current_val)
            end
            idx = findfirst('=', line)
            current_key = strip(line[1:idx-1])
            current_val = strip(line[idx+1:end])
        else
            # Continuation line: append to current value
            current_key !== nothing && (current_val *= " " * line)
        end
    end
    # Flush final key
    if current_group !== nothing && current_key !== nothing
        groups[current_group][current_key] = strip(current_val)
    end

    return groups
end

"""
    _strip_inline_comment(line) -> String

Remove everything after an unquoted `!` character.
"""
function _strip_inline_comment(line::AbstractString)
    in_quote = false
    for (i, c) in enumerate(line)
        c == '"' && (in_quote = !in_quote)
        !in_quote && c == '!' && return strip(line[1:i-1])
    end
    return line
end

# ---------------------------------------------------------------------------
# Type-directed value parsing
# ---------------------------------------------------------------------------

"""
    parse_nml_value(::Type{T}, s) -> T

Parse a raw namelist string `s` into Julia type `T`.
"""
parse_nml_value(::Type{Bool}, s::AbstractString) =
    lowercase(strip(s)) in ("true", ".true.", "t", "1")

parse_nml_value(::Type{Int}, s::AbstractString) =
    parse(Int, strip(s))

parse_nml_value(::Type{Float64}, s::AbstractString) =
    parse(Float64, replace(strip(s), r"[dD]" => "e"))  # Fortran D-exponent

parse_nml_value(::Type{String}, s::AbstractString) =
    String(strip(s, [' ', '"', '\'']))

function parse_nml_value(::Type{Vector{Float64}}, s::AbstractString)
    parts = split(strip(s), r"[\s,]+"; keepempty=false)
    return parse.(Float64, replace.(parts, r"[dD]" => "e"))
end

function parse_nml_value(::Type{Vector{String}}, s::AbstractString)
    # Match all quoted tokens
    ms = collect(eachmatch(r"\"([^\"]*)\"|'([^']*)'", s))
    isempty(ms) && return String[]
    return [something(m[1], m[2]) for m in ms]
end

# Fallback for unexpected types
parse_nml_value(::Type{T}, s::AbstractString) where {T} = parse(T, strip(s))

# ---------------------------------------------------------------------------
# Struct reconstruction from a flat Dict{String,String}
# ---------------------------------------------------------------------------

"""
    struct_from_dict(::Type{S}, d, group; strict=true) -> S

Reconstruct struct `S` from a `Dict{String,String}` of raw namelist values.
Fields absent from `d` keep their default. A key in `d` that is not a
parameter of `S` is an error (as in Fortran's `nml_validate`), or a
warning with `strict = false`.
"""
function struct_from_dict(::Type{S}, d::Dict{String,String}, group::AbstractString;
                          strict::Bool = true) where {S}
    defaults = S()   # zero-arg @kwdef constructor gives us all defaults
    known = Set{String}()
    kwargs = Dict{Symbol,Any}()
    for fname in fieldnames(S)
        def = getfield(defaults, fname)
        for (k, _) in _nml_entries(fname, def)
            push!(known, k)
        end
        kwargs[fname] = _from_nml(fieldtype(S, fname), fname, d, def)
    end
    unknown = setdiff(keys(d), known)
    if !isempty(unknown)
        msg = "read_nml: unknown parameter(s) in &$(group): " * join(sort(collect(unknown)), ", ")
        strict ? error(msg) : @warn(msg * " (ignored)")
    end
    return S(; kwargs...)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

const GROUP_TYPES = (yelmo = YelmoParams, ytopo = YtopoParams, ycalv = YcalvParams,
                     ydyn = YdynParams, ytill = YtillParams, yhyd = YhydParams,
                     ymat = YmatParams, ytrc = YtrcParams, ytherm = YthermParams,
                     yelmo_masks = YelmoMasksParams, yelmo_init_topo = YelmoInitTopoParams,
                     yelmo_data = YelmoDataParams)

"""
    read_nml(filename; strict=true) -> YelmoParameters

Read a Yelmo namelist file and return a fully populated `YelmoParameters`.
Groups or keys absent from the file take the defaults; unknown keys in a
Yelmo group are an error. Other groups (a driver's `&ctrl`) are ignored.
`strict = false` drops unknown keys with a warning instead, to read
namelists written for an older Yelmo: renamed keys then take their
defaults, so check the warning.

# Example
```julia
p = read_nml("run.nml")
println(p.ydyn.solver)   # "diva"
```
"""
function read_nml(filename::AbstractString; strict::Bool = true)
    raw = parse_nml_file(filename)
    groups = (g => struct_from_dict(GROUP_TYPES[g], get(raw, string(g), Dict{String,String}()),
                                    string(g); strict)
              for g in GROUPS)
    return YelmoParameters(splitext(basename(filename))[1]; groups...)   # name = stem of filename
end


## Comparison

# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

function Base.:(==)(a::YelmoParameters, b::YelmoParameters)
    for fname in fieldnames(YelmoParameters)
        fname == :name && continue
        getfield(a, fname) == getfield(b, fname) || return false
    end
    return true
end

for S in values(GROUP_TYPES)
    @eval function Base.:(==)(a::$S, b::$S)
        for fname in fieldnames($S)
            getfield(a, fname) == getfield(b, fname) || return false
        end
        return true
    end
end

# ---------------------------------------------------------------------------
# Diff printing
# ---------------------------------------------------------------------------

"""
    compare([io,] p1, p2; include_name=false)

Print all fields that differ between `p1` and `p2`, grouped by namelist group.
Identical groups are skipped entirely.
"""
function compare(io::IO, p1::YelmoParameters, p2::YelmoParameters; include_name=false)
    any_diff = false
    if include_name && p1.name != p2.name
        any_diff = true
        println(io, "  $(rpad("name", 24))  \"$(p1.name)\"  =>  \"$(p2.name)\"\n")
    end
    for g in GROUPS
        g1, g2 = getfield(p1, g), getfield(p2, g)
        g1 == g2 && continue
        any_diff = true
        println(io, "&$(g)")
        for sfield in fieldnames(typeof(g1))
            e1 = Dict(_nml_entries(sfield, getfield(g1, sfield)))
            e2 = Dict(_nml_entries(sfield, getfield(g2, sfield)))
            for (k, v1) in e1
                v1 == e2[k] && continue
                println(io, "  $(rpad(k, 24))  $(format_value(v1))  =>  $(format_value(e2[k]))")
            end
        end
        println(io, "/\n")
    end
    any_diff || println(io, "(no differences)")
    return nothing
end

compare(p1::YelmoParameters, p2::YelmoParameters; kw...) = compare(stdout, p1, p2; kw...)

include("YelmoParPort.jl")

end # module YelmoPar
