"""
    YelmoModelThrm

Thermodynamics (`thrm`) component for the pure-Julia `YelmoModel`.
Computes ice and bedrock temperature / enthalpy state, basal water,
internal heat sources, and pressure-melting diagnostics from the
current `tpo`/`dyn`/`mat`/`bnd` state.

Public surface: `therm_step!(y::YelmoModel, dt)`, dispatched from
`YelmoCore.step!(::YelmoModel, dt)` AFTER `mat_step!` in the fixed
phase order. Matches the Fortran per-step loop at
`yelmo_ice.f90:268-286` (`calc_ytopo` → `calc_ydyn` → `calc_ymat` →
`calc_ytherm` → corrector topo).

Port plan (incremental):

  - **PR1**: foundation. Module scaffolding, `therm_step!` dispatching
    `method = "fixed"` as a no-op, foundation kernels (`solve_tridiag!`,
    properties, `calc_dzeta_terms!`).
  - **PR2**: analytic solvers `linear`, `robin`, `robin-cold`.
    Adds the per-step property update (cp / kt / T_pmp), the
    Q_rock-from-Q_geo fallback, the analytic dispatch, and the
    universal post-step diagnostics (`T_prime`, `T_prime_b`, `f_pmp`,
    `enth`).
  - **PR3 (this commit)**: heat sources (`Q_strn`, `Q_b`) and basal
    water (`H_w`, `dHwdt`). Adds `qb_method ∈ {1, 2}` dispatch, the
    `use_strain_sia` toggle for the SIA-style strain-heating
    approximation, and the per-step `H_w` mass-balance update inside
    a `dt > 0` gate. With analytic / fixed methods the Fortran RK2
    half-then-full split degenerates to a single full-dt update; PR4
    will reintroduce the split when the implicit solver writes
    `bmb_grnd`.
  - **PR4**: implicit `temp` solver + bedrock column (production
    target). Adds `calc_advec_horizontal_3D!`,
    `calc_temp_3D!` / `calc_temp_column!` /
    `_calc_temp_column_internal!`, `define_temp_bedrock_3D!`
    (equil) + `define_temp_bedrock_active_3D!` (active), and the
    Fortran-style RK2 H_w split.
  - **PR5**: `enth` solver (port-only, not benchmarked). Adds
    `calc_enth_3D!` / `calc_enth_column!` /
    `_calc_enth_column_internal!`, `calc_enth_diffusivity!`, and
    `convert_from_enthalpy_column!`.
  - **PR6 (this commit)**: ice-margin extrapolation. Adds the
    Fortran `calc_ytherm_enthalpy_3D:444-477` post-pass that fills
    ice-free / partial-ice cells with a 3×3 neighbour-average of
    `enth` / `T_ice` / `omega` / `T_pmp` from fully-iced
    neighbours, so newly-advected cells inherit reasonable values
    for downstream consumers (mat's rate factor with rf_method=1).
    With this, the `temp` and `enth` solvers are at near-Fortran
    parity inside `therm_step!`.

Out of scope: `enth-poly` (~1929 lines of adaptive CTS solver) and
`ice_tracer` (belongs to `mat`).

Open follow-ups (not blocking thrm closeout):

  - Julia-native benchmark fixture for the `enth` solver — Mirror's
    `enth` path is unreliable on the Fortran side, so a Julia
    self-validation is the right next step.
  - `mat` `rf_method = 1` (temperature- and water-coupled
    Arrhenius) gating now has the upstream therm fields available;
    enabling it is a `mat` follow-up.

KNOWN VERTICAL CONVENTION ISSUE — see `.claude/PlanVerticalSplit.md`
for the full plan. Yelmo's grid loader builds an Oceananigans
`Center`-staggered z axis from the file's `zeta_ac` (Face values)
and loads file `T_ice` / `enth` / `T_pmp` etc. (length Nz_file =
file zeta length, includes endpoints 0 and 1) into Yelmo's
`Center` Nz fields by **verbatim index copy**. Because Center cells
are forced to be midpoints of Faces, file center positions and Yelmo
center positions don't align — file's `T_ice[:,:,1]` (basal value at
z=0) ends up at Yelmo's `zeta_aa[1] ≈ 0.003`, and file's
`T_ice[:,:,Nz_file]` (surface value at z=1) ends up at Yelmo's
`zeta_aa[Nz] ≈ 0.948` — a 5% mis-alignment at the surface. The
implicit column solver in this module uses `T_ice[:,:,1]` and
`T_ice[:,:,Nz]` AS basal / surface BC values, which makes the
mis-alignment cancel for thrm's own purposes — but any other consumer
that interprets `T_ice[:,:,1]` as an interior cell value reads
mis-positioned data. The same issue affects dyn (`ux`, `uy`,
`uz_star`) and mat (`visc`, `ATT`, `enh`). Path B fix (true
interior + 2D `_b` / `_s` boundary fields) is documented in
`.claude/PlanVerticalSplit.md`; deferred to a dedicated
cross-cutting refactor branch.

`therm_step!` does NOT advance `y.time` — that is owned by
`topo_step!`, matching the dyn/mat convention.
"""
module YelmoModelThrm

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using Oceananigans.Grids: znodes, topology, AbstractTopology
using Oceananigans.Fields: interior

using ..YelmoCore: AbstractYelmoModel, YelmoModel
using ..YelmoUtils: solve_tridiag!,
                    gq2d_nodes_2pt, gq2d_interp_to_node,
                    _neighbor_im1, _neighbor_ip1,
                    _neighbor_jm1, _neighbor_jp1,
                    _ip1_modular,  _jp1_modular

import ..YelmoCore: therm_step!

export therm_step!,
       calc_specific_heat_capacity, calc_thermal_conductivity,
       calc_T_pmp,
       calc_cp_3D!, calc_kt_3D!, calc_T_pmp_3D!, calc_T_pmp_boundaries_2D!, calc_f_pmp!,
       calc_dzeta_terms!,
       convert_to_enthalpy, convert_to_enthalpy_3D!,
       define_temp_linear_3D!, define_temp_robin_3D!,
       define_temp_linear_column!, define_temp_robin_column!,
       calc_strain_heating!, calc_strain_heating_sia!,
       calc_basal_heating_simplestagger!, calc_basal_heating_nodes!,
       calc_basal_water_local!,
       calc_advec_horizontal_3D!,
       calc_temp_column!, calc_temp_3D!,
       calc_enth_column!, calc_enth_3D!,
       calc_enth_diffusivity!, convert_from_enthalpy_column!,
       define_temp_bedrock_column!, define_temp_bedrock_3D!,
       define_temp_bedrock_active_3D!, calc_Q_bedrock_column

include("properties.jl")
include("dzeta.jl")
include("enthalpy.jl")
include("solvers_analytic.jl")
include("strain_heating.jl")
include("basal_heating.jl")
include("basal_water.jl")
include("helpers.jl")
include("advection.jl")
include("column_solver.jl")
include("temp_solver.jl")
include("enth_solver.jl")
include("bedrock.jl")

"""
    therm_step!(y::YelmoModel, dt) -> y

Update thermodynamic state (`y.thrm`) for the current step. Dispatches
on `y.p.ytherm.method`.

Phase order (Fortran `calc_ytherm`, yelmo_thermodynamics.f90:22):

  1. **Properties** — refresh `cp`, `kt` from current `T_ice` (or pin
     to `const_cp` / `const_kt`); refresh `T_pmp` from current `H_ice`.
  2. **Basal heat Q_b** — `calc_basal_heating_simplestagger!`
     (`qb_method = 1`) or `calc_basal_heating_nodes!`
     (`qb_method = 2`, default).
  3. **Strain heat Q_strn** — `calc_strain_heating!` (general
     `4 * visc * de^2`) or `calc_strain_heating_sia!`
     (`use_strain_sia = true`).
  4. **dQsdT = 0** — Fortran-faithful (the `dQsdT` derivative is
     gated behind a hard-coded `false` in calc_ytherm).
  5. **Q_rock fallback** — if `Q_rock` is identically zero (first
     step), seed from `bnd.Q_geo` for a non-degenerate basal flux.
  6. **Time-stepped block** (`dt > 0` only):
       a. `calc_basal_water_local!` — single full-dt update of `H_w`
          / `dHwdt` (Fortran's RK2 half-then-full split degenerates
          to a single update for analytic / fixed methods; PR4 will
          reintroduce the split).
       b. **Method dispatch** —
            - `fixed`        : pass-through (no T_ice update).
            - `linear`       : `define_temp_linear_3D!`.
            - `robin`        : `define_temp_robin_3D!(cold=false)`.
            - `robin-cold`   : `define_temp_robin_3D!(cold=true)`.
            - `temp`         : not yet ported (PR4).
            - `enth`         : not yet ported (PR5).
            - `enth-poly`    : out of scope.
  7. **Enthalpy** — `convert_to_enthalpy_3D!` keeps `enth` consistent
     with the just-written `(T_ice, omega)`.
  8. **Diagnostics** — `T_prime = T_ice - T_pmp`, `T_prime_b =
     T_prime[:, :, 1]`, `f_pmp` via `calc_f_pmp!`.

Steps 1–5, 7, 8 always run, even for `method = "fixed"` and at
`dt = 0`. Step 6 runs only when `dt > 0`.

`therm_step!` does NOT advance `y.time`.
"""
function therm_step!(y::YelmoModel, dt::Float64)
    y.p === nothing && error(
        "therm_step!: y.p must be non-nothing once the physics body is wired. " *
        "Construct YelmoModel with a YelmoModelParameters value, or skip " *
        "the orchestrator's per-component chain.")

    par     = y.p.ytherm
    method  = par.method
    c       = y.c
    # Path B (commit 5a): use the cached `Vector{Float64}` snapshots
    # of the ice grid axes from `y.thrm.scratch` instead of allocating
    # a fresh `collect(znodes(...))` per `therm_step!` call.
    zeta_aa = y.thrm.scratch.zeta_aa
    zeta_ac = y.thrm.scratch.zeta_ac

    # 1. Thermal property update.
    if par.use_const_cp
        fill!(interior(y.thrm.cp), par.const_cp)
    else
        calc_cp_3D!(y.thrm.cp, y.thrm.T_ice)
    end
    if par.use_const_kt
        fill!(interior(y.thrm.kt), par.const_kt)
    else
        calc_kt_3D!(y.thrm.kt, y.thrm.T_ice, c.sec_year)
    end
    calc_T_pmp_3D!(y.thrm.T_pmp, y.tpo.H_ice, zeta_aa,
                   c.T0, c.T_pmp_beta, c.rho_ice, c.g)
    # Path B: also fill the dedicated 2D basal / surface T_pmp fields
    # registered by `PATH_B_REGISTRY_ICE`, since `T_pmp[:,:,1]` is now
    # the first interior layer at ζ_aa[1], not the basal boundary at
    # ζ = 0.
    calc_T_pmp_boundaries_2D!(y.thrm.T_pmp_b, y.thrm.T_pmp_s,
                              y.tpo.H_ice,
                              c.T0, c.T_pmp_beta, c.rho_ice, c.g)

    # 2. Heat sources — basal frictional heating Q_b.
    if par.qb_method == 1
        calc_basal_heating_simplestagger!(y.thrm.Q_b,
                                          y.dyn.ux_b, y.dyn.uy_b,
                                          y.dyn.taub_acx, y.dyn.taub_acy,
                                          c.sec_year)
    elseif par.qb_method == 2
        calc_basal_heating_nodes!(y.thrm.Q_b,
                                  y.dyn.ux_b, y.dyn.uy_b,
                                  y.dyn.taub_acx, y.dyn.taub_acy,
                                  y.tpo.f_ice, c.sec_year)
    else
        error("therm_step!: unknown qb_method=$(par.qb_method); supported: 1 (\"aa\"), 2 (\"nodes\").")
    end

    # 3. Heat sources — internal strain heating Q_strn.
    if par.use_strain_sia
        calc_strain_heating_sia!(y.thrm.Q_strn,
                                 y.dyn.ux, y.dyn.uy,
                                 y.tpo.dzsdx, y.tpo.dzsdy,
                                 y.tpo.H_ice,
                                 zeta_aa, zeta_ac, c.rho_ice, c.g)
    else
        calc_strain_heating!(y.thrm.Q_strn, y.dyn.strn_de, y.mat.visc)
    end

    # 4. dQ_strn/dT — Fortran gates this behind a hard-coded
    #    `calculate_Q_strn_derivative = .FALSE.`, leaving dQsdT at zero.
    #    Mirror that.
    fill!(interior(y.thrm.dQsdT), 0.0)

    # 5. Q_rock fallback: first step seeds Q_rock from Q_geo so the
    #    column solvers have a non-zero basal heat flux. Mirrors
    #    Fortran `calc_ytherm` lines 122-124.
    Q_rock_int = interior(y.thrm.Q_rock)
    if maximum(Q_rock_int) == 0.0
        copyto!(Q_rock_int, interior(y.bnd.Q_geo))
    end

    # 6. Time-stepped block (basal water + method dispatch).
    #    Fortran `calc_ytherm` runs the H_w update + analytic / implicit
    #    solver only when `dt > 0`. With analytic methods the
    #    H_w-half-step + H_w-full-step RK2 split degenerates to a
    #    single full-dt update (the analytic solver does not touch
    #    `bmb_grnd`), so we collapse to one `calc_basal_water_local!`
    #    call here. PR4 will reintroduce the two-stage split once the
    #    implicit solver writes `bmb_grnd`.
    if dt > 0.0
        # 6a. Vertical-discretisation weights for the implicit solvers.
        #     Cheap allocation per call — could later be cached on
        #     `y.thrm` if benchmarking shows it matters.
        Nz_aa   = y.thrm.T_ice.grid.Nz
        dzeta_a = Vector{Float64}(undef, Nz_aa)
        dzeta_b = Vector{Float64}(undef, Nz_aa)
        calc_dzeta_terms!(dzeta_a, dzeta_b,
                          collect(Float64, zeta_aa),
                          collect(Float64, zeta_ac))

        # 6b. Basal water — RK2 split. Snapshot H_w before any update,
        #     advance with `dt/2` using the *previous* step's
        #     `bmb_grnd`, then restore + advance with full `dt` using
        #     the *new* `bmb_grnd` after the method-dispatch block.
        Nx_2D = y.thrm.H_w.grid.Nx
        Ny_2D = y.thrm.H_w.grid.Ny
        bmb_w_scratch = Array{Float64}(undef, Nx_2D, Ny_2D, 1)
        H_w_now       = copy(interior(y.thrm.H_w))
        _fill_bmb_w_scratch!(bmb_w_scratch, y.thrm.bmb_grnd,
                             c.rho_ice, c.rho_w)
        _calc_basal_water_local_kernel!(y.thrm.H_w.data, y.thrm.dHwdt.data,
                                        y.tpo.f_ice.data, y.tpo.f_grnd.data,
                                        bmb_w_scratch,
                                        0.5 * dt, Float64(par.till_rate),
                                        Float64(par.H_w_max),
                                        Nx_2D, Ny_2D)

        # 6c. Method dispatch — write `T_ice`, `omega`, possibly
        #     `bmb_grnd` (temp solver only).
        if method == "fixed"
            # Pass-through. Leave T_ice and omega alone.
        elseif method == "linear"
            define_temp_linear_3D!(y.thrm.T_ice, y.thrm.omega,
                                   y.tpo.H_ice, y.bnd.T_srf,
                                   zeta_aa,
                                   c.T0, c.T_pmp_beta, c.rho_ice, c.g)
        elseif method == "robin"
            define_temp_robin_3D!(y.thrm.T_ice, y.thrm.omega,
                                  y.thrm.T_pmp, y.thrm.cp, y.thrm.kt,
                                  y.thrm.Q_rock, y.bnd.T_srf, y.tpo.H_ice,
                                  y.bnd.smb_ref, y.thrm.bmb_grnd, y.tpo.f_grnd,
                                  zeta_aa, c.rho_ice, c.sec_year;
                                  cold=false)
        elseif method == "robin-cold"
            define_temp_robin_3D!(y.thrm.T_ice, y.thrm.omega,
                                  y.thrm.T_pmp, y.thrm.cp, y.thrm.kt,
                                  y.thrm.Q_rock, y.bnd.T_srf, y.tpo.H_ice,
                                  y.bnd.smb_ref, y.thrm.bmb_grnd, y.tpo.f_grnd,
                                  zeta_aa, c.rho_ice, c.sec_year;
                                  cold=true)
        elseif method == "temp"
            # Pre-compute horizontal advection of `T_ice`.
            calc_advec_horizontal_3D!(y.thrm.advecxy, y.thrm.T_ice,
                                       y.dyn.ux, y.dyn.uy, _dx_thrm(y.g))
            # Implicit cold-ice column solve.
            calc_temp_3D!(y.thrm.enth, y.thrm.T_ice, y.thrm.omega,
                          y.thrm.bmb_grnd, y.thrm.Q_ice_b, y.thrm.H_cts,
                          y.thrm.T_pmp, y.thrm.cp, y.thrm.kt, y.thrm.advecxy,
                          y.dyn.uz_star, y.thrm.Q_strn,
                          y.thrm.Q_b, y.thrm.Q_rock,
                          y.bnd.T_srf, y.bnd.T_shlf,
                          y.tpo.H_ice, y.tpo.f_ice, y.thrm.H_w,
                          y.tpo.f_grnd, y.tpo.H_grnd,
                          zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                          par.omega_max, c.T0, c.rho_ice, c.rho_sw,
                          c.rho_w, c.L_ice, c.sec_year, dt;
                          path_b        = true,
                          T_ice_b_field = y.thrm.T_ice_b,
                          T_pmp_b_field = y.thrm.T_pmp_b,
                          T_ice_s_field = y.thrm.T_ice_s)
        elseif method == "enth"
            # Pre-compute horizontal advection of `enth`.
            calc_advec_horizontal_3D!(y.thrm.advecxy, y.thrm.enth,
                                       y.dyn.ux, y.dyn.uy, _dx_thrm(y.g))
            calc_enth_3D!(y.thrm.enth, y.thrm.T_ice, y.thrm.omega,
                           y.thrm.bmb_grnd, y.thrm.Q_ice_b, y.thrm.H_cts,
                           y.thrm.T_pmp, y.thrm.cp, y.thrm.kt, y.thrm.advecxy,
                           y.dyn.uz_star, y.thrm.Q_strn,
                           y.thrm.Q_b, y.thrm.Q_rock,
                           y.bnd.T_srf, y.bnd.T_shlf,
                           y.tpo.H_ice, y.tpo.f_ice, y.thrm.H_w,
                           y.tpo.f_grnd, y.tpo.H_grnd,
                           zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                           par.enth_cr, par.omega_max, c.T0,
                           c.rho_ice, c.rho_sw, c.rho_w,
                           c.L_ice, c.sec_year, dt)
        elseif method == "enth-poly"
            error("therm_step!: method=\"$(method)\" (polythermal CTS-adaptive " *
                  "solver) is out of scope for the Yelmo.jl thrm port. The " *
                  "Fortran `ice_enthalpy_poly.f90` (~1929 lines) is not " *
                  "planned for porting. Use \"temp\" (production).")
        else
            error("therm_step!: unknown method=\"$(method)\". Supported: " *
                  "\"fixed\", \"linear\", \"robin\", \"robin-cold\", \"temp\". " *
                  "Coming: \"enth\" (PR5).")
        end

        # 6c-extra. Extrapolate the thermodynamic state to ice-free /
        #           ice-margin cells from full-ice 3×3 neighbours. Helps
        #           mat's rate factor (rf_method=1) at advected points.
        #           Fortran calc_ytherm_enthalpy_3D:444-477. Skip for
        #           method = "fixed" (T_ice unchanged so no extrap needed).
        if method != "fixed"
            _extrapolate_thrm_margin!(y.thrm.enth.data, y.thrm.T_ice.data,
                                       y.thrm.omega.data, y.thrm.T_pmp.data,
                                       y.tpo.f_ice.data,
                                       y.thrm.T_ice.grid.Nx,
                                       y.thrm.T_ice.grid.Ny,
                                       y.thrm.T_ice.grid.Nz)
        end

        # 6d. Basal water — full-dt update from initial snapshot using
        #     the (possibly updated) `bmb_grnd`.
        Hw_int = interior(y.thrm.H_w)
        copyto!(Hw_int, H_w_now)
        _fill_bmb_w_scratch!(bmb_w_scratch, y.thrm.bmb_grnd,
                             c.rho_ice, c.rho_w)
        _calc_basal_water_local_kernel!(y.thrm.H_w.data, y.thrm.dHwdt.data,
                                        y.tpo.f_ice.data, y.tpo.f_grnd.data,
                                        bmb_w_scratch,
                                        dt, Float64(par.till_rate),
                                        Float64(par.H_w_max),
                                        Nx_2D, Ny_2D)

        # 6e. Bedrock dispatch.
        rock_method = par.rock_method
        if rock_method == "fixed"
            # Pass-through; T_rock / enth_rock / Q_rock unchanged.
        elseif rock_method == "equil"
            zeta_aa_rock = znodes(y.gr, Center())
            define_temp_bedrock_3D!(y.thrm.enth_rock, y.thrm.T_rock,
                                     y.thrm.Q_rock,
                                     y.thrm.T_ice,
                                     y.bnd.Q_geo,
                                     par.cp_rock, par.kt_rock, par.H_rock,
                                     zeta_aa_rock, c.sec_year;
                                     path_b       = true,
                                     T_ice_b_field = y.thrm.T_ice_b)
            # T_rock_b: deep-boundary diagnostic (deepest bedrock layer, ζ≈0).
            interior(y.thrm.T_rock_b) .= view(interior(y.thrm.T_rock), :, :, 1)
        elseif rock_method == "active"
            zeta_aa_rock = znodes(y.gr, Center())
            zeta_ac_rock = znodes(y.gr, Face())
            Nzr_aa       = y.thrm.T_rock.grid.Nz
            dzeta_a_rock = Vector{Float64}(undef, Nzr_aa)
            dzeta_b_rock = Vector{Float64}(undef, Nzr_aa)
            calc_dzeta_terms!(dzeta_a_rock, dzeta_b_rock,
                              collect(Float64, zeta_aa_rock),
                              collect(Float64, zeta_ac_rock))
            define_temp_bedrock_active_3D!(y.thrm.enth_rock, y.thrm.T_rock,
                                            y.thrm.Q_rock,
                                            y.thrm.T_ice,
                                            y.bnd.Q_geo,
                                            par.cp_rock, par.kt_rock,
                                            c.rho_rock, par.H_rock,
                                            zeta_aa_rock, zeta_ac_rock,
                                            dzeta_a_rock, dzeta_b_rock,
                                            c.sec_year, dt;
                                            path_b       = true,
                                            T_ice_b_field = y.thrm.T_ice_b)
            # T_rock_b: deep-boundary diagnostic (deepest bedrock layer, ζ≈0).
            interior(y.thrm.T_rock_b) .= view(interior(y.thrm.T_rock), :, :, 1)
        else
            error("therm_step!: unknown rock_method=\"$(rock_method)\". " *
                  "Supported: \"fixed\", \"equil\", \"active\".")
        end
    end  # end if dt > 0

    # 7. Update enth from (T_ice, omega, T_pmp, cp, L). For analytic
    #    methods omega is identically zero, so enth = cp * T_ice; the
    #    formula is general and handles the temperate enth solver
    #    once it lands in PR5.
    convert_to_enthalpy_3D!(y.thrm.enth, y.thrm.T_ice, y.thrm.omega,
                            y.thrm.T_pmp, y.thrm.cp, c.L_ice)

    # 8. Homologous-temperature + pressure-melting fraction.
    #    Path B: T_prime_b / T_prime_s come from the dedicated
    #    boundary fields (T_ice_b - T_pmp_b, T_ice_s - T_pmp_s),
    #    not from T_prime[:, :, 1] / [:, :, end] (which are interior
    #    layers under Path B). f_pmp reads T_ice_b / T_pmp_b.
    _calc_T_prime_3D!(y.thrm.T_prime, y.thrm.T_prime_b, y.thrm.T_prime_s,
                      y.thrm.T_ice, y.thrm.T_pmp,
                      y.thrm.T_ice_b, y.thrm.T_pmp_b,
                      y.thrm.T_ice_s, y.thrm.T_pmp_s)
    calc_f_pmp!(y.thrm.f_pmp, y.thrm.T_ice_b, y.thrm.T_pmp_b, y.tpo.f_grnd;
                gamma=par.gamma)

    return y
end

# Helper: fill a 2D scratch buffer with `bmb_w = -bmb_grnd * (rho_ice
# / rho_w)`. Mirrors the inline expression at Fortran calc_ytherm:134
# / 206. The buffer is shaped (Nx, Ny, 1) to match the OffsetArray
# convention of `interior(field)`.
function _fill_bmb_w_scratch!(bmb_w::AbstractArray{Float64,3},
                              bmb_grnd_field,
                              rho_ice::Real, rho_w::Real)
    Bg     = bmb_grnd_field.data
    Nx     = bmb_grnd_field.grid.Nx
    Ny     = bmb_grnd_field.grid.Ny
    factor = -Float64(rho_ice) / Float64(rho_w)
    @inbounds for j in 1:Ny, i in 1:Nx
        bmb_w[i, j, 1] = factor * Bg[i, j, 1]
    end
    return bmb_w
end

# Compute T_prime = T_ice - T_pmp (3D interior) plus the Path B
# boundary fields T_prime_b = T_ice_b - T_pmp_b (basal, ζ=0) and
# T_prime_s = T_ice_s - T_pmp_s (surface, ζ=1). Mirrors the diagnostic
# lines in Fortran `calc_ytherm` lines 248-249, but with the boundary
# values pulled from their dedicated 2D fields rather than from
# `T_prime[:, :, 1]` / `T_prime[:, :, end]` (which are interior
# layer values, not boundary values, under Path B).
function _calc_T_prime_3D!(T_prime_field, T_prime_b_field, T_prime_s_field,
                            T_ice_field, T_pmp_field,
                            T_ice_b_field, T_pmp_b_field,
                            T_ice_s_field, T_pmp_s_field)
    Tp_d   = T_prime_field.data
    Tpb_d  = T_prime_b_field.data
    Tps_d  = T_prime_s_field.data
    Ti_d   = T_ice_field.data
    Tm_d   = T_pmp_field.data
    Tib_d  = T_ice_b_field.data
    Tmb_d  = T_pmp_b_field.data
    Tis_d  = T_ice_s_field.data
    Tms_d  = T_pmp_s_field.data
    Nx     = T_ice_field.grid.Nx
    Ny     = T_ice_field.grid.Ny
    Nz     = T_ice_field.grid.Nz
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Tp_d[i, j, k] = Ti_d[i, j, k] - Tm_d[i, j, k]
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        Tpb_d[i, j, 1] = Tib_d[i, j, 1] - Tmb_d[i, j, 1]
        Tps_d[i, j, 1] = Tis_d[i, j, 1] - Tms_d[i, j, 1]
    end
    return nothing
end

# Horizontal grid spacing for the thrm domain. Matches the dyn
# convention (uniform grids only for now).
function _dx_thrm(grid::RectilinearGrid)
    Δx = grid.Δxᶜᵃᵃ
    Δx isa Number || error("YelmoModelThrm requires uniform x-spacing (got $(typeof(Δx))).")
    return abs(Δx)
end

end # module YelmoModelThrm
