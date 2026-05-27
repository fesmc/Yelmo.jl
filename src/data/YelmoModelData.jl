"""
    YelmoModelData

Reference / present-day data (`dta`) component for the pure-Julia
`YelmoModel`. Holds observational fields used for online comparison
of the simulated state against present-day (or "target") topography,
climate forcing, and surface velocity. Mirrors the Fortran
`yelmo_data` module (`yelmo/src/yelmo_data.f90`).

Public surface:

  - `init_topo_load!(y)`         — read initial-topography fields
    from NetCDF into `y.tpo.H_ice / y.bnd.z_bed / y.bnd.z_bed_sd`
    according to `y.p.yelmo_init_topo`. Applies `z_bed_f_sd`.
    Mirrors the topo-loading half of Fortran `yelmo_init_topo`.
  - `data_load!(y; par_path)`    — read PD reference fields from
    NetCDF into `y.dta.pd_*` according to `y.p.yelmo_data`.
    Mirrors `ydata_load`. Reuses the SAME `z_bed_f_sd` from
    `y.p.yelmo_init_topo`, so runtime and reference bedrock stay
    in lockstep (Fortran `yelmo_data.f90:223-229`).
  - `data_compare!(y)`           — compute `pd_err_*` fields and
    update the scalar RMSEs on `y.dta.rmse`. Mirrors
    `ydata_compare`. Call this manually before `write_output!` if
    you want fresh diagnostics in the output file.
  - `RMSEStats`                  — mutable struct holding
    `H, zsrf, uxy, loguxy` scalars (initialised to `NaN`).

Deviations from Fortran (intentional, scope-bounded):

  - **Isochrones / age** (`pd_age_iso`, `pd_depth_iso`,
    `pd_err_depth_iso`, `rmse_iso`): allocated by the variable table
    (`yelmo-variables-ydata.md`) but neither read nor compared here.
    They will follow once the corresponding `mat` isochrone
    diagnostics land.
  - **`pd_lsf`**: not present in the Julia variable table; the
    Fortran call to `LSFinit` on the reference-data lsf is dropped.
  - **Sub-grid `mask_bed` reclassification**: derived from the loaded
    `H_ice / z_srf / H_grnd` exactly as in Fortran.

Field-name convention: the `dta` group flattens Fortran's
`dta%pd%H_ice` to `y.dta.pd_H_ice`. Errors live at
`y.dta.pd_err_H_ice`. Scalar RMSEs are NOT in the variable table —
they live on the `RMSEStats` struct exposed as `y.dta.rmse`.
"""
module YelmoModelData

using NCDatasets
using Oceananigans.Fields: Field, CenterField, interior

using ..YelmoCore: AbstractYelmoModel, YelmoModel, RMSEStats,
                   MASK_BED_OCEAN, MASK_BED_LAND,
                   MASK_BED_FLOAT, MASK_BED_FROZEN,
                   MASK_ICE_NONE, MASK_ICE_FIXED, MASK_ICE_DYNAMIC
using ..YelmoUtils: remove_englacial_lakes!, adjust_topography_gradients!,
                    smooth_gauss_2D!

export init_topo_load!, init_masks!, define_mask_ice!, data_load!, data_compare!
# Note: `RMSEStats` is defined and exported in `YelmoCore` so the type
# is in scope inside `_alloc_yelmo_groups` without forward references.
# We re-import it here for `data_compare!`.

# ---------------------------------------------------------------------------
# Path substitution
# ---------------------------------------------------------------------------

# `{domain}` → domain, `{grid_name}` → grid_name. Mirrors
# `yelmo_parse_path` in Fortran.
function _parse_path(path::AbstractString, domain::AbstractString, grid_name::AbstractString)
    path = replace(path, "{domain}"    => domain)
    path = replace(path, "{grid_name}" => grid_name)
    return path
end

# ---------------------------------------------------------------------------
# init_topo_load!
# ---------------------------------------------------------------------------

# `init_topo_names[i]` is treated as "absent" if it equals the empty
# string or one of the Fortran sentinel values. Mirrors the
# `trim(name) ∉ {"", "none", "None"}` checks in
# `yelmo_init_topo`/`ydata_load`.
@inline _name_present(s::AbstractString) =
    !(isempty(strip(s)) || lowercase(strip(s)) == "none")

"""
    init_topo_load!(y::YelmoModel; grad_lim_zb::Real = 0.05,
                                   boundaries::AbstractString = "infinite") -> y

Read the initial-topography fields from NetCDF into the runtime model
state, mirroring the topo-loading half of Fortran `yelmo_init_topo`
(`yelmo/src/yelmo_ice.f90:939`). Reads
`y.p.yelmo_init_topo.init_topo_path` with field-name vector
`init_topo_names = [H_ice, z_bed, z_bed_sd, z_srf]`, then:

  1. `z_bed += z_bed_f_sd · z_bed_sd` (when `init_topo_names[3]` is
     present). The same `z_bed_f_sd` is reused by `data_load!` to
     keep runtime and reference bedrock in lockstep.
  2. `remove_englacial_lakes!` against `bnd.z_sl` (when
     `init_topo_names[4]` is present).
  3. `smooth_gauss_2D!(H_ice; f_sigma = smooth_H_ice)` and the same
     for `z_bed`, when those parameters are `>= 1`.
  4. Clamp `H_ice < 0.1` → 0; lift `0.1 ≤ H_ice < 10` → 10.
  5. `adjust_topography_gradients!(z_bed, H_ice, grad_lim_zb, dx,
     boundaries)`.
  6. Apply `init_topo_state`:
       * `0` — keep loaded topography,
       * `1` — `H_ice .= 0` (remove ice, leave bedrock),
       * `2` — isostatic rebound: `z_bed += (ρ_ice / ρ_a) · H_ice`,
              then `H_ice .= 0`.
  7. Write back into `y.tpo.H_ice`, `y.bnd.z_bed`, `y.bnd.z_bed_sd`.

Out of scope (handled elsewhere in Yelmo.jl):

  - **Restart-file path**: in Fortran the same routine merges with a
    restart file when `use_restart=true`. In Yelmo.jl that is the
    job of `YelmoModel(restart_file, ...)` /  `load_state!`; this
    function only handles the parameter-driven NetCDF load.
  - **Mass-balance "G_boundaries" / `apply_tendency` step** at the
    end of Fortran's routine. Depends on `tpo` runtime state not
    necessarily ready at this point; can be added when needed.
  - **`LSFinit`**: the level-set field is not in the Yelmo.jl
    init-topo path yet.

`grad_lim_zb` and `boundaries` mirror the Fortran arguments
`tpo%par%grad_lim_zb` / `tpo%par%boundaries`. Defaulting them lets
this function work even when the topo parameter struct hasn't pinned
those down — adjust to match your `ytopo` config when tuning.
"""
function init_topo_load!(y::YelmoModel;
                         grad_lim_zb::Real = 0.05,
                         boundaries::AbstractString = "infinite")
    y.p === nothing && error(
        "init_topo_load!: y.p must be a YelmoParameters value " *
        "(with `yelmo_init_topo` filled in).")

    par       = y.p.yelmo_init_topo
    domain    = y.p.yelmo.domain
    grid_name = y.p.yelmo.grid_name
    rho_ice   = y.c.rho_ice
    rho_sw    = y.c.rho_sw
    rho_a     = y.c.rho_a
    dx        = _grid_dx(y.g)

    if !par.init_topo_load
        @info "init_topo_load!: init_topo_load=false — skipping NetCDF load."
        return y
    end

    filename = _parse_path(par.init_topo_path, domain, grid_name)
    nms      = par.init_topo_names
    length(nms) >= 4 || error(
        "init_topo_load!: init_topo_names must have length >= 4 " *
        "(got $(length(nms))).")

    @info "init_topo_load!: reading initial topography from $(filename)"

    H_ice    = @view interior(y.tpo.H_ice)[:, :, 1]
    z_bed    = @view interior(y.bnd.z_bed)[:, :, 1]
    z_bed_sd = @view interior(y.bnd.z_bed_sd)[:, :, 1]
    z_sl     = @view interior(y.bnd.z_sl)[:, :, 1]
    nx, ny   = size(H_ice)

    _read_to!(H_ice, filename, nms[1])
    _read_to!(z_bed, filename, nms[2])

    # Step 1: z_bed_sd (third name) — apply z_bed_f_sd scaling.
    if _name_present(nms[3])
        _read_to!(z_bed_sd, filename, nms[3])
        @inbounds for j in 1:ny, i in 1:nx
            z_bed[i, j] += par.z_bed_f_sd * z_bed_sd[i, j]
        end
    else
        fill!(z_bed_sd, 0.0)
    end

    # Step 2: z_srf (fourth name) — englacial-lake removal.
    if _name_present(nms[4])
        z_srf_tmp = Array{Float64}(undef, nx, ny)
        _read_to!(z_srf_tmp, filename, nms[4])
        remove_englacial_lakes!(H_ice, z_bed, z_srf_tmp, z_sl, rho_ice, rho_sw)
        @info "init_topo_load!: removed englacial lakes."
    end

    # Step 3: optional smoothing.
    if par.smooth_H_ice >= 1.0
        smooth_gauss_2D!(H_ice, dx, par.smooth_H_ice)
    end
    if par.smooth_z_bed >= 1.0
        smooth_gauss_2D!(z_bed, dx, par.smooth_z_bed)
    end

    # Step 4: H_ice clean-up.
    @inbounds for j in 1:ny, i in 1:nx
        h = H_ice[i, j]
        if h < 0.1
            H_ice[i, j] = 0.0
        elseif h < 10.0
            H_ice[i, j] = 10.0
        end
    end

    # Step 5: gradient-limited smoothing.
    adjust_topography_gradients!(z_bed, H_ice, grad_lim_zb, dx, boundaries)

    # Step 6: init_topo_state.
    if par.init_topo_state == 0
        # Keep topography as loaded.
    elseif par.init_topo_state == 1
        fill!(H_ice, 0.0)
    elseif par.init_topo_state == 2
        @inbounds for j in 1:ny, i in 1:nx
            z_bed[i, j] += (rho_ice / rho_a) * H_ice[i, j]
            H_ice[i, j]  = 0.0
        end
    else
        error("init_topo_load!: init_topo_state=$(par.init_topo_state) " *
              "not recognised. Supported: 0, 1, 2.")
    end

    @info "init_topo_load!: range(H_ice)    = " minimum(H_ice)    maximum(H_ice)
    @info "init_topo_load!: range(z_bed)    = " minimum(z_bed)    maximum(z_bed)
    @info "init_topo_load!: range(z_bed_sd) = " minimum(z_bed_sd) maximum(z_bed_sd)
    @info "init_topo_load!: z_bed_f_sd      = " par.z_bed_f_sd

    return y
end

# ---------------------------------------------------------------------------
# init_masks!
# ---------------------------------------------------------------------------

# Domain-default region indices, matching Fortran `bnd%index_*` in
# `yelmo_defs.f90:755-757`. Used when `regions_load = false` (or the
# regions file is absent) so the regional mask in `data_compare!`
# still works on real-domain runs.
const _DOMAIN_REGION_INDEX = Dict(
    "Greenland"  => 1.3,
    "Antarctica" => 2.0,
    "North"      => 1.0,
)

"""
    init_masks!(y::YelmoModel) -> y

Load the basin- and region-mask boundary fields (`y.bnd.basins`,
`y.bnd.basin_mask`, `y.bnd.regions`, `y.bnd.region_mask`) from the
NetCDF files specified in `y.p.yelmo_masks`. Mirrors Fortran
`ybound_load_masks` (`yelmo/src/yelmo_boundaries.f90:108-189`).

Behaviour:

  - `basin_mask` and `basins` are seeded with `1.0` everywhere so a
    `basins_load = false` run still has a sensible default.
  - `region_mask` is seeded with `1.0`. `regions` is seeded with the
    domain-specific default index from `_DOMAIN_REGION_INDEX`
    (Greenland → 1.3, Antarctica → 2.0, North → 1.0, otherwise 1.0).
    This matches the Fortran `select case(domain)` block.
  - When `basins_load = true`, reads `basins_nms[1]` into `bnd.basins`
    and (if `basins_nms[2] != "None"`) `basins_nms[2]` into
    `bnd.basin_mask`.
  - When `regions_load = true`, reads `regions_nms[1]` into
    `bnd.regions` and (if `regions_nms[2] != "None"`) `regions_nms[2]`
    into `bnd.region_mask`.
  - Paths support the `{domain}` / `{grid_name}` placeholders (same
    `_parse_path` helper as `init_topo_load!`).

This is a load-time helper. Call once after `YelmoModel` construction
and before `init_state!` if you need `bnd.basins` / `bnd.regions`
populated for downstream regional diagnostics (`data_compare!`,
custom region masks via `add_region!`, basin-aware physics hooks).
"""
function init_masks!(y::YelmoModel)
    y.p === nothing && error(
        "init_masks!: y.p must be a YelmoParameters value " *
        "(with `yelmo_masks` filled in).")

    par       = y.p.yelmo_masks
    domain    = y.p.yelmo.domain
    grid_name = y.p.yelmo.grid_name

    basins      = @view interior(y.bnd.basins)[:, :, 1]
    basin_mask  = @view interior(y.bnd.basin_mask)[:, :, 1]
    regions     = @view interior(y.bnd.regions)[:, :, 1]
    region_mask = @view interior(y.bnd.region_mask)[:, :, 1]

    # ----- Defaults --------------------------------------------------
    fill!(basins,     1.0)
    fill!(basin_mask, 1.0)
    fill!(region_mask, 1.0)
    fill!(regions, get(_DOMAIN_REGION_INDEX, domain, 1.0))

    # ----- basins ----------------------------------------------------
    if par.basins_load
        filename = _parse_path(par.basins_path, domain, grid_name)
        nms = par.basins_nms
        length(nms) >= 1 || error(
            "init_masks!: basins_nms must have length >= 1 " *
            "(got $(length(nms))).")
        @info "init_masks!: reading basins from $(filename)"
        _read_to!(basins, filename, nms[1])
        if length(nms) >= 2 && _name_present(nms[2])
            _read_to!(basin_mask, filename, nms[2])
        end
    end

    # ----- regions ---------------------------------------------------
    if par.regions_load
        filename = _parse_path(par.regions_path, domain, grid_name)
        nms = par.regions_nms
        length(nms) >= 1 || error(
            "init_masks!: regions_nms must have length >= 1 " *
            "(got $(length(nms))).")
        @info "init_masks!: reading regions from $(filename)"
        _read_to!(regions, filename, nms[1])
        if length(nms) >= 2 && _name_present(nms[2])
            _read_to!(region_mask, filename, nms[2])
        end
    end

    @info "init_masks!: range(basins)  = " minimum(basins)  maximum(basins)
    @info "init_masks!: range(regions) = " minimum(regions) maximum(regions)

    # Paint the per-cell ice domain mask from the domain definition +
    # regions, now that `regions` is populated. Restart / benchmark ICs
    # may override `bnd.mask_ice` afterward.
    define_mask_ice!(y)

    return y
end

"""
    define_mask_ice!(y::YelmoModel) -> y

Set `y.bnd.mask_ice` according to the domain definition, marking each
cell as dynamic (`MASK_ICE_DYNAMIC`), prescribed (`MASK_ICE_FIXED`), or
forced to zero (`MASK_ICE_NONE`). Faithful port of
`ybound_define_mask_ice` in `yelmo/src/yelmo_boundaries.f90:195`.

Defaults to all-dynamic, then applies domain-specific patterns keyed
off `bnd.regions` and the grid edges. Regional domains (e.g. Greenland,
Eurasia) use this to force zero ice outside the region of interest and
to fix the domain borders, which is what enables them to run without
spurious ice growth at the edges. Unknown domains fall through to the
DEFAULT case: dynamic interior, prescribed (`MASK_ICE_FIXED`) borders.

Note: unlike the Fortran routine this does not touch `bnd.calv_mask`
(initialised separately in Yelmo.jl).
"""
function define_mask_ice!(y::YelmoModel)
    domain   = strip(y.p.yelmo.domain)
    mask_ice = @view interior(y.bnd.mask_ice)[:, :, 1]
    regions  = @view interior(y.bnd.regions)[:, :, 1]
    nx, ny   = size(mask_ice)

    none    = Float64(MASK_ICE_NONE)
    fixed   = Float64(MASK_ICE_FIXED)
    dynamic = Float64(MASK_ICE_DYNAMIC)

    # Initially mark all points as dynamic (ice is solved).
    fill!(mask_ice, dynamic)

    # Region-code comparisons below cast to Float32 to match Fortran's
    # `wp = sp` semantics: region codes like 1.3 / 1.11 are not exactly
    # representable, so a Float64 `== 1.3` would fail against the Float32
    # value loaded from NetCDF (1.2999999523…). Fortran compares in
    # single precision, where both sides round to the same bits.

    if domain == "North"
        # Allow ice everywhere except the open ocean.
        @inbounds for j in 1:ny, i in 1:nx
            Float32(regions[i, j]) == 1.0f0 && (mask_ice[i, j] = none)
        end
        mask_ice[1, :]  .= none
        mask_ice[nx, :] .= none
        mask_ice[:, 1]  .= none
        mask_ice[:, ny] .= none

    elseif domain == "Eurasia"
        # Allow ice only in the Eurasia domain (1.2*).
        @inbounds for j in 1:ny, i in 1:nx
            r = Float32(regions[i, j])
            (r < 1.2f0 || r > 1.29f0) && (mask_ice[i, j] = none)
        end
        mask_ice[1, :]  .= none
        mask_ice[nx, :] .= none
        mask_ice[:, 1]  .= none
        mask_ice[:, ny] .= none

    elseif domain == "Greenland"
        fill!(mask_ice, none)
        @inbounds for j in 1:ny, i in 1:nx
            r = Float32(regions[i, j])
            if r == 1.3f0 || r == 1.11f0 || r == 1.0f0
                # Main Greenland (1.3), Ellesmere Island (1.11),
                # open-ocean connections (1.0).
                mask_ice[i, j] = dynamic
            end
        end

    elseif domain == "Antarctica"
        @inbounds for j in 1:ny, i in 1:nx
            Float32(regions[i, j]) == 2.0f0 && (mask_ice[i, j] = none)
        end
        mask_ice[1, :]  .= none
        mask_ice[nx, :] .= none
        mask_ice[:, 1]  .= none
        mask_ice[:, ny] .= none

    elseif domain == "EISMINT"
        # Ice can grow everywhere, except borders.
        mask_ice[1, :]  .= none
        mask_ice[nx, :] .= none
        mask_ice[:, 1]  .= none
        mask_ice[:, ny] .= none

    elseif domain in ("MISMIP", "MISMIP+", "TROUGH", "TROUGH-F17")
        # Ice can grow everywhere, except the farthest x-border.
        mask_ice[nx, :] .= none

    else
        # Unknown domain: dynamic interior, prescribed borders
        # (mask_ice can always be modified later).
        mask_ice[1, :]  .= fixed
        mask_ice[nx, :] .= fixed
        mask_ice[:, 1]  .= fixed
        mask_ice[:, ny] .= fixed
    end

    return y
end

# ---------------------------------------------------------------------------
# data_load!
# ---------------------------------------------------------------------------

"""
    data_load!(y::YelmoModel; par_path::Union{Nothing,AbstractString} = nothing,
               grad_lim_zb::Real = 0.05, boundaries::AbstractString = "infinite",
               group::AbstractString = "yelmo_init_topo") -> y

Read present-day reference data from NetCDF files described by
`y.p.yelmo_data` and write into `y.dta.pd_*` Fields. Optional flags
mirror the corresponding Fortran arguments to `ydata_load`:

  - `grad_lim_zb` — bedrock gradient limit for
    `adjust_topography_gradients!`. Default 0.05 matches the
    typical Fortran setting.
  - `boundaries`  — boundary string passed through to
    `adjust_topography_gradients!` (currently advisory; see helper).

Per-section behaviour:

  - **`pd_topo_load`**: read `H_ice`, `z_bed`, `z_srf` from
    `pd_topo_path`. If `pd_topo_names[3]` (z_bed_sd) is present,
    apply the same `z_bed_f_sd` scaling as `init_topo_load!`
    (read from `y.p.yelmo_init_topo`), so reference and runtime
    bedrock stay in lockstep. Then:
      * `remove_englacial_lakes!` against `z_sl = 0`,
      * clamp `H_ice < 1.0 m` → 0,
      * `adjust_topography_gradients!`,
      * delete ice from `bnd.mask_ice == MASK_ICE_NONE` cells (set
        `z_srf = max(z_bed, 0)` there),
      * recompute `pd_H_grnd = H_ice − (ρ_sw/ρ_ice) max(0 − z_bed, 0)`,
      * derive `pd_mask_bed`.
  - **`pd_tsrf_load`**: read `pd_tsrf_path[pd_tsrf_name]`. If
    `pd_tsrf_monthly`, take an annual mean from the third dim. If
    the resulting min < 100, assume °C and add 273.15.
  - **`pd_smb_load`**: read `pd_smb_path[pd_smb_name]`. Annual mean
    from monthly if requested. Apply unit conversion:
      * monthly → mm/d we → m/yr ie: × 1e-3 · 365 · ρ_w/ρ_ice
      * annual  → mm/yr we → m/yr ie: × 1e-3 · ρ_w/ρ_ice
    Clamp `|smb| < 1e-3` → 0.
  - **`pd_vel_load`**: read `pd_vel_path[pd_vel_names]`, derive
    `uxy_s = sqrt(ux² + uy²)`. If `pd_topo_load` was on, zero
    velocities where `H_ice < 1.0`.
  - **`pd_age_load`**: deferred. Variable table allocates `pd_age_iso`
    and `pd_depth_iso`, but the loader does not touch them. Set
    `pd_age_load = false` (default) until the isochrone port lands.

This is a load-time helper — operates on plain `interior(field)`
matrices rather than going through halos / Field BCs.
"""
function data_load!(y::YelmoModel;
                    par_path::Union{Nothing,AbstractString} = nothing,
                    grad_lim_zb::Real = 0.05,
                    boundaries::AbstractString = "infinite")
    y.p === nothing && error(
        "data_load!: y.p must be a YelmoParameters value (with " *
        "`yelmo_data` filled in). Construct YelmoModel with parameters " *
        "before calling data_load!.")

    par         = y.p.yelmo_data
    par_init    = y.p.yelmo_init_topo
    domain      = y.p.yelmo.domain
    grid_name   = y.p.yelmo.grid_name
    rho_ice     = y.c.rho_ice
    rho_w       = y.c.rho_w
    rho_sw      = y.c.rho_sw

    conv_we_ie       = rho_w / rho_ice
    conv_mmawe_maie  = 1e-3 * conv_we_ie
    conv_mmdwe_maie  = 1e-3 * 365.0 * conv_we_ie

    dx = _grid_dx(y.g)

    # 2D interior views of the dta Fields. Mutating these mutates
    # the underlying field storage in place; no writeback needed.
    H_ice      = @view interior(y.dta.pd_H_ice)[:, :, 1]
    z_bed      = @view interior(y.dta.pd_z_bed)[:, :, 1]
    z_srf      = @view interior(y.dta.pd_z_srf)[:, :, 1]
    H_grnd     = @view interior(y.dta.pd_H_grnd)[:, :, 1]
    mask_bed   = @view interior(y.dta.pd_mask_bed)[:, :, 1]
    T_srf      = @view interior(y.dta.pd_T_srf)[:, :, 1]
    smb_ref    = @view interior(y.dta.pd_smb_ref)[:, :, 1]
    ux_s       = @view interior(y.dta.pd_ux_s)[:, :, 1]
    uy_s       = @view interior(y.dta.pd_uy_s)[:, :, 1]
    uxy_s      = @view interior(y.dta.pd_uxy_s)[:, :, 1]

    nx, ny = size(H_ice)
    z_sl   = zeros(Float64, nx, ny)  # PD relative sea level = 0

    # ----- pd_topo_load -----------------------------------------------
    if par.pd_topo_load
        filename = _parse_path(par.pd_topo_path, domain, grid_name)
        nms      = par.pd_topo_names
        length(nms) >= 4 || error(
            "data_load!: pd_topo_names must have length >= 4 " *
            "(got $(length(nms))).")
        @info "data_load!: reading PD topography from $(filename)"

        _read_to!(H_ice, filename, nms[1])
        _read_to!(z_bed, filename, nms[2])

        # nms[3] (z_bed_sd): apply z_bed_f_sd scaling. Same factor as
        # init_topo_load! so runtime z_bed and reference pd_z_bed stay
        # consistent (Fortran yelmo_data.f90:223-229).
        if _name_present(nms[3])
            z_bed_sd_tmp = Array{Float64}(undef, nx, ny)
            _read_to!(z_bed_sd_tmp, filename, nms[3])
            @inbounds for j in 1:ny, i in 1:nx
                z_bed[i, j] += par_init.z_bed_f_sd * z_bed_sd_tmp[i, j]
            end
        end

        _read_to!(z_srf, filename, nms[4])

        # Remove englacial lakes (use z_sl = 0 for PD).
        remove_englacial_lakes!(H_ice, z_bed, z_srf, z_sl, rho_ice, rho_sw)
        @info "data_load!: removed englacial lakes from PD reference ice thickness."

        # Clean tiny ice values.
        @inbounds for j in 1:ny, i in 1:nx
            H_ice[i, j] < 1.0 && (H_ice[i, j] = 0.0)
        end

        adjust_topography_gradients!(z_bed, H_ice, grad_lim_zb, dx, boundaries)

        # Delete ice from cells where the boundary mask forbids it
        # (MASK_ICE_NONE). Mirrors Fortran yelmo_data.f90:262.
        mask_ice = @view interior(y.bnd.mask_ice)[:, :, 1]
        @inbounds for j in 1:ny, i in 1:nx
            if mask_ice[i, j] == Float64(MASK_ICE_NONE)
                H_ice[i, j] = 0.0
                z_srf[i, j] = max(z_bed[i, j], 0.0)
            end
        end

        # H_grnd diagnostic (sea level = 0 here).
        @inbounds for j in 1:ny, i in 1:nx
            H_grnd[i, j] = H_ice[i, j] - (rho_sw / rho_ice) * max(0.0 - z_bed[i, j], 0.0)
        end

        # mask_bed reclassification.
        @inbounds for j in 1:ny, i in 1:nx
            mb = MASK_BED_OCEAN
            if H_ice[i, j] == 0.0 && z_srf[i, j] > 0.0
                mb = MASK_BED_LAND
            elseif H_ice[i, j] > 0.0 && H_grnd[i, j] < 0.0
                mb = MASK_BED_FLOAT
            elseif H_ice[i, j] > 0.0 && H_grnd[i, j] >= 0.0
                mb = MASK_BED_FROZEN
            end
            mask_bed[i, j] = Float64(mb)
        end
    end

    # ----- pd_tsrf_load -----------------------------------------------
    if par.pd_tsrf_load
        filename = _parse_path(par.pd_tsrf_path, domain, grid_name)
        @info "data_load!: reading PD T_srf from $(filename)"
        if par.pd_tsrf_monthly
            _read_annual_mean_to!(T_srf, filename, par.pd_tsrf_name)
        else
            _read_to!(T_srf, filename, par.pd_tsrf_name)
        end
        # Convert °C → K when the reference field looks like Celsius.
        if minimum(T_srf) < 100.0
            T_srf .+= 273.15
        end
    end

    # ----- pd_smb_load ------------------------------------------------
    if par.pd_smb_load
        filename = _parse_path(par.pd_smb_path, domain, grid_name)
        @info "data_load!: reading PD smb from $(filename)"
        if par.pd_smb_monthly
            _read_annual_mean_to!(smb_ref, filename, par.pd_smb_name)
            smb_ref .*= conv_mmdwe_maie
        else
            _read_to!(smb_ref, filename, par.pd_smb_name)
            smb_ref .*= conv_mmawe_maie
        end
        @inbounds for j in 1:ny, i in 1:nx
            abs(smb_ref[i, j]) < 1e-3 && (smb_ref[i, j] = 0.0)
        end
    end

    # ----- pd_vel_load ------------------------------------------------
    if par.pd_vel_load
        filename = _parse_path(par.pd_vel_path, domain, grid_name)
        @info "data_load!: reading PD surface velocity from $(filename)"
        _read_to!(ux_s, filename, par.pd_vel_names[1])
        _read_to!(uy_s, filename, par.pd_vel_names[2])
        @inbounds for j in 1:ny, i in 1:nx
            uxy_s[i, j] = sqrt(ux_s[i, j]^2 + uy_s[i, j]^2)
        end
        if par.pd_topo_load
            @inbounds for j in 1:ny, i in 1:nx
                if H_ice[i, j] < 1.0
                    ux_s[i, j]  = 0.0
                    uy_s[i, j]  = 0.0
                    uxy_s[i, j] = 0.0
                end
            end
        end
    end

    # ----- pd_age_load — deferred -------------------------------------
    if par.pd_age_load
        @warn "data_load!: pd_age_load=true but isochrone reference loading is deferred. Skipping."
    end

    @info "data_load!: range(H_ice)   = " minimum(H_ice)   maximum(H_ice)
    @info "data_load!: range(z_srf)   = " minimum(z_srf)   maximum(z_srf)
    @info "data_load!: range(z_bed)   = " minimum(z_bed)   maximum(z_bed)
    @info "data_load!: range(T_srf)   = " minimum(T_srf)   maximum(T_srf)
    @info "data_load!: range(smb_ref) = " minimum(smb_ref) maximum(smb_ref)
    @info "data_load!: range(uxy_s)   = " minimum(uxy_s)   maximum(uxy_s)

    return y
end

# Read a 2D variable from a NetCDF file into a preallocated matrix.
# `missing` entries are silently passed through as `NaN` (Float64).
function _read_to!(out::AbstractMatrix{Float64},
                   filename::AbstractString, varname::AbstractString)
    NCDataset(filename) do ds
        v = ds[varname][:, :]
        size(v) == size(out) || error(
            "data_load!: variable \"$(varname)\" in $(filename) has " *
            "shape $(size(v)); expected $(size(out)).")
        @inbounds for j in axes(out, 2), i in axes(out, 1)
            out[i, j] = ismissing(v[i, j]) ? NaN : Float64(v[i, j])
        end
    end
    return out
end

# Read monthly NetCDF data and write annual mean into `out`.
function _read_annual_mean_to!(out::AbstractMatrix{Float64},
                                filename::AbstractString,
                                varname::AbstractString)
    NCDataset(filename) do ds
        v = ds[varname][:, :, :]
        size(v, 3) == 12 || error(
            "data_load!: monthly variable \"$(varname)\" in $(filename) " *
            "has third-dim length $(size(v, 3)); expected 12.")
        nx, ny = size(v, 1), size(v, 2)
        size(out) == (nx, ny) || error(
            "data_load!: variable \"$(varname)\" in $(filename) " *
            "has horizontal shape $((nx, ny)); expected $(size(out)).")
        @inbounds for j in 1:ny, i in 1:nx
            s = 0.0
            for m in 1:12
                s += ismissing(v[i, j, m]) ? 0.0 : Float64(v[i, j, m])
            end
            out[i, j] = s / 12.0
        end
    end
    return out
end

# Uniform-grid Δx (assumes uniform spacing — same convention as
# `_dx_thrm` in YelmoModelThrm).
function _grid_dx(grid)
    Δx = grid.Δxᶜᵃᵃ
    Δx isa Number || error("YelmoModelData requires uniform x-spacing (got $(typeof(Δx))).")
    return abs(Float64(Δx))
end

# ---------------------------------------------------------------------------
# data_compare!
# ---------------------------------------------------------------------------

"""
    data_compare!(y::YelmoModel) -> y

Update the comparison-error fields and the scalar RMSEs on `y.dta`
from the current model state. Mirrors Fortran `ydata_compare`.

Errors written:

  - `pd_err_H_ice    = tpo.H_ice − dta.pd_H_ice`
  - `pd_err_z_srf    = tpo.z_srf − dta.pd_z_srf`
  - `pd_err_smb_ref  = bnd.smb_ref − dta.pd_smb_ref`
  - `pd_err_uxy_s    = dyn.uxy_s − dta.pd_uxy_s`

(`pd_err_z_bed`, `pd_err_depth_iso` are not updated here — `z_bed`
parity is handled via `tpo.z_bed`/`bnd.z_bed`, and isochrone
comparison is deferred.)

RMSEs written to `y.dta.rmse`:

  - `H`       : RMSE over `H_ice ≠ 0 ∨ pd_H_ice ≠ 0`
  - `zsrf`    : RMSE over non-zero `pd_err_z_srf`
  - `uxy`     : RMSE over non-zero `pd_err_uxy_s`
  - `loguxy`  : RMSE of `log(uxy_s) − log(pd_uxy_s)` (with the
                Fortran convention `log(x) := x` when `x ≤ 0`)

For `domain == "Greenland"`, every mask is intersected with
`bnd.regions ≈ 1.3` (continental Greenland), matching Fortran
`ydata_compare`.

Each metric is set to `NaN` when the contributing mask is empty or
the result is exactly zero (Fortran assigns `mv` in both cases).
"""
function data_compare!(y::YelmoModel)
    y.p === nothing && error(
        "data_compare!: y.p must be non-nothing so the domain string " *
        "is available for the regional masking step.")

    domain = y.p.yelmo.domain

    H_now    = @view interior(y.tpo.H_ice)[:, :, 1]
    H_ref    = @view interior(y.dta.pd_H_ice)[:, :, 1]
    zs_now   = @view interior(y.tpo.z_srf)[:, :, 1]
    zs_ref   = @view interior(y.dta.pd_z_srf)[:, :, 1]
    smb_now  = @view interior(y.bnd.smb_ref)[:, :, 1]
    smb_ref  = @view interior(y.dta.pd_smb_ref)[:, :, 1]
    u_now    = @view interior(y.dyn.uxy_s)[:, :, 1]
    u_ref    = @view interior(y.dta.pd_uxy_s)[:, :, 1]

    err_H    = @view interior(y.dta.pd_err_H_ice)[:, :, 1]
    err_zs   = @view interior(y.dta.pd_err_z_srf)[:, :, 1]
    err_smb  = @view interior(y.dta.pd_err_smb_ref)[:, :, 1]
    err_u    = @view interior(y.dta.pd_err_uxy_s)[:, :, 1]

    nx, ny = size(H_now)

    # ----- Errors ------------------------------------------------------
    @inbounds for j in 1:ny, i in 1:nx
        err_H[i, j]   = H_now[i, j]   - H_ref[i, j]
        err_zs[i, j]  = zs_now[i, j]  - zs_ref[i, j]
        err_smb[i, j] = smb_now[i, j] - smb_ref[i, j]
        err_u[i, j]   = u_now[i, j]   - u_ref[i, j]
    end

    # ----- Regional mask ----------------------------------------------
    mask_region = trues(nx, ny)
    if domain == "Greenland"
        regions = @view interior(y.bnd.regions)[:, :, 1]
        @inbounds for j in 1:ny, i in 1:nx
            mask_region[i, j] = abs(regions[i, j] - 1.3) < 1e-6
        end
    end

    rmse = y.dta.rmse  # mutate in place

    # ----- rmse_H -------------------------------------------------------
    s = 0.0; n = 0
    @inbounds for j in 1:ny, i in 1:nx
        if (H_now[i, j] != 0.0 || H_ref[i, j] != 0.0) && mask_region[i, j]
            d = H_now[i, j] - H_ref[i, j]
            s += d * d
            n += 1
        end
    end
    rmse.H = (n > 0 && s > 0.0) ? sqrt(s / n) : NaN

    # ----- rmse_zsrf ----------------------------------------------------
    s = 0.0; n = 0
    @inbounds for j in 1:ny, i in 1:nx
        if err_zs[i, j] != 0.0 && mask_region[i, j]
            s += err_zs[i, j]^2
            n += 1
        end
    end
    rmse.zsrf = (n > 0 && s > 0.0) ? sqrt(s / n) : NaN

    # ----- rmse_uxy -----------------------------------------------------
    s = 0.0; n = 0
    @inbounds for j in 1:ny, i in 1:nx
        if err_u[i, j] != 0.0 && mask_region[i, j]
            s += err_u[i, j]^2
            n += 1
        end
    end
    rmse.uxy = (n > 0 && s > 0.0) ? sqrt(s / n) : NaN

    # ----- rmse_loguxy --------------------------------------------------
    # Fortran convention: tmp = u_ref; where(u_ref > 0) tmp = log(tmp)
    # (i.e. zero / negative entries are passed through unchanged); same
    # for u_now. Mask = (tmp != 0 || tmp1 != 0) ∧ region.
    s = 0.0; n = 0
    @inbounds for j in 1:ny, i in 1:nx
        a = u_ref[i, j] > 0.0 ? log(u_ref[i, j]) : u_ref[i, j]
        b = u_now[i, j] > 0.0 ? log(u_now[i, j]) : u_now[i, j]
        if (a != 0.0 || b != 0.0) && mask_region[i, j]
            s += (b - a)^2
            n += 1
        end
    end
    rmse.loguxy = (n > 0 && s > 0.0) ? sqrt(s / n) : NaN

    return y
end

end # module YelmoModelData
