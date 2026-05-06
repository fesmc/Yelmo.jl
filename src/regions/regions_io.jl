# ----------------------------------------------------------------------
# NetCDF I/O for `YelmoRegions` time-series files. One file per region.
#
# Layout (mirrors Fortran `yelmo_regions.f90:426 yelmo_write_reg_init` /
# `:450 yelmo_write_reg_step`):
#
#   dims:
#     xc      = Nx           (cell-centre x, km)
#     yc      = Ny           (cell-centre y, km)
#     time    = unlimited    (years)
#
#   static vars:
#     xc(xc), yc(yc)
#     mask(xc, yc)           (Int8: 0/1, the region selector)
#
#   time-series vars (one float64 per timestep):
#     time(time)
#     H_ice(time), z_srf(time), dHidt(time), H_ice_max(time), dzsdt(time),
#     V_ice(time), A_ice(time), dVidt(time), fwf(time),
#     dmb(time), cmb(time), cmb_flt(time), cmb_grnd(time),
#     V_sl(time), V_sle(time),
#     uxy_bar(time), uxy_s(time), uxy_b(time),
#     z_bed(time), smb(time), T_srf(time), bmb(time),
#     H_ice_g(time), z_srf_g(time), V_ice_g(time), A_ice_g(time),
#     uxy_bar_g(time), uxy_s_g(time), uxy_b_g(time),
#     f_pmp(time), H_w(time), bmb_g(time),
#     H_ice_f(time), V_ice_f(time), A_ice_f(time),
#     uxy_bar_f(time), uxy_s_f(time), uxy_b_f(time),
#     z_sl(time), bmb_shlf(time), T_shlf(time)
#
# Unit conventions follow the Fortran writer; the Fortran writer
# multiplies a few volumes by 1e-6 ("1e6 km^3" units). We write the
# same — V_ice / V_ice_g / V_ice_f / V_sl in 1e6 km^3, A_ice variants
# in 1e6 km^2, V_sle in m sle.
# ----------------------------------------------------------------------

# Variable schema: list of `(field_symbol, varname, unit_string,
# scale, long_name)` tuples. `field_symbol` is the
# `RegionDiagnostics` field; `scale` is the multiplier applied at
# write time (so volumes go to "1e6 km^3" etc.).
const _REGION_VARS = (
    # Total ice
    (:H_ice,     "H_ice",     "m",         1.0,  "Mean ice thickness"),
    (:z_srf,     "z_srf",     "m",         1.0,  "Mean surface elevation"),
    (:dHidt,     "dHidt",     "m/a",       1.0,  "Mean rate ice thickness change"),
    (:H_ice_max, "H_ice_max", "m",         1.0,  "Max ice thickness"),
    (:dzsdt,     "dzsdt",     "m/a",       1.0,  "Mean rate surface elevation change"),

    (:V_ice,     "V_ice",     "1e6 km^3",  1e-6, "Ice volume"),
    (:A_ice,     "A_ice",     "1e6 km^2",  1e-6, "Ice area"),
    (:dVidt,     "dVidt",     "km^3/a",    1.0,  "Rate volume change"),
    (:fwf,       "fwf",       "Sv",        1.0,  "Freshwater flux"),

    (:dmb,       "dmb",       "m^3/yr",    1.0,  "Discharge mass balance rate"),
    (:cmb,       "cmb",       "m^3/yr",    1.0,  "Calving mass balance rate"),
    (:cmb_flt,   "cmb_flt",   "m^3/yr",    1.0,  "Calving mass balance rate (floating)"),
    (:cmb_grnd,  "cmb_grnd",  "m^3/yr",    1.0,  "Calving mass balance rate (grounded)"),

    (:V_sl,      "V_sl",      "1e6 km^3",  1e-6, "Ice volume above flotation"),
    (:V_sle,     "V_sle",     "m sle",     1.0,  "Sea-level equivalent volume"),

    (:uxy_bar,   "uxy_bar",   "m/a",       1.0,  "Mean depth-averaged velocity"),
    (:uxy_s,     "uxy_s",     "m/a",       1.0,  "Mean surface velocity"),
    (:uxy_b,     "uxy_b",     "m/a",       1.0,  "Mean basal velocity"),

    (:z_bed,     "z_bed",     "m",         1.0,  "Mean bedrock elevation"),
    (:smb,       "smb",       "m/a",       1.0,  "Mean surface mass balance"),
    (:T_srf,     "T_srf",     "K",         1.0,  "Mean surface temperature"),
    (:bmb,       "bmb",       "m/a",       1.0,  "Mean total basal mass balance"),

    # Grounded
    (:H_ice_g,   "H_ice_g",   "m",         1.0,  "Mean ice thickness (grounded)"),
    (:z_srf_g,   "z_srf_g",   "m",         1.0,  "Mean surface elevation (grounded)"),
    (:V_ice_g,   "V_ice_g",   "1e6 km^3",  1e-6, "Ice volume (grounded)"),
    (:A_ice_g,   "A_ice_g",   "1e6 km^2",  1e-6, "Ice area (grounded)"),
    (:uxy_bar_g, "uxy_bar_g", "m/a",       1.0,  "Mean depth-averaged velocity (grounded)"),
    (:uxy_s_g,   "uxy_s_g",   "m/a",       1.0,  "Mean surface velocity (grounded)"),
    (:uxy_b_g,   "uxy_b_g",   "m/a",       1.0,  "Mean basal velocity (grounded)"),
    (:f_pmp,     "f_pmp",     "1",         1.0,  "Temperate fraction (grounded)"),
    (:H_w,       "H_w",       "m",         1.0,  "Mean basal water thickness (grounded)"),
    (:bmb_g,     "bmb_g",     "m/a",       1.0,  "Mean basal mass balance (grounded)"),

    # Floating
    (:H_ice_f,   "H_ice_f",   "m",         1.0,  "Mean ice thickness (floating)"),
    (:V_ice_f,   "V_ice_f",   "1e6 km^3",  1e-6, "Ice volume (floating)"),
    (:A_ice_f,   "A_ice_f",   "1e6 km^2",  1e-6, "Ice area (floating)"),
    (:uxy_bar_f, "uxy_bar_f", "m/a",       1.0,  "Mean depth-averaged velocity (floating)"),
    (:uxy_s_f,   "uxy_s_f",   "m/a",       1.0,  "Mean surface velocity (floating)"),
    (:uxy_b_f,   "uxy_b_f",   "m/a",       1.0,  "Mean basal velocity (floating)"),
    (:z_sl,      "z_sl",      "m",         1.0,  "Mean sea level (floating)"),
    (:bmb_shlf,  "bmb_shlf",  "m/a",       1.0,  "Mean basal mass balance (floating)"),
    (:T_shlf,    "T_shlf",    "K",         1.0,  "Mean marine shelf temperature (floating)"),
)

# Initialise the NetCDF file: dims, static `xc` / `yc` axes, the
# 2D region mask, and the unlimited `time` dimension. Each per-step
# scalar variable is defined here so subsequent `_write_region_step`
# calls only assign at the new time index.
function _write_region_init(path::AbstractString, y::YelmoModel,
                            mask::AbstractMatrix{Bool})
    Nx, Ny = size(mask)
    # Cell centres in km. We don't have direct access to a cell-centre
    # coordinate vector on the YelmoModel struct, so derive from
    # spacing and origin of the underlying RectilinearGrid.
    dx_g = y.g.Δxᶜᵃᵃ
    dy_g = y.g.Δyᵃᶜᵃ
    dx = abs(Float64(dx_g))
    dy = abs(Float64(dy_g))
    # Origin: assume xc[1] = dx/2, yc[1] = dy/2 (Oceananigans Center
    # convention for a (0, Lx) × (0, Ly) grid). Convert to km.
    xc_km = [(i - 0.5) * dx * 1e-3 for i in 1:Nx]
    yc_km = [(j - 0.5) * dy * 1e-3 for j in 1:Ny]

    NCDataset(path, "c") do ds
        defDim(ds, "xc",   Nx)
        defDim(ds, "yc",   Ny)
        defDim(ds, "time", Inf)         # unlimited

        let v = defVar(ds, "xc",   Float64, ("xc",))
            v[:] = xc_km
            v.attrib["units"] = "kilometers"
            v.attrib["long_name"] = "x cell centres"
        end
        let v = defVar(ds, "yc",   Float64, ("yc",))
            v[:] = yc_km
            v.attrib["units"] = "kilometers"
            v.attrib["long_name"] = "y cell centres"
        end
        let v = defVar(ds, "mask", Int8, ("xc", "yc"))
            mi = Int8.(mask)
            v[:, :] = mi
            v.attrib["units"] = "1"
            v.attrib["long_name"] = "Region mask"
        end
        let v = defVar(ds, "time", Float64, ("time",))
            v.attrib["units"] = "years"
            v.attrib["long_name"] = "Simulation time"
        end

        for (_, vname, units, _, longname) in _REGION_VARS
            v = defVar(ds, vname, Float64, ("time",))
            v.attrib["units"]     = units
            v.attrib["long_name"] = longname
        end
    end
    return path
end

# Append a single time-step record. Time is keyed by exact match
# against the existing `time` axis; if `time` is not already present,
# a new index at the end is created.
function _write_region_step(path::AbstractString,
                            diag::RegionDiagnostics,
                            time::Float64)
    NCDataset(path, "a") do ds
        # Resolve the time index — exact-match against the existing
        # axis, or append at the end.
        tvar = ds["time"]
        existing = Vector{Float64}(tvar[:])
        n = findfirst(==(time), existing)
        idx = n === nothing ? length(existing) + 1 : n

        ds["time"][idx] = time
        for (sym, vname, _, scale, _) in _REGION_VARS
            ds[vname][idx] = scale * getfield(diag, sym)
        end
    end
    return path
end
