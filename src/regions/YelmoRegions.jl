"""
    YelmoRegions

Regional analysis + time-series I/O for `YelmoModel`. Port of
Fortran `yelmo/src/yelmo_regions.f90`.

A "region" is a 2D Boolean mask over the model domain plus a
NetCDF output file. After each model step the user calls
`update_regions!` to recompute regional aggregate diagnostics
(volume, area, mean velocity, mean SMB, etc.) for every region,
then `write_regions!(regs, y, time)` to append the current
diagnostics to each region's time-series file.

Typical usage:

```julia
y    = YelmoModel(...)
regs = init_regions(y)                # default: whole-domain region
add_region!(regs, y, "shelf", mask_shelf)

for n in 1:N
    step!(y, dt)
    update_regions!(regs, y)
    write_regions!(regs, y, y.time)
end
```

Mirrors Fortran `yelmo_calc_region` (~38 aggregate diagnostics
across total / grounded / floating sub-masks) and the NetCDF
schema of `yelmo_write_reg_init` / `yelmo_write_reg_step`.
"""
module YelmoRegions

using NCDatasets
using Oceananigans.Fields: interior

using ..YelmoCore: YelmoModel

export RegionDiagnostics, YelmoRegion, RegionsCollection
export init_regions, add_region!, update_regions!, write_regions!
export calc_region_diagnostics!

# ---------------------------------------------------------------------
# Public types — declared before the includes that consume them.
# ---------------------------------------------------------------------

"""
    RegionDiagnostics

Mutable container for the ~38 scalar regional aggregates defined in
Fortran `yelmo_regions.f90:208 yelmo_calc_region`. Grouped by sub-mask
(total ice, grounded, floating).

Units (mirroring Fortran):

  - `H_ice*`, `z_srf*`, `z_bed`, `z_sl`, `H_w` : m
  - `dHidt`, `dzsdt`, `uxy_*`, `smb`, `bmb*`   : m/yr
  - `T_srf`, `T_shlf`                          : K
  - `f_pmp`                                    : fraction
  - `V_ice*`                                   : km^3
  - `A_ice*`                                   : km^2
  - `dVidt`                                    : km^3/yr
  - `dmb`, `cmb`, `cmb_flt`, `cmb_grnd`        : m^3/yr
  - `fwf`                                      : Sv
  - `V_sl`                                     : km^3
  - `V_sle`                                    : m sea-level equivalent
"""
mutable struct RegionDiagnostics
    # Total ice
    H_ice::Float64
    z_srf::Float64
    dHidt::Float64
    H_ice_max::Float64
    dzsdt::Float64

    V_ice::Float64
    A_ice::Float64
    dVidt::Float64
    fwf::Float64

    dmb::Float64
    cmb::Float64
    cmb_flt::Float64
    cmb_grnd::Float64

    V_sl::Float64
    V_sle::Float64

    uxy_bar::Float64
    uxy_s::Float64
    uxy_b::Float64

    z_bed::Float64
    smb::Float64
    T_srf::Float64
    bmb::Float64

    # Grounded ice
    H_ice_g::Float64
    z_srf_g::Float64
    V_ice_g::Float64
    A_ice_g::Float64
    uxy_bar_g::Float64
    uxy_s_g::Float64
    uxy_b_g::Float64
    f_pmp::Float64
    H_w::Float64
    bmb_g::Float64

    # Floating ice
    H_ice_f::Float64
    V_ice_f::Float64
    A_ice_f::Float64
    uxy_bar_f::Float64
    uxy_s_f::Float64
    uxy_b_f::Float64
    z_sl::Float64
    bmb_shlf::Float64
    T_shlf::Float64
end

RegionDiagnostics() = RegionDiagnostics(ntuple(_ -> 0.0, fieldcount(RegionDiagnostics))...)

"""
    YelmoRegion

A single named region — its mask, its NetCDF output path (empty string
if I/O is disabled for this region), and its current diagnostics.
"""
mutable struct YelmoRegion
    name::String
    mask::Matrix{Bool}
    outfile::String
    diag::RegionDiagnostics
end

"""
    RegionsCollection

A collection of `YelmoRegion`s. Built via `init_regions(y)` and grown
via `add_region!`.
"""
mutable struct RegionsCollection
    regions::Vector{YelmoRegion}
end

Base.length(rs::RegionsCollection) = length(rs.regions)
Base.iterate(rs::RegionsCollection, args...) = iterate(rs.regions, args...)
Base.getindex(rs::RegionsCollection, i) = rs.regions[i]

# ---------------------------------------------------------------------
# Implementation files (depend on the types above).
# ---------------------------------------------------------------------

include("calc_region.jl")
include("regions_io.jl")

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

"""
    init_regions(y::YelmoModel;
                 outdir = y.rundir,
                 include_whole_domain = true,
                 whole_domain_name    = "domain") -> RegionsCollection

Construct an empty `RegionsCollection` collection. When
`include_whole_domain = true` (default) a region covering the entire
model domain is added automatically with name `whole_domain_name` and
output file `<outdir>/region_<name>.nc`. The init NetCDF is written
synchronously so subsequent `write_regions!` calls only append.
"""
function init_regions(y::YelmoModel;
                      outdir::AbstractString = y.rundir,
                      include_whole_domain::Bool = true,
                      whole_domain_name::AbstractString = "domain")
    regs = RegionsCollection(YelmoRegion[])
    if include_whole_domain
        Nx = size(y.tpo.H_ice, 1)
        Ny = size(y.tpo.H_ice, 2)
        full_mask = trues(Nx, Ny)
        add_region!(regs, y, String(whole_domain_name), full_mask;
                    outdir = outdir)
    end
    return regs
end

"""
    add_region!(regs::RegionsCollection, y::YelmoModel,
                name::AbstractString, mask::AbstractMatrix{Bool};
                outdir          = y.rundir,
                write_to_file   = true) -> YelmoRegion

Register a new region. `mask` must have shape `(Nx, Ny)` where Nx, Ny
are the model's grid sizes. When `write_to_file = true` the init
NetCDF is created at `<outdir>/region_<name>.nc` and the static mask
variable is written; subsequent `write_regions!` calls append to
this file. Pass `write_to_file = false` for a memory-only region.
"""
function add_region!(regs::RegionsCollection, y::YelmoModel,
                     name::AbstractString,
                     mask::AbstractMatrix{Bool};
                     outdir::AbstractString = y.rundir,
                     write_to_file::Bool = true)
    Nx = size(y.tpo.H_ice, 1)
    Ny = size(y.tpo.H_ice, 2)
    size(mask) == (Nx, Ny) ||
        error("add_region!: mask shape $(size(mask)) does not match grid " *
              "($(Nx), $(Ny)).")

    outfile = ""
    if write_to_file
        isdir(outdir) || mkpath(outdir)
        outfile = joinpath(outdir, "region_$(name).nc")
        _write_region_init(outfile, y, mask)
    end

    reg = YelmoRegion(String(name), Matrix{Bool}(mask), outfile,
                      RegionDiagnostics())
    push!(regs.regions, reg)
    return reg
end

"""
    update_regions!(regs::RegionsCollection, y::YelmoModel) -> regs

Recompute every region's `diag` from the current `y` state.
"""
function update_regions!(regs::RegionsCollection, y::YelmoModel)
    for reg in regs.regions
        calc_region_diagnostics!(reg.diag, y, reg.mask)
    end
    return regs
end

"""
    write_regions!(regs::RegionsCollection, y::YelmoModel, time::Real) -> regs

Append the currently-stored diagnostics to each region's NetCDF
file at the time index corresponding to `time`. Regions with empty
`outfile` (in-memory only) are skipped.
"""
function write_regions!(regs::RegionsCollection, y::YelmoModel, time::Real)
    for reg in regs.regions
        isempty(reg.outfile) && continue
        _write_region_step(reg.outfile, reg.diag, Float64(time))
    end
    return regs
end

end # module YelmoRegions
