# InitMIPBenchmark — real-data ice-sheet benchmark (Greenland or Antarctica).
#
# This is a thin grid-spec struct: it reads xc/yc from a REGIONS.nc
# file and exposes them so the YelmoBenchmarks extension can build the
# Oceananigans grid. All physical fields (topography, climate, GHF) are
# loaded separately in run.jl via init_topo_load! / data_load!.

struct InitMIPBenchmark <: AbstractBenchmark
    xc::Vector{Float64}   # cell-centre x [m]
    yc::Vector{Float64}   # cell-centre y [m]
end

"""
    InitMIPBenchmark(regions_nc::AbstractString) -> InitMIPBenchmark

Read `xc`/`yc` coordinates from a REGIONS NetCDF file and construct the
benchmark. Coordinates are converted to metres if the file stores them
in kilometres. Works for any real-data domain (e.g. GRL-16KM,
ANT-32KM).
"""
function InitMIPBenchmark(regions_nc::AbstractString)
    NCDataset(regions_nc) do ds
        xc = Vector{Float64}(ds["xc"][:])
        yc = Vector{Float64}(ds["yc"][:])
        xu = lowercase(strip(get(ds["xc"].attrib, "units", "")))
        yu = lowercase(strip(get(ds["yc"].attrib, "units", "")))
        (xu == "km" || xu == "kilometers") && (xc .*= 1e3)
        (yu == "km" || yu == "kilometers") && (yc .*= 1e3)
        return InitMIPBenchmark(xc, yc)
    end
end

function state(b::InitMIPBenchmark, ::Real)
    return (xc = b.xc, yc = b.yc)
end
