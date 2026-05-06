# ----------------------------------------------------------------------
# SCRIP-style horizontal regridding for 2D fields read from a restart
# file. Vendored from `palma-ice/ScripMap.jl` (the Julia port of
# Yelmo Fortran's `mapping_scrip` module). Only the core load + apply
# routines are inlined here; the optional fill / filter helpers (which
# would pull in `NearestNeighbors` and `ImageFiltering`) are omitted.
#
# MIGRATION NOTE: this is a stop-gap. Once `ScripMap.jl` is registered
# in the General registry (or we are willing to track it via a `Pkg`
# develop / git URL dependency in `Project.toml`), this file should
# be removed in favour of `using ScripMap`. The user-facing API
# (`gen_map_filename`, `map_scrip_load`, `map_scrip_field`,
# `vec_stat`) is held byte-identical to the upstream package so that
# swap is mechanical.
#
# Usage by the Yelmo restart loader:
#
#   if restart_grid_name != target_grid_name
#       mps = map_scrip_load(restart_grid_name, target_grid_name,
#                            "maps"; method="con")
#       # then per 2D variable read from the restart NetCDF:
#       _, field2d = map_scrip_field(mps, varname, raw_2d)
#   end
#
# Source / target grid sizes, cell areas and remap weights are loaded
# directly from the SCRIP NetCDF; the mapping itself is a sparse
# weighted reduction over `(src_address, dst_address, remap_matrix)`
# triplets.
# ----------------------------------------------------------------------

"""
    gen_map_filename(src_name, dst_name, fldr, method) -> String

Standard SCRIP-map filename: `<fldr>/scrip-<method>_<src>_<dst>.nc`.
Mirrors the Fortran `gen_map_filename` convention; CDO writes maps
under this naming scheme.
"""
function gen_map_filename(src_name::AbstractString, dst_name::AbstractString,
                          fldr::AbstractString, method::AbstractString)
    return joinpath(fldr, "scrip-" * method * "_" * src_name * "_" * dst_name * ".nc")
end

"""
    map_scrip_load(src_name, dst_name, fldr; method="con") -> Dict

Load a SCRIP map (CDO-generated NetCDF) into a `Dict` keyed by the
standard SCRIP variable names (`src_address`, `dst_address`,
`remap_matrix`, `src_grid_dims`, etc.). The `Dict` shape is held
byte-identical to upstream `ScripMap.jl` so call-sites can swap to
`using ScripMap` later without code changes.

`method` selects the suffix in the filename (`"con"` for
conservative mapping). Errors with a clear message if the file is
missing — the typical cause is "user forgot to generate the map
externally with CDO".
"""
function map_scrip_load(src_name::AbstractString, dst_name::AbstractString,
                        fldr::AbstractString; method::AbstractString = "con")
    filename = gen_map_filename(src_name, dst_name, fldr, method)
    isfile(filename) || error(
        "map_scrip_load: SCRIP map file not found at $(filename). " *
        "These files are generated externally by CDO; see Yelmo " *
        "Fortran documentation under maps/ for the conventional " *
        "naming + generation pipeline.")

    map = Dict{String,Any}()
    map["src_name"]  = String(src_name)
    map["dst_name"]  = String(dst_name)
    map["map_fname"] = filename

    NCDataset(filename) do nc
        map["src_grid_size"]  = nc.dim["src_grid_size"]
        map["dst_grid_size"]  = nc.dim["src_grid_size"]
        map["dst_grid_corners"] = haskey(nc, "dst_grid_corner_lat") ?
            nc.dim["dst_grid_corners"] : 0
        map["src_grid_rank"]  = nc.dim["src_grid_rank"]
        map["dst_grid_rank"]  = nc.dim["dst_grid_rank"]
        map["num_wgts"]       = nc.dim["num_wgts"]
        map["num_links"]      = nc.dim["num_links"]

        # Static metadata. Center lat/lon and frac/area are loaded
        # for downstream callers that may need them; the actual
        # mapping itself only needs (src_address, dst_address,
        # remap_matrix, dst_grid_dims).
        map["src_grid_dims"]       = nc["src_grid_dims"][:]
        map["dst_grid_dims"]       = nc["dst_grid_dims"][:]
        map["src_grid_center_lat"] = nc["src_grid_center_lat"][:]
        map["dst_grid_center_lat"] = nc["dst_grid_center_lat"][:]
        map["src_grid_center_lon"] = nc["src_grid_center_lon"][:]
        map["dst_grid_center_lon"] = nc["dst_grid_center_lon"][:]
        if haskey(nc, "dst_grid_corner_lat")
            map["dst_grid_corner_lat"] = nc["dst_grid_corner_lat"][:]
        end
        if haskey(nc, "dst_grid_corner_lon")
            map["dst_grid_corner_lon"] = nc["dst_grid_corner_lon"][:]
        end
        map["src_grid_imask"] = nc["src_grid_imask"][:]
        map["dst_grid_imask"] = nc["dst_grid_imask"][:]
        if haskey(nc, "src_grid_area")
            map["src_grid_area"] = nc["src_grid_area"][:]
        end
        if haskey(nc, "dst_grid_area")
            map["dst_grid_area"] = nc["dst_grid_area"][:]
        end
        map["src_grid_frac"] = nc["src_grid_frac"][:]
        map["dst_grid_frac"] = nc["dst_grid_frac"][:]
        map["src_address"]   = nc["src_address"][:]
        map["dst_address"]   = nc["dst_address"][:]
        # `[:, :]` preserves the (num_wgts, num_links) 2D shape — `[:]`
        # flattens, breaking the `remap_matrix[1, j1:j2]` reads in
        # `map_scrip_field`. The upstream ScripMap.jl uses `[:]`; the
        # 2D form is the corrected version for current NCDatasets.
        map["remap_matrix"]  = nc["remap_matrix"][:, :]
    end

    return map
end

"""
    map_scrip_field(map::Dict, var_name, var1::AbstractMatrix; method="mean")
        -> (mask2, var2)

Apply a SCRIP map to a 2D source field `var1` of shape matching
`map["src_grid_size"]`, returning a 2D destination field of shape
`Tuple(map["dst_grid_dims"])`. `mask2` is a Bool matrix marking
which destination cells received any contribution.

`method = "mean"` (default) is the Fortran `normalize_opt = "fracarea"`
variant — area-weighted mean of source contributors. `"count"` and
`"stdev"` are also available (see `vec_stat`).

NOTE: the optional `fill_method` / `filt_method` keyword arguments
from upstream `ScripMap.jl` are NOT included here — they pull in
`NearestNeighbors` and `ImageFiltering`, and the Yelmo restart loader
does not use them. Cells with no source contribution are left as
`NaN` in `var2` and `false` in `mask2`. Callers that need post-fill
should run their own pass.
"""
function map_scrip_field(map::Dict, var_name::AbstractString,
                         var1::AbstractMatrix{T};
                         method::AbstractString = "mean") where T

    @assert length(var1) == map["src_grid_size"] "map_scrip_field: " *
        "src array length $(length(var1)) does not match map " *
        "src_grid_size $(map["src_grid_size"]) for $(var_name)."

    npts2 = map["dst_grid_dims"][1] * map["dst_grid_dims"][2]
    var1_vec = reshape(var1, length(var1))

    var2_vec  = fill(NaN,   npts2)
    mask2_vec = fill(false, npts2)

    dst_address = map["dst_address"]
    src_address = map["src_address"]
    remap_matrix = map["remap_matrix"]
    num_links = map["num_links"]

    j1 = 0
    j2 = 0
    @inbounds for k in 1:npts2
        j1 = j2 + 1
        j1 > length(dst_address) && break

        # If the next link's destination address is past the current
        # k, this k has no incoming links — skip and rewind.
        if dst_address[j1] > k
            j1 -= 1
            continue
        end

        # Walk forward until dst_address changes, defining [j1, j2].
        for j in j1:num_links
            if dst_address[j] == dst_address[j1]
                j2 = j
            else
                break
            end
        end

        # Gather source values + weights for this destination cell.
        idx_src = src_address[j1:j2]
        var1_now = var1_vec[idx_src]
        # remap_matrix is shape (num_wgts, num_links); we use the
        # first weight (fracarea) as in the Fortran reference.
        wts1_now = remap_matrix[1, j1:j2]

        var2_vec[k]  = vec_stat(var1_now; wts = wts1_now, method = method)
        mask2_vec[k] = true
    end

    var2  = reshape(var2_vec,  Tuple(map["dst_grid_dims"]))
    mask2 = reshape(mask2_vec, Tuple(map["dst_grid_dims"]))

    # Match the input element type when possible (NaN → NaN, real → T).
    var2_T = convert.(T, var2)
    return mask2, var2_T
end

"""
    vec_stat(var::Vector; wts, method="mean") -> Float64

Weighted statistic over a vector with NaN handling. Supported
`method`s: `"mean"` (default, area-weighted average), `"count"`
(most-frequent value), `"stdev"` (unbiased weighted standard
deviation). Returns `NaN` if no non-NaN points have positive total
weight.
"""
function vec_stat(var::AbstractVector{<:Number};
                  wts::AbstractVector{<:Number} = ones(length(var)),
                  method::AbstractString = "mean")
    kk = findall(.!isnan.(var))
    wts_tot = sum(wts[kk])

    if wts_tot <= 0.0
        return NaN
    end

    if method == "mean"
        return sum((wts[kk] ./ wts_tot) .* var[kk])
    elseif method == "count"
        # Most frequently occurring value, weighted by area.
        counts = Dict{eltype(var), Float64}()
        for (val, wt) in zip(var[kk], wts[kk])
            isa(val, Number) && isnan(val) && continue
            counts[val] = get(counts, val, 0.0) + wt
        end
        return findmax(counts)[2]
    elseif method == "stdev"
        npt = length(kk)
        if npt > 2
            mean_val = sum((wts[kk] ./ wts_tot) .* var[kk])
            var_out  = (npt / (npt - 1.0)) *
                       sum((wts[kk] ./ wts_tot) .* (var[kk] .- mean_val) .^ 2)
            return sqrt(var_out)
        else
            return 0.0
        end
    else
        error("vec_stat: method not recognized: $(method).")
    end
end
