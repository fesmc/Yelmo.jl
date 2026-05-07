# ----------------------------------------------------------------------
# SCRIP-style horizontal regridding for 2D fields read from a restart
# file. Vendored from `palma-ice/ScripMap.jl` (the Julia port of
# Yelmo Fortran's `mapping_scrip` module). Core load + apply routines
# are inlined here; the optional fill / filter helpers from upstream
# (`fill_weighted!`, `fill_nearest!`, Gaussian filter) are
# re-implemented in pure Julia rather than depending on
# `NearestNeighbors` and `ImageFiltering` — the regular-grid case
# does not need a KD-tree, and a separable 2D Gaussian is ~30 lines.
#
# MIGRATION NOTE: this is a stop-gap. Once `ScripMap.jl` is registered
# in the General registry (or we are willing to track it via a `Pkg`
# develop / git URL dependency in `Project.toml`), this file should
# be removed in favour of `using ScripMap`. The user-facing API
# (`gen_map_filename`, `map_scrip_load`, `map_scrip_field`,
# `vec_stat`) is held byte-identical to the upstream package so that
# swap is mechanical. The pure-Julia fill / filter helpers added
# below diverge from upstream's KD-tree / `imfilter` implementations
# in details (and may differ slightly in numerics for sparse fills),
# but the public signatures match.
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
    map_scrip_field(map::Dict, var_name, var1::AbstractMatrix;
                    method="mean",
                    fill_method=nothing, filt_method=nothing,
                    filt_par=nothing, verbose=false) -> (mask2, var2)

Apply a SCRIP map to a 2D source field `var1` of shape matching
`map["src_grid_size"]`, returning a 2D destination field of shape
`Tuple(map["dst_grid_dims"])`. `mask2` is a Bool matrix marking
which destination cells received any contribution.

`method = "mean"` (default) is the Fortran `normalize_opt = "fracarea"`
variant — area-weighted mean of source contributors. `"count"` and
`"stdev"` are also available (see `vec_stat`).

Optional post-processing (matches upstream `ScripMap.jl`):

  - `fill_method`: post-fill the NaN destination cells with values from
    nearby valid cells. Recognised values:
      * `"weighted"` — weighted average of up to `nmax` nearest valid
        cells (annular search, weights `1/(distance + 1e-10)`). See
        `fill_weighted!`.
      * `"nn"` — nearest non-NaN neighbour (`fill_nearest!`).
      * `"none"` / `nothing` (default) — leave NaNs in place.
  - `filt_method`: smooth the destination field. Recognised values:
      * `"gaussian"` — 2D separable Gaussian with `filt_par = [sigma, dx]`
        where `sigma` and `dx` are in the same length unit; the
        convolution uses `sigma_norm = sigma/dx` (cells). 3-sigma
        truncation, replicate boundaries. See `gaussian_filter!`.
      * `"none"` / `nothing` (default) — no smoothing.
    Apply AFTER `fill_method` if both are requested — Gaussian
    convolution propagates NaNs from any window with a NaN entry.

`verbose = true` prints a one-line summary after the map is applied.
"""
function map_scrip_field(map::Dict, var_name::AbstractString,
                         var1::AbstractMatrix{T};
                         method::AbstractString = "mean",
                         fill_method::Union{AbstractString,Nothing} = nothing,
                         filt_method::Union{AbstractString,Nothing} = nothing,
                         filt_par::Union{AbstractVector,Nothing} = nothing,
                         verbose::Bool = false) where T

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

    # ----- Optional NaN fill -----
    if fill_method === nothing || fill_method == "none"
        # No fill — leave NaNs in place.
    elseif fill_method == "weighted"
        fill_weighted!(var2)
    elseif fill_method == "nn"
        fill_nearest!(var2)
    else
        error("map_scrip_field: fill_method=\"$(fill_method)\" not recognised. " *
              "Use \"weighted\", \"nn\", \"none\", or `nothing`.")
    end

    # ----- Optional smoothing -----
    if filt_method === nothing || filt_method == "none"
        # No filter.
    elseif filt_method == "gaussian"
        filt_par === nothing &&
            error("map_scrip_field: filt_method=\"gaussian\" requires " *
                  "`filt_par = [sigma, dx]`.")
        length(filt_par) >= 2 ||
            error("map_scrip_field: filt_par for gaussian must hold at least " *
                  "[sigma, dx] (got length $(length(filt_par))).")
        sigma = Float64(filt_par[1])
        dx    = Float64(filt_par[2])
        gaussian_filter!(var2; sigma = sigma, dx = dx)
    else
        error("map_scrip_field: filt_method=\"$(filt_method)\" not recognised. " *
              "Use \"gaussian\", \"none\", or `nothing`.")
    end

    if verbose
        @info "map_scrip_field: mapped $(var_name) (" *
              "fill=$(repr(fill_method)), filt=$(repr(filt_method)))"
    end

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

# ----------------------------------------------------------------------
# Fill / filter helpers — pure-Julia replacements for ScripMap.jl's
# upstream KD-tree-based fill_weighted! / fill_nearest! and
# ImageFiltering-based Gaussian filter. The regular-grid case lets us
# replace the KD-tree with an annular (square-ring) search around each
# NaN cell; for ice-sheet regridding the typical NaN halo is only a
# few cells thick so the search is bounded.
# ----------------------------------------------------------------------

"""
    fill_weighted!(var::AbstractMatrix; nmax=10) -> var

Fill `NaN` cells in `var` with the inverse-distance-weighted average
of up to `nmax` nearest non-NaN neighbours. The search is an annular
(square-ring) walk starting at radius `r=1` and growing outward until
`nmax` valid neighbours have been collected. If a cell has no
reachable valid neighbours after a full pass it is skipped; the
outer loop iterates until either no NaNs remain or no progress is
made (the latter case leaves residual NaNs in fully-isolated holes).

Mirrors upstream `ScripMap.jl::fill_weighted!` semantically. Pure
Julia — no `NearestNeighbors` dependency.
"""
function fill_weighted!(var::AbstractMatrix{<:Number}; nmax::Integer = 10)
    @assert nmax > 0 "fill_weighted!: nmax must be positive, got $nmax."
    Nx, Ny = size(var)
    R_max  = max(Nx, Ny)

    while true
        nan_idxs = findall(isnan, var)
        isempty(nan_idxs) && break

        # Snapshot — we read from `prev` and write to `var` so cells
        # filled this pass don't act as sources for their neighbours
        # within the same pass (matches upstream's pre-snapshot loop).
        prev = copy(var)
        progress = false

        for ci in nan_idxs
            i, j = Tuple(ci)
            vals  = Float64[]
            dists = Float64[]
            r = 1
            while length(vals) < nmax && r <= R_max
                _collect_ring!(vals, dists, prev, i, j, r, Nx, Ny)
                r += 1
            end
            isempty(vals) && continue

            # Take the up to `nmax` nearest by distance.
            if length(vals) > nmax
                perm  = sortperm(dists)
                vals  = vals[perm[1:nmax]]
                dists = dists[perm[1:nmax]]
            end
            wts = 1.0 ./ (dists .+ 1e-10)
            var[i, j] = sum(vals .* wts) / sum(wts)
            progress = true
        end

        progress || break
    end
    return var
end

"""
    fill_nearest!(var::AbstractMatrix) -> var

Fill `NaN` cells in `var` with the value of the single nearest
non-NaN neighbour. Equivalent to `fill_weighted!(var; nmax=1)` modulo
floating-point details (no weight averaging when only one neighbour
is consulted).

Mirrors upstream `ScripMap.jl::fill_nearest!`.
"""
fill_nearest!(var::AbstractMatrix{<:Number}) = fill_weighted!(var; nmax = 1)

# Append all non-NaN cells at Chebyshev distance `r` from `(i, j)`
# into `vals` / `dists`. Skips out-of-bounds neighbours.
@inline function _collect_ring!(vals::Vector{Float64}, dists::Vector{Float64},
                                prev::AbstractMatrix{<:Number},
                                i::Int, j::Int, r::Int, Nx::Int, Ny::Int)
    @inbounds for di in -r:r, dj in -r:r
        max(abs(di), abs(dj)) == r || continue
        ii = i + di; jj = j + dj
        (1 <= ii <= Nx && 1 <= jj <= Ny) || continue
        v = prev[ii, jj]
        isnan(v) && continue
        push!(vals,  Float64(v))
        push!(dists, sqrt(Float64(di) * di + Float64(dj) * dj))
    end
    return nothing
end

"""
    gaussian_filter!(var::AbstractMatrix; sigma, dx) -> var

In-place 2D Gaussian smoothing via separable 1D convolutions. `sigma`
and `dx` are in the same length unit; the kernel is built in cell
units as `sigma_norm = sigma / dx`. Truncated at 3 sigma (kernel
half-width = `ceil(3 * sigma_norm)`); replicate (clamped-edge)
boundary conditions on both axes.

Mirrors upstream `ScripMap.jl`'s `imfilter`-based call but is pure
Julia — no `ImageFiltering` dependency. Note that `NaN` cells in
`var` poison their convolution windows; run `fill_weighted!` /
`fill_nearest!` first if a NaN-clean output is required.
"""
function gaussian_filter!(var::AbstractMatrix{<:Number};
                          sigma::Real, dx::Real)
    sigma > 0 || error("gaussian_filter!: sigma must be > 0, got $sigma.")
    dx    > 0 || error("gaussian_filter!: dx must be > 0, got $dx.")
    sigma_norm = Float64(sigma) / Float64(dx)

    radius = max(1, ceil(Int, 3.0 * sigma_norm))
    ks     = collect(-radius:radius)
    kernel = exp.(-(ks .^ 2) ./ (2.0 * sigma_norm * sigma_norm))
    kernel ./= sum(kernel)

    Nx, Ny = size(var)
    tmp    = similar(var, Float64)

    # Convolve along x with replicate boundary.
    @inbounds for j in 1:Ny, i in 1:Nx
        s = 0.0
        for (k, w) in zip(ks, kernel)
            ii = clamp(i + k, 1, Nx)
            s += w * Float64(var[ii, j])
        end
        tmp[i, j] = s
    end

    # Convolve along y with replicate boundary, write back into var.
    @inbounds for j in 1:Ny, i in 1:Nx
        s = 0.0
        for (k, w) in zip(ks, kernel)
            jj = clamp(j + k, 1, Ny)
            s += w * tmp[i, jj]
        end
        var[i, j] = s
    end
    return var
end
