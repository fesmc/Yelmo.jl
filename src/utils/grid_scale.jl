# ----------------------------------------------------------------------
# Conservative remapping between two uniform-Cartesian resolutions of
# the SAME grid. Specialised case of the general SCRIP regridding —
# `grid_hi` cells evenly subdivide `grid_lo` cells with no fractional
# overlap, so the conservative weights degenerate to `1/s^2` for every
# hi-cell contributing to its enclosing lo-cell (`s` = integer
# refinement factor).
#
# This is NOT a feature of Fortran Yelmo. Useful for:
#
#   - Sub-cycling a high-resolution dynamics step inside a coarser
#     thermo / topo step (or vice versa).
#   - Producing low-resolution time-series outputs from high-resolution
#     state without going through the full SCRIP pipeline.
#   - Quick fixed-resolution coarsening for diagnostics.
#
# The two mapping directions assume the field is intensive (per-area),
# i.e. concentration / temperature / thickness:
#
#   - `map_field_to_lo` : each lo-cell value = arithmetic mean of its
#     `s × s` hi-cells. Conservative for intensive fields when the
#     volume integral is `value × area`.
#   - `map_field_to_hi` : replicate each lo-cell value into its `s × s`
#     hi-cells. Also conservative under the same convention.
#
# Extensive (per-cell) fields would need a `mode = :extensive` switch
# (sum on coarsen, divide on refine) — not implemented yet; flag with
# an issue if needed.
# ----------------------------------------------------------------------

using Oceananigans.Grids: RectilinearGrid
using Base.Threads: @threads

export GridScaleWeights, map_field_to_lo, map_field_to_lo!,
       map_field_to_hi, map_field_to_hi!

# ----------------------------------------------------------------------
# Stencil helpers — explicit s×s block access patterns. Inlined so the
# call sites are zero-overhead. Each helper operates on a single
# block of the high-resolution grid; the outer kernel drives them
# over (i, j) lo-cells.
# ----------------------------------------------------------------------

# Sum over an s×s block of `src` rooted at `(i0+1, j0+1)`. The inner
# loop is over the column-major fast axis (`ii`) so it can SIMD-vectorise.
@inline function _block_sum(src::AbstractMatrix, i0::Int, j0::Int, s::Int)
    acc = zero(eltype(src))
    @inbounds for jj in 1:s
        @simd for ii in 1:s
            acc += src[i0 + ii, j0 + jj]
        end
    end
    return acc
end

# Replicate a scalar value into an s×s block of `dst`.
@inline function _block_replicate!(dst::AbstractMatrix, v, i0::Int, j0::Int, s::Int)
    @inbounds for jj in 1:s
        @simd for ii in 1:s
            dst[i0 + ii, j0 + jj] = v
        end
    end
    return nothing
end

"""
    GridScaleWeights(Nx_lo, Ny_lo, s)
    GridScaleWeights(Nx_lo, Ny_lo, Nx_hi, Ny_hi)
    GridScaleWeights(grid_lo, grid_hi)

Pre-computed weights for conservative remapping between aligned
uniform Cartesian grids `grid_lo` and `grid_hi` where
`Nx_hi = s · Nx_lo` and `Ny_hi = s · Ny_lo` for some integer `s ≥ 1`.

The remapping itself is trivial — block-mean for `hi → lo`, replicate
for `lo → hi`, both with uniform `1/s^2` weights — so the struct
only stores `s` and the four grid sizes for sanity-check at apply
time.

# Fields

  - `s::Int`         — integer refinement factor (`Nx_hi = s · Nx_lo`).
  - `Nx_lo`, `Ny_lo` — low-resolution grid sizes.
  - `Nx_hi`, `Ny_hi` — high-resolution grid sizes.
"""
struct GridScaleWeights
    s::Int
    Nx_lo::Int
    Ny_lo::Int
    Nx_hi::Int
    Ny_hi::Int

    function GridScaleWeights(Nx_lo::Integer, Ny_lo::Integer,
                              Nx_hi::Integer, Ny_hi::Integer)
        Nx_lo > 0 && Ny_lo > 0 && Nx_hi > 0 && Ny_hi > 0 ||
            error("GridScaleWeights: all grid sizes must be positive.")
        sx, rx = divrem(Nx_hi, Nx_lo)
        sy, ry = divrem(Ny_hi, Ny_lo)
        rx == 0 ||
            error("GridScaleWeights: Nx_hi=$(Nx_hi) is not an integer multiple of Nx_lo=$(Nx_lo).")
        ry == 0 ||
            error("GridScaleWeights: Ny_hi=$(Ny_hi) is not an integer multiple of Ny_lo=$(Ny_lo).")
        sx == sy ||
            error("GridScaleWeights: refinement factor differs between axes — Nx ratio $(sx), Ny ratio $(sy). " *
                  "Aligned uniform-resolution remapping requires a single isotropic factor.")
        sx >= 1 || error("GridScaleWeights: refinement factor must be ≥ 1, got $(sx).")
        return new(Int(sx), Int(Nx_lo), Int(Ny_lo), Int(Nx_hi), Int(Ny_hi))
    end
end

# Convenience: derive Nx_hi / Ny_hi from a refinement factor.
function GridScaleWeights(Nx_lo::Integer, Ny_lo::Integer, s::Integer)
    s >= 1 || error("GridScaleWeights: refinement factor must be ≥ 1, got $(s).")
    return GridScaleWeights(Nx_lo, Ny_lo, Nx_lo * s, Ny_lo * s)
end

# Construct from an Oceananigans RectilinearGrid pair. Validates that
# both grids span the same physical extent (within a 1e-9 relative
# tolerance) so the cell alignment is exact.
function GridScaleWeights(grid_lo::RectilinearGrid, grid_hi::RectilinearGrid)
    Nx_lo = size(grid_lo, 1); Ny_lo = size(grid_lo, 2)
    Nx_hi = size(grid_hi, 1); Ny_hi = size(grid_hi, 2)
    w = GridScaleWeights(Nx_lo, Ny_lo, Nx_hi, Ny_hi)

    # Sanity: same physical extent on both axes (cell alignment).
    Lx_lo = abs(Float64(grid_lo.Δxᶜᵃᵃ)) * Nx_lo
    Lx_hi = abs(Float64(grid_hi.Δxᶜᵃᵃ)) * Nx_hi
    Ly_lo = abs(Float64(grid_lo.Δyᵃᶜᵃ)) * Ny_lo
    Ly_hi = abs(Float64(grid_hi.Δyᵃᶜᵃ)) * Ny_hi
    isapprox(Lx_lo, Lx_hi; rtol = 1e-9) ||
        error("GridScaleWeights: physical x extent disagrees (lo=$(Lx_lo), hi=$(Lx_hi)). " *
              "Grids must cover the same domain for aligned remapping.")
    isapprox(Ly_lo, Ly_hi; rtol = 1e-9) ||
        error("GridScaleWeights: physical y extent disagrees (lo=$(Ly_lo), hi=$(Ly_hi)). " *
              "Grids must cover the same domain for aligned remapping.")
    return w
end

# ----------------------------------------------------------------------
# hi → lo: conservative coarsening (block-mean of s×s hi cells)
# ----------------------------------------------------------------------

"""
    map_field_to_lo!(dst, src, w::GridScaleWeights) -> dst

Conservative coarsening (intensive field): each lo-cell `(i, j)`
takes the arithmetic mean of the `s × s` block of hi-cells at
indices `((i-1)·s+1 : i·s, (j-1)·s+1 : j·s)`. The integral
`Σ value · area` is preserved when the field is intensive (uniform
hi-cell areas; the lo-cell area equals `s²` hi-cell areas).

Mutates `dst` in place.
"""
function map_field_to_lo!(dst::AbstractMatrix, src::AbstractMatrix,
                          w::GridScaleWeights)
    size(src) == (w.Nx_hi, w.Ny_hi) ||
        error("map_field_to_lo!: src shape $(size(src)) does not match " *
              "weights' (Nx_hi, Ny_hi) = ($(w.Nx_hi), $(w.Ny_hi)).")
    size(dst) == (w.Nx_lo, w.Ny_lo) ||
        error("map_field_to_lo!: dst shape $(size(dst)) does not match " *
              "weights' (Nx_lo, Ny_lo) = ($(w.Nx_lo), $(w.Ny_lo)).")
    s = w.s
    inv_s2 = 1.0 / (s * s)
    Nx_lo = w.Nx_lo
    Ny_lo = w.Ny_lo
    # Each lo-cell `(i, j)` reads from a disjoint `s × s` stencil
    # block of `src` and writes to a single slot in `dst`. Threading
    # on the outer `j` axis is race-free — different `j` strips of
    # `dst` belong to different threads.
    @threads for j in 1:Ny_lo
        @inbounds for i in 1:Nx_lo
            i0 = (i - 1) * s
            j0 = (j - 1) * s
            dst[i, j] = _block_sum(src, i0, j0, s) * inv_s2
        end
    end
    return dst
end

# 3D overload — per-layer coarsening.
function map_field_to_lo!(dst::AbstractArray{<:Any,3},
                          src::AbstractArray{<:Any,3},
                          w::GridScaleWeights)
    Nz = size(src, 3)
    size(dst, 3) == Nz ||
        error("map_field_to_lo! (3D): vertical dim mismatch (src=$(Nz), dst=$(size(dst, 3))).")
    @inbounds for k in 1:Nz
        map_field_to_lo!(view(dst, :, :, k), view(src, :, :, k), w)
    end
    return dst
end

"""
    map_field_to_lo(src, w::GridScaleWeights) -> dst

Allocating variant of `map_field_to_lo!`. `dst` matches `src`'s
element type and dimensionality. Returns a fresh array.
"""
function map_field_to_lo(src::AbstractMatrix{T}, w::GridScaleWeights) where T
    dst = Matrix{T}(undef, w.Nx_lo, w.Ny_lo)
    return map_field_to_lo!(dst, src, w)
end
function map_field_to_lo(src::AbstractArray{T,3}, w::GridScaleWeights) where T
    dst = Array{T,3}(undef, w.Nx_lo, w.Ny_lo, size(src, 3))
    return map_field_to_lo!(dst, src, w)
end

# ----------------------------------------------------------------------
# lo → hi: conservative refinement (replicate s×s hi cells)
# ----------------------------------------------------------------------

"""
    map_field_to_hi!(dst, src, w::GridScaleWeights) -> dst

Conservative refinement (intensive field): each lo-cell value is
replicated into its `s × s` block of hi-cells. The integral
`Σ value · area` is preserved.

Mutates `dst` in place.
"""
function map_field_to_hi!(dst::AbstractMatrix, src::AbstractMatrix,
                          w::GridScaleWeights)
    size(src) == (w.Nx_lo, w.Ny_lo) ||
        error("map_field_to_hi!: src shape $(size(src)) does not match " *
              "weights' (Nx_lo, Ny_lo) = ($(w.Nx_lo), $(w.Ny_lo)).")
    size(dst) == (w.Nx_hi, w.Ny_hi) ||
        error("map_field_to_hi!: dst shape $(size(dst)) does not match " *
              "weights' (Nx_hi, Ny_hi) = ($(w.Nx_hi), $(w.Ny_hi)).")
    s = w.s
    Nx_lo = w.Nx_lo
    Ny_lo = w.Ny_lo
    # Each lo-cell `(i, j)` writes to a disjoint `s × s` stencil
    # block of `dst`. Different `j` strips of lo-cells map to
    # different `j` strips of hi-cells, so threading on `j` is
    # race-free.
    @threads for j in 1:Ny_lo
        @inbounds for i in 1:Nx_lo
            v  = src[i, j]
            i0 = (i - 1) * s
            j0 = (j - 1) * s
            _block_replicate!(dst, v, i0, j0, s)
        end
    end
    return dst
end

# 3D overload — per-layer refinement.
function map_field_to_hi!(dst::AbstractArray{<:Any,3},
                          src::AbstractArray{<:Any,3},
                          w::GridScaleWeights)
    Nz = size(src, 3)
    size(dst, 3) == Nz ||
        error("map_field_to_hi! (3D): vertical dim mismatch (src=$(Nz), dst=$(size(dst, 3))).")
    @inbounds for k in 1:Nz
        map_field_to_hi!(view(dst, :, :, k), view(src, :, :, k), w)
    end
    return dst
end

"""
    map_field_to_hi(src, w::GridScaleWeights) -> dst

Allocating variant of `map_field_to_hi!`. `dst` matches `src`'s
element type and dimensionality. Returns a fresh array.
"""
function map_field_to_hi(src::AbstractMatrix{T}, w::GridScaleWeights) where T
    dst = Matrix{T}(undef, w.Nx_hi, w.Ny_hi)
    return map_field_to_hi!(dst, src, w)
end
function map_field_to_hi(src::AbstractArray{T,3}, w::GridScaleWeights) where T
    dst = Array{T,3}(undef, w.Nx_hi, w.Ny_hi, size(src, 3))
    return map_field_to_hi!(dst, src, w)
end
