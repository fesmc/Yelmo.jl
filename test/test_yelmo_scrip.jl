## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Tests for the vendored SCRIP-mapping helpers (`src/utils/scrip_map.jl`)
# and for the regridding path through `YelmoModel(restart_file, time;
# target_grid_file=...)`.

using Test
using Yelmo
using NCDatasets
using Oceananigans

# Bring the unexported helpers into scope.
const _gen_map_filename = Yelmo.YelmoUtils.gen_map_filename
const _map_scrip_load   = Yelmo.YelmoUtils.map_scrip_load
const _map_scrip_field  = Yelmo.YelmoUtils.map_scrip_field
const _vec_stat         = Yelmo.YelmoUtils.vec_stat
const _fill_weighted!   = Yelmo.YelmoUtils.fill_weighted!
const _fill_nearest!    = Yelmo.YelmoUtils.fill_nearest!
const _gaussian_filter! = Yelmo.YelmoUtils.gaussian_filter!

# ---------------------------------------------------------------------
# Helper: hand-craft a minimal SCRIP NetCDF with a known weights table.
# ---------------------------------------------------------------------

"""
Write a SCRIP-format NetCDF to `path` describing a `Nx_src × Ny_src`
source grid mapped to a `Nx_dst × Ny_dst` target grid via the
caller-supplied (src_address, dst_address, weights) link triplets.
1-indexed addresses (Fortran/SCRIP convention).
"""
function _write_scrip_fixture(path::AbstractString;
                              src_dims::Tuple{Int,Int},
                              dst_dims::Tuple{Int,Int},
                              src_addr::Vector{Int},
                              dst_addr::Vector{Int},
                              weights::Vector{Float64})
    @assert length(src_addr) == length(dst_addr) == length(weights)
    Nx_src, Ny_src = src_dims
    Nx_dst, Ny_dst = dst_dims
    n_src = Nx_src * Ny_src
    n_dst = Nx_dst * Ny_dst
    n_links = length(src_addr)

    NCDataset(path, "c") do ds
        defDim(ds, "src_grid_size",    n_src)
        defDim(ds, "dst_grid_size",    n_dst)
        defDim(ds, "src_grid_corners", 4)
        defDim(ds, "dst_grid_corners", 4)
        defDim(ds, "src_grid_rank",    2)
        defDim(ds, "dst_grid_rank",    2)
        defDim(ds, "num_wgts",         1)
        defDim(ds, "num_links",        n_links)

        defVar(ds, "src_grid_dims",       Int32, ("src_grid_rank",))[:] = [Nx_src, Ny_src]
        defVar(ds, "dst_grid_dims",       Int32, ("dst_grid_rank",))[:] = [Nx_dst, Ny_dst]

        # Latitude/longitude / area metadata are not used by
        # map_scrip_field but must be present.
        defVar(ds, "src_grid_center_lat", Float64, ("src_grid_size",))[:] = zeros(n_src)
        defVar(ds, "src_grid_center_lon", Float64, ("src_grid_size",))[:] = zeros(n_src)
        defVar(ds, "dst_grid_center_lat", Float64, ("dst_grid_size",))[:] = zeros(n_dst)
        defVar(ds, "dst_grid_center_lon", Float64, ("dst_grid_size",))[:] = zeros(n_dst)
        defVar(ds, "src_grid_imask",      Int32,   ("src_grid_size",))[:] = ones(Int32, n_src)
        defVar(ds, "dst_grid_imask",      Int32,   ("dst_grid_size",))[:] = ones(Int32, n_dst)
        defVar(ds, "src_grid_frac",       Float64, ("src_grid_size",))[:] = ones(n_src)
        defVar(ds, "dst_grid_frac",       Float64, ("dst_grid_size",))[:] = ones(n_dst)

        defVar(ds, "src_address",         Int32,   ("num_links",))[:] = src_addr
        defVar(ds, "dst_address",         Int32,   ("num_links",))[:] = dst_addr
        defVar(ds, "remap_matrix",        Float64, ("num_wgts", "num_links"))[:, :] =
            reshape(weights, 1, n_links)
    end
    return path
end

# ---------------------------------------------------------------------
# vec_stat
# ---------------------------------------------------------------------

@testset "scrip: vec_stat — mean / count / stdev" begin
    @test _vec_stat([1.0, 2.0, 3.0]) ≈ 2.0
    @test _vec_stat([1.0, 2.0, 3.0]; wts = [0.0, 1.0, 0.0]) ≈ 2.0
    # NaNs are skipped.
    @test _vec_stat([1.0, NaN, 3.0]) ≈ 2.0
    # All weights zero ⇒ NaN.
    @test isnan(_vec_stat([1.0, 2.0]; wts = [0.0, 0.0]))
    # method = "count" picks the most-frequent value.
    @test _vec_stat([1.0, 1.0, 2.0]; method = "count") == 1.0
end

# ---------------------------------------------------------------------
# gen_map_filename
# ---------------------------------------------------------------------

@testset "scrip: gen_map_filename round-trip" begin
    path = _gen_map_filename("ANT-32KM", "ANT-16KM", "maps", "con")
    @test path == joinpath("maps", "scrip-con_ANT-32KM_ANT-16KM.nc")
end

# ---------------------------------------------------------------------
# map_scrip_field — identity map
# ---------------------------------------------------------------------

@testset "scrip: map_scrip_field — identity 3×3 → 3×3" begin
    tdir = mktempdir(; prefix = "scrip_id_")
    path = joinpath(tdir, "scrip-con_A_A.nc")
    Nx, Ny = 3, 3
    n = Nx * Ny
    _write_scrip_fixture(path;
        src_dims = (Nx, Ny), dst_dims = (Nx, Ny),
        src_addr = collect(1:n),
        dst_addr = collect(1:n),
        weights  = ones(n))

    mps = _map_scrip_load("A", "A", tdir; method = "con")
    src = Float64.(reshape(1:n, Nx, Ny))
    _, dst = _map_scrip_field(mps, "test", src)
    @test dst == src
end

# ---------------------------------------------------------------------
# map_scrip_field — 4×1 → 2×1 averaging (each target = mean of 2 source)
# ---------------------------------------------------------------------

@testset "scrip: map_scrip_field — 4×1 → 2×1 area-weighted mean" begin
    tdir = mktempdir(; prefix = "scrip_avg_")
    path = joinpath(tdir, "scrip-con_FINE_COARSE.nc")
    # Source: 4×1 grid (cells 1..4). Target: 2×1 grid.
    # Each target cell averages 2 source cells with equal weight.
    _write_scrip_fixture(path;
        src_dims = (4, 1), dst_dims = (2, 1),
        src_addr = [1, 2, 3, 4],
        dst_addr = [1, 1, 2, 2],
        weights  = [0.5, 0.5, 0.5, 0.5])

    mps = _map_scrip_load("FINE", "COARSE", tdir; method = "con")
    src = reshape([10.0, 20.0, 30.0, 40.0], 4, 1)
    _, dst = _map_scrip_field(mps, "T", src)
    @test dst[1, 1] ≈ 15.0   # mean(10, 20)
    @test dst[2, 1] ≈ 35.0   # mean(30, 40)
end

# ---------------------------------------------------------------------
# map_scrip_field — empty target cells get NaN
# ---------------------------------------------------------------------

@testset "scrip: map_scrip_field — unmapped target cells stay NaN" begin
    tdir = mktempdir(; prefix = "scrip_nan_")
    path = joinpath(tdir, "scrip-con_A_B.nc")
    # Source: 2×1, Target: 3×1. Only target cell 2 has any links.
    _write_scrip_fixture(path;
        src_dims = (2, 1), dst_dims = (3, 1),
        src_addr = [1, 2],
        dst_addr = [2, 2],
        weights  = [0.5, 0.5])

    mps = _map_scrip_load("A", "B", tdir; method = "con")
    _, dst = _map_scrip_field(mps, "T", reshape([10.0, 20.0], 2, 1))
    @test isnan(dst[1, 1])
    @test dst[2, 1] ≈ 15.0
    @test isnan(dst[3, 1])
end

# ---------------------------------------------------------------------
# Missing SCRIP file → clear error.
# ---------------------------------------------------------------------

@testset "scrip: map_scrip_load — missing file errors clearly" begin
    @test_throws ErrorException _map_scrip_load("NOSUCH_SRC", "NOSUCH_DST",
        mktempdir(; prefix = "scrip_missing_"))
end

# ---------------------------------------------------------------------
# fill_nearest! / fill_weighted!
# ---------------------------------------------------------------------

@testset "scrip: fill_nearest! — single NaN cell takes nearest neighbour" begin
    a = Float64[1.0  2.0  3.0;
                4.0  NaN  6.0;
                7.0  8.0  9.0]
    _fill_nearest!(a)
    @test !any(isnan, a)
    # The NaN at (2,2) had eight 8-neighbours all valid; ties broken
    # by "first encountered" — the value lands somewhere in the
    # neighbour set. Just assert that the chosen value is in {1..9}.
    @test 1.0 <= a[2, 2] <= 9.0
end

@testset "scrip: fill_weighted! — symmetric pattern recovers mean" begin
    # 5x5 with the centre cell NaN and a uniform-1 ring around it.
    # The annular weighted average must land at 1.0 to f.p. precision.
    a = ones(Float64, 5, 5)
    a[3, 3] = NaN
    _fill_weighted!(a)
    @test a[3, 3] ≈ 1.0 atol = 1e-12
    @test !any(isnan, a)
end

@testset "scrip: fill_weighted! — NaN halo at boundary fills inward" begin
    # Source domain: 3x3 valid block surrounded by 1-cell NaN halo.
    a = fill(NaN, 5, 5)
    @inbounds for j in 2:4, i in 2:4
        a[i, j] = 7.0
    end
    _fill_weighted!(a)
    @test !any(isnan, a)
    # Halo cells should fill to 7.0 (the only available source value).
    # Tolerance allows for floating-point roundoff in the weighted
    # average — all source values are 7.0 so the analytic result is
    # exactly 7.0 but the divisions introduce ~1 ULP of error.
    @test maximum(abs.(a .- 7.0)) < 1e-12
end

@testset "scrip: fill_weighted! — no valid cells leaves NaNs" begin
    a = fill(NaN, 3, 3)
    _fill_weighted!(a)
    @test all(isnan, a)
end

# ---------------------------------------------------------------------
# gaussian_filter!
# ---------------------------------------------------------------------

@testset "scrip: gaussian_filter! — uniform input is invariant" begin
    a = fill(5.0, 8, 6)
    _gaussian_filter!(a; sigma = 2.0, dx = 1.0)
    @test all(abs.(a .- 5.0) .< 1e-12)
end

@testset "scrip: gaussian_filter! — small sigma preserves peak shape qualitatively" begin
    Nx, Ny = 11, 11
    a = zeros(Float64, Nx, Ny)
    a[6, 6] = 1.0          # delta function at the centre
    _gaussian_filter!(a; sigma = 1.0, dx = 1.0)
    # After convolution the peak is at the centre and decreasing
    # monotonically along x and y axes.
    @test a[6, 6] >= a[6, 5] >= a[6, 4]
    @test a[6, 6] >= a[5, 6] >= a[4, 6]
    # Symmetry along x and y.
    @test a[6, 5] ≈ a[6, 7] atol = 1e-12
    @test a[5, 6] ≈ a[7, 6] atol = 1e-12
    # Mass is approximately preserved (replicate boundaries inflate
    # the integral slightly because the delta is far from edges, but
    # for our 3-sigma kernel within an 11x11 array the loss is small).
    @test sum(a) ≈ 1.0 atol = 1e-6
end

@testset "scrip: gaussian_filter! — invalid sigma / dx error" begin
    a = ones(Float64, 4, 4)
    @test_throws ErrorException _gaussian_filter!(a; sigma = 0.0, dx = 1.0)
    @test_throws ErrorException _gaussian_filter!(a; sigma = 1.0, dx = 0.0)
end

# ---------------------------------------------------------------------
# map_scrip_field with fill_method / filt_method wiring
# ---------------------------------------------------------------------

@testset "scrip: map_scrip_field — fill_method=\"nn\" removes NaNs" begin
    tdir = mktempdir(; prefix = "scrip_fill_")
    path = joinpath(tdir, "scrip-con_A_B.nc")
    # 2×1 → 3×1 with only middle target cell mapped (same fixture as
    # the unmapped-cells test above; reusing the assertion that those
    # NaNs would normally remain).
    _write_scrip_fixture(path;
        src_dims = (2, 1), dst_dims = (3, 1),
        src_addr = [1, 2],
        dst_addr = [2, 2],
        weights  = [0.5, 0.5])

    mps = _map_scrip_load("A", "B", tdir; method = "con")
    src = reshape([10.0, 20.0], 2, 1)
    _, dst = _map_scrip_field(mps, "T", src; fill_method = "nn")
    @test !any(isnan, dst)
    # All three cells should equal the only mapped value (15.0).
    @test all(dst .≈ 15.0)
end

@testset "scrip: map_scrip_field — filt_method=\"gaussian\" smooths" begin
    tdir = mktempdir(; prefix = "scrip_filt_")
    path = joinpath(tdir, "scrip-con_A_A.nc")
    Nx, Ny = 7, 7
    n = Nx * Ny
    _write_scrip_fixture(path;
        src_dims = (Nx, Ny), dst_dims = (Nx, Ny),
        src_addr = collect(1:n),
        dst_addr = collect(1:n),
        weights  = ones(n))

    mps = _map_scrip_load("A", "A", tdir; method = "con")
    src = zeros(Float64, Nx, Ny)
    src[4, 4] = 1.0
    _, dst = _map_scrip_field(mps, "T", src;
                              filt_method = "gaussian",
                              filt_par    = [1.0, 1.0])
    # Identity map then Gaussian smooth: peak is reduced, neighbours rise.
    @test 0.0 < dst[4, 4] < 1.0
    @test dst[4, 5] > 0.0
    @test dst[3, 4] > 0.0
end

@testset "scrip: map_scrip_field — gaussian without filt_par errors" begin
    tdir = mktempdir(; prefix = "scrip_filt_err_")
    path = joinpath(tdir, "scrip-con_A_A.nc")
    n = 4
    _write_scrip_fixture(path;
        src_dims = (2, 2), dst_dims = (2, 2),
        src_addr = collect(1:n),
        dst_addr = collect(1:n),
        weights  = ones(n))
    mps = _map_scrip_load("A", "A", tdir; method = "con")
    src = ones(Float64, 2, 2)
    @test_throws ErrorException _map_scrip_field(mps, "T", src;
        filt_method = "gaussian")
end

# ---------------------------------------------------------------------
# Integration: YelmoModel constructor with target_grid_file kwarg.
# ---------------------------------------------------------------------

const RESTART_PATH = "/Users/alrobi001/models/yelmox/output/16KM/test/restart-0.000-kyr/yelmo_restart.nc"

# Write a synthetic target-grid NetCDF: same xc/yc as the restart (so
# the SCRIP map is identity), but a different `grid_name` global
# attribute so the constructor's mismatch-detection path fires.
function _write_target_grid_copy(path::AbstractString, restart::AbstractString,
                                 target_grid_name::AbstractString)
    NCDataset(restart) do src
        xc = Vector{Float64}(src["xc"][:])
        yc = Vector{Float64}(src["yc"][:])
        x_units = String(get(src["xc"].attrib, "units", "km"))
        y_units = String(get(src["yc"].attrib, "units", "km"))
        NCDataset(path, "c") do tgt
            tgt.attrib["grid_name"] = target_grid_name
            defDim(tgt, "xc", length(xc))
            defDim(tgt, "yc", length(yc))
            let v = defVar(tgt, "xc", Float64, ("xc",))
                v[:] = xc; v.attrib["units"] = x_units
            end
            let v = defVar(tgt, "yc", Float64, ("yc",))
                v[:] = yc; v.attrib["units"] = y_units
            end
        end
    end
end

# Identity SCRIP map for an Nx*Ny grid, in 1-indexed lexicographic
# order (matches Yelmo / Fortran SCRIP convention).
function _write_identity_scrip(path::AbstractString,
                               src_name::AbstractString,
                               dst_name::AbstractString,
                               Nx::Int, Ny::Int)
    n = Nx * Ny
    _write_scrip_fixture(path;
        src_dims = (Nx, Ny), dst_dims = (Nx, Ny),
        src_addr = collect(1:n), dst_addr = collect(1:n),
        weights  = ones(n))
end

@testset "scrip: YelmoModel(target_grid_file=) — identity regrid round-trip" begin
    isfile(RESTART_PATH) || (@test_skip "restart fixture missing"; return)

    # Read restart's grid_name (typically GRL-16KM) and dims.
    restart_grid_name = NCDataset(RESTART_PATH) do ds
        String(ds.attrib["grid_name"])
    end
    Nx, Ny = NCDataset(RESTART_PATH) do ds
        length(ds["xc"][:]), length(ds["yc"][:])
    end

    target_name = restart_grid_name * "-COPY"

    tdir = mktempdir(; prefix = "scrip_e2e_")
    target_grid_file = joinpath(tdir, "target_grid.nc")
    _write_target_grid_copy(target_grid_file, RESTART_PATH, target_name)

    # SCRIP map under <tdir>/maps/scrip-con_<src>_<dst>.nc.
    maps_dir = joinpath(tdir, "maps")
    mkpath(maps_dir)
    _write_identity_scrip(_gen_map_filename(restart_grid_name, target_name,
                                            maps_dir, "con"),
                          restart_grid_name, target_name, Nx, Ny)

    # Reference load (no regrid).
    y_ref = YelmoModel(RESTART_PATH, 0.0;
                       alias  = "scrip-ref",
                       rundir = tdir,
                       strict = false)

    # Regrid load with identity map.
    y_rg = YelmoModel(RESTART_PATH, 0.0;
                      alias            = "scrip-rg",
                      rundir           = tdir,
                      strict           = false,
                      target_grid_file = target_grid_file,
                      maps_dir         = maps_dir,
                      regrid_method    = "con")

    # Same grid sizes.
    @test size(y_rg.tpo.H_ice) == size(y_ref.tpo.H_ice)
    @test size(y_rg.dyn.ux_bar) == size(y_ref.dyn.ux_bar)

    # H_ice should round-trip exactly under identity mapping (Center
    # field, no boundary-replicate gymnastics).
    h_ref = interior(y_ref.tpo.H_ice)
    h_rg  = interior(y_rg.tpo.H_ice)
    @test maximum(abs.(h_rg .- h_ref)) < 1e-9

    # z_bed too.
    @test maximum(abs.(interior(y_rg.bnd.z_bed) .-
                       interior(y_ref.bnd.z_bed))) < 1e-9
end

@testset "scrip: YelmoModel(target_grid_file=) — matching grid_name warns + falls back" begin
    isfile(RESTART_PATH) || (@test_skip "restart fixture missing"; return)

    restart_grid_name = NCDataset(RESTART_PATH) do ds
        String(ds.attrib["grid_name"])
    end

    tdir = mktempdir(; prefix = "scrip_warn_")
    # Target grid file with the *same* grid_name → the constructor
    # should emit a warning and fall through to the no-regrid path.
    target_grid_file = joinpath(tdir, "target_grid.nc")
    _write_target_grid_copy(target_grid_file, RESTART_PATH, restart_grid_name)

    y = @test_logs (:warn, r"target_grid_file grid_name matches") match_mode = :any begin
        YelmoModel(RESTART_PATH, 0.0;
                   alias            = "scrip-warn",
                   rundir           = tdir,
                   strict           = false,
                   target_grid_file = target_grid_file)
    end
    @test y isa YelmoModel
end

@testset "scrip: YelmoModel(target_grid_file=) — missing grid_name attr errors" begin
    tdir = mktempdir(; prefix = "scrip_noattr_")
    bad_grid = joinpath(tdir, "no_grid_name.nc")
    NCDataset(bad_grid, "c") do ds
        # No grid_name attribute. Just dummy xc/yc.
        defDim(ds, "xc", 3); defDim(ds, "yc", 3)
        defVar(ds, "xc", Float64, ("xc",))[:] = [0.0, 1.0, 2.0]
        defVar(ds, "yc", Float64, ("yc",))[:] = [0.0, 1.0, 2.0]
    end

    @test_throws ErrorException YelmoModel(RESTART_PATH, 0.0;
                                           strict           = false,
                                           target_grid_file = bad_grid,
                                           rundir           = tdir)
end
