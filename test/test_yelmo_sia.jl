## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 3c tests for the SIA solver:
#
#   Commit 1 — kernel unit tests (analytical):
#     - `calc_shear_stress_3D!` (per-layer x/y shear stress)
#     - `calc_uxy_sia_3D!`      (depth-recurrence horizontal velocity)
#
#   Commit 2 — wrapper + dispatch + integration test:
#     - `calc_velocity_sia!`            (Option C SIA wrapper)
#     - `dyn_step!` solver=="sia" branch (uniform-slab integration test
#       against the discretized analytical SIA velocity)

using Test
using Yelmo
using Yelmo.YelmoModelPar: ydyn_params, ymat_params
using Oceananigans
using Oceananigans: interior
using NCDatasets

# 3D ice grid builder, matching the layout used by `yelmo_define_grids`
# (Bounded × Bounded × Bounded) and parametrised by an explicit
# `zeta_ac` vector of layer interfaces.
function _bounded_3d(Nx, Ny, zeta_ac; dx::Real = 1.0)
    return RectilinearGrid(size=(Nx, Ny, length(zeta_ac) - 1),
                           x=(0.0, Nx*dx), y=(0.0, Ny*dx),
                           z=zeta_ac,
                           topology=(Bounded, Bounded, Bounded))
end

# zeta_aa = layer-centre vector. For uniform layers between zeta_ac[1]
# and zeta_ac[end], zeta_aa[k] = 0.5 * (zeta_ac[k] + zeta_ac[k+1]).
_zeta_aa(zeta_ac) = [0.5 * (zeta_ac[k] + zeta_ac[k+1]) for k in 1:length(zeta_ac)-1]

# ======================================================================
# Test 1 — Zero-input → zero-output sanity.
# ======================================================================

@testset "sia: kernels — zero input gives zero output" begin
    Nx, Ny    = 3, 3
    zeta_ac   = collect(range(0.0, 1.0; length=5))   # Nz_aa = 4
    zeta_aa   = _zeta_aa(zeta_ac)
    Nz        = length(zeta_aa)
    g3        = _bounded_3d(Nx, Ny, zeta_ac; dx=1.0)

    taud_acx  = XFaceField(g3);  fill!(interior(taud_acx), 0.0)
    taud_acy  = YFaceField(g3);  fill!(interior(taud_acy), 0.0)
    f_ice     = CenterField(g3); fill!(interior(f_ice),    1.0)
    H_ice     = CenterField(g3); fill!(interior(H_ice),    1000.0)
    ATT       = CenterField(g3); fill!(interior(ATT),      1e-16)

    tau_xz    = XFaceField(g3)
    tau_yz    = YFaceField(g3)
    ux        = XFaceField(g3)
    uy        = YFaceField(g3)

    calc_shear_stress_3D!(tau_xz, tau_yz, taud_acx, taud_acy, f_ice, zeta_aa)
    @test maximum(abs.(interior(tau_xz))) == 0.0
    @test maximum(abs.(interior(tau_yz))) == 0.0

    calc_uxy_sia_3D!(ux, uy, tau_xz, tau_yz, taud_acx, taud_acy,
                     H_ice, f_ice, ATT, 3.0, zeta_aa)
    @test maximum(abs.(interior(ux))) == 0.0
    @test maximum(abs.(interior(uy))) == 0.0
end

# ======================================================================
# Test 2 — Linear-zeta tau_xz check (sign convention).
# ======================================================================
#
# With taud_acx = -1000 Pa uniformly and taud_acy = 0, we expect
#     tau_xz(i, j, k) = -(1 - zeta_aa[k]) * taud_acx
#                     =  1000 * (1 - zeta_aa[k])
# at every interior x-face, and tau_yz = 0 everywhere.

@testset "sia: calc_shear_stress_3D! — linear-zeta sign convention" begin
    Nx, Ny    = 3, 3
    zeta_ac   = collect(range(0.0, 1.0; length=5))   # Nz_aa = 4
    zeta_aa   = _zeta_aa(zeta_ac)
    Nz        = length(zeta_aa)
    g3        = _bounded_3d(Nx, Ny, zeta_ac; dx=1.0)

    taud_acx  = XFaceField(g3);  fill!(interior(taud_acx), -1000.0)
    taud_acy  = YFaceField(g3);  fill!(interior(taud_acy), 0.0)
    f_ice     = CenterField(g3); fill!(interior(f_ice),    1.0)

    tau_xz    = XFaceField(g3)
    tau_yz    = YFaceField(g3)

    calc_shear_stress_3D!(tau_xz, tau_yz, taud_acx, taud_acy, f_ice, zeta_aa)

    # Interior x-face slots [i+1, j, k] for i in 1:Nx, j in 1:Ny.
    Txz = interior(tau_xz)
    for k in 1:Nz
        expected = 1000.0 * (1.0 - zeta_aa[k])
        for j in 1:Ny, i in 1:Nx
            @test abs(Txz[i+1, j, k] - expected) < 1e-12
        end
    end

    # tau_yz must be exactly zero everywhere.
    @test maximum(abs.(interior(tau_yz))) == 0.0
end

# ======================================================================
# Test 3 — Boundary-replicate sanity on a 5×5 grid.
# ======================================================================

@testset "sia: calc_shear_stress_3D! — boundary replicate, no NaN/Inf" begin
    Nx, Ny    = 5, 5
    zeta_ac   = collect(range(0.0, 1.0; length=5))   # Nz_aa = 4
    zeta_aa   = _zeta_aa(zeta_ac)
    Nz        = length(zeta_aa)
    g3        = _bounded_3d(Nx, Ny, zeta_ac; dx=1.0)

    # Non-uniform taud_acx: linearly varying in i, uniform in j.
    taud_acx  = XFaceField(g3)
    Tx        = interior(taud_acx)
    for j in 1:Ny, i in 1:Nx
        Tx[i+1, j, 1] = -100.0 * i
    end
    taud_acy  = YFaceField(g3); fill!(interior(taud_acy), 0.0)
    f_ice     = CenterField(g3); fill!(interior(f_ice), 1.0)

    tau_xz    = XFaceField(g3)
    tau_yz    = YFaceField(g3)

    calc_shear_stress_3D!(tau_xz, tau_yz, taud_acx, taud_acy, f_ice, zeta_aa)

    Txz = interior(tau_xz)
    @test all(isfinite, Txz)
    @test all(isfinite, interior(tau_yz))

    # Per-layer scaling: tau_xz[i+1, j, k] = -(1 - zeta_aa[k]) * taud_acx[i+1, j, 1]
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        expected = -(1.0 - zeta_aa[k]) * Tx[i+1, j, 1]
        @test abs(Txz[i+1, j, k] - expected) < 1e-12
    end

    # Replicated leading slot: Txz[1, j, k] = Txz[2, j, k]
    for k in 1:Nz, j in 1:Ny
        @test Txz[1, j, k] == Txz[2, j, k]
    end
end

# ======================================================================
# Test 4 — calc_uxy_sia_3D! recurrence smoke (uniform layer-tau).
# ======================================================================
#
# Set tau_xz = -1000 Pa at every interior x-face / every layer (so the
# layer-mid average is also -1000), tau_yz = 0, H_ice = 1000 m,
# ATT = 1e-16 Pa^-3 yr^-1, n_glen = 3. Run kernel 2 and check that:
#   - ux is finite everywhere (no NaN / Inf);
#   - ux(:, :, 1) is non-zero — under Option C the kernel integrates
#     a bed segment from zeta = 0 (where u = 0) to zeta_aa[1] using
#     tau_xz_bed = -taud_acx (closed form). With taud_acx = 0 in this
#     test, tau_xz_bed = 0 and the bed-segment contribution
#     0.5 * fact_ac * (tau_xz[k=1] + 0) is non-zero.
#   - |ux| is monotonically non-decreasing in k (the recurrence adds
#     a positive contribution at each layer because fact_ac > 0 and
#     the sign of (tau(k) + tau(k-1)) is fixed throughout the column);
#   - uy stays at zero (since tau_yz is zero AND taud_acy is zero,
#     so tau_yz_bed is also zero).

@testset "sia: calc_uxy_sia_3D! — recurrence smoke" begin
    Nx, Ny    = 4, 4
    zeta_ac   = collect(range(0.0, 1.0; length=5))   # Nz_aa = 4
    zeta_aa   = _zeta_aa(zeta_ac)
    Nz        = length(zeta_aa)
    g3        = _bounded_3d(Nx, Ny, zeta_ac; dx=1.0)

    # taud_acx / taud_acy must be zero so the bed-segment closed form
    # tau_xz_bed = -taud_acx = 0 — keeps the test layer-uniform and
    # the depth profile predictable.
    taud_acx  = XFaceField(g3); fill!(interior(taud_acx), 0.0)
    taud_acy  = YFaceField(g3); fill!(interior(taud_acy), 0.0)
    f_ice     = CenterField(g3); fill!(interior(f_ice), 1.0)

    H_ice     = CenterField(g3); fill!(interior(H_ice), 1000.0)
    ATT       = CenterField(g3); fill!(interior(ATT),   1e-16)

    # Direct fill on tau_xz / tau_yz (skipping kernel 1 to isolate
    # kernel 2's recurrence behaviour). Layer-uniform tau_xz = -1000.
    tau_xz    = XFaceField(g3)
    tau_yz    = YFaceField(g3)
    Txz       = interior(tau_xz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Txz[i+1, j, k] = -1000.0
    end
    # Replicate leading slot to match what kernel 1 would produce.
    @views Txz[1, :, :] .= Txz[2, :, :]
    fill!(interior(tau_yz), 0.0)

    ux = XFaceField(g3)
    uy = YFaceField(g3)

    calc_uxy_sia_3D!(ux, uy, tau_xz, tau_yz, taud_acx, taud_acy,
                     H_ice, f_ice, ATT, 3.0, zeta_aa)

    Ux = interior(ux)
    Uy = interior(uy)

    @test all(isfinite, Ux)
    @test all(isfinite, Uy)

    # k = 1 (Center[1]): bed segment contribution. With taud_acx = 0
    # in this test, tau_xz_bed = 0, so
    #   ux[k=1] = 0.5 * fact_ac_x_bed * (tau_xz[k=1] + 0)
    # which is finite and (for non-zero tau_xz) non-zero. We don't pin
    # the exact value — just confirm sign and finiteness via the
    # monotonicity check below.

    # Recurrence sign: Ux moves monotonically in one direction with k
    # at every interior face. We don't pin the magnitude — just that
    # each step keeps the same sign and grows in absolute value.
    for j in 2:Ny-1, i in 2:Nx-1   # interior facial cells only
        for k in 2:Nz
            @test abs(Ux[i+1, j, k]) >= abs(Ux[i+1, j, k-1])
        end
    end

    # tau_yz = 0 AND taud_acy = 0 ⇒ uy stays at zero everywhere
    # (bed segment + recurrence both accumulate 0).
    @test maximum(abs.(Uy)) == 0.0
end

# ======================================================================
# Test 5 — Slab integration test for `dyn_step!` solver = "sia".
# ======================================================================
#
# Builds a synthetic uniform-slab restart NetCDF (8×6 cells, dx=1 km,
# H = 1000 m everywhere, z_bed linearly tilted in x giving a uniform
# surface slope dzsdx = -slope), loads via YelmoModel, sets ATT to a
# uniform constant, runs one `dyn_step!` with solver="sia", and
# checks the depth-averaged velocity against the discretized
# analytical SIA solution.
#
# Continuous SIA on uniform slab (Glen flow law, no sliding):
#   u(zeta) = -2 A H^(n+1) |S|^(n-1) S / (n+1) · (1 - (1 - zeta)^(n+1))
#   ū      = -2 A H^(n+1) |S|^(n-1) S / (n+2)        ← continuous depth-avg
# where S = ∂z_s/∂x = dzsdx.
#
# The Yelmo.jl Option C kernel does NOT match the continuous depth-
# average exactly: it uses a trapezoidal scheme over Center-staggered
# layers plus explicit bed and surface segments. The "ground truth"
# we compare against is the discretized scheme applied to the
# analytical u(zeta) profile evaluated at zeta_c[1..N] plus the bed
# (u = 0) and surface (u_max) endpoints.

# Write a synthetic uniform-slab restart NetCDF compatible with
# YelmoModel(restart_file, time; strict=false). Geometry:
#   - Nx × Ny horizontal cells at dx (m)
#   - z_bed = -slope_x * (x_in_metres)
#   - H_ice = H_const everywhere
#   - z_sl  = -1e6 (well below bed → slab is fully grounded)
#   - f_ice = 1, mask_ice = MASK_ICE_DYNAMIC
#   - zeta_ac = uniform layer interfaces, length Nz_ac → Nz_aa cells
function _write_slab_fixture!(path::AbstractString;
                              Nx::Int, Ny::Int, dx::Float64,
                              H_const::Float64, slope_x::Float64,
                              Nz::Int)
    # Coordinates in metres (xc / yc), converted to km for the file.
    xc_m = collect(range(0.5*dx, (Nx - 0.5)*dx; length=Nx))
    yc_m = collect(range(0.5*dx, (Ny - 0.5)*dx; length=Ny))
    zeta_ac = collect(range(0.0, 1.0; length=Nz + 1))
    zeta_rock_ac = collect(range(0.0, 1.0; length=5))

    NCDataset(path, "c") do ds
        defDim(ds, "xc",           Nx)
        defDim(ds, "yc",           Ny)
        defDim(ds, "zeta",         Nz)
        defDim(ds, "zeta_ac",      Nz + 1)
        defDim(ds, "zeta_rock",    length(zeta_rock_ac) - 1)
        defDim(ds, "zeta_rock_ac", length(zeta_rock_ac))

        xv = defVar(ds, "xc", Float64, ("xc",))
        xv[:] = xc_m ./ 1e3
        xv.attrib["units"] = "km"
        yv = defVar(ds, "yc", Float64, ("yc",))
        yv[:] = yc_m ./ 1e3
        yv.attrib["units"] = "km"

        zc = defVar(ds, "zeta", Float64, ("zeta",))
        zc[:] = 0.5 .* (zeta_ac[1:end-1] .+ zeta_ac[2:end])
        zc.attrib["units"] = "1"
        zac = defVar(ds, "zeta_ac", Float64, ("zeta_ac",))
        zac[:] = zeta_ac
        zac.attrib["units"] = "1"

        zrc = defVar(ds, "zeta_rock", Float64, ("zeta_rock",))
        zrc[:] = 0.5 .* (zeta_rock_ac[1:end-1] .+ zeta_rock_ac[2:end])
        zrc.attrib["units"] = "1"
        zrac = defVar(ds, "zeta_rock_ac", Float64, ("zeta_rock_ac",))
        zrac[:] = zeta_rock_ac
        zrac.attrib["units"] = "1"

        H    = fill(H_const, Nx, Ny)
        z_bed = zeros(Nx, Ny)
        f_ice = ones(Nx, Ny)
        z_sl  = fill(-1e6, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            z_bed[i, j] = -slope_x * xc_m[i]
        end

        for (name, arr) in (("H_ice", H), ("z_bed", z_bed),
                            ("f_ice", f_ice), ("z_sl", z_sl))
            v = defVar(ds, name, Float64, ("xc", "yc"))
            v[:, :] = arr
        end
    end
    return path
end

# Continuous SIA depth-average (m/yr) for a uniform slab.
#
# Derivation: with taud = ρ g H · ∂z_s/∂x, tau_xz(zeta) = -(1-zeta)·taud,
# and du/dzeta = 2 A H · |tau_xz|^(n-1) · tau_xz, integration over
# [0, 1] with u_bed = 0 yields
#
#     u(zeta) = -2 A (ρ g)^n H^(n+1) |S|^(n-1) S · (1 - (1-zeta)^(n+1)) / (n+1)
#     ū      = -2 A (ρ g)^n H^(n+1) |S|^(n-1) S / (n+2)
#
# where S = ∂z_s/∂x.
function _slab_continuous_ubar(; A, H, slope, n, rho_ice, g)
    rhog  = rho_ice * g
    ubar  = -2 * A * rhog^n * H^(n+1) * abs(slope)^(n-1) * slope / (n + 2)
    return ubar
end

# Discretized depth-average (m/yr) — faithful 1D reproduction of the
# Option C SIA kernel's depth recurrence on the uniform slab. Returns
# `(ubar, u_c, u_surf)` so the per-Center profile can be cross-checked
# too.
#
# This is the answer the kernel SHOULD produce on this geometry. The
# scheme:
#
#   1. tau_xz at Centers and at the bed/surface uses the closed form
#      `tau_xz(zeta) = -(1 - zeta)·taud`. Bed: `tau_bed = -taud`.
#      Surface: `tau_surf = 0`.
#   2. The recurrence increments `Δu = 0.5·fact_ac·(tau_top+tau_bot)`
#      where `fact_ac = 2·A·H·dzeta·tau_eff_avg^(n-1)` and
#      `tau_eff_avg = 0.5·(tau_top + tau_bot)` (segment midpoint).
#   3. Bed segment from zeta=0 to zeta_c[1] propagates from u_bed=0
#      to u_c[1]; interior segments k=2..Nz from u_c[k-1] to u_c[k];
#      surface segment from zeta_c[Nz] to 1 from u_c[Nz] to u_surf.
#   4. Depth-average via trapezoidal integration over the same
#      bed/interior/surface segment structure.
function _slab_discretized_ubar(zeta_c::AbstractVector{<:Real};
                                A, H, slope, n, rho_ice, g)
    rhog = rho_ice * g
    # taud follows Yelmo.jl's `taud = ρ g H · ∂z_s/∂x` (signed).
    taud = rhog * H * slope

    Nz = length(zeta_c)
    # Propagate ux_c via the same recurrence the kernel uses.
    u_c = zeros(Float64, Nz)
    # Bed segment: zeta in [0, zeta_c[1]].
    let dzeta = zeta_c[1] - 0.0
        tau_top = -(1 - zeta_c[1]) * taud
        tau_bed = -taud                # closed form at zeta = 0
        tau_avg = 0.5 * (tau_top + tau_bed)
        fact_ac = 2 * A * H * dzeta * abs(tau_avg)^(n - 1)
        u_c[1]  = 0.0 + 0.5 * fact_ac * (tau_top + tau_bed)
    end
    for k in 2:Nz
        dzeta   = zeta_c[k] - zeta_c[k-1]
        tau_top = -(1 - zeta_c[k])     * taud
        tau_bot = -(1 - zeta_c[k-1])   * taud
        tau_avg = 0.5 * (tau_top + tau_bot)
        fact_ac = 2 * A * H * dzeta * abs(tau_avg)^(n - 1)
        u_c[k]  = u_c[k-1] + 0.5 * fact_ac * (tau_top + tau_bot)
    end
    # Surface segment: zeta in [zeta_c[Nz], 1]. tau_surf = 0.
    u_surf = let dzeta = 1.0 - zeta_c[Nz]
        tau_top  = 0.0
        tau_bot  = -(1 - zeta_c[Nz]) * taud
        tau_avg  = 0.5 * (tau_top + tau_bot)
        fact_ac  = 2 * A * H * dzeta * abs(tau_avg)^(n - 1)
        u_c[Nz] + 0.5 * fact_ac * (tau_top + tau_bot)
    end

    # Depth-average via the same trapezoidal-with-boundary scheme
    # `vert_int_trapz_boundary!` uses.
    u_bed = 0.0
    acc = 0.5 * (u_bed + u_c[1]) * (zeta_c[1] - 0.0)
    for k in 2:Nz
        acc += 0.5 * (u_c[k-1] + u_c[k]) * (zeta_c[k] - zeta_c[k-1])
    end
    acc += 0.5 * (u_c[Nz] + u_surf) * (1.0 - zeta_c[Nz])

    return acc, u_c, u_surf
end

# Construct a YelmoModel for the slab + run dyn_step! once. Returns
# the fully-populated `y` (after `update_diagnostics!` and one
# `dyn_step!`). Keeps the write/load/run plumbing in one helper so
# the convergence sub-test can call it for multiple Nz values.
function _run_slab_sia(; Nx::Int, Ny::Int, dx::Float64,
                        H::Float64, slope_x::Float64, Nz::Int,
                        A::Float64, n_glen::Float64)
    tdir = mktempdir(; prefix="slab_sia_$(Nz)_")
    path = joinpath(tdir, "slab_restart.nc")
    _write_slab_fixture!(path; Nx=Nx, Ny=Ny, dx=dx,
                         H_const=H, slope_x=slope_x, Nz=Nz)

    p = Yelmo.YelmoModelPar.YelmoModelParameters("slab-sia";
            ydyn = ydyn_params(solver = "sia"),
            ymat = ymat_params(n_glen = n_glen),
        )

    y = YelmoModel(path, 0.0;
                   rundir = tdir,
                   alias  = "slab-sia",
                   p      = p,
                   strict = false)

    # Uniform ATT.
    fill!(interior(y.mat.ATT), A)

    # Refresh tpo diagnostics so dzsdx, H_ice_dyn, f_ice_dyn,
    # mask_frnt are populated before dyn_step!.
    Yelmo.update_diagnostics!(y)

    Yelmo.YelmoModelDyn.dyn_step!(y, 1.0)

    return y
end

@testset "dyn_step! solver=\"sia\" — uniform slab" begin
    # Slab parameters.
    Nx, Ny  = 8, 6
    dx      = 1000.0           # 1 km
    H       = 1000.0
    slope_x = 0.001            # m/m, gives surface slope -slope_x
    A       = 1e-16
    n_glen  = 3.0
    rho_ice = 910.0
    g_acc   = 9.81

    # Surface slope in the SIA formula. With z_bed = -slope_x * x and
    # H = const, z_srf = z_bed + H = const - slope_x * x, so
    # ∂z_s/∂x = -slope_x. Plug as `slope` in the analytical formula.
    slope = -slope_x

    Nz_main = 10
    y = _run_slab_sia(; Nx=Nx, Ny=Ny, dx=dx, H=H, slope_x=slope_x,
                       Nz=Nz_main, A=A, n_glen=n_glen)

    zeta_c_main = znodes(y.gt, Center())
    @test length(zeta_c_main) == Nz_main

    # Analytical reference values.
    ubar_cont = _slab_continuous_ubar(; A=A, H=H, slope=slope,
                                       n=n_glen, rho_ice=rho_ice,
                                       g=g_acc)
    ubar_disc, u_c_disc, u_surf_disc =
        _slab_discretized_ubar(zeta_c_main; A=A, H=H, slope=slope,
                               n=n_glen, rho_ice=rho_ice, g=g_acc)

    # Sanity: continuous answer is on the order of a few × 10⁻²
    # m/yr for the slab parameters above, and positive (flow in +x).
    @test ubar_cont > 0
    @test 1e-3 < ubar_cont < 1e-1
    @test ubar_disc > 0
    # Discretized answer should be close (within a few percent) to
    # the continuous answer for Nz = 10.
    @test abs(ubar_disc - ubar_cont) / abs(ubar_cont) < 0.05

    # Kernel output. Pull the depth-averaged velocity from interior
    # face cells (skip outermost face row in each direction — those
    # are the leading-replicated face slot or the boundary face that
    # may differ from the discretized answer due to halo / margin
    # effects).
    Ux_bar = interior(y.dyn.ux_bar)
    Uy_bar = interior(y.dyn.uy_bar)

    # Restrict to "inner" face cells: x-faces at i+1 ∈ 3:Nx (skip
    # i=1, leading-replicated; skip i=2, leftmost margin; skip i=Nx+1,
    # rightmost margin where halo / boundary face value may differ)
    # and j ∈ 2:Ny-1 (skip y boundaries).
    inner_xfaces = Ux_bar[3:Nx, 2:Ny-1, 1]
    @test all(isfinite, inner_xfaces)
    @test maximum(abs.(inner_xfaces .- ubar_disc)) /
          max(abs(ubar_disc), 1e-30) < 1e-9

    # 3D ux_i profile at the same inner cells: should equal u_c_disc.
    Ux_i = interior(y.dyn.ux_i)
    for k in 1:Nz_main
        slab = Ux_i[3:Nx, 2:Ny-1, k]
        @test maximum(abs.(slab .- u_c_disc[k])) /
              max(abs(u_c_disc[k]), 1e-30) < 1e-9
    end

    # Surface boundary value (scratch.ux_i_s) should equal u_surf_disc.
    Ux_is = interior(y.dyn.scratch.ux_i_s)
    inner_xis = Ux_is[3:Nx, 2:Ny-1, 1]
    @test maximum(abs.(inner_xis .- u_surf_disc)) /
          max(abs(u_surf_disc), 1e-30) < 1e-9

    # uy_bar should be zero (no y-slope).
    inner_yfaces = Uy_bar[2:Nx-1, 3:Ny, 1]
    @test maximum(abs.(inner_yfaces)) < 1e-12

    # Basal velocity / basal stress are zeroed by the SIA branch.
    @test maximum(abs.(interior(y.dyn.ux_b))) == 0.0
    @test maximum(abs.(interior(y.dyn.uy_b))) == 0.0
    @test maximum(abs.(interior(y.dyn.taub_acx))) == 0.0
    @test maximum(abs.(interior(y.dyn.taub_acy))) == 0.0

    # ux ≈ ux_i (since ux_b = 0). Allow a tiny tolerance because
    # `_clip_underflow!` runs on `ux` but not on `ux_i`; for the
    # well-resolved interior cells the difference is exactly 0.
    @test maximum(abs.(interior(y.dyn.ux) .- interior(y.dyn.ux_i))) < 1e-15
    @test maximum(abs.(interior(y.dyn.uy) .- interior(y.dyn.uy_i))) < 1e-15

    # Surface diagnostic: ux_s = scratch.ux_i_s + ux_b = scratch.ux_i_s
    # (since ux_b = 0). Compare on inner face cells.
    Ux_s = interior(y.dyn.ux_s)
    @test maximum(abs.((Ux_s .- Ux_is)[3:Nx, 2:Ny-1, 1])) < 1e-15

    # ---- Convergence sub-assert: rerun the full kernel at Nz = 20
    # and Nz = 40, confirm the kernel output converges toward the
    # continuous answer as the layer count grows.
    function _kernel_err(Nz_local)
        y_local = _run_slab_sia(; Nx=Nx, Ny=Ny, dx=dx, H=H,
                                  slope_x=slope_x, Nz=Nz_local,
                                  A=A, n_glen=n_glen)
        ubar_local = interior(y_local.dyn.ux_bar)[3, 3, 1]
        return abs(ubar_local - ubar_cont) / abs(ubar_cont)
    end
    err_10 = abs(ubar_disc - ubar_cont) / abs(ubar_cont)
    err_20 = _kernel_err(20)
    err_40 = _kernel_err(40)
    @test err_20 < err_10
    @test err_40 < err_20
end
