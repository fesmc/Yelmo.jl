## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 3a tests for `dyn`. Three groups:
#
#   1. Analytical kernel benchmarks — `calc_driving_stress!`,
#      `calc_lateral_bc_stress_2D!`, `calc_ice_flux!`,
#      `calc_magnitude_from_staggered!`, `calc_vel_ratio!` on
#      synthetic geometries with closed-form expected values.
#
#   2. ISMIP-HOM Experiment A driving-stress shape — sinusoidal
#      surface, periodic in x, slope-only in y. Verifies the kernel
#      reproduces the analytic spatial pattern.
#
#   3. Real-restart consistency — load the 16 km Greenland restart,
#      snapshot the loaded (Fortran-computed) dyn diagnostics, refresh
#      tpo + run `dyn_step!(y, 0.0)`, then compare the recomputed
#      magnitudes / fluxes / stresses against the snapshot. Boundary
#      face rows are skipped (BC handling differs between the Fortran
#      `infinite` code and our Oceananigans halo BCs); interior cells
#      should match within Float32 NetCDF rounding (`tol = 1e-3`).

using Test
using Yelmo
using Yelmo.YelmoModelPar: ydyn_params
using Oceananigans
using Oceananigans: interior
using Oceananigans.BoundaryConditions: fill_halo_regions!
using NCDatasets

const RESTART_PATH = "/Users/alrobi001/models/yelmox/output/16KM/test/restart-0.000-kyr/yelmo_restart.nc"
const NML_PATH     = "/Users/alrobi001/models/yelmox/output/16KM/test/yelmo_Greenland_rembo.nml"

# rel-L∞ helpers — same shape as `test_yelmo_topo.jl`. `rel_linf_inner`
# strips the outermost face row/col on every side, which is where the
# Oceananigans halo BC and the Fortran "infinite" BC code diverge.
function _rel_linf(now::AbstractArray, ref::AbstractArray)
    max_diff = maximum(abs.(now .- ref))
    max_ref  = maximum(abs.(ref))
    return max_ref > 0 ? max_diff / max_ref : max_diff
end

function _rel_linf_inner(now::AbstractArray, ref::AbstractArray; trim::Int = 1)
    nx, ny = size(now, 1), size(now, 2)
    slc = (now .- ref)[1+trim:nx-trim, 1+trim:ny-trim, :]
    max_diff = maximum(abs.(slc))
    max_ref  = maximum(abs.(ref[1+trim:nx-trim, 1+trim:ny-trim, :]))
    return max_ref > 0 ? max_diff / max_ref : max_diff
end

_bounded_2d(Nx, Ny; dx=1.0) = RectilinearGrid(size=(Nx, Ny),
                                                x=(0.0, Nx*dx), y=(0.0, Ny*dx),
                                                topology=(Bounded, Bounded, Flat))


# ======================================================================
# Group 1: Analytical kernel benchmarks
# ======================================================================

@testset "dyn: calc_driving_stress! — uniform slab" begin
    # Uniform 1000 m slab on a constant 0.001 surface slope in x.
    # Expected: τ_d = ρ_i · g · H · slope = 910 · 9.81 · 1000 · 0.001
    #                ≈ 8927.1 Pa, identical at every interior face.
    Nx, Ny = 8, 6
    dx = 1000.0
    g  = _bounded_2d(Nx, Ny; dx=dx)

    H = CenterField(g);  fill!(interior(H), 1000.0)
    f = CenterField(g);  fill!(interior(f), 1.0)
    sx = CenterField(g); fill!(interior(sx), 0.001)
    sy = CenterField(g); fill!(interior(sy), 0.0)

    Tx = XFaceField(g)
    Ty = YFaceField(g)

    rho_ice = 910.0
    grav    = 9.81
    expected_x = rho_ice * grav * 1000.0 * 0.001

    calc_driving_stress!(Tx, Ty, H, f, sx, sy,
                         dx, 1e10, rho_ice, grav)

    # Interior face slots `[2:Nx, 2:Ny-1, 1]` correspond to Fortran
    # `taud_acx[1:Nx-1, 2:Ny-1]` — full ice cells on both sides of
    # each face, no margin. (The eastmost interior slot `[Nx+1, ...]`
    # uses halo H_ice = 0 from the Dirichlet BC, which would zero the
    # face; trim it.)
    @test maximum(abs.(interior(Tx)[2:Nx, 2:Ny-1, 1] .- expected_x)) < 1e-9
    @test maximum(abs.(interior(Ty)[2:Nx-1, 2:Ny, 1])) < 1e-9
end

@testset "dyn: calc_driving_stress! — taud_lim clamp" begin
    # Build geometry with τ_d > taud_lim → expected to clamp.
    Nx, Ny = 4, 3
    g = _bounded_2d(Nx, Ny; dx=1.0)
    H = CenterField(g);  fill!(interior(H), 3000.0)
    f = CenterField(g);  fill!(interior(f), 1.0)
    sx = CenterField(g); fill!(interior(sx), 0.05)   # very steep
    sy = CenterField(g); fill!(interior(sy), 0.0)

    Tx = XFaceField(g)
    Ty = YFaceField(g)

    taud_lim = 2e5
    calc_driving_stress!(Tx, Ty, H, f, sx, sy,
                         1.0, taud_lim, 910.0, 9.81)
    raw = 910.0 * 9.81 * 3000.0 * 0.05   # ≈ 1.34e6 Pa
    @test raw > taud_lim
    @test maximum(interior(Tx)[2:Nx, :, 1]) ≈ taud_lim atol = 1e-9
end

@testset "dyn: calc_lateral_bc_stress_2D! — floating-front formula" begin
    # 6×3 grid: cell i=3 is fully covered, marine front; cell i=4 is
    # ice-free margin neighbour (`mask_frnt = -1`). Expected face
    # value at the i=3↔i=4 face matches the Lipscomb 2019 formula.
    Nx, Ny = 6, 3
    g = _bounded_2d(Nx, Ny; dx=1000.0)

    rho_ice = 910.0
    rho_sw  = 1028.0
    grav    = 9.81

    H_val   = 800.0
    z_sl_val = 0.0
    # Floating ice surface elevation per the buoyancy relation:
    z_srf_val = (1.0 - rho_ice / rho_sw) * H_val   # ≈ 95.5 m
    f_submerged = 1.0 - min((z_srf_val - z_sl_val) / H_val, 1.0)
    H_ocn = H_val * f_submerged
    expected = 0.5 * rho_ice * grav * H_val^2 -
               0.5 * rho_sw  * grav * H_ocn^2

    H_ice  = CenterField(g);   fill!(interior(H_ice),  H_val)
    f_ice  = CenterField(g);   fill!(interior(f_ice),  1.0)
    z_srf  = CenterField(g);   fill!(interior(z_srf),  z_srf_val)
    z_sl   = CenterField(g);   fill!(interior(z_sl),   z_sl_val)
    mfield = CenterField(g);   fill!(interior(mfield), 0.0)

    # Front at column i=3 (marine, mask_frnt=+1) bordered by ice-free
    # i=4 (mask_frnt=-1).
    interior(mfield)[3, :, 1] .=  1.0
    interior(mfield)[4, :, 1] .= -1.0

    Tx = XFaceField(g)
    Ty = YFaceField(g)

    calc_lateral_bc_stress_2D!(Tx, Ty, mfield, H_ice, f_ice,
                                z_srf, z_sl, rho_ice, rho_sw, grav)

    # Face between cells 3 and 4 in x is interior(Tx)[4, j, 1].
    @test all(abs.(interior(Tx)[4, :, 1] .- expected) .< 1e-6)
    # All other faces should be zero — no front configured.
    Tx_int = copy(interior(Tx))
    Tx_int[4, :, 1] .= 0.0       # zero out the front face
    Tx_int[1, :, 1] .= 0.0       # the replicate slot mirrors front=0 elsewhere
    @test maximum(abs.(Tx_int)) < 1e-9
    @test maximum(abs.(interior(Ty))) < 1e-9
end

@testset "dyn: calc_ice_flux! — uniform u_bar / H" begin
    Nx, Ny = 5, 4
    dx = 2000.0
    g  = _bounded_2d(Nx, Ny; dx=dx)

    H = CenterField(g); fill!(interior(H), 500.0)
    u = XFaceField(g);  fill!(interior(u), 100.0)
    v = YFaceField(g);  fill!(interior(v), -50.0)

    qx = XFaceField(g)
    qy = YFaceField(g)

    calc_ice_flux!(qx, qy, u, v, H, dx, dx)

    expected_qx =  0.5 * (500.0 + 500.0) * dx * 100.0
    expected_qy =  0.5 * (500.0 + 500.0) * dx * (-50.0)

    # Fortran loops i = 1..Nx-1 only, so eastmost face is left at 0;
    # interior(qx)[i+1, j, 1] for i = 1..Nx-1 → interior pos 2..Nx.
    @test all(abs.(interior(qx)[2:Nx, :, 1] .- expected_qx) .< 1e-9)
    # Eastmost face slot (interior pos Nx+1) was never written.
    @test all(interior(qx)[Nx+1, :, 1] .== 0.0)
    @test all(abs.(interior(qy)[:, 2:Ny, 1] .- expected_qy) .< 1e-9)
end

@testset "dyn: calc_magnitude_from_staggered! — known u, v" begin
    Nx, Ny = 4, 3
    g = _bounded_2d(Nx, Ny; dx=1.0)

    f = CenterField(g); fill!(interior(f), 1.0)

    u = XFaceField(g);  fill!(interior(u), 3.0)
    v = YFaceField(g);  fill!(interior(v), 4.0)

    mag = CenterField(g)
    calc_magnitude_from_staggered!(mag, u, v, f)

    # Cell-centre u = ½(3+3) = 3, v = ½(4+4) = 4 → mag = 5
    @test all(abs.(interior(mag)[:, :, 1] .- 5.0) .< 1e-12)

    # f_ice = 0 in one cell zeros that cell's magnitude.
    interior(f)[2, 2, 1] = 0.0
    calc_magnitude_from_staggered!(mag, u, v, f)
    @test interior(mag)[2, 2, 1] == 0.0
end

@testset "dyn: calc_vel_ratio! — edge cases" begin
    Nx, Ny = 3, 2
    g  = _bounded_2d(Nx, Ny; dx=1.0)
    fb = CenterField(g)
    ub = CenterField(g);  fill!(interior(ub), 0.0)
    us = CenterField(g);  fill!(interior(us), 0.0)

    # uxy_s = 0 everywhere ⇒ f_vbvs = 1
    calc_vel_ratio!(fb, ub, us)
    @test all(interior(fb) .== 1.0)

    # uxy_s > 0, uxy_b = ½ uxy_s ⇒ f_vbvs = 0.5
    fill!(interior(us), 100.0)
    fill!(interior(ub),  50.0)
    calc_vel_ratio!(fb, ub, us)
    @test all(interior(fb) .== 0.5)

    # uxy_b > uxy_s ⇒ clamp at 1
    fill!(interior(ub), 200.0)
    calc_vel_ratio!(fb, ub, us)
    @test all(interior(fb) .== 1.0)
end

# ======================================================================
# Group 2: ISMIP-HOM-A driving-stress shape
# ======================================================================
#
# The full ISMIP-HOM A benchmark (sinusoidal bed, periodic-x, slope-only
# in y; Pattyn et al. 2008) is deferred to milestone 3c when the SIA
# solver lands and we have a velocity to compare against. The driving
# stress kernel currently assumes Bounded XFace indexing
# (interior shape = Nx+1 in x), so the ac-staggering on a Periodic-x
# grid (interior shape = Nx) needs a topology-aware write index. That
# generalisation is bundled with the SIA / SSA solver work where
# periodic grids actually matter for the lockstep tests.

# ======================================================================
# Group 3: Real-restart consistency
# ======================================================================

@testset "dyn: real-restart diagnostic-chain consistency" begin
    @assert isfile(RESTART_PATH) "Restart fixture not found at $(RESTART_PATH)"
    @assert isfile(NML_PATH)     "Namelist fixture not found at $(NML_PATH)"

    # Load the actual run namelist so parameter-sensitive kernels
    # (`grad_lim`, `gl_sep`, `taud_lim`, etc.) match the Fortran
    # configuration that wrote this restart. Then override
    # `ydyn.solver = "fixed"` so `(ux, uy, ux_bar, uy_bar, …)` stay
    # at their restart-loaded values; `dyn_step!` only refreshes the
    # diagnostic outputs (driving stress, lateral stress, ice flux,
    # magnitudes, surface / basal slices, `f_vbvs`).
    p_nml = Yelmo.YelmoModelPar.read_nml(NML_PATH)
    p = Yelmo.YelmoModelPar.YelmoModelParameters("dyn-consistency";
            yelmo           = p_nml.yelmo,
            ytopo           = p_nml.ytopo,
            ycalv           = p_nml.ycalv,
            ydyn            = ydyn_params(solver = "fixed",
                                          taud_lim       = p_nml.ydyn.taud_lim,
                                          taud_gl_method = p_nml.ydyn.taud_gl_method),
            ytill           = p_nml.ytill,
            yneff           = p_nml.yneff,
            ymat            = p_nml.ymat,
            ytherm          = p_nml.ytherm,
            yelmo_masks     = p_nml.yelmo_masks,
            yelmo_init_topo = p_nml.yelmo_init_topo,
            yelmo_data      = p_nml.yelmo_data,
        )

    y = YelmoModel(RESTART_PATH, 0.0;
                   rundir = mktempdir(; prefix="dyn_diag_"),
                   alias  = "dyn-consistency",
                   p      = p,
                   groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                   strict = false)

    # Snapshot the dyn fields as written by Fortran (still in the
    # YelmoModel after `load_state!`).
    snap_fields = (:taud_acx, :taud_acy, :taud, :taub,
                   :taul_int_acx, :taul_int_acy,
                   :qq_acx, :qq_acy, :qq,
                   :uxy_bar, :uxy_b, :uxy_s, :uxy_i_bar,
                   :uz_b, :uz_s, :ux_s, :uy_s,
                   :f_vbvs)
    snap = NamedTuple{snap_fields}(copy(interior(getfield(y.dyn, k))) for k in snap_fields)

    # Refresh tpo diagnostics first so dyn reads the right
    # `H_ice_dyn`, `f_ice_dyn`, `dzsdx`, `dzsdy`, `mask_frnt`. Then
    # run `dyn_step!(y, 0.0)` so `duxydt = 0` (no time advance) and we
    # only check the spatial diagnostics.
    update_diagnostics!(y)
    Yelmo.YelmoModelDyn.dyn_step!(y, 0.0)

    # --- Tight: deterministic per-cell formulas. Driving stress and
    # lateral stress are independent of solver state and depend only on
    # the topo / boundary inputs. Compare interior only — boundary face
    # rows differ between the Fortran `infinite` BC code and our halo
    # BCs (Dirichlet H = 0 on the eastern / northern edge for `H_ice`).
    for k in (:taud_acx, :taud_acy)
        err = _rel_linf_inner(interior(getfield(y.dyn, k)), snap[k])
        @test err < 1e-3
    end
    for k in (:taul_int_acx, :taul_int_acy)
        err = _rel_linf_inner(interior(getfield(y.dyn, k)), snap[k])
        @test err < 1e-3
    end

    # Ice flux (deterministic from u_bar · H_face · dx).
    for k in (:qq_acx, :qq_acy)
        err = _rel_linf_inner(interior(getfield(y.dyn, k)), snap[k])
        @test err < 1e-3
    end

    # Magnitudes (deterministic from staggered velocity / stress).
    # Float32 NetCDF storage limits the achievable tolerance.
    for k in (:taud, :taub, :qq,
              :uxy_bar, :uxy_b, :uxy_s, :uxy_i_bar)
        err = _rel_linf_inner(interior(getfield(y.dyn, k)), snap[k])
        @test err < 1e-3
    end

    # Surface / basal velocity slices — Julia recomputes by indexing
    # the appropriate z-layer. Should match the loaded Fortran slice
    # exactly (modulo Float32 rounding).
    for k in (:uz_b, :uz_s, :ux_s, :uy_s)
        err = _rel_linf_inner(interior(getfield(y.dyn, k)), snap[k])
        @test err < 1e-3
    end

    # f_vbvs — element-wise ratio.
    err = _rel_linf_inner(interior(y.dyn.f_vbvs), snap.f_vbvs)
    @test err < 1e-3
end
