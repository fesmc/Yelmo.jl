# ----------------------------------------------------------------------
# BUELER-B smoke benchmark — analytical Halfar fixture.
#
# BUELER-B has a closed-form analytical solution (the Halfar
# similarity profile, Bueler et al. 2005 Eqs. 10–11), so the
# "fixture" is just the analytical state at the chosen output time.
# We don't need YelmoMirror to produce it — the
# `bueler_test_BC!` Julia port from `bueler.jl` is the source of
# truth.
#
# Fixture: 31×31 grid at dx = 50 km, isothermal SIA dome with
# H₀ = 3600 m, R₀ = 750 km, λ = 0 (no mass balance), evaluated at
# t = 1000 yr after the analytical IC at t = 0.
#
# Validation by milestone:
#   - **scaffolding (this PR)**: smoke test loads the fixture via
#     YelmoModel and checks dome geometry / radial symmetry. Proves
#     the scaffolding (analytical writer + YelmoModel loader)
#     round-trips cleanly.
#   - **3c (SIA solver)**: lockstep test loads the t=0 IC via the
#     same path, runs YelmoModel's SIA solver to t=1000, compares
#     against the analytical Halfar at t=1000. Validates the SIA
#     solver against the analytical reference.
# ----------------------------------------------------------------------

using NCDatasets

# Halfar / BUELER-B physical parameters (mirror the Fortran defaults
# in `yelmo_benchmarks.f90:233-238`).
const _BUELER_B_DX_KM   = 50.0
const _BUELER_B_R0_KM   = 750.0
const _BUELER_B_NX      = Int(2 * _BUELER_B_R0_KM / _BUELER_B_DX_KM) + 1   # 31
const _BUELER_B_H0      = 3600.0    # initial dome thickness [m]
const _BUELER_B_LAMBDA  = 0.0       # no mass balance
const _BUELER_B_N       = 3.0
const _BUELER_B_A       = 1e-16     # [Pa^-3 yr^-1]
const _BUELER_B_RHO_ICE = 910.0
const _BUELER_B_G       = 9.81

function _bueler_b_axes()
    half_m = _BUELER_B_R0_KM * 1e3
    xc = collect(range(-half_m, half_m; length=_BUELER_B_NX))
    return xc, copy(xc)
end

# Write the analytical Halfar state at `time` into a freshly-opened
# NetCDF dataset. Coordinates are already laid down by
# `write_analytical_fixture!`; we just `defVar` the state fields we
# care about. For BUELER-B these are `H_ice` (the analytical Halfar
# profile), `z_bed` (flat at 0), and `smb_ref` (the analytical mass
# balance — exactly 0 for λ = 0).
function _bueler_b_write_fields!(ds::NCDataset, time::Float64)
    xc_m = ds["xc"][:] .* 1e3   # convert km back to metres
    yc_m = ds["yc"][:] .* 1e3
    Nx, Ny = length(xc_m), length(yc_m)

    H    = zeros(Nx, Ny)
    smb  = zeros(Nx, Ny)
    bueler_test_BC!(H, smb, xc_m, yc_m, time;
                    R0     = _BUELER_B_R0_KM,
                    H0     = _BUELER_B_H0,
                    lambda = _BUELER_B_LAMBDA,
                    n      = _BUELER_B_N,
                    A      = _BUELER_B_A,
                    rho_ice = _BUELER_B_RHO_ICE,
                    g      = _BUELER_B_G)

    H_var = defVar(ds, "H_ice", Float64, ("xc", "yc"))
    H_var[:, :] = H
    H_var.attrib["units"]     = "m"
    H_var.attrib["long_name"] = "Ice thickness (analytical Halfar)"

    smb_var = defVar(ds, "smb_ref", Float64, ("xc", "yc"))
    smb_var[:, :] = smb
    smb_var.attrib["units"]     = "m/yr"
    smb_var.attrib["long_name"] = "Surface mass balance (analytical)"

    z_bed_var = defVar(ds, "z_bed", Float64, ("xc", "yc"))
    z_bed_var[:, :] = zeros(Nx, Ny)
    z_bed_var.attrib["units"]     = "m"
    z_bed_var.attrib["long_name"] = "Bedrock elevation (flat)"

    # Provenance
    ds.attrib["benchmark"]     = "BUELER-B"
    ds.attrib["solution_type"] = "analytical-halfar"
    ds.attrib["time_yr"]       = time
    ds.attrib["R0_km"]         = _BUELER_B_R0_KM
    ds.attrib["H0_m"]          = _BUELER_B_H0
    ds.attrib["lambda"]        = _BUELER_B_LAMBDA
    ds.attrib["n_glen"]        = _BUELER_B_N
    ds.attrib["A_Pa-3yr-1"]    = _BUELER_B_A

    return nothing
end

const BUELER_B_SMOKE_SPEC = let
    xc, yc = _bueler_b_axes()
    AnalyticalSpec(
        name          = "bueler_b_smoke",
        output_times  = [1000.0],
        xc            = xc,
        yc            = yc,
        write_fields! = _bueler_b_write_fields!,
    )
end
