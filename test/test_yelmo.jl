## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

using BenchmarkTools
using Revise
using CairoMakie
using Oceananigans.Fields
using Oceananigans.Grids
using Yelmo

# Define parameters and write parameter file
p = YelmoParameters("Greenland")

# Initialize Yelmo
ylmo = YelmoMirror(p, 0.0; rundir="run01", overwrite=true);

# Populate boundary fields
ylmo.bnd.H_sed .= 100.0;

# Initialize Yelmo state
init_state!(ylmo, 0.0, "robin-cold");

# Initialize output file
out = init_output(ylmo, joinpath(ylmo.rundir,"yelmo.nc"),
    selection = OutputSelection(
        groups  = [:tpo,:dyn,:thrm,:mat,:bnd],
    )
)

# Write initial state
write_output!(out, ylmo)
#close(out)

time_init, time_end, dt = 0.0, 5.0, 1.0;

for t in time_init:dt:time_end

    # Update isostasy
    ylmo.bnd.z_bed .+= 100.0

    # Advance by dt
    time_step!(ylmo,t-ylmo.time);

    write_output!(out, ylmo)
end

# Close output file
close(out)

# Plot some data
heatmap(ylmo.dyn.uxy_s,colorscale=log10)


## Parameter sets

p1 = YelmoParameters("Greenland");
p2 = YelmoParameters("Greenland";ydyn  = ydyn_params(solver="ssa"));

p3 = YelmoParameters("/Users/alrobi001/models/yelmo/output/grl-diva-test/yelmo_initmip.nml","Greenland")


## YelmoMirror Writing

# Default: write everything
out = init_output(ylmo, "output/yelmo.nc")

# Only dynamics and thermodynamics, excluding face-staggered fields
out = init_output(ylmo, "output/yelmo.nc";
    selection = OutputSelection(
        groups  = [:dyn, :thrm],
        exclude = vcat(XFACE_VARIABLES, YFACE_VARIABLES, ZFACE_VARIABLES),
    )
)

# Time loop
time_init, time_end, dt = 0.0, 5.0, 1.0;

for t in time_init:dt:time_end
    time_step!(ylmo,t-ylmo.time);
    write_output!(out, ylmo)
end

close(out)