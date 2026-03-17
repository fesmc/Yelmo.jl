## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

using BenchmarkTools
using Revise
#using CairoMakie
#using Oceananigans.Fields
using Yelmo

# Define parameters and write parameter file
p = YelmoParameters("Greenland")

# Initialize Yelmo
ylmo = YelmoMirror(p, 0.0; overwrite=true);

# Populate boundary fields
ylmo.bnd.H_sed .= 100.0;

# Initialize Yelmo state
@btime init_state!(ylmo, 0.0, "robin-cold");

time_init, time_end, dt = 0.0, 5.0, 1.0;

for t in time_init:dt:time_end

    # Advance by dt
    @btime time_step!(ylmo,t-ylmo.time);

    # Update boundary fields
    ylmo.bnd.z_bed .+= 100.0

end

# Plot some data
heatmap(ylmo.dyn.uxy_s,colorscale=log10)


## Parameter sets

p1 = YelmoParameters("Greenland");
p2 = YelmoParameters("Greenland";ydyn  = ydyn_params(solver="ssa"));

p3 = YelmoParameters("/Users/alrobi001/models/yelmo/output/grl-diva-test/yelmo_initmip.nml","Greenland")
