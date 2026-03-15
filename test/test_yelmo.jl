## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

#using Revise
using CairoMakie
using Yelmo

# Define parameters and write parameter file
p = YelmoParameters("Greenland")

# Initialize Yelmo
ylmo = YelmoMirror(p, 0.0);

# Populate boundary fields
ylmo.bnd.H_sed .= 100.0

# Initialize Yelmo state
init_state!(ylmo, 0.0, "robin-cold");

time_init, time_end, dt = 0.0, 5.0, 1.0;

for t in time_init:dt:time_end

    # Advance by dt
    time_step!(ylmo,t-ylmo.time);

    # Update boundary fields
    ylmo.bnd.z_bed .+= 100.0

end

# Plot some data
heatmap(ylmo.g.xc,ylmo.g.yc,log10.(ylmo.dyn.uxy_s))

