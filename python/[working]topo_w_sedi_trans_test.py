# Viet's play ground
# packages for the model
from landlab.components import (
    LinearDiffuser, OverlandFlow, FlowDirectorSteepest, FlowAccumulator, 
    SedDepEroder, ChannelProfiler, PriorityFloodFlowRouter, DepressionFinderAndRouter, Space
)
from landlab import RasterModelGrid, imshow_grid
from landlab.plot.graph import plot_graph
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#define the grid
size_x = 100
size_y = 200
spacing = 1.0
grid = RasterModelGrid((size_y, size_x), xy_spacing=spacing)

# morph the grid with a sine function
ice_height = 10
bedrock_slope = 0.22
soil_thickness = 5
moraine_disappears_at_y = 100

# add the topography    
z_ice = grid.add_zeros('ice_thickness', at='node')
z_soil = grid.add_zeros('soil_thickness', at='node')
z_bed = grid.add_zeros('bedrock_thickness', at='node')
elev = grid.add_zeros('topographic__elevation', at='node')

x = grid.x_of_node
y = grid.y_of_node

# setting ice elevation respected to the base elevation
z = ice_height * np.sin(np.pi*x / (size_x/3)) 
# get z where y is less than moraine_disappears_at_y
z[y<moraine_disappears_at_y] += (y[y<moraine_disappears_at_y] - moraine_disappears_at_y) * ice_height/moraine_disappears_at_y
z[z<0] = 0 # cut the sine function at 0

z_ice += z
z_soil += soil_thickness
z_bed += bedrock_slope * y
elev = z_ice + z_soil + z_bed

# reassign the values to the grid just to be sure
grid.at_node['ice_thickness'] = z_ice
grid.at_node['soil_thickness'] = z_soil
grid.at_node['bedrock_thickness'] = z_bed
grid.at_node['topographic__elevation'] = elev

# grid.imshow(grid.at_node["topographic__elevation"], color_for_closed = 'm')
# plt.show()

# fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(12, 10))

# def plot_each_axs(axs,size_x,size_y,topo, title="Topography"):
#     X, Y = np.meshgrid(np.arange(0,size_x,spacing),np.arange(0,size_y,spacing))
#     Z = topo.reshape(size_y, size_x)
#     axs.set_box_aspect((1, size_y/size_x, Z.max()/size_x))
#     axs.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#     axs.set_title(title)
#     cbar = fig.colorbar(axs.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False), ax=axs, shrink=0.5, aspect=10)

# # Plot Ice Thickness
# plot_each_axs(axs[0, 0], size_x, size_y, grid.at_node['ice_thickness'], title="Ice Thickness")

# # Plot Soil Thickness
# plot_each_axs(axs[0, 1], size_x, size_y, grid.at_node['soil_thickness'], title="Soil Thickness")

# # Plot Bedrock Layer
# plot_each_axs(axs[1, 0], size_x, size_y, grid.at_node['bedrock_thickness'], title="Bedrock Layer")

# # Plot Topography
# plot_each_axs(axs[1, 1], size_x, size_y, grid.at_node['topographic__elevation'], title="Topography")

# Adjust layout
# plt.tight_layout()
# plt.show()

# Add field "soil__depth" to the grid
grid.add_zeros("soil__depth", at="node", dtype = "float")
# Getting the calculated elevation that assigned into the soil_thickness at the beginning
# and assign it to the soil__depth
grid.at_node["soil__depth"] = grid.at_node["soil_thickness"]

grid.add_ones("bedrock__elevation", at="node", dtype = "float")
# Getting the calculated elevation that assigned into the ice_thickness at the beginning
# and assign it to the bedrock__elevation
# assuming ice is also bedrock
grid.at_node["bedrock__elevation"] = grid.at_node["ice_thickness"] + grid.at_node["bedrock_thickness"]

# set constant random seed for consistent topographic roughness
np.random.seed(seed=5000)

# impose topography values on model grid
elev += np.random.uniform(0.0, 1.0, size=elev.shape)
# add boundaries for the ridge of the moraine
grid.status_at_node[grid.x_of_node < (100/6)] = grid.BC_NODE_IS_CLOSED
grid.status_at_node[grid.x_of_node > (100-100/6)] = grid.BC_NODE_IS_CLOSED

# Set lower-edge as an open boundary
grid.set_watershed_boundary_condition_outlet_id(
    50 , grid.at_node["topographic__elevation"], -9999.0
)

# grid.imshow(grid.at_node['topographic__elevation'], color_for_closed = 'm')
# plt.show()


# ChannelProfiler parameters
number_of_watersheds = 1
minimum_channel_threshold=100
# PriorityFloodFLowRouter parameters
flow_metric = "D8"
phi_FR = 0.0

# Instantiate flow router
fr = FlowAccumulator(grid, flow_director="FlowDirectorD8")

# Flow routing
# fr = PriorityFloodFlowRouter(
# 	grid,
# 	surface = "topographic__elevation",
# 	flow_metric = flow_metric,
# 	depression_handler = "fill",
# 	accumulate_flow = True,
# 	separate_hill_flow = True,
# 	accumulate_flow_hill = True,
# 	)

# Instantiate depression finder and router; optional
df = DepressionFinderAndRouter(grid)

# Instantiate SPACE model with chosen parameters
sp = Space(
    grid,
    K_sed=0.01, # Sediment erodibility, Governs the rate of sediment entrainment; may be specified as a single floating point number, an array of length equal to the number of grid nodes, or a string naming an existing grid field.
    K_br=0.00001, # Bedrock erodibility, Governs the rate of bedrock erosion; may be specified as a single floating point number, an array of length equal to the number of grid nodes, or a string naming an existing grid field.
    F_f=0.0, # Fraction of fine sediment, (unitless, 0-1) fraction of rock that does not get converted to sediment but assumed to exit model domain as wash load
    phi=0.0, # Sediment porosity
    H_star=1.0, # Sediment entrainment length scale - think of reflecting bedrock surface roughness
    v_s=5.0, # Effective settling velocity
    m_sp=0.5, # Stream power exponent on drainage area or discharge in the stream power framework. Generally 0.5
    n_sp=1.0, # Stream power exponent n  on channel slope around 1
    sp_crit_sed=0, 
    sp_crit_br=0,
)

# Set model timestep
timestep = 50.0  # years

# Set elapsed time to zero
elapsed_time = 0.0  # years

# Set timestep count to zero
count = 0

# Set model run time
run_time = 200  # years

# Array to save sediment flux values
sed_flux = np.zeros(int(run_time // timestep))  # Adjusted size
node_next_to_outlet = 151

while elapsed_time < run_time:  # Changed condition
    # Run the flow router
    fr.run_one_step()

    # Run the depression finder and router
    df.map_depressions()

    # Run SPACE for one time step
    sp.run_one_step(dt=timestep)

    # Save sediment flux value to array
    sed_flux[count] = grid.at_node["sediment__flux"][node_next_to_outlet]

    # Add to value of elapsed time
    elapsed_time += timestep

    # Increase timestep count
    count += 1
    # Print progress
    print(f"Elapsed time: {elapsed_time:.2f} years, Sediment flux: {sed_flux[count-1]:.2f} m^3/yr")


# Create a figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot topographic elevation in the first subplot
plt.sca(axes[0])  # Set the current axis to the first subplot
imshow_grid(
    grid,
    "topographic__elevation",
    plot_name="Topographic Elevation",
    var_name="Topographic Elevation",
    var_units="m",
    grid_units=("m", "m"),
    color_for_closed='black',
)

# Plot sediment flux in the second subplot
plt.sca(axes[1])  # Set the current axis to the second subplot
imshow_grid(
    grid,
    "sediment__flux",
    plot_name="Sediment Flux",
    var_name="Sediment Flux",
    var_units=r"m$^3$/yr",
    grid_units=("m", "m"),
    cmap="terrain",
)

plt.tight_layout()
plt.show()

# channel deposition figure etc.
# Profiles with sediment
profiler = ChannelProfiler(grid)
profiler.run_one_step()
fig, ax1 = plt.subplots(figsize=(10,5))
riv_nodes = profiler.nodes
grid.at_node["bedrock_elevation"]=grid.at_node["topographic__elevation"]-grid.at_node["soil__depth"]
 
dist_atNodes=profiler.distance_along_profile[0]
el = grid.at_node["topographic__elevation"][riv_nodes]
sd = grid.at_node["soil__depth"][riv_nodes]
br = el- sd

 # Use ravel() to flatten el and br to 1D
el = el.ravel()
sd = sd.ravel()
br = br.ravel()
 
 # Calculate minimum elevation and relief
min_elevation = min(np.min(el), np.min(br), 0)
el_relief = el - min_elevation
br_relief = br - min_elevation
 
qs = grid.at_node["sediment__flux"][riv_nodes]
qs[np.where(qs<10)] = np.nan
 
plt_fontsize =10
 
#fig, ax1 = plt.subplots(figsize=(5,3.7), dpi=400)
ax2 = ax1.twinx()
dist = (max(dist_atNodes)-dist_atNodes)*1e-3
 
brown_c = [205/255,133/255,63/255]
 
# Fill the entire area from max relief to 0
#ax1.fill_between(dist, np.max(el_relief), 0, color='lightgrey', label='total area')
 
# Fill the bedrtock and colluvial material relief
ax1.fill_between(dist, br_relief, 0, color='grey', label='bedrock')
ax1.fill_between(dist, el_relief, br_relief, color=brown_c, label='colluvial material')        
 
 # Plot bedrock and topographic relief
lns2 = ax1.plot(dist, br_relief, color='k', label='bedrock', linewidth=0.8, zorder=3)
lns1 = ax1.plot(dist, el_relief, color='brown', label='topo', linewidth=1, zorder=4)
 
 # set up labels
ax1.set_xlabel('Distance Along Profile, km', fontweight='normal', fontsize=plt_fontsize)
ax1.tick_params(labelsize=plt_fontsize)
ax1.set_ylabel('Relief, m', fontweight='normal', fontsize=plt_fontsize)
ax1.set_ylim((0, np.max(el_relief)*1.1))
 
 # plot sediment thickness
lns3 = ax2.plot(dist, sd, color='orange', label ='sediment', linewidth=0.8)
ax2.set_ylabel('Sediment thickness, m', fontweight='normal', fontsize=plt_fontsize)
ax2.set_ylim((0, np.nanmax(sd) * 1.4))
ax2.set_xlim((0, dist[2]))
ax2.tick_params(labelsize=plt_fontsize)
 
 # Merge legends
lns = lns1 + lns2 + lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize=plt_fontsize)

# Set y-axis to start from 0
ax1.set_ylim(bottom=0)
plt.show()

# Instantiate figure
fig = plt.figure(figsize=(10, 6))

# Instantiate subplot
sedfluxplot = plt.subplot()

# Create time array based on actual model timesteps
# Since you're using timestep=50 and have 10 points, create proper x-axis
time_values = np.arange(0, run_time, timestep)  # [0, 50, 100, 150, ..., 450]

# Plot data - now x and y will have the same length (10 elements)
sedfluxplot.plot(time_values, sed_flux, color="k", linewidth=3.0)

# Add axis labels
sedfluxplot.set_xlabel("Time [yr]")
sedfluxplot.set_ylabel(r"Sediment flux [m$^3$/yr]")

# Add grid for better readability
sedfluxplot.grid(True, linestyle='--', alpha=0.7)

# Add a title
sedfluxplot.set_title("Sediment Flux Over Time")

# Display the plot
plt.tight_layout()
plt.show()