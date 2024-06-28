from VSM.WingAerodynamicModel import WingAerodynamics
from VSM.WingGeometry import Wing, Section
from VSM.Solver import Solver
import numpy as np

# Use example
################# CAREFULL WITH REFERENCE FRAMES, CHANGING FROM ORIGINAL CODE #################
# Aircraft reference frame
# x: forward
# y: right
# z: down
# Create a wing object
wing = Wing(n_panels=10)

# Add sections to the wing
# arguments are: (leading edge position [x,y,z], trailing edge position [x,y,z], airfoil data)
# airfoil data can be:
# ['inviscid']
# ['lei_airfoil_breukels', [tube_diameter, chamber_height]]
# ['polars', []]
span = 20
wing.add_section([-1, -span/2, 0], [0, -span/2, 0], ["inviscid"])
wing.add_section([-1, span/2, 0], [0, span/2, 0], ["inviscid"])





# Initialize wing aerodynamics
# Default parameters are used (elliptic circulation distribution, 5 filaments per ring)
wing_aero = WingAerodynamics([wing])


# Initialize solver
# Default parameters are used (VSM, no artificial damping)
VSM = Solver()

Umag = 20
aoa = 3
aoa = aoa*np.pi/180
Uinf = np.array([np.cos(aoa),0,np.sin(aoa)])*Umag
# Define inflow conditions
wing_aero.va = Uinf

# Plotting the wing
wing_aero.plot()

# solve the aerodynamics
results, wing_aero = VSM.solve(wing_aero)

# Print
print(results)


import matplotlib.pyplot as plt
# Plot Gamma distribution
plt.figure()
plt.plot(wing_aero.gamma_distribution)
plt.show()

# TODOs
# 1. update_wake, called in WingAerodynamicModel, def va setter
