from VSM.WingAerodynamicModel import WingAerodynamics
from VSM.WingGeometry import Wing, Section
from VSM.Solver import Solver
import numpy as np
from copy import deepcopy

# Use example
################# CAREFULL WITH REFERENCE FRAMES, CHANGING FROM ORIGINAL CODE #################
# Aircraft reference frame
# x: forward
# y: right
# z: down
# Create a wing object
wing = Wing(n_panels=50)

# Add sections to the wing
# arguments are: (leading edge position [x,y,z], trailing edge position [x,y,z], airfoil data)
# airfoil data can be:
# ['inviscid']
# ['lei_airfoil_breukels', [tube_diameter, chamber_height]]
# ['polars', []]
span = 20
wing.add_section([0, -span / 2, 0], [-1, -span / 2, 0], ["inviscid"])
wing.add_section([0, span / 2, 0], [-1, span / 2, 0], ["inviscid"])

# Initialize wing aerodynamics
# Default parameters are used (elliptic circulation distribution, 5 filaments per ring)
wing_aero = WingAerodynamics([wing])


# Initialize solver
# Default parameters are used (VSM, no artificial damping)
LLT = Solver(aerodynamic_model_type="LLT")
VSM = Solver(aerodynamic_model_type="VSM")

Umag = -20
aoa = 3
aoa = aoa * np.pi / 180
Uinf = np.array([np.cos(aoa), 0, np.sin(aoa)]) * Umag
# Define inflow conditions
wing_aero.va = Uinf
wing_aero_LLT = deepcopy(wing_aero)
# Plotting the wing
wing_aero.plot()

# solve the aerodynamics
results_VSM, wing_aero_VSM = VSM.solve(wing_aero)
results_LLT, wing_aero_LLT = LLT.solve(wing_aero_LLT)


# Print
print(results_VSM)


import matplotlib.pyplot as plt

# Plot Gamma distribution
plt.figure()
plt.plot(wing_aero_VSM.gamma_distribution, label="VSM")
plt.plot(wing_aero_LLT.gamma_distribution, label="LLT")
plt.legend()
plt.show()


def is_symmetric_1d(array, tol=1e-8):
    return np.allclose(array, array[::-1], atol=tol)


print(f"VSM is symmetric: {is_symmetric_1d(wing_aero_VSM.gamma_distribution)}")
print(f"LLT is symmetric: {is_symmetric_1d(wing_aero_LLT.gamma_distribution)}")


# TODOs
# 1. update_wake, called in WingAerodynamicModel, def va setter
