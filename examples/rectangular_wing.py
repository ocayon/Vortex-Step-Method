from VSM.WingAerodynamics import WingAerodynamics
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

Umag = 20
aoa = 3
aoa = np.deg2rad(aoa)
Uinf = np.array([np.cos(aoa), 0, -np.sin(aoa)]) * -Umag

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


###############
# Plotting
###############
import matplotlib.pyplot as plt


def plot_a_distribution(title, VSM_distribution, LLT_distribution):
    plt.figure()
    plt.title(title)
    plt.plot(VSM_distribution, label="VSM")
    plt.plot(LLT_distribution, label="LLT")
    plt.legend()


plot_a_distribution(
    "gamma_distribution",
    results_VSM["gamma_distribution"],
    results_LLT["gamma_distribution"],
)
plot_a_distribution("cl_distrbution", results_VSM["cl"], results_LLT["cl"])
plot_a_distribution(
    "alpha_at_ac", results_VSM["alpha_at_ac"], results_LLT["alpha_at_ac"]
)
plot_a_distribution(
    "alpha_uncorrected",
    results_VSM["alpha_uncorrected"],
    results_LLT["alpha_uncorrected"],
)
plot_a_distribution(
    "cl_over_cd",
    results_VSM["cl"] / results_VSM["cd"],
    results_LLT["cl"] / results_LLT["cd"],
)
plt.show()

# Check if the gamma distribution is symmetric


def is_symmetric_1d(array, tol=1e-8):
    return np.allclose(array, array[::-1], atol=tol)


print(f"VSM is symmetric: {is_symmetric_1d(wing_aero_VSM.gamma_distribution)}")
print(f"LLT is symmetric: {is_symmetric_1d(wing_aero_LLT.gamma_distribution)}")


# TODOs
# 1. update_wake, called in WingAerodynamicModel, def va setter
