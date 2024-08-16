from VSM.WingAerodynamics import WingAerodynamics
from VSM.WingGeometry import Wing
from VSM.Solver import Solver
import numpy as np
import matplotlib.pyplot as plt
import logging
from VSM.color_palette import set_plot_style
from copy import deepcopy

set_plot_style()


logging.basicConfig(level=logging.INFO)


# Body EastNorthUp (ENU) Reference Frame (aligned with Earth direction)
# x: along the chord / parallel to flow direction
# y: left
# z: upwards

## Create a wing object
# optional arguments are:
#   spanwise_panel_distribution: str = "linear"
#   - "linear"
#   - "cosine"
#   - "cosine_van_Garrel" (http://dx.doi.org/10.13140/RG.2.1.2773.8000)
# spanwise_direction: np.array = np.array([0, 1, 0])
wing = Wing(n_panels=20, spanwise_panel_distribution="split_provided")

## Add sections to the wing
# MUST be done in order from left-to-right
# Sections MUST be defined perpendicular to the quarter-chord line
# arguments are: (leading edge position [x,y,z], trailing edge position [x,y,z], airfoil data)
# airfoil data can be:
# ['inviscid']
# ['lei_airfoil_breukels', [tube_diameter, chamber_height]]
# ['polar_data', [[alpha[rad], cl, cd, cm]]]

## Rectangular wing
span = 20
# wing.add_section([0, -span / 2, 0], [1, -span / 2, 0], ["inviscid"])
wing.add_section([0, span / 2, 0], [1, span / 2, 0], ["inviscid"])
# wing.add_section([0, span / 4, 0], [1, span / 4, 0], ["inviscid"])
# wing.add_section([0, 0, 0], [1, 0, 0], ["inviscid"])
# wing.add_section([0, -span / 4, 0], [1, -span / 4, 0], ["inviscid"])
wing.add_section([0, -span / 2, 0], [1, -span / 2, 0], ["inviscid"])

# Initialize wing aerodynamics
# Default parameters are used (elliptic circulation distribution, 5 filaments per ring)
wing_aero = WingAerodynamics([wing])

# Initialize solver
# Default parameters are used (VSM, no artificial damping)
LLT = Solver(aerodynamic_model_type="LLT")
VSM = Solver(aerodynamic_model_type="VSM")

Umag = 20
aoa = 30
aoa = np.deg2rad(aoa)
Uinf = np.array([np.cos(aoa), 0, np.sin(aoa)]) * Umag

# Define inflow conditions
wing_aero.va = Uinf
wing_aero_LLT = deepcopy(wing_aero)
# Plotting the wing
# wing_aero.plot()
import VSM.plotting as plotting

plotting.plot_geometry(
    wing_aero,
    is_save=True,
    title="rectangular_wing_geometry",
    data_type=".pdf",
    save_path="./",
    is_show=True,
    view_elevation=15,
    view_azimuth=-120,
)

## Solve the aerodynamics
# cl,cd,cs coefficients are flipped to "normal ref frame"
# x (+) downstream, y(+) left and z-up reference frame
results_VSM, wing_aero_VSM = VSM.solve(wing_aero)
results_LLT, wing_aero_LLT = LLT.solve(wing_aero_LLT)

# plotting distributions

plotting.plot_distribution(
    results_list=[results_VSM, results_LLT],
    label_list=["VSM", "LLT"],
    title="spanwise_distributions",
    data_type=".pdf",
    save_path="./",
    is_save=True,
    is_show=True,
)


# # Check if the gamma distribution is symmetric
# def is_symmetric_1d(array, tol=1e-8):
#     return np.allclose(array, array[::-1], atol=tol)


# print(f"VSM is symmetric: {is_symmetric_1d(results_VSM['gamma_distribution'])}")
# print(f"LLT is symmetric: {is_symmetric_1d(results_LLT['gamma_distribution'])}")


# TODOs
# 1. update_wake, called in WingAerodynamicModel, def va setter
