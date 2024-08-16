import numpy as np
import logging
import os
from pathlib import Path
from copy import deepcopy
from VSM.WingAerodynamics import WingAerodynamics
from VSM.WingGeometry import Wing
from VSM.Solver import Solver
import VSM.plotting as plotting


# Find the root directory of the repository
root_dir = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
    root_dir = os.path.abspath(os.path.join(root_dir, ".."))
    if root_dir == "/":
        raise FileNotFoundError("Could not find the root directory of the repository.")

# setting the log level
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

# ### Plotting the wing
save_path = Path(root_dir) / "examples" / "rectangular_wing" / "results"
plotting.plot_geometry(
    wing_aero,
    title="rectangular_wing_geometry",
    data_type=".pdf",
    save_path=save_path,
    is_save=True,
    is_show=True,
)

### Solve the aerodynamics
# cl,cd,cs coefficients are flipped to "normal ref frame"
# x (+) downstream, y(+) left and z-up reference frame
results_VSM, wing_aero_VSM = VSM.solve(wing_aero)
results_LLT, wing_aero_LLT = LLT.solve(wing_aero_LLT)

### plotting distributions
plotting.plot_distribution(
    results_list=[results_VSM, results_LLT],
    label_list=["VSM", "LLT"],
    title="spanwise_distributions",
    data_type=".pdf",
    save_path=save_path,
    is_save=True,
    is_show=True,
)

### plotting polar
path_cfd_lebesque = (
    Path(root_dir)
    / "examples"
    / "LEI_kite_V3"
    / "data"
    / "V3_CL_CD_RANS_CFD_lebesque_2020_Rey_1e6.csv"
)
path_wind_tunnel_poland = (
    Path(root_dir)
    / "examples"
    / "LEI_kite_V3"
    / "data"
    / "V3_CL_CD_WindTunnel_Poland_2024_Rey_56e4.csv"
)
plotting.plot_polars(
    solver_list=[LLT, VSM],
    wing_aero_list=[wing_aero, wing_aero],
    label_list=["LLT", "VSM", "CFD_Lebesque", "WindTunnel_Poland"],
    literature_path_list=[path_cfd_lebesque, path_wind_tunnel_poland],
    angle_range=np.linspace(0, 20, 4),
    angle_type="angle_of_attack",
    angle_of_attack=0,
    side_slip=0,
    yaw_rate=0,
    Umag=10,
    title="rectangular_wing_polars",
    data_type=".pdf",
    save_path=save_path,
    is_save=True,
    is_show=True,
)


# Check if the gamma distribution is symmetric
def is_symmetric_1d(array, tol=1e-8):
    return np.allclose(array, array[::-1], atol=tol)


print(f"VSM is symmetric: {is_symmetric_1d(results_VSM['gamma_distribution'])}")
print(f"LLT is symmetric: {is_symmetric_1d(results_LLT['gamma_distribution'])}")


# TODOs
# 1. update_wake, called in WingAerodynamicModel, def va setter
