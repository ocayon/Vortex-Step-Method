import numpy as np
import logging
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver
from VSM.plotting import plot_polars, plot_distribution

# Find the root directory of the repository
root_dir = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
    root_dir = os.path.abspath(os.path.join(root_dir, ".."))
    if root_dir == "/":
        raise FileNotFoundError("Could not find the root directory of the repository.")
save_folder = Path(root_dir) / "results" / "TUDELFT_V3_LEI_KITE"

### rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
csv_file_path = (
    Path(root_dir)
    / "processed_data"
    / "TUDELFT_V3_LEI_KITE"
    / "rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.csv"
)
(
    LE_x_array,
    LE_y_array,
    LE_z_array,
    TE_x_array,
    TE_y_array,
    TE_z_array,
    d_tube_array,
    camber_array,
) = np.loadtxt(csv_file_path, delimiter=",", skiprows=1, unpack=True)
rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs = []
for i in range(len(LE_x_array)):
    LE = np.array([LE_x_array[i], LE_y_array[i], LE_z_array[i]])
    TE = np.array([TE_x_array[i], TE_y_array[i], TE_z_array[i]])
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.append(
        [LE, TE, ["lei_airfoil_breukels", [d_tube_array[i], camber_array[i]]]]
    )

# Defining discretisation
spanwise_panel_distribution = "split_provided"


# ### n_panels = 9
# n_panels = 9
# CAD_wing = Wing(n_panels, spanwise_panel_distribution)
# for i, CAD_rib_i in enumerate(
#     rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
# ):
#     CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
# wing_aero_CAD_19ribs_9panels = WingAerodynamics([CAD_wing])

### n_panels = 18
n_panels = 18
CAD_wing = Wing(n_panels, spanwise_panel_distribution)
for i, CAD_rib_i in enumerate(
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
):
    CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
wing_aero_CAD_19ribs_18panels = WingAerodynamics([CAD_wing])

### n_panels = 36
n_panels = 36
CAD_wing = Wing(n_panels, spanwise_panel_distribution)
for i, CAD_rib_i in enumerate(
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
):
    CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
wing_aero_CAD_19ribs_36panels = WingAerodynamics([CAD_wing])

### n_panels = 45
n_panels = 45
CAD_wing = Wing(n_panels, spanwise_panel_distribution)
for i, CAD_rib_i in enumerate(
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
):
    CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
wing_aero_CAD_19ribs_45panels = WingAerodynamics([CAD_wing])

### n_panels = 54
n_panels = 54
CAD_wing = Wing(n_panels, spanwise_panel_distribution)
for i, CAD_rib_i in enumerate(
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
):
    CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
wing_aero_CAD_19ribs_54panels = WingAerodynamics([CAD_wing])

## n_panels = 72
n_panels = 72
CAD_wing = Wing(n_panels, spanwise_panel_distribution)
for i, CAD_rib_i in enumerate(
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
):
    CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
wing_aero_CAD_19ribs_72panels = WingAerodynamics([CAD_wing])

## n_panels = 117
n_panels = 117
CAD_wing = Wing(n_panels, spanwise_panel_distribution)
for i, CAD_rib_i in enumerate(
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
):
    CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
wing_aero_CAD_19ribs_117panels = WingAerodynamics([CAD_wing])


# Solvers
VSM_stall = Solver(
    aerodynamic_model_type="VSM",
    is_with_artificial_damping=True,
)

### plotting polar
save_path = Path(root_dir) / "results" / "TUD_V3_LEI_KITE"
path_cfd_lebesque = (
    Path(root_dir)
    / "data"
    / "TUDELFT_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
)
path_wind_tunnel_poland = (
    Path(root_dir)
    / "data"
    / "TUDELFT_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_Wind_Tunnel_Poland_2024_Rey_56e4.csv"
)
plot_polars(
    solver_list=[
        # VSM_stall,
        VSM_stall,
        VSM_stall,
        VSM_stall,
        VSM_stall,
        VSM_stall,
        VSM_stall,
    ],
    wing_aero_list=[
        # wing_aero_CAD_19ribs_9panels,
        wing_aero_CAD_19ribs_18panels,
        wing_aero_CAD_19ribs_36panels,
        wing_aero_CAD_19ribs_45panels,
        wing_aero_CAD_19ribs_54panels,
        wing_aero_CAD_19ribs_72panels,
        wing_aero_CAD_19ribs_117panels,
    ],
    label_list=[
        # "  9panels VSM stall CAD 19ribs",
        " 18panels VSM stall CAD 19ribs",
        " 36panels VSM stall CAD 19ribs",
        " 45panels VSM stall CAD 19ribs",
        " 54panels VSM stall CAD 19ribs",
        " 72panels VSM stall CAD 19ribs",
        "117panels VSM stall CAD 19ribs",
        "CFD_Lebesque Rey 30e5",
        "WindTunnel_Poland Rey 5.6e5",
    ],
    literature_path_list=[path_cfd_lebesque, path_wind_tunnel_poland],
    angle_range=np.linspace(-4, 24, 28),
    angle_type="angle_of_attack",
    angle_of_attack=0,
    side_slip=0,
    yaw_rate=0,
    Umag=10,
    title=f"discretisation_study_CAD_19ribs_distribution_{spanwise_panel_distribution}",
    data_type=".pdf",
    save_path=Path(save_folder) / "polars",
    is_save=True,
    is_show=True,
)
