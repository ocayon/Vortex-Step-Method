import numpy as np
import logging
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver
from VSM.plotting import plot_polars, plot_distribution, plot_geometry

# Find the root directory of the repository
root_dir = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
    root_dir = os.path.abspath(os.path.join(root_dir, ".."))
    if root_dir == "/":
        raise FileNotFoundError("Could not find the root directory of the repository.")

# Defining settings
n_panels = 36
spanwise_panel_distribution = "split_provided"

### rib_list_from_Surfplan_19ribs
csv_file_path = (
    Path(root_dir)
    / "processed_data"
    / "TUDELFT_V3_LEI_KITE"
    / "rib_list_from_Surfplan_19ribs.csv"
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
rib_list_from_Surfplan_19ribs = []
for i in range(len(LE_x_array)):
    LE = np.array([LE_x_array[i], LE_y_array[i], LE_z_array[i]])
    TE = np.array([TE_x_array[i], TE_y_array[i], TE_z_array[i]])
    rib_list_from_Surfplan_19ribs.append(
        [LE, TE, ["lei_airfoil_breukels", [d_tube_array[i], camber_array[i]]]]
    )
surfplan_wing = Wing(n_panels, spanwise_panel_distribution)
for i, surfplan_rib_i in enumerate(rib_list_from_Surfplan_19ribs):
    surfplan_wing.add_section(surfplan_rib_i[0], surfplan_rib_i[1], surfplan_rib_i[2])
wing_aero_surfplan_19ribs = WingAerodynamics([surfplan_wing])

### rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_10ribs
csv_file_path = (
    Path(root_dir)
    / "processed_data"
    / "TUDELFT_V3_LEI_KITE"
    / "rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_10ribs.csv"
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
rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_10ribs = []
for i in range(len(LE_x_array)):
    LE = np.array([LE_x_array[i], LE_y_array[i], LE_z_array[i]])
    TE = np.array([TE_x_array[i], TE_y_array[i], TE_z_array[i]])
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_10ribs.append(
        [LE, TE, ["lei_airfoil_breukels", [d_tube_array[i], camber_array[i]]]]
    )
CAD_wing = Wing(n_panels, spanwise_panel_distribution)

for i, CAD_rib_i in enumerate(
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_10ribs
):
    CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
wing_aero_CAD_10ribs = WingAerodynamics([CAD_wing])

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
CAD_wing = Wing(n_panels, spanwise_panel_distribution)

for i, CAD_rib_i in enumerate(
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
):
    CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
wing_aero_CAD_19ribs = WingAerodynamics([CAD_wing])

### Solvers
VSM = Solver(
    aerodynamic_model_type="VSM",
)
VSM_stall = Solver(
    aerodynamic_model_type="VSM",
    is_with_artificial_damping=True,
)

save_folder = Path(root_dir) / "results" / "TUDELFT_V3_LEI_KITE"

# ### plotting distributions
# surfplan_y_coordinates = [
#     panels.aerodynamic_center[1] for panels in wing_aero_surfplan_19ribs.panels
# ]
# CAD_y_coordinates = [panels.aerodynamic_center[1] for panels in wing_aero_CAD_10ribs.panels]

# angle_of_attack_range = np.linspace(0, 20, 2)
# side_slip = 0
# yaw_rate = 0
# Umag = 15
# for angle_of_attack in angle_of_attack_range:
#     aoa_rad = np.deg2rad(angle_of_attack)
#     wing_aero_CAD_10ribs.va = (
#         np.array(
#             [
#                 np.cos(aoa_rad) * np.cos(side_slip),
#                 np.sin(side_slip),
#                 np.sin(aoa_rad),
#             ]
#         )
#         * Umag,
#         yaw_rate,
#     )
#     wing_aero_surfplan_19ribs.va = (
#         np.array(
#             [
#                 np.cos(aoa_rad) * np.cos(side_slip),
#                 np.sin(side_slip),
#                 np.sin(aoa_rad),
#             ]
#         )
#         * Umag,
#         yaw_rate,
#     )
#     CAD_results, _ = VSM.solve(wing_aero_CAD_10ribs)
#     surfplan_results, _ = VSM.solve(wing_aero_surfplan_19ribs)
#     plot_distribution(
#         y_coordinates_list=[CAD_y_coordinates, surfplan_y_coordinates],
#         results_list=[CAD_results, surfplan_results],
#         label_list=["CAD", "surfplan"],
#         title=f"CAD_spanwise_distributions_alpha_{angle_of_attack:.2f}_beta_{side_slip}_yaw_{yaw_rate}_Umag_{Umag}",
#         data_type=".pdf",
#         save_path=Path(save_folder) / "spanwise_distributions",
#         is_save=True,
#         is_show=True,
#     )

### Setting va for each wing
aoa_rad = np.deg2rad(10)
side_slip = 0
yaw_rate = 0
Umag = 15
va = (
    np.array(
        [
            np.cos(aoa_rad) * np.cos(side_slip),
            np.sin(side_slip),
            np.sin(aoa_rad),
        ]
    )
    * Umag
)
wing_aero_CAD_10ribs.va = va, yaw_rate
wing_aero_CAD_19ribs.va = va, yaw_rate
wing_aero_surfplan_19ribs.va = va, yaw_rate

### plotting polar
angle_of_attack_range = np.linspace(-5, 25, 15)
path_cfd_lebesque_3e6 = (
    Path(root_dir)
    / "data"
    / "TUDELFT_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_RANS_CFD_lebesque_2020_Rey_3e6.csv"
)
path_cfd_lebesque_100e4 = (
    Path(root_dir)
    / "data"
    / "TUDELFT_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_RANS_Lebesque_2024_Rey_100e4.csv"
)
path_cfd_lebesque_300e4 = (
    Path(root_dir)
    / "data"
    / "TUDELFT_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
)
path_wind_tunnel_poland_56e4 = (
    Path(root_dir)
    / "data"
    / "TUDELFT_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_Wind_Tunnel_Poland_2024_Rey_56e4.csv"
)
plot_polars(
    solver_list=[VSM, VSM, VSM, VSM_stall, VSM_stall, VSM_stall],
    wing_aero_list=[
        wing_aero_surfplan_19ribs,
        wing_aero_CAD_10ribs,
        wing_aero_CAD_19ribs,
        wing_aero_surfplan_19ribs,
        wing_aero_CAD_10ribs,
        wing_aero_CAD_19ribs,
    ],
    label_list=[
        f"Surfplan 19ribs",
        f"CAD 10ribs",
        f"CAD 19ribs",
        f"Surfplan 19ribs with stall correction",
        f"CAD 10ribs with stall correction",
        f"CAD 19ribs with stall correction",
        f"RANS CFD Lebesque Rey = 10e5 (2020)",
        f"RANS CFD Lebesque Rey = 30e5 (2020)",
        f"Wind Tunnel Poland Rey = 5.6e5 (2024)",
    ],
    literature_path_list=[
        path_cfd_lebesque_100e4,
        path_cfd_lebesque_300e4,
        path_wind_tunnel_poland_56e4,
    ],
    angle_range=angle_of_attack_range,
    angle_type="angle_of_attack",
    angle_of_attack=0,
    side_slip=side_slip,
    yaw_rate=yaw_rate,
    Umag=Umag,
    title=f"polars_comparing_CAD_vs_surfplan_n_panels_{int(n_panels):.0f}_distribution_{spanwise_panel_distribution}",
    data_type=".pdf",
    save_path=Path(save_folder) / "polars",
    is_save=True,
    is_show=True,
)

# import pandas as pd
# # Save straight polars
# polar_df = pd.DataFrame(
#     {
#         "angle_of_attack": np.deg2rad(aoas),
#         "lift_coefficient": cl_straight,
#         "drag_coefficient": cd_straight,
#     }
# )

# polar_df.to_csv("./data/polars/straight_powered_polars.csv", index=False)

# # Save turn polars
# polar_df = pd.DataFrame(
#     {
#         "angle_of_attack": np.deg2rad(aoas),
#         "lift_coefficient": cl_turn,
#         "drag_coefficient": cd_turn,
#     }
# )

# polar_df.to_csv("./data/polars/turn_powered_polars.csv", index=False)

# %%
