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

# Load from Pickle file
surfplan_path = (
    Path(root_dir)
    / "processed_data"
    / "TUD_V3_LEI_KITE"
    / "surfplan_extracted_input_rib_list.pkl"
)
with open(surfplan_path, "rb") as file:
    surfplan_input_rib_list = pickle.load(file)

CAD_path = (
    Path(root_dir)
    / "processed_data"
    / "TUD_V3_LEI_KITE"
    / "CAD_extracted_input_rib_list.pkl"
)
with open(CAD_path, "rb") as file:
    CAD_input_rib_list = pickle.load(file)

# Create wing geometry
n_panels = 41
spanwise_panel_distribution = "linear"
surfplan_wing = Wing(n_panels, spanwise_panel_distribution)
CAD_wing = Wing(n_panels, spanwise_panel_distribution)

# Populate the wing geometry
for i, surfplan_rib_i in enumerate(surfplan_input_rib_list):
    surfplan_wing.add_section(surfplan_rib_i[0], surfplan_rib_i[1], surfplan_rib_i[2])

for i, CAD_rib_i in enumerate(CAD_input_rib_list):
    CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])

# Create wing aerodynamics objects
surfplan_wing_aero = WingAerodynamics([surfplan_wing])
CAD_wing_aero = WingAerodynamics([CAD_wing])

# Solvers
VSM = Solver(
    aerodynamic_model_type="VSM",
)

save_folder = Path(root_dir) / "results" / "TUD_V3_LEI_KITE"

# ### plotting distributions
# surfplan_y_coordinates = [
#     panels.aerodynamic_center[1] for panels in surfplan_wing_aero.panels
# ]
# CAD_y_coordinates = [panels.aerodynamic_center[1] for panels in CAD_wing_aero.panels]

# angle_of_attack_range = np.linspace(0, 20, 2)
# side_slip = 0
# yaw_rate = 0
# Umag = 15
# for angle_of_attack in angle_of_attack_range:
#     aoa_rad = np.deg2rad(angle_of_attack)
#     CAD_wing_aero.va = (
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
#     surfplan_wing_aero.va = (
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
#     CAD_results, _ = VSM.solve(CAD_wing_aero)
#     surfplan_results, _ = VSM.solve(surfplan_wing_aero)
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

### plotting shapes
aoa_rad = np.deg2rad(10)
side_slip = 0
yaw_rate = 0
Umag = 15
CAD_wing_aero.va = (
    np.array(
        [
            np.cos(aoa_rad) * np.cos(side_slip),
            np.sin(side_slip),
            np.sin(aoa_rad),
        ]
    )
    * Umag,
    yaw_rate,
)
plot_geometry(
    CAD_wing_aero,
    title="CAD_wing_geometry",
    data_type=".pdf",
    save_path=None,
    is_save=False,
    is_show=True,
    view_elevation=15,
    view_azimuth=-120,
)
surfplan_wing_aero.va = (
    np.array(
        [
            np.cos(aoa_rad) * np.cos(side_slip),
            np.sin(side_slip),
            np.sin(aoa_rad),
        ]
    )
    * Umag,
    yaw_rate,
)
plot_geometry(
    surfplan_wing_aero,
    title="surfplan_wing_geometry",
    data_type=".pdf",
    save_path=None,
    is_save=False,
    is_show=True,
    view_elevation=15,
    view_azimuth=-120,
)

### plotting polar
angle_of_attack_range = np.linspace(-5, 25, 15)
path_cfd_lebesque_3e6 = (
    Path(root_dir)
    / "data"
    / "TUD_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_RANS_CFD_lebesque_2020_Rey_3e6.csv"
)
path_cfd_lebesque_100e4 = (
    Path(root_dir)
    / "data"
    / "TUD_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_RANS_Lebesque_2024_Rey_100e4.csv"
)
path_cfd_lebesque_300e4 = (
    Path(root_dir)
    / "data"
    / "TUD_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
)
path_wind_tunnel_poland_56e4 = (
    Path(root_dir)
    / "data"
    / "TUD_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_Wind_Tunnel_Poland_2024_Rey_56e4.csv"
)
plot_polars(
    solver_list=[VSM, VSM],
    wing_aero_list=[
        surfplan_wing_aero,
        CAD_wing_aero,
    ],
    label_list=[
        f"Surfplan",
        f"CAD",
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
    title="polars_CAD_vs_surfplan",
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
