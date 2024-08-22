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

# Load from Pickle file
surfplan_path = (
    Path(root_dir)
    / "processed_data"
    / "TUD_V3_LEI_KITE"
    / "surfplan_extracted_input_rib_list.pkl"
)
with open(surfplan_path, "rb") as file:
    surfplan_input_rib_list = pickle.load(file)

# Create wing geometry
n_panels = 20
spanwise_panel_distribution = "unchanged"
surfplan_wing = Wing(n_panels, spanwise_panel_distribution)

# Populate the wing geometry
for i, surfplan_rib_i in enumerate(surfplan_input_rib_list):
    surfplan_wing.add_section(surfplan_rib_i[0], surfplan_rib_i[1], surfplan_rib_i[2])

# Create wing aerodynamics objects
surfplan_wing_aero = WingAerodynamics([surfplan_wing])

# Solvers
VSM = Solver(
    aerodynamic_model_type="VSM",
)
VSM_with_stall_correction = Solver(
    aerodynamic_model_type="VSM",
    is_with_artificial_damping=True,
)

### plotting distributions
save_folder = Path(root_dir) / "results" / "TUD_V3_LEI_KITE"
surfplan_y_coordinates = [
    panels.aerodynamic_center[1] for panels in surfplan_wing_aero.panels
]
angle_of_attack_range = np.linspace(0, 20, 2)

for angle_of_attack in angle_of_attack_range:
    aoa_rad = np.deg2rad(angle_of_attack)

    ## straight case
    side_slip = 0
    yaw_rate = 0
    Umag = 15
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
    results = VSM.solve(surfplan_wing_aero)
    results_with_stall_correction = VSM_with_stall_correction.solve(surfplan_wing_aero)
    plot_distribution(
        y_coordinates_list=[surfplan_y_coordinates, surfplan_y_coordinates],
        results_list=[results, results_with_stall_correction],
        label_list=["VSM", "VSM with stall correction"],
        title=f"surfplan_spanwise_distributions_alpha_{angle_of_attack:.1f}_beta_{side_slip:.1f}_yaw_{yaw_rate:.1f}_Umag_{Umag:.1f}",
        data_type=".pdf",
        save_path=Path(save_folder) / "spanwise_distributions",
        is_save=True,
        is_show=True,
    )

    # ## turn case
    # side_slip = 0
    # yaw_rate = 1.5
    # Umag = 15
    # surfplan_wing_aero.va = (
    #     np.array(
    #         [
    #             np.cos(aoa_rad) * np.cos(side_slip),
    #             np.sin(side_slip),
    #             np.sin(aoa_rad),
    #         ]
    #     )
    #     * Umag,
    #     yaw_rate,
    # )
    # results, _ = VSM.solve(surfplan_wing_aero)
    # results_with_stall_correction, _ = VSM_with_stall_correction.solve(surfplan_wing_aero)
    # plot_distribution(
    #     y_coordinates=surfplan_y_coordinates,
    #     results_list=[results, results_with_stall_correction],
    #     label_list=["VSM", "VSM with stall correction"],
    #     title=f"surfplan_spanwise_distributions_alpha_{angle_of_attack:.1f}_beta_{side_slip:.1f}_yaw_{yaw_rate:.1f}_Umag_{Umag:.1f}",
    #     data_type=".pdf",
    #     save_path=Path(save_folder) / "spanwise_distributions",
    #     is_save=True,
    #     is_show=True,
    # )


### plotting polar
save_path = Path(root_dir) / "results" / "TUD_V3_LEI_KITE"
path_cfd_lebesque = (
    Path(root_dir)
    / "data"
    / "TUD_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_RANS_Lebesque_2024_Rey_100e4.csv"
)
path_wind_tunnel_poland = (
    Path(root_dir)
    / "data"
    / "TUD_V3_LEI_KITE"
    / "literature_results"
    / "V3_CL_CD_Wind_Tunnel_Poland_2024_Rey_56e4.csv"
)
plot_polars(
    solver_list=[VSM, VSM_with_stall_correction, VSM, VSM_with_stall_correction],
    wing_aero_list=[
        surfplan_wing_aero,
        surfplan_wing_aero,
    ],
    label_list=[
        "VSM from surfplan",
        "VSM from surfplan (with correction)",
        "CFD_Lebesque (10e5)",
        "WindTunnel_Poland (5.6e5)",
    ],
    literature_path_list=[path_cfd_lebesque, path_wind_tunnel_poland],
    angle_range=np.linspace(0, 22, 11),
    angle_type="angle_of_attack",
    angle_of_attack=0,
    side_slip=0,
    yaw_rate=0,
    Umag=10,
    title="rectangular_wing_polars",
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
