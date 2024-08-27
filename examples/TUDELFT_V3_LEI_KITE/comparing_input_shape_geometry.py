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
save_results_folder = Path(root_dir) / "results" / "TUDELFT_V3_LEI_KITE"

# Defining settings
n_panels = 18
spanwise_panel_distribution = "unchanged"

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

### Printing differences
print(
    "rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_10ribs",
    len(rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_10ribs),
)
print(
    "rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_10ribs",
    len(rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_10ribs),
)
print("rib_list_from_Surfplan_19ribs", len(rib_list_from_Surfplan_19ribs))


for cad, surfplan in zip(
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_10ribs,
    rib_list_from_Surfplan_19ribs,
):
    print("LE_delta", cad[0] - surfplan[0])
    print("TE_delta", cad[1] - surfplan[1])
    print(
        "d_tube",
        cad[2][1][0] - surfplan[2][1][0],
        "d_tube surfplan",
        surfplan[2][1][0],
        "d_tube cad",
        cad[2][1][0],
    )
    print(
        "camber",
        cad[2][1][1] - surfplan[2][1][1],
        "camber surfplan",
        surfplan[2][1][1],
        "camber cad",
        cad[2][1][1],
    )
    print("\n")


# Plotting the differences

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


### plotting shapes
plot_geometry(
    wing_aero_CAD_10ribs,
    title="wing_aero_CAD_10ribs",
    data_type=".pdf",
    save_path=Path(save_results_folder) / "geometry",
    is_save=True,
    is_show=True,
    view_elevation=15,
    view_azimuth=-120,
)
plot_geometry(
    wing_aero_CAD_19ribs,
    title="wing_aero_CAD_19ribs",
    data_type=".pdf",
    save_path=Path(save_results_folder) / "geometry",
    is_save=True,
    is_show=True,
    view_elevation=15,
    view_azimuth=-120,
)
plot_geometry(
    wing_aero_surfplan_19ribs,
    title="wing_aero_surfplan_19ribs",
    data_type=".pdf",
    save_path=Path(save_results_folder) / "geometry",
    is_save=True,
    is_show=True,
    view_elevation=15,
    view_azimuth=-120,
)
