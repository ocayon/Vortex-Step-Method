import numpy as np

import logging
import pickle
import os
from pathlib import Path
from VSM.WingGeometry import Wing, flip_created_coord_in_pairs_if_needed

# Find the root directory of the repository
root_dir = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
    root_dir = os.path.abspath(os.path.join(root_dir, ".."))
    if root_dir == "/":
        raise FileNotFoundError("Could not find the root directory of the repository.")


# TODO: Convert into a Kite class
def Ry(theta):
    return np.matrix(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def struct2aero_geometry(coord_struc):

    coord = np.empty((20, 3))

    coord[0, :] = coord_struc[20, :]
    coord[1, :] = coord_struc[10, :]

    coord[2, :] = coord_struc[9, :]
    coord[3, :] = coord_struc[11, :]

    coord[4, :] = coord_struc[8, :]
    coord[5, :] = coord_struc[12, :]

    coord[6, :] = coord_struc[7, :]
    coord[7, :] = coord_struc[13, :]

    coord[8, :] = coord_struc[6, :]
    coord[9, :] = coord_struc[14, :]

    coord[10, :] = coord_struc[5, :]
    coord[11, :] = coord_struc[15, :]

    coord[12, :] = coord_struc[4, :]
    coord[13, :] = coord_struc[16, :]

    coord[14, :] = coord_struc[3, :]
    coord[15, :] = coord_struc[17, :]

    coord[16, :] = coord_struc[2, :]
    coord[17, :] = coord_struc[18, :]

    coord[18, :] = coord_struc[19, :]
    coord[19, :] = coord_struc[1, :]

    return coord


def refine_LEI_mesh(coord, N_sect, N_split):
    refined_coord = []

    for i_sec in range(N_sect):
        temp_coord = np.empty((int(N_split * 2), 3))
        for i_spl in range(N_split):
            temp_coord[2 * i_spl] = (
                coord[2 * i_sec, :] * (N_split - i_spl) / N_split
                + coord[2 * (i_sec + 1), :] * (i_spl) / N_split
            )
            temp_coord[2 * i_spl + 1] = (
                coord[2 * i_sec + 1, :] * (N_split - i_spl) / N_split
                + coord[2 * (i_sec + 1) + 1, :] * (i_spl) / N_split
            )
        if i_sec == 0:
            refined_coord = temp_coord
        else:
            refined_coord = np.append(refined_coord, temp_coord, axis=0)

    refined_coord = np.append(
        refined_coord, [coord[2 * N_sect, :], coord[2 * N_sect + 1, :]], axis=0
    )

    return refined_coord


# %% Read the coordinates from the CAD file
CAD_path = (
    Path(root_dir)
    / "data"
    / "TUD_V3_LEI_KITE"
    / "geometry"
    / "CAD_extracted_coords_v3_kite.csv"
)
coord_struct = np.loadtxt(CAD_path, delimiter=",")

## Convert the coordinates to the aero coordinates
coord_aero = struct2aero_geometry(coord_struct) / 1000
n_aero = len(coord_aero) // 2
N_splits = 2
coord = refine_LEI_mesh(coord_aero, n_aero - 1, N_splits)
coord = flip_created_coord_in_pairs_if_needed(coord)

n_sections = len(coord) // 2
n_panels = n_sections - 1
# thickness of the leading edge tube
# LE thickness at each section [m]
t = [
    0.118753,
    0.151561,
    0.178254,
    0.19406,
    0.202418,
    0.202418,
    0.19406,
    0.178254,
    0.151561,
    0.118753,
]
# Create an array of indices for the original thickness distribution
original_indices = np.arange(len(t))

# Create an array of indices for the interpolated thickness distribution
interpolated_indices = np.linspace(0, len(t) - 1, n_sections)

# Interpolate the thickness distribution
LE_thickness_dimensional = np.interp(interpolated_indices, original_indices, t)

# Create wing geometry
rib_input_list = []
for idx, idx2 in enumerate(range(0, len(coord), 2)):

    logging.debug(f"idx: {idx} | coord[{idx2}] = {coord[idx2]}")
    coord_length = np.linalg.norm(coord[idx2] - coord[idx2 + 1])
    LE_thickness = LE_thickness_dimensional[idx] / coord_length
    camber = 0.095
    LE = coord[idx2]
    TE = coord[idx2 + 1]
    airfoil_input = ["lei_airfoil_breukels", [LE_thickness, camber]]
    rib_input_list.append([LE, TE, airfoil_input])

save_path = Path(root_dir) / "processed_data" / "CAD_extracted_input_rib_list.pkl"
with open(save_path, "wb") as file:
    pickle.dump(rib_input_list, file)
