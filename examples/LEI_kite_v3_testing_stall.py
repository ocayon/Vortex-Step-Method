import numpy as np
from VSM.WingGeometry import Wing, flip_created_coord_in_pairs_if_needed
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver
from VSM.color_palette import set_plot_style, get_color
import logging
import matplotlib.pyplot as plt

set_plot_style()


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
coord_struct = np.loadtxt("../data/coordinates/coords_v3_kite.csv", delimiter=",")

## Convert the coordinates to the aero coordinates
coord_aero = struct2aero_geometry(coord_struct) / 1000
n_aero = len(coord_aero) // 2
coord = refine_LEI_mesh(coord_aero, n_aero - 1, 5)
coord = flip_created_coord_in_pairs_if_needed(coord)

n_sections = len(coord) // 2
n_panels = n_sections - 1
# thickness of the leading edge tube
LE_thicc = np.ones(n_sections) * 0.1

# camber of the leading edge airfoil
camber = np.ones(n_sections) * 0.095

# Create wing geometry
wing = Wing(n_panels, "unchanged")
for idx, idx2 in enumerate(range(0, len(coord), 2)):
    logging.debug(f"coord[{idx2}] = {coord[idx2]}")
    wing.add_section(
        coord[idx2],
        coord[idx2 + 1],
        ["lei_airfoil_breukels", [LE_thicc[idx], camber[idx]]],
    )
wing_aero = WingAerodynamics([wing])

# VSM
VSM = Solver(
    aerodynamic_model_type="VSM",
    artificial_damping={"k2": 0, "k4": 0},
)
VSM_with_stall_correction = Solver(
    aerodynamic_model_type="VSM",
)

# aoas = np.arange(0, 21, 1)
# aoas = [10, 15, 16, 20, 25]
aoas = [10, 15, 20]
cl_straight = np.zeros(len(aoas))
cl_straight_with_correction = np.zeros(len(aoas))
cd_straight = np.zeros(len(aoas))
cd_straight_with_correction = np.zeros(len(aoas))
cs_straight = np.zeros(len(aoas))
cs_straight_with_correction = np.zeros(len(aoas))
gamma_straight = np.zeros((len(aoas), len(wing_aero.panels)))
gamma_straight_with_correction = np.zeros((len(aoas), len(wing_aero.panels)))
gamma_turn = np.zeros((len(aoas), len(wing_aero.panels)))
gamma_turn_with_correction = np.zeros((len(aoas), len(wing_aero.panels)))
cl_distribution_straight = np.zeros((len(aoas), len(wing_aero.panels)))
cl_distribution_straight_with_correction = np.zeros((len(aoas), len(wing_aero.panels)))
cl_distribution_turn = np.zeros((len(aoas), len(wing_aero.panels)))
cl_distribution_turn_with_correction = np.zeros((len(aoas), len(wing_aero.panels)))
cl_turn = np.zeros(len(aoas))
cl_turn_with_correction = np.zeros(len(aoas))
cd_turn = np.zeros(len(aoas))
cd_turn_with_correction = np.zeros(len(aoas))
cs_turn = np.zeros(len(aoas))
cs_turn_with_correction = np.zeros(len(aoas))

yaw_rate = 1.5
for i, aoa in enumerate(aoas):
    print(f" ")
    print(f"Calculating for AOA: {aoa}")
    aoa = np.deg2rad(aoa)
    sideslip = 0
    Umag = 25

    wing_aero.va = (
        np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)])
        * Umag,
        0,
    )
    results, _ = VSM.solve(wing_aero)
    cl_straight[i] = results["cl"]
    cd_straight[i] = results["cd"]
    cs_straight[i] = results["cs"]
    gamma_straight[i] = results["gamma_distribution"]
    cl_distribution_straight[i] = results["cl_distribution"]

    # with stall correction
    print(f"    Running with stall correction")
    results_with_stall_correction, _ = VSM_with_stall_correction.solve(wing_aero)
    cl_straight_with_correction[i] = results_with_stall_correction["cl"]
    cd_straight_with_correction[i] = results_with_stall_correction["cd"]
    cs_straight_with_correction[i] = results_with_stall_correction["cs"]
    gamma_straight_with_correction[i] = results_with_stall_correction[
        "gamma_distribution"
    ]
    cl_distribution_straight_with_correction[i] = results_with_stall_correction[
        "cl_distribution"
    ]

    ## Turn
    print(f" <--> Calculating for Turn")
    wing_aero.va = (
        np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)])
        * Umag,
        yaw_rate,
    )
    results, _ = VSM.solve(wing_aero)
    cl_turn[i] = results["cl"]
    cd_turn[i] = results["cd"]
    cs_turn[i] = results["cs"]
    cl_distribution_turn[i] = results["cl_distribution"]
    gamma_turn[i] = results["gamma_distribution"]

    # with stall correction
    print(f"    Running with stall correction")
    results_with_stall_correction, _ = VSM_with_stall_correction.solve(wing_aero)
    cl_turn_with_correction[i] = results_with_stall_correction["cl"]
    cd_turn_with_correction[i] = results_with_stall_correction["cd"]
    cs_turn_with_correction[i] = results_with_stall_correction["cs"]
    gamma_turn_with_correction[i] = results_with_stall_correction["gamma_distribution"]
    cl_distribution_turn_with_correction[i] = results_with_stall_correction[
        "cl_distribution"
    ]


# %% plotting the results

# plt.figure()
# plt.plot(aoas)

# # Plot the CL and CD alpha plots
fig, axs = plt.subplots(3, figsize=(10, 15))
axs[0].plot(aoas, cl_straight, label="$C_L$ straight")
axs[0].plot(aoas, cl_straight_with_correction, label="$C_L$ straight with correction")
axs[0].plot(aoas, cl_turn, label="$C_L$ turn")
axs[0].plot(aoas, cl_turn_with_correction, label="$C_L$ turn with correction")
axs[0].legend()
axs[0].set_ylabel("CL")
axs[0].set_xlabel("AOA [deg]")

axs[1].plot(aoas, cd_straight, label="$C_D$ straight")
axs[1].plot(aoas, cd_straight_with_correction, label="$C_D$ straight with correction")
axs[1].plot(aoas, cd_turn, label="$C_D$ turn")
axs[1].plot(aoas, cd_turn_with_correction, label="$C_D$ turn with correction")
axs[1].legend()
axs[1].set_ylabel("CD")
axs[1].set_xlabel("AOA [deg]")

axs[2].plot(aoas, cs_straight, label="$C_S$ straight")
axs[2].plot(aoas, cs_straight_with_correction, label="$C_S$ straight with correction")
axs[2].plot(aoas, cs_turn, label="$C_S$ turn")
axs[2].plot(aoas, cs_turn_with_correction, label="$C_S$ turn with correction")
axs[2].legend()
axs[2].set_ylabel("CS")
axs[2].set_xlabel("AOA [deg]")
plt.show()

# plot gamma distributions and cl distributions for each the aoas
fig, axs = plt.subplots(len(aoas), 4, figsize=(15, 15))
for i, (ax1, ax2, ax3, ax4) in enumerate(axs):
    ax1.plot(gamma_straight[i], label=f"No stall correction")
    ax1.plot(gamma_straight_with_correction[i], label=f"With stall correction")
    ax1.legend()
    ax1.set_title(r"Straight $\alpha$ = {}".format(aoas[i]))
    ax1.set_ylabel("Gamma")
    ax1.set_xlabel("spanwise position")

    ax2.plot(cl_distribution_straight[i], label=f"No stall correction")
    ax2.plot(
        cl_distribution_straight_with_correction[i], label=f"With stall correction"
    )
    ax2.legend()
    ax2.set_title(r"Straight $\alpha$ = {}".format(aoas[i]))
    ax2.set_ylabel("CL")
    ax2.set_xlabel("spanwise position")

    ax3.plot(gamma_turn[i], label=f"No stall correction")
    ax3.plot(gamma_turn_with_correction[i], label=f"With stall correction")
    ax3.legend()
    ax3.set_title(r"Turn $\alpha$ = {}".format(aoas[i]) + f" - Yaw rate: {yaw_rate}")
    ax3.set_ylabel("Gamma")
    ax3.set_xlabel("spanwise position")

    ax4.plot(cl_distribution_turn[i], label=f"No stall correction")
    ax4.plot(cl_distribution_turn_with_correction[i], label=f"With stall correction")
    ax4.legend()
    ax4.set_title(r"Turn $\alpha$ = {}".format(aoas[i]) + f" - Yaw rate: {yaw_rate}")
    ax4.set_ylabel("CL")
    ax4.set_xlabel("spanwise position")

plt.show()


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
