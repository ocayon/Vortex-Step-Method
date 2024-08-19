import numpy as np
import logging
import pickle
import matplotlib.pyplot as plt
import os
from pathlib import Path
from VSM.WingGeometry import Wing, flip_created_coord_in_pairs_if_needed
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver
from VSM.color_palette import set_plot_style, get_color

# Find the root directory of the repository
root_dir = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
    root_dir = os.path.abspath(os.path.join(root_dir, ".."))
    if root_dir == "/":
        raise FileNotFoundError("Could not find the root directory of the repository.")

# Load from Pickle file
CAD_path = (
    Path(root_dir)
    / "processed_data"
    / "TUD_V3_LEI_KITE"
    / "CAD_extracted_input_rib_list.pkl"
)
with open(CAD_path, "rb") as file:
    input_rib_list = pickle.load(file)

# Create wing geometry
n_panels = 20
spanwise_panel_distribution = "unchanged"
wing = Wing(n_panels, spanwise_panel_distribution)

# Populate the wing geometry
for i, rib_i in enumerate(input_rib_list):
    wing.add_section(rib_i[0], rib_i[1], rib_i[2])

    print(f"after: {rib_i}")

# Create wing aerodynamics objecty
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
# aoas = [14, 16, 18, 20]
aoas = np.arange(10, 21, 1)
cl_straight = np.zeros(len(aoas))
cl_straight_with_correction = np.zeros(len(aoas))
cd_straight = np.zeros(len(aoas))
cd_straight_with_correction = np.zeros(len(aoas))
cs_straight = np.zeros(len(aoas))
cs_straight_with_correction = np.zeros(len(aoas))
gamma_straight = np.zeros((len(aoas), len(wing_aero.panels)))
gamma_straight_with_correction = np.zeros((len(aoas), len(wing_aero.panels)))
cl_distribution_straight = np.zeros((len(aoas), len(wing_aero.panels)))
cl_distribution_straight_with_correction = np.zeros((len(aoas), len(wing_aero.panels)))

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


# %% plotting the results

# grabbing data from literature
CL_CFD_lebesque, CD_CFD_lebesque, aoa_CFD_lebesque = np.loadtxt(
    "./data/V3_CL_CD_RANS_CFD_lebesque_2020_Rey_1e6.csv",
    delimiter=",",
    skiprows=1,
    unpack=True,
)
CL_WindTunnel_Poland, CD_WindTunnel_Poland, aoa_WindTunnel_Poland = np.loadtxt(
    "./data/V3_CL_CD_WindTunnel_Poland_2024_Rey_56e4.csv",
    delimiter=",",
    skiprows=1,
    unpack=True,
)


plt.figure()

# # Plot the CL and CD alpha plots
fig, axs = plt.subplots(2, figsize=(10, 15))
axs[0].plot(aoas, cl_straight, label="$C_L$ straight")
axs[0].plot(aoas, cl_straight_with_correction, label="$C_L$ straight with correction")
axs[0].plot(aoa_CFD_lebesque, CL_CFD_lebesque, label="CFD Lebesque", linestyle="dashed")
axs[0].plot(
    aoa_WindTunnel_Poland,
    CL_WindTunnel_Poland,
    label="Wind Tunnel Poland",
    linestyle="dashed",
)
axs[0].legend()
axs[0].set_ylabel("CL")
axs[0].set_xlabel("AOA [deg]")

axs[1].plot(aoas, cd_straight, label="$C_D$ straight")
axs[1].plot(aoas, cd_straight_with_correction, label="$C_D$ straight with correction")
axs[1].plot(aoa_CFD_lebesque, CD_CFD_lebesque, label="CFD Lebesque", linestyle="dashed")
axs[1].plot(
    aoa_WindTunnel_Poland,
    CD_WindTunnel_Poland,
    label="Wind Tunnel Poland",
    linestyle="dashed",
)
axs[1].legend()
axs[1].set_ylabel("CD")
axs[1].set_xlabel("AOA [deg]")

fig.savefig("./results/CAD_straight_cl_cd_polars.pdf")

# plot gamma distributions and cl distributions for each the aoas
fig, axs = plt.subplots(len(aoas), 2, figsize=(15, 15))
for i, (ax1, ax2) in enumerate(axs):
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

fig.savefig("./results/CAD_straight_gamma_and_cl_distribution.pdf")


# %% Doing it for the turning cases


# gamma_turn = np.zeros((len(aoas), len(wing_aero.panels)))
# gamma_turn_with_correction = np.zeros((len(aoas), len(wing_aero.panels)))
# cl_distribution_straight = np.zeros((len(aoas), len(wing_aero.panels)))
# cl_distribution_straight_with_correction = np.zeros((len(aoas), len(wing_aero.panels)))
# cl_distribution_turn = np.zeros((len(aoas), len(wing_aero.panels)))
# cl_distribution_turn_with_correction = np.zeros((len(aoas), len(wing_aero.panels)))
# cl_turn = np.zeros(len(aoas))
# cl_turn_with_correction = np.zeros(len(aoas))
# cd_turn = np.zeros(len(aoas))
# cd_turn_with_correction = np.zeros(len(aoas))
# cs_turn = np.zeros(len(aoas))
# cs_turn_with_correction = np.zeros(len(aoas))

# yaw_rate = 1.5
# for i, aoa in enumerate(aoas):
#     ## Turn
#     print(f" <--> Calculating for Turn")
#     wing_aero.va = (
#         np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)])
#         * Umag,
#         yaw_rate,
#     )
#     results, _ = VSM.solve(wing_aero)
#     cl_turn[i] = results["cl"]
#     cd_turn[i] = results["cd"]
#     cs_turn[i] = results["cs"]
#     cl_distribution_turn[i] = results["cl_distribution"]
#     gamma_turn[i] = results["gamma_distribution"]

#     # with stall correction
#     print(f"    Running with stall correction")
#     results_with_stall_correction, _ = VSM_with_stall_correction.solve(wing_aero)
#     cl_turn_with_correction[i] = results_with_stall_correction["cl"]
#     cd_turn_with_correction[i] = results_with_stall_correction["cd"]
#     cs_turn_with_correction[i] = results_with_stall_correction["cs"]
#     gamma_turn_with_correction[i] = results_with_stall_correction["gamma_distribution"]
#     cl_distribution_turn_with_correction[i] = results_with_stall_correction[
#         "cl_distribution"
#     ]

# # # Plot the CL and CD alpha plots
# fig, axs = plt.subplots(3, figsize=(10, 15))
# axs[0].plot(aoas, cl_turn, label="$C_L$ turn")
# axs[0].plot(aoas, cl_turn_with_correction, label="$C_L$ turn with correction")
# axs[0].legend()
# axs[0].set_ylabel("CL")
# axs[0].set_xlabel("AOA [deg]")

# axs[1].plot(aoas, cd_turn, label="$C_D$ turn")
# axs[1].plot(aoas, cd_turn_with_correction, label="$C_D$ turn with correction")
# axs[1].legend()
# axs[1].set_ylabel("CD")
# axs[1].set_xlabel("AOA [deg]")
# axs[2].plot(aoas, cs_turn, label="$C_S$ turn")
# axs[2].plot(aoas, cs_turn_with_correction, label="$C_S$ turn with correction")
# axs[2].legend()
# axs[2].set_ylabel("CS")
# axs[2].set_xlabel("AOA [deg]")
# plt.show()

# # plot gamma distributions and cl distributions for each the aoas
# fig, axs = plt.subplots(len(aoas), 2, figsize=(15, 15))
# for i, (ax3, ax4) in enumerate(axs):
#     ax3.plot(gamma_turn[i], label=f"No stall correction")
#     ax3.plot(gamma_turn_with_correction[i], label=f"With stall correction")
#     ax3.legend()
#     ax3.set_title(r"Turn $\alpha$ = {}".format(aoas[i]) + f" - Yaw rate: {yaw_rate}")
#     ax3.set_ylabel("Gamma")
#     ax3.set_xlabel("spanwise position")

#     ax4.plot(cl_distribution_turn[i], label=f"No stall correction")
#     ax4.plot(cl_distribution_turn_with_correction[i], label=f"With stall correction")
#     ax4.legend()
#     ax4.set_title(r"Turn $\alpha$ = {}".format(aoas[i]) + f" - Yaw rate: {yaw_rate}")
#     ax4.set_ylabel("CL")
#     ax4.set_xlabel("spanwise position")

# plt.show()

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
