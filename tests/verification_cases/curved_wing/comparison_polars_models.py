# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:09:16 2022

@author: oriol2
"""


import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.insert(0, "../functions/")
import functions_VSM_LLT as VSM

# %%  Input DATA
N = 60
#   Rectangular wing coordinates
R = 4.673
theta = 45 * np.pi / 180
span = 6.969
chord = 2.18
AR = span / chord
dist = "lin"
coord = VSM.generate_coordinates_curved_wing(chord, span, theta, R, N, dist)
Atot = 16
# Atot = 16
# Wind speed mag and direction
Umag = 22
aoa = 5
aoa = aoa * np.pi / 180
Uinf = np.array([np.cos(aoa), 0, np.sin(aoa)]) * Umag

# Convergence criteria
conv_crit = {"Niterations": 1500, "error": 1e-5, "Relax_factor": 0.03}

#   Model and program specifics
ring_geo = "5fil"
model = "VSM"

data_airf = np.loadtxt(r"./polars/clarky_maneia.csv", delimiter=",")
# %% SOLVER + OUTPUT
start_time = time.time()
aoas = np.arange(-4, 24, 1) / 180 * np.pi
CL1 = np.zeros(len(aoas))
CL2 = np.zeros(len(aoas))
CD1 = np.zeros(len(aoas))
CD2 = np.zeros(len(aoas))
Gamma0 = np.zeros(N - 1)
for i in range(len(aoas)):

    Gamma0 = np.zeros(N - 1)
    Uinf = np.array([np.cos(aoas[i]), 0, np.sin(aoas[i])]) * Umag
    model = "LLT"
    # Define system of vorticity
    controlpoints, rings, bladepanels, ringvec, coord_L = VSM.create_geometry_general(
        coord, Uinf, N, ring_geo, model
    )
    # Solve for Gamma
    Fmag, Gamma, aero_coeffs = (
        VSM.solve_lifting_line_system_matrix_approach_semiinfinite(
            ringvec, controlpoints, rings, Uinf, Gamma0, data_airf, conv_crit, model
        )
    )
    # Output forces
    F_rel, F_gl, Ltot, Dtot, CL1[i], CD1[i], CS = VSM.output_results(
        Fmag, aero_coeffs, ringvec, Uinf, controlpoints, Atot
    )

    Gamma0 = np.zeros(N - 1)
    model = "VSM"
    # Define system of vorticity
    controlpoints, rings, bladepanels, ringvec, coord_L = VSM.create_geometry_general(
        coord, Uinf, N, ring_geo, model
    )
    # Solve for Gamma
    Fmag, Gamma, aero_coeffs = (
        VSM.solve_lifting_line_system_matrix_approach_semiinfinite(
            ringvec, controlpoints, rings, Uinf, Gamma0, data_airf, conv_crit, model
        )
    )
    # Output forces
    F_rel, F_gl, Ltot, Dtot, CL2[i], CD2[i], CS = VSM.output_results(
        Fmag, aero_coeffs, ringvec, Uinf, controlpoints, Atot
    )

    print(str(round((i + 1) / len(aoas) * 100, 1)) + " %")
end_time = time.time()
print("Time employed: " + str(end_time - start_time) + "seconds")

# %% PLOTS

# polars_VLM = pd.read_csv('./XFLR5/Rect_AR_10_VLM1_Inv.csv', skiprows=6)
# polars_LLT = pd.read_csv('./XFLR5/Rect_AR_5_LLT.csv', skiprows=6)

polars_Maneia = np.loadtxt("./polars/curved_wing_polars_maneia.csv", delimiter=",")

plt_path = "./plots/"
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)
plt.rcParams.update({"font.size": 10})

plt.figure(figsize=(6, 4))

plt.plot(polars_Maneia[:, 0], polars_Maneia[:, 1], marker="x")
plt.plot(aoas * 180 / np.pi, CL1, marker=".", alpha=0.8)
plt.plot(aoas * 180 / np.pi, CL2, marker=".", alpha=0.8)
plt.legend(["RANS", "LLT", "VSM"])
plt.xlabel(r"$\alpha$ ($^\circ$)")
plt.ylabel("$C_L$ ()")
xdata = np.arange(-10, 30, 1)
plt.grid()
plt.savefig(plt_path + "maneia_CL_alpha.pdf", bbox_inches="tight")


plt.figure(figsize=(6, 4))
plt.plot(polars_Maneia[:, 2], polars_Maneia[:, 1], marker="x")
plt.plot(CD1, CL1, marker=".", alpha=0.8)
plt.plot(CD2, CL2, marker=".", alpha=0.8)
plt.legend(["RANS", "LLT", "VSM"])
plt.xlabel("$C_D$ ()")
plt.ylabel("$C_L$ ()")
# plt.title('Curved wing AR =' + str(round(AR,2)))
plt.grid()
plt.savefig(plt_path + "maneia_CL_CD.pdf", bbox_inches="tight")

plt.figure(figsize=(6, 4))
plt.plot(polars_Maneia[:, 0], polars_Maneia[:, 1] / polars_Maneia[:, 2], marker="x")
plt.plot(aoas * 180 / np.pi, CL1 / CD1, marker=".", alpha=0.8)
plt.plot(aoas * 180 / np.pi, CL2 / CD2, marker=".", alpha=0.8)
plt.legend(["RANS", "LLT", "VSM"])
plt.xlabel(r"$\alpha$ ($^\circ$)")
plt.ylabel("$C_L/C_D$ ()")
# plt.title('Curved wing AR =' + str(round(AR,2)))
plt.grid()
plt.savefig(plt_path + "maneia_CLCD_alpha.pdf", bbox_inches="tight")

plt.figure(figsize=(6, 4))
plt.plot(polars_Maneia[:, 0], polars_Maneia[:, 2], marker="x")
plt.plot(aoas * 180 / np.pi, CD1, marker=".", alpha=0.8)
plt.plot(aoas * 180 / np.pi, CD2, marker=".", alpha=0.8)
plt.legend(["RANS", "LLT", "VSM"])
plt.xlabel(r"$\alpha$ ($^\circ$)")
plt.ylabel("$C_D $ ()")
# plt.title('Curved wing AR =' + str(round(AR,2)))
plt.grid()
plt.savefig(plt_path + "maneia_CD_alpha.pdf", bbox_inches="tight")
