# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 00:26:08 2022

@author: oriol
"""

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
N = 50
#   Rectangular wing coordinates
R = 4.673
theta = 45 * np.pi / 180
span = 6.969
chord = 2.18
AR = span / chord
dist = "lin"
coord = VSM.generate_coordinates_curved_wing(chord, span, theta, R, N, dist)
Atot = 14.956
# Atot = 16

# Wind speed mag and direction
Umag = 22
aoa = 20
aoa = aoa * np.pi / 180
Uinf = np.array([np.cos(aoa), 0, np.sin(aoa)]) * Umag

Gamma0 = 0 * 0.5 * abs(np.pi * aoa * (1 + 2 / AR) * chord * 0.5 * Umag)
#   Model and program specifics
ring_geo = "5fil"
model = "VSM"

data_airf = np.loadtxt(r"./polars/clarky_maneia.csv", delimiter=",")

# %% SOLVER
start_time = time.time()
# Define system of vorticity
controlpoints, rings, bladepanels, ringvec, coord_L = VSM.create_geometry_general(
    coord, Uinf, N, ring_geo, model
)
# Solve for Gamma
Fmag, Gamma, aero_coeffs = VSM.solve_lifting_line_system_matrix_approach_semiinfinite(
    ringvec, controlpoints, rings, Uinf, data_airf, True, Gamma0, model
)


end_time = time.time()
# %% OUTPUT Results

F_rel, F_gl, Ltot, Dtot, CL, CD = VSM.output_results(
    Fmag, aero_coeffs, ringvec, Uinf, controlpoints, Atot
)

print(CL, CD)
# %% PLOTS
plt_path = "./plots/spanwise_dist/"

fig = plt.figure(figsize=(6, 5))
plt.plot(coord_L[:, 1] / max(coord_L[:, 1]), aero_coeffs[:, 1], label=model)
plt.title("Curved wing AR =" + str(AR) + r"$, \alpha = $" + str(aoa * 180 / np.pi))
plt.xlabel(r"$y/s$")
plt.ylabel(r"$C_l$")
plt.legend()

plt.savefig(plt_path + model + "_maneia_CL_y.png", dpi=150)

fig = plt.figure(figsize=(6, 5))
plt.plot(
    coord_L[:, 1] / max(coord_L[:, 1]), aero_coeffs[:, 0] * 180 / np.pi, label=model
)
plt.title("Curved wing AR =" + str(AR) + r"$, \alpha = $" + str(aoa * 180 / np.pi))
plt.xlabel(r"$y/s$")
plt.ylabel(r"$\alpha$")
plt.legend()

plt.savefig(plt_path + model + "_maneia_alpha_y.png", dpi=150)

fig = plt.figure(figsize=(6, 5))
plt.plot(coord_L[:, 1] / max(coord_L[:, 1]), aero_coeffs[:, 2], label=model)
plt.title("Curved wing AR =" + str(AR) + r"$, \alpha = $" + str(aoa * 180 / np.pi))
plt.xlabel(r"$y/s$")
plt.ylabel(r"Viscous $C_d$")
plt.legend()

plt.savefig(plt_path + model + "_maneia_Cdvisc_y.png", dpi=150)

fig = plt.figure(figsize=(6, 5))
plt.plot(coord_L[:, 1] / max(coord_L[:, 1]), Gamma, label=model)
plt.title("Curved wing AR =" + str(AR) + r"$, \alpha = $" + str(aoa * 180 / np.pi))
plt.xlabel(r"$y/s$")
plt.ylabel(r"$\Gamma$")
plt.legend()

plt.savefig(plt_path + model + "_maneia_gamma_y.png", dpi=150)


VSM.plot_geometry(bladepanels, controlpoints, rings, F_gl, coord_L, "False")
