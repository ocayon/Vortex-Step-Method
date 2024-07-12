# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:09:16 2022

@author: oriol2
"""


import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import logging
import sys
import os

# Go back to root folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)
import tests.thesis_functions_oriol_cayon as thesis_functions

from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver


def calculating_cl_cd_for_alpha_range(aoas):
    ## INPUT DATA
    N = 40
    max_chord = 1
    span = 2.36
    AR = span**2 / (np.pi * span * max_chord / 4)
    dist = "cos"
    coord = thesis_functions.generate_coordinates_el_wing(max_chord, span, N, dist)
    Atot = max_chord / 2 * span / 2 * np.pi

    Umag = 20
    aoa = 5.7106 * np.pi / 180
    Uinf = np.array([np.cos(aoa), 0, np.sin(aoa)]) * Umag
    # Uinf = np.array([np.sqrt(0.99),0,0.1])

    conv_crit = {"Niterations": 1500, "error": 1e-5, "Relax_factor": 0.05}

    Gamma0 = np.zeros(N - 1)

    ring_geo = "5fil"
    model = "VSM"

    alpha_airf = np.arange(-10, 30)
    data_airf = np.zeros((len(alpha_airf), 4))
    data_airf[:, 0] = alpha_airf
    data_airf[:, 1] = alpha_airf / 180 * np.pi * 2 * np.pi
    data_airf[:, 2] = alpha_airf * 0
    data_airf[:, 3] = alpha_airf * 0

    ###% SOLVER + OUTPUT F
    start_time = time.time()
    CL_LLT = np.zeros(len(aoas))
    CD_LLT = np.zeros(len(aoas))
    gamma_LLT = np.zeros((len(aoas), N - 1))
    CL_VSM = np.zeros(len(aoas))
    CD_VSM = np.zeros(len(aoas))
    gamma_VSM = np.zeros((len(aoas), N - 1))
    CL_LLT_new = np.zeros(len(aoas))
    CD_LLT_new = np.zeros(len(aoas))
    gamma_LLT_new = np.zeros((len(aoas), N - 1))
    CL_VSM_new = np.zeros(len(aoas))
    CD_VSM_new = np.zeros(len(aoas))
    gamma_VSM_new = np.zeros((len(aoas), N - 1))

    core_radius_fraction = 1e-20

    for i, aoa_i in enumerate(aoas):

        Uinf = np.array([np.cos(aoa_i), 0, np.sin(aoa_i)]) * Umag
        model = "LLT"

        ### thesis
        # Define system of vorticity
        controlpoints, rings, bladepanels, ringvec, coord_L = (
            thesis_functions.create_geometry_general(coord, Uinf, N, ring_geo, model)
        )
        # Solve for Gamma
        Fmag, Gamma, aero_coeffs = (
            thesis_functions.solve_lifting_line_system_matrix_approach_semiinfinite(
                ringvec, controlpoints, rings, Uinf, Gamma0, data_airf, conv_crit, model
            )
        )
        # Output forces
        F_rel, F_gl, Ltot, Dtot, CL_LLT[i], CD_LLT[i], CS = (
            thesis_functions.output_results(
                Fmag, aero_coeffs, ringvec, Uinf, controlpoints, Atot
            )
        )
        gamma_LLT[i] = Gamma

        ### new object-oriented
        wing_LLT = Wing(N, "unchanged")
        for idx in range(int(len(coord) / 2)):
            wing_LLT.add_section(coord[2 * idx], coord[2 * idx + 1], ["inviscid"])
        wing_aero_LTT = WingAerodynamics([wing_LLT])
        wing_aero_LTT.va = Uinf
        LLT = Solver(
            aerodynamic_model_type=model, core_radius_fraction=core_radius_fraction
        )
        results_LLT, wing_aero_LLT = LLT.solve(wing_aero_LTT)
        CL_LLT_new[i] = results_LLT["cl"]
        CD_LLT_new[i] = results_LLT["cd"]
        gamma_LLT_new[i] = results_LLT["gamma_distribution"]

        #############
        #### VSM ####
        #############
        model = "VSM"
        # Define system of vorticity
        controlpoints, rings, bladepanels, ringvec, coord_L = (
            thesis_functions.create_geometry_general(coord, Uinf, N, ring_geo, model)
        )
        # Solve for Gamma
        Fmag, Gamma, aero_coeffs = (
            thesis_functions.solve_lifting_line_system_matrix_approach_semiinfinite(
                ringvec, controlpoints, rings, Uinf, Gamma0, data_airf, conv_crit, model
            )
        )
        # Output forces
        F_rel, F_gl, Ltot, Dtot, CL_VSM[i], CD_VSM[i], CS = (
            thesis_functions.output_results(
                Fmag, aero_coeffs, ringvec, Uinf, controlpoints, Atot
            )
        )
        gamma_VSM[i] = Gamma

        ### new object-oriented
        wing_VSM = Wing(N, "unchanged")
        for idx in range(int(len(coord) / 2)):
            wing_VSM.add_section(coord[2 * idx], coord[2 * idx + 1], ["inviscid"])
        wing_aero_VSM = WingAerodynamics([wing_VSM])
        wing_aero_VSM.va = Uinf
        VSM = Solver(
            aerodynamic_model_type=model, core_radius_fraction=core_radius_fraction
        )
        results_VSM, wing_aero_VSM = VSM.solve(wing_aero_VSM)
        CL_VSM_new[i] = results_VSM["cl"]
        CD_VSM_new[i] = results_VSM["cd"]
        gamma_VSM_new[i] = results_VSM["gamma_distribution"]

        Gamma0 = Gamma
        print(str((i + 1) / len(aoas) * 100) + " %")
    end_time = time.time()
    print(end_time - start_time)

    return (
        CL_LLT,
        CD_LLT,
        gamma_LLT,
        CL_VSM,
        CD_VSM,
        gamma_VSM,
        CL_LLT_new,
        CD_LLT_new,
        gamma_LLT_new,
        CL_VSM_new,
        CD_VSM_new,
        gamma_VSM_new,
        AR,
    )


def plotting(
    CL_LLT,
    CD_LLT,
    gamma_LLT,
    CL_VSM,
    CD_VSM,
    gamma_VSM,
    CL_LLT_new,
    CD_LLT_new,
    gamma_LLT_new,
    CL_VSM_new,
    CD_VSM_new,
    gamma_VSM_new,
    AR,
):
    colors = sns.color_palette()
    CL_th = 2 * np.pi * aoas / (1 + 2 / AR)
    CDi_th = CL_th**2 / np.pi / AR

    legend = ["Analytic LLT", "LLT", "VSM", "LLT_new", "VSM_new"]
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
    plt.plot(aoas * 180 / np.pi, CL_th, marker="x", color=colors[0])
    # plt.plot(aoas * 180 / np.pi, CL_LLT, marker=".", alpha=0.8, color=colors[1])
    # plt.plot(aoas * 180 / np.pi, CL_VSM, marker=".", alpha=0.8, color=colors[2])
    # plt.plot(aoas * 180 / np.pi, CL_LLT_new, marker=".", alpha=0.8, color=colors[3])
    plt.plot(aoas * 180 / np.pi, CL_VSM_new, marker=".", alpha=0.8, color=colors[4])
    plt.legend(legend)
    plt.xlabel(r"$\alpha$ ($^\circ$)")
    plt.ylabel("$C_L$ ()")
    # plt.title('Elliptic wing AR =' + str(round(AR,1)))
    plt.grid()
    plt.savefig(
        plt_path + str(round(AR, 1)) + "_AR_Ell_CL_alpha.pdf", bbox_inches="tight"
    )

    plt.figure(figsize=(6, 4))
    plt.plot(CDi_th, CL_th, marker="x", color=colors[0])
    # plt.plot(CD_LLT, CL_LLT, marker=".", alpha=0.8, color=colors[1])
    # plt.plot(CD_VSM, CL_VSM, marker=".", alpha=0.8, color=colors[2])
    # plt.plot(CD_LLT_new, CL_LLT_new, marker=".", alpha=0.8, color=colors[3])
    plt.plot(CD_VSM_new, CL_VSM_new, marker=".", alpha=0.8, color=colors[4])
    plt.legend(legend)
    plt.xlabel("$C_D$")
    plt.ylabel("$C_L$")
    # plt.title('Elliptic wing AR =' + str(round(AR,1)))
    plt.grid()
    plt.savefig(
        plt_path + str(round(AR, 1)) + "_AR_Rect_CL_CD.pdf", bbox_inches="tight"
    )

    plt.figure(figsize=(6, 4))
    plt.plot(aoas * 180 / np.pi, CL_th / CDi_th, marker=".", color=colors[0])
    # plt.plot(
    #     aoas * 180 / np.pi, CL_LLT / CD_LLT, marker=".", alpha=0.8, color=colors[1]
    # )
    # plt.plot(
    #     aoas * 180 / np.pi, CL_VSM / CD_VSM, marker=".", alpha=0.8, color=colors[2]
    # )
    # plt.plot(
    #     aoas * 180 / np.pi,
    #     CL_LLT_new / CD_LLT_new,
    #     marker=".",
    #     alpha=0.8,
    #     color=colors[3],
    # )
    plt.plot(
        aoas * 180 / np.pi,
        CL_VSM_new / CD_VSM_new,
        marker=".",
        alpha=0.8,
        color=colors[4],
    )
    plt.legend(legend)
    plt.xlabel(r"$\alpha$ ($^\circ$)")
    plt.ylabel("$C_L/C_D$")
    # plt.title('Elliptic wing AR =' + str(round(AR,1)))
    plt.grid()
    plt.savefig(
        plt_path + str(round(AR, 1)) + "_AR_Rect_CLCD_alpha.pdf", bbox_inches="tight"
    )

    plt.figure(figsize=(6, 4))
    plt.plot(aoas * 180 / np.pi, CDi_th, marker="x", alpha=0.3, color=colors[0])
    # plt.plot(aoas * 180 / np.pi, CD_LLT, marker=".", alpha=0.3, color=colors[1])
    # plt.plot(aoas * 180 / np.pi, CD_VSM, marker=".", alpha=0.8, color=colors[2])
    # plt.plot(aoas * 180 / np.pi, CD_LLT_new, marker=".", alpha=0.8, color=colors[3])
    plt.plot(aoas * 180 / np.pi, CD_VSM_new, marker=".", alpha=0.8, color=colors[4])
    plt.legend(legend)
    plt.xlabel(r"$\alpha$ ($^\circ$)")
    plt.ylabel("$C_D$")
    # plt.title('Elliptic wing AR =' + str(round(AR,1)))
    plt.grid()
    plt.savefig(
        plt_path + str(round(AR, 1)) + "_AR_Ell_CD_alpha.pdf", bbox_inches="tight"
    )

    for i, aoa in enumerate(aoas):
        plt.figure(figsize=(6, 4))
        plt.plot(gamma_LLT[i], marker=".", alpha=0.3, color=colors[1])
        plt.plot(gamma_VSM[i], marker=".", alpha=0.3, color=colors[2])
        plt.plot(gamma_LLT_new[i], marker=".", alpha=0.8, color=colors[3])
        plt.plot(gamma_VSM_new[i], marker=".", alpha=0.8, color=colors[4])
        plt.legend(legend[1:])
        plt.xlabel("Section")
        plt.ylabel("$\Gamma$")
        plt.grid()
        plt.savefig(
            plt_path
            + str(round(AR, 1))
            + "_AR_Ell_Gamma_aoa_"
            + str(round(np.rad2deg(aoa), 1))
            + ".pdf",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    aoas = np.arange(0, 20, 1) / 180 * np.pi
    aoas = np.arange(2, 16, 6) / 180 * np.pi
    (
        CL_LLT,
        CD_LLT,
        gamma_LLT,
        CL_VSM,
        CD_VSM,
        gamma_VSM,
        CL_LLT_new,
        CD_LLT_new,
        gamma_LLT_new,
        CL_VSM_new,
        CD_VSM_new,
        gamma_VSM_new,
        AR,
    ) = calculating_cl_cd_for_alpha_range(aoas)
    plotting(
        CL_LLT,
        CD_LLT,
        gamma_LLT,
        CL_VSM,
        CD_VSM,
        gamma_VSM,
        CL_LLT_new,
        CD_LLT_new,
        gamma_LLT_new,
        CL_VSM_new,
        CD_VSM_new,
        gamma_VSM_new,
        AR,
    )
    logging.info(f"CL_VSM = {CL_VSM}, CL_VSM_new = {CL_VSM_new}")
    logging.info(f"CD_VSM = {CD_VSM}, CD_VSM_new = {CD_VSM_new}")
    plt.show()
