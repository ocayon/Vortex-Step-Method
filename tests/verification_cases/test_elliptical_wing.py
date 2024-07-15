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
from copy import deepcopy

# Go back to root folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)
import tests.thesis_functions_oriol_cayon as thesis_functions
from tests.utils import flip_created_coord_in_pairs

from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver


def calculate_OLD_for_alpha_range(N, max_chord, span, AR, Umag, aoas):
    ##INPUT DATA
    dist = "cos"
    coord = thesis_functions.generate_coordinates_el_wing(max_chord, span, N, dist)
    Atot = max_chord / 2 * span / 2 * np.pi

    # aoa = 5.7106 * np.pi / 180
    # Uinf = np.array([np.cos(aoa), 0, np.sin(aoa)]) * Umag
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
    ## SOLVER + OUTPUT F
    start_time = time.time()
    CL1 = np.zeros(len(aoas))
    CL2 = np.zeros(len(aoas))
    CD1 = np.zeros(len(aoas))
    CD2 = np.zeros(len(aoas))
    Gamma_LLT = []
    Gamma_VSM = []
    for i in range(len(aoas)):

        Uinf = np.array([np.cos(aoas[i]), 0, np.sin(aoas[i])]) * Umag
        model = "LLT"
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
        F_rel, F_gl, Ltot, Dtot, CL1[i], CD1[i], CS = thesis_functions.output_results(
            Fmag, aero_coeffs, ringvec, Uinf, controlpoints, Atot
        )
        Gamma_LLT.append(Gamma)
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
        Gamma_VSM.append(Gamma)
        # Output forces
        F_rel, F_gl, Ltot, Dtot, CL2[i], CD2[i], CS = thesis_functions.output_results(
            Fmag, aero_coeffs, ringvec, Uinf, controlpoints, Atot
        )

        Gamma0 = Gamma
        print(str((i + 1) / len(aoas) * 100) + " %")
    # end_time = time.time()
    # print(end_time - start_time)

    return CL1, CD1, CL2, CD2, Gamma_LLT, Gamma_VSM


def calculate_NEW_for_alpha_range(
    N, max_chord, span, AR, Umag, aoas, is_plotting=False
):
    dist = "cos"
    core_radius_fraction = 1e-20
    coord = thesis_functions.generate_coordinates_el_wing(max_chord, span, N, dist)
    coord_left_to_right = flip_created_coord_in_pairs(deepcopy(coord))
    wing = Wing(N, "unchanged")
    for idx in range(int(len(coord_left_to_right) / 2)):
        logging.debug(f"coord_left_to_right[idx] = {coord_left_to_right[idx]}")
        wing.add_section(
            coord_left_to_right[2 * idx],
            coord_left_to_right[2 * idx + 1],
            ["inviscid"],
        )
    wing_aero = WingAerodynamics([wing])

    # initializing zero lists
    wing_aero = WingAerodynamics([wing])
    CL_LLT_new = np.zeros(len(aoas))
    CD_LLT_new = np.zeros(len(aoas))
    gamma_LLT_new = np.zeros((len(aoas), N - 1))
    CL_VSM_new = np.zeros(len(aoas))
    CD_VSM_new = np.zeros(len(aoas))
    gamma_VSM_new = np.zeros((len(aoas), N - 1))
    controlpoints_list = []

    for i, aoa_i in enumerate(aoas):
        logging.debug(f"aoa_i = {np.rad2deg(aoa_i)}")
        Uinf = np.array([np.cos(aoa_i), 0, np.sin(aoa_i)]) * Umag
        wing_aero.va = Uinf
        if i == 0 and is_plotting:
            wing_aero.plot()
        # LLT
        LLT = Solver(
            aerodynamic_model_type="LLT", core_radius_fraction=core_radius_fraction
        )
        results_LLT, wing_aero_LLT = LLT.solve(wing_aero)
        CL_LLT_new[i] = results_LLT["cl"]
        CD_LLT_new[i] = results_LLT["cd"]
        gamma_LLT_new[i] = results_LLT["gamma_distribution"]

        # VSM
        VSM = Solver(
            aerodynamic_model_type="VSM", core_radius_fraction=core_radius_fraction
        )
        results_VSM, wing_aero_VSM = VSM.solve(wing_aero)
        CL_VSM_new[i] = results_VSM["cl"]
        CD_VSM_new[i] = results_VSM["cd"]
        gamma_VSM_new[i] = results_VSM["gamma_distribution"]

        logging.debug(f"CD_LLT_new = {results_LLT['cd']}")
        logging.debug(f"CD_VSM_new = {results_VSM['cd']}")

        controlpoints_list.append(
            [panel.aerodynamic_center for panel in wing_aero_LLT.panels]
        )
    panel_y = [panel.aerodynamic_center[1] for panel in wing_aero_LLT.panels]
    return (
        CL_LLT_new,
        CD_LLT_new,
        CL_VSM_new,
        CD_VSM_new,
        gamma_LLT_new,
        gamma_VSM_new,
        panel_y,
    )


def plotting(
    aoas,
    CL_th,
    CDi_th,
    CL1,
    CD1,
    CL2,
    CD2,
    CL_LLT_new,
    CD_LLT_new,
    CL_VSM_new,
    CD_VSM_new,
):
    colors = sns.color_palette()

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
    plt.plot(
        aoas * 180 / np.pi, CL_th, marker="x", color=colors[0], label="Analytic LLT"
    )
    plt.plot(
        aoas * 180 / np.pi, CL1, marker=".", alpha=0.8, color=colors[1], label="LLT"
    )
    # plt.plot(
    #     aoas * 180 / np.pi, CL2, marker=".", alpha=0.8, color=colors[2], label="VSM"
    # )
    plt.plot(
        aoas * 180 / np.pi,
        CL_LLT_new,
        marker=".",
        alpha=0.8,
        color=colors[3],
        label="LLT_new",
    )
    # plt.plot(
    #     aoas * 180 / np.pi,
    #     CL_VSM_new,
    #     marker=".",
    #     alpha=0.8,
    #     color=colors[4],
    #     label="VSM_new",
    # )
    plt.legend()
    plt.xlabel(r"$\alpha$ ($^\circ$)")
    plt.ylabel("$C_L$ ()")
    # plt.title('Elliptic wing AR =' + str(round(AR,1)))
    plt.grid()
    plt.savefig(
        plt_path + str(round(AR, 1)) + "_AR_Ell_CL_alpha_LLT.pdf", bbox_inches="tight"
    )

    plt.figure(figsize=(6, 4))
    plt.plot(
        aoas * 180 / np.pi,
        CL_th,
        marker="x",
        alpha=0.3,
        color=colors[0],
        label="Analytic LLT",
    )
    # plt.plot(
    #     aoas * 180 / np.pi, CL1, marker=".", alpha=0.8, color=colors[1], label="LLT"
    # )
    # plt.plot(
    #     aoas * 180 / np.pi,
    #     CL_LLT_new,
    #     marker=".",
    #     alpha=0.8,
    #     color=colors[3],
    #     label="LLT_new",
    # )
    plt.plot(
        aoas * 180 / np.pi, CL2, marker=".", alpha=0.8, color=colors[2], label="VSM"
    )

    plt.plot(
        aoas * 180 / np.pi,
        CL_VSM_new,
        marker=".",
        alpha=0.8,
        color=colors[4],
        label="VSM_new",
    )
    plt.legend()
    plt.xlabel(r"$\alpha$ ($^\circ$)")
    plt.ylabel("$C_L$ ()")
    # plt.title('Elliptic wing AR =' + str(round(AR,1)))
    plt.grid()
    plt.savefig(
        plt_path + str(round(AR, 1)) + "_AR_Ell_CL_alpha_VSM.pdf", bbox_inches="tight"
    )

    # plt.figure(figsize=(6, 4))
    # plt.plot(CDi_th, CL_th, marker="x", color=colors[0], label="Analytic LLT")
    # plt.plot(
    #     CD1,
    #     CL1,
    #     marker=".",
    #     alpha=0.8,
    #     color=colors[1],
    # )
    # plt.plot(CD2, CL2, marker=".", alpha=0.8, color=colors[2])
    # plt.plot(CD_LLT_new, CL_LLT_new, marker=".", alpha=0.8, color=colors[3])
    # plt.plot(CD_VSM_new, CL_VSM_new, marker=".", alpha=0.8, color=colors[4])
    # plt.legend(["Analytic LLT", "LLT", "VSM", "LLT_new", "VSM_new"])
    # plt.xlabel("$C_D$")
    # plt.ylabel("$C_L$")
    # # plt.title('Elliptic wing AR =' + str(round(AR,1)))
    # plt.grid()
    # plt.savefig(
    #     plt_path + str(round(AR, 1)) + "_AR_Rect_CL_CD.pdf", bbox_inches="tight"
    # )

    # plt.figure(figsize=(6, 4))
    # plt.plot(aoas * 180 / np.pi, CL_th / CDi_th, marker=".")
    # plt.plot(aoas * 180 / np.pi, CL1 / CD1, marker=".", alpha=0.8)
    # plt.plot(aoas * 180 / np.pi, CL2 / CD2, marker=".", alpha=0.8)
    # plt.legend(["Analytic LLT", "LLT", "VSM", "LLT_new", "VSM_new"])
    # plt.xlabel(r"$\alpha$ ($^\circ$)")
    # plt.ylabel("$C_L/C_D$")
    # # plt.title('Elliptic wing AR =' + str(round(AR,1)))
    # plt.grid()
    # plt.savefig(
    #     plt_path + str(round(AR, 1)) + "_AR_Rect_CLCD_alpha.pdf", bbox_inches="tight"
    # )

    plt.figure(figsize=(6, 4))
    plt.plot(aoas * 180 / np.pi, CDi_th, marker="x", color=colors[0])
    plt.plot(aoas * 180 / np.pi, CD1, marker=".", alpha=0.8, color=colors[1])
    # plt.plot(aoas * 180 / np.pi, CD2, marker=".", alpha=0.8, color=colors[2])
    plt.plot(aoas * 180 / np.pi, CD_LLT_new, marker=".", alpha=0.8, color=colors[3])
    # plt.plot(aoas * 180 / np.pi, CD_VSM_new, marker=".", alpha=0.8, color=colors[4])
    plt.legend(["Analytic LLT", "LLT", "LLT_new"])
    plt.xlabel(r"$\alpha$ ($^\circ$)")
    plt.ylabel("$C_D$")
    # plt.title('Elliptic wing AR =' + str(round(AR,1)))
    plt.grid()
    plt.savefig(
        plt_path + str(round(AR, 1)) + "_AR_Ell_CD_alpha_LLT.pdf", bbox_inches="tight"
    )

    plt.figure(figsize=(6, 4))
    plt.plot(aoas * 180 / np.pi, CDi_th, marker="x", alpha=0.3, color=colors[0])
    # plt.plot(aoas * 180 / np.pi, CD1, marker=".", alpha=0.8, color=colors[1])
    # plt.plot(aoas * 180 / np.pi, CD_LLT_new, marker=".", alpha=0.8, color=colors[3])
    plt.plot(aoas * 180 / np.pi, CD2, marker=".", alpha=0.8, color=colors[2])
    plt.plot(aoas * 180 / np.pi, CD_VSM_new, marker=".", alpha=0.8, color=colors[4])
    plt.legend(["Analytic LLT", "VSM", "VSM_new"])
    plt.xlabel(r"$\alpha$ ($^\circ$)")
    plt.ylabel("$C_D$")
    # plt.title('Elliptic wing AR =' + str(round(AR,1)))
    plt.grid()
    plt.savefig(
        plt_path + str(round(AR, 1)) + "_AR_Ell_CD_alpha_VSM.pdf", bbox_inches="tight"
    )


def plotting_gamma_distrbution(
    aoa, panel_y, gamma_LLT, gamma_VSM, gamma_LLT_new, gamma_VSM_new
):
    colors = sns.color_palette()

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
    plt.plot(panel_y, gamma_LLT, marker="x", alpha=0.8, color=colors[1], label="LLT")
    # plt.plot(panel_y, gamma_VSM, marker=".", alpha=0.8, color=colors[2], label="VSM")
    plt.plot(
        panel_y, gamma_LLT_new, marker="x", alpha=0.8, color=colors[3], label="LLT_new"
    )
    # plt.plot(
    #     panel_y, gamma_VSM_new, marker=".", alpha=0.8, color=colors[4], label="VSM_new"
    # )
    plt.legend()
    plt.xlabel(r"$y$")
    plt.ylabel(r"$\Gamma$")
    plt.title(
        f"LLT Circulation at aoa: {np.rad2deg(aoa)}, Elliptic wing AR = {str(round(AR,1))}"
    )
    plt.grid()
    plt.savefig(plt_path + str(round(AR, 1)) + "gamma_LLT.pdf", bbox_inches="tight")

    plt.rcParams.update({"font.size": 10})
    plt.figure(figsize=(6, 4))
    plt.plot(panel_y, gamma_LLT, marker="x", alpha=0.3, color=colors[1], label="LLT")
    plt.plot(panel_y, gamma_VSM, marker=".", alpha=0.8, color=colors[2], label="VSM")
    plt.plot(
        panel_y, gamma_LLT_new, marker="x", alpha=0.3, color=colors[3], label="LLT_new"
    )
    plt.plot(
        panel_y, gamma_VSM_new, marker=".", alpha=0.8, color=colors[4], label="VSM_new"
    )
    plt.legend()
    plt.xlabel(r"$y$")
    plt.ylabel(r"$\Gamma$")
    plt.title(
        f"VSM Circulation at aoa: {np.rad2deg(aoa)}, Elliptic wing AR = {str(round(AR,1))}"
    )
    plt.grid()
    plt.savefig(plt_path + str(round(AR, 1)) + "gamma_VSM.pdf", bbox_inches="tight")


def test_elliptical():
    aoas = np.deg2rad([5, 10])
    N = 40
    max_chord = 1
    span = 15.709  # AR = 20
    # span = 2.36  # AR = 3
    Umag = 20
    AR = span**2 / (np.pi * span * max_chord / 4)
    # analytical solution
    CL_th = 2 * np.pi * aoas / (1 + 2 / AR)
    CDi_th = CL_th**2 / np.pi / AR
    # OLD numerical
    CL1, CD1, CL2, CD2, gamma_LLT, gamma_VSM = calculate_OLD_for_alpha_range(
        N, max_chord, span, AR, Umag, aoas
    )
    # NEW numerical
    (
        CL_LLT_new,
        CD_LLT_new,
        CL_VSM_new,
        CD_VSM_new,
        gamma_LLT_new,
        gamma_VSM_new,
        panel_y,
    ) = calculate_NEW_for_alpha_range(N, max_chord, span, AR, Umag, aoas)
    for aoa in aoas:
        aoa_deg = np.rad2deg(aoa)
        # checking all LLTs to be close
        assert np.allclose(CL_th, CL1, atol=1e-2)
        assert np.allclose(CDi_th, CD1, atol=1e-4)
        assert np.allclose(CL_th, CL_LLT_new, atol=1e-2)
        assert np.allclose(CDi_th, CD_LLT_new, atol=1e-4)
        assert np.allclose(gamma_LLT, gamma_LLT_new, atol=1e-2)

        # checking VSMs to be close to one another
        assert np.allclose(CL2, CL_VSM_new, atol=1e-2)
        assert np.allclose(CD2, CD_VSM_new, atol=1e-4)

        # checking the LLT to be close to the VSM, with HIGHER tolerance
        tol_llt_to_vsm_CL = 1e-1
        tol_llt_to_vsm_CD = 1e-3
        assert np.allclose(CL_th, CL2, atol=tol_llt_to_vsm_CL)
        assert np.allclose(CDi_th, CD2, atol=tol_llt_to_vsm_CD)
        assert np.allclose(CL_th, CL_VSM_new, atol=tol_llt_to_vsm_CL)
        assert np.allclose(CDi_th, CD_VSM_new, atol=tol_llt_to_vsm_CD)
        assert np.allclose(CL1, CL2, atol=tol_llt_to_vsm_CL)
        assert np.allclose(CD1, CD2, atol=tol_llt_to_vsm_CD)
        assert np.allclose(CL_LLT_new, CL_VSM_new, atol=tol_llt_to_vsm_CL)
        assert np.allclose(CD_LLT_new, CD_VSM_new, atol=tol_llt_to_vsm_CD)


if __name__ == "__main__":
    aoas = np.arange(0, 20, 1) / 180 * np.pi
    N = 40
    max_chord = 1
    span = 15.709  # AR = 20
    # span = 2.36  # AR = 3
    Umag = 20
    AR = span**2 / (np.pi * span * max_chord / 4)

    # analytical solution
    CL_th = 2 * np.pi * aoas / (1 + 2 / AR)
    CDi_th = CL_th**2 / np.pi / AR
    # OLD numerical
    CL1, CD1, CL2, CD2, gamma_LLT, gamma_VSM = calculate_OLD_for_alpha_range(
        N, max_chord, span, AR, Umag, aoas
    )
    # NEW numerical
    (
        CL_LLT_new,
        CD_LLT_new,
        CL_VSM_new,
        CD_VSM_new,
        gamma_LLT_new,
        gamma_VSM_new,
        panel_y,
    ) = calculate_NEW_for_alpha_range(N, max_chord, span, AR, Umag, aoas)

    logging.debug(f"CD_LLT_new = {CD_LLT_new}")
    logging.debug(f"CD_VSM_new = {CD_VSM_new}")

    for i, aoa in enumerate(aoas):
        print(f"aoa = {np.rad2deg(aoa)}")
        print(
            f"CL_th: {CL_th[i]}, CL_LLT: {CL1[i]}, CL_VSM: {CL1[i]}, CL_LLT_new: {CL_LLT_new[i]}, CL_VSM_new: {CL_VSM_new[i]}"
        )
        print(
            f"CDi_th: {CDi_th[i]}, CD_LLT: {CD1[i]}, CD_VSM: {CD2[i]}, CD_LLT_new: {CD_LLT_new[i]}, CD_VSM_new: {CD_VSM_new[i]}"
        )
    idx = 5
    plotting_gamma_distrbution(
        aoas[idx],
        panel_y,
        gamma_LLT[idx],
        gamma_VSM[idx],
        gamma_LLT_new[idx],
        gamma_VSM_new[idx],
    )

    plotting(
        aoas,
        CL_th,
        CDi_th,
        CL1,
        CD1,
        CL2,
        CD2,
        CL_LLT_new,
        CD_LLT_new,
        CL_VSM_new,
        CD_VSM_new,
    )
