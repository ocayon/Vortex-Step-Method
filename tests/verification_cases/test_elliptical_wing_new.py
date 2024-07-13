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


def calculating_cl_cd_for_alpha_range(aoas, span, AR, max_chord, n_sections):
    ## INPUT DATA
    N = n_sections
    dist = "cos"
    coord = thesis_functions.generate_coordinates_el_wing(max_chord, span, N, dist)
    coord_left_to_right = np.flip(coord, axis=0)
    logging.debug(f"coord = {coord}")
    logging.debug(f"coord_left_to_right = {coord_left_to_right}")

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
    controlpoints_list = []

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
        for idx in range(int(len(coord_left_to_right) / 2)):
            logging.info(f"coord_left_to_right[idx] = {coord_left_to_right[idx]}")
            wing_LLT.add_section(
                coord_left_to_right[2 * idx],
                coord_left_to_right[2 * idx + 1],
                ["inviscid"],
            )
        wing_aero_LTT = WingAerodynamics([wing_LLT])
        wing_aero_LTT.va = Uinf
        wing_aero_LTT.plot()
        LLT = Solver(
            aerodynamic_model_type=model, core_radius_fraction=core_radius_fraction
        )
        results_LLT, wing_aero_LLT = LLT.solve(wing_aero_LTT)
        CL_LLT_new[i] = results_LLT["cl"]
        CD_LLT_new[i] = results_LLT["cd"]
        gamma_LLT_new[i] = results_LLT["gamma_distribution"]

        logging.info(f"aoa_i = {np.rad2deg(aoa_i)}")
        logging.info(f"CL_LLT_new = {CL_LLT_new}")
        logging.info(f"CD_LLT_new = {CD_LLT_new}")
        breakpoint()
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
        # wing_VSM = Wing(N, "unchanged")
        # for idx in range(int(len(coord) / 2)):
        #     wing_VSM.add_section(coord[2 * idx], coord[2 * idx + 1], ["inviscid"])
        # wing_aero_VSM = WingAerodynamics([wing_VSM])
        # wing_aero_VSM.va = Uinf
        VSM = Solver(
            aerodynamic_model_type=model, core_radius_fraction=core_radius_fraction
        )
        results_VSM, wing_aero_VSM = VSM.solve(wing_aero_LLT)
        CL_VSM_new[i] = results_VSM["cl"]
        CD_VSM_new[i] = results_VSM["cd"]
        gamma_VSM_new[i] = results_VSM["gamma_distribution"]

        Gamma0 = Gamma
        controlpoints_list.append(
            [panel.aerodynamic_center for panel in wing_aero_LLT.panels]
        )
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
        controlpoints_list,
    )


def plot_4_parameters(
    x_axis,
    y_axis_list,
    legend,
    alpha_list,
    x_label=r"$\alpha$ ($^\circ$)",
    y_label="$C_L$ ()",
    title="",
    plt_path="./plots/",
):
    colors = sns.color_palette()
    plt.figure(figsize=(6, 4))
    for i, y_axis in enumerate(y_axis_list):
        plt.plot(x_axis, y_axis, marker=".", alpha=alpha_list[i], color=colors[i])
    plt.legend(legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.savefig(plt_path + title + ".png", bbox_inches="tight")


def plotting(
    aoas,
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
    controlpoints_list,
):

    CL_th = 2 * np.pi * aoas / (1 + 2 / AR)
    CDi_th = CL_th**2 / np.pi / AR

    legend = ["Analytic LLT", "LLT", "VSM", "LLT_new", "VSM_new"]
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )
    plt.rcParams.update({"font.size": 10})

    CL_list = [CL_th, CL_LLT, CL_VSM, CL_LLT_new, CL_VSM_new]
    CD_list = [CDi_th, CD_LLT, CD_VSM, CD_LLT_new, CD_VSM_new]
    gamma_list = [gamma_LLT, gamma_VSM, gamma_LLT_new, gamma_VSM_new]
    aoa_list = aoas * 180 / np.pi
    alpha_list = [1, 0.8, 0.8, 0.8, 0.8]
    # Cl -alpha
    plot_4_parameters(
        aoa_list,
        CL_list,
        legend,
        alpha_list,
        x_label=r"$\alpha$ ($^\circ$)",
        y_label="$C_L$ ()",
        title=str(round(AR, 1)) + "_AR_Ell_CL_alpha",
    )
    # # Cl - Cd
    # plot_4_parameters(
    #     CD_list,
    #     CL_list,
    #     legend,
    #     alpha_list,
    #     x_label="$C_D$",
    #     y_label="$C_L$ ()",
    #     title=str(round(AR, 1)) + "_AR_Rect_CL_CD",
    # )
    # Cl/Cd - alpha
    plot_4_parameters(
        aoa_list,
        [CL_list[i] / CD_list[i] for i in range(len(CL_list))],
        legend,
        alpha_list,
        x_label=r"$\alpha$ ($^\circ$)",
        y_label="$C_L/C_D$",
        title=str(round(AR, 1)) + "_AR_Rect_CLCD_alpha",
    )
    # Cd - alpha
    plot_4_parameters(
        aoa_list,
        CD_list,
        legend,
        alpha_list,
        x_label=r"$\alpha$ ($^\circ$)",
        y_label="$C_D$",
        title=str(round(AR, 1)) + "_AR_Ell_CD_alpha",
    )
    cp_y_aoa_list = []
    for cp in controlpoints_list:
        cp_y_list = []
        for cp_i in cp:
            cp_y_list.append(cp_i["coordinates"][1])
        cp_y_aoa_list.append(cp_y_list)

    # plotting gammas
    for i, aoa in enumerate(aoas):
        plot_4_parameters(
            cp_y_aoa_list[i],
            [gamma_LLT[i], gamma_VSM[i], gamma_LLT_new[i], gamma_VSM_new[i]],
            ["LLT", "VSM", "LLT_new", "VSM_new"],
            alpha_list,
            x_label="Section",
            y_label="$\Gamma$",
            title=str(round(AR, 1))
            + "_AR_Ell_Gamma_aoa_"
            + str(round(np.rad2deg(aoa), 1)),
        )


if __name__ == "__main__":
    # elliptical geometry
    max_chord = 1
    span = 2.36
    n_sections = 3
    AR = span**2 / (np.pi * span * max_chord / 4)

    # range of aoas
    aoas = np.arange(0, 20, 1) / 180 * np.pi
    aoas = np.arange(3, 16, 10) / 180 * np.pi

    # analytical solution
    CL_th = 2 * np.pi * aoas / (1 + 2 / AR)
    CDi_th = CL_th**2 / np.pi / AR

    # numerical solution
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
        controlpoints_list,
    ) = calculating_cl_cd_for_alpha_range(aoas, span, AR, max_chord, n_sections)

    # plotting
    plotting(
        aoas,
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
        controlpoints_list,
    )

    logging.info(f"CL")
    logging.info(f"CL_th= {CL_th}")
    logging.info(f"CL_LLT = {CL_LLT}")
    logging.info(f"CL_LLT_new = {CL_LLT_new}")
    logging.info(f"CL_VSM = {CL_VSM}")
    logging.info(f"CL_VSM_new = {CL_VSM_new}")
    logging.info(f"CD")
    logging.info(f"CD_th= {CDi_th}")
    logging.info(f"CD_LLT = {CD_LLT}")
    logging.info(f"CD_LLT_new = {CD_LLT_new}")
    logging.info(f"CD_VSM = {CD_VSM}")
    logging.info(f"CD_VSM_new = {CD_VSM_new}")
