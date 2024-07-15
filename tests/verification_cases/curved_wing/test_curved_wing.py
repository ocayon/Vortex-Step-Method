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
        controlpoints, rings, bladepanels, ringvec, coord_L = (
            thesis_functions.create_geometry_general(coord, Uinf, N, ring_geo, model)
        )
        Fmag, Gamma, aero_coeffs = (
            thesis_functions.solve_lifting_line_system_matrix_approach_semiinfinite(
                ringvec, controlpoints, rings, Uinf, Gamma0, data_airf, conv_crit, model
            )
        )
        F_rel, F_gl, Ltot, Dtot, CL1[i], CD1[i], CS = thesis_functions.output_results(
            Fmag, aero_coeffs, ringvec, Uinf, controlpoints, Atot
        )
        Gamma_LLT.append(Gamma)
        
        model = "VSM"
        controlpoints, rings, bladepanels, ringvec, coord_L = (
            thesis_functions.create_geometry_general(coord, Uinf, N, ring_geo, model)
        )
        Fmag, Gamma, aero_coeffs = (
            thesis_functions.solve_lifting_line_system_matrix_approach_semiinfinite(
                ringvec, controlpoints, rings, Uinf, Gamma0, data_airf, conv_crit, model
            )
        )
        Gamma_VSM.append(Gamma)
        F_rel, F_gl, Ltot, Dtot, CL2[i], CD2[i], CS = thesis_functions.output_results(
            Fmag, aero_coeffs, ringvec, Uinf, controlpoints, Atot
        )

        Gamma0 = Gamma
        print(str((i + 1) / len(aoas) * 100) + " %")
    
    end_time = time.time()
    print('Time employed: ' + str(end_time - start_time) + ' seconds')

    return CL1, CD1, CL2, CD2, Gamma_LLT, Gamma_VSM


def calculate_NEW_for_alpha_range(N, max_chord, span, AR, Umag, aoas, is_plotting=False):
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
        LLT = Solver(aerodynamic_model_type="LLT", core_radius_fraction=
