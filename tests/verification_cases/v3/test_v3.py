# -*- coding: utf-8 -*-
import numpy as np
import logging
import sys
import os
import matplotlib.pyplot as plt
import pytest

# Go back to root folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, root_path)
import tests.utils as test_utils
import tests.thesis_functions_oriol_cayon as thesis_functions


@pytest.fixture(autouse=True)
def change_test_dir(request):
    # Get the directory of the current test file
    test_dir = os.path.dirname(request.fspath)
    # Change the working directory to the test file's directory
    os.chdir(test_dir)


def get_v3_case_params():

    wing_type = "LEI_kite"
    dist = "lin"
    N_split = 4
    aoas = np.arange(-4, 24, 2)
    Umag = 22
    Atot = 19.753
    # convergence criteria
    max_iterations = 1500
    allowed_error = 1e-5
    relaxation_factor = 0.03
    core_radius_fraction = 1e-20

    # Wing geometry
    coord_struc = thesis_functions.get_CAD_matching_uri()
    coord = thesis_functions.struct2aero_geometry(coord_struc) / 1000

    N = len(coord) // 2

    # LE thickness at each section [m]
    # 10 sections
    LE_thicc = 0.1 

    # Camber for each section (ct in my case)
    camber = 0.095

    # Number of splits per panel
    N_split = 5
    # Refine structrural mesh into more panels
    coord = thesis_functions.refine_LEI_mesh(coord, N - 1, N_split)
    N = int(len(coord) / 2)  # Number of section after refining the mesh

    # Definition of airfoil coefficients
    # Based on Breukels (2011) correlation model
    aoas = np.arange(-20, 21, 1)
    data_airf = np.empty((len(aoas), 4))
    t_c = np.empty(N - 1)
    for j in range(len(aoas)):
        alpha = aoas[j]
        Cl, Cd, Cm = thesis_functions.LEI_airf_coeff(LE_thicc, camber, alpha)
        data_airf[j, 0] = alpha
        data_airf[j, 1] = Cl
        data_airf[j, 2] = Cd
        data_airf[j, 3] = Cm
    coord_input_params = [coord,LE_thicc,camber]
    case_parameters = [
        coord_input_params,
        aoas,
        wing_type,
        Umag,
        0,
        Atot,
        max_iterations,
        allowed_error,
        relaxation_factor,
        core_radius_fraction,
        data_airf,
    ]

    return case_parameters


def test_v3():
    case_params = get_v3_case_params()
    # making sure not too many points are tested
    case_params[1] = np.deg2rad(np.array([3, 6, 9]))
    # comparison solution
    aoas = case_params[1]
    # OLD numerical
    CL_LLT, CD_LLT, CL_VSM, CD_VSM, gamma_LLT, gamma_VSM = (
        test_utils.calculate_old_for_alpha_range(case_params)
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
    ) = test_utils.calculate_new_for_alpha_range(
        case_params,
        is_plotting=False,
    )
    # checking LTT old close to LLT new
    assert np.allclose(CL_LLT, CL_LLT_new, atol=2e-2)
    assert np.allclose(CD_LLT, CD_LLT_new, atol=2e-3)

    # checking VSMs to be close to one another
    assert np.allclose(CL_VSM, CL_VSM_new, atol=4e-2)
    assert np.allclose(CD_VSM, CD_VSM_new, atol=4e-3)

    # checking the LLT to be close to the VSM, with HIGHER tolerance
    tol_llt_to_vsm_CL = 2e-1
    tol_llt_to_vsm_CD = 4e-2
    assert np.allclose(CL_LLT, CL_VSM, atol=tol_llt_to_vsm_CL)
    assert np.allclose(CD_LLT, CD_VSM, atol=tol_llt_to_vsm_CD)
    assert np.allclose(CL_LLT_new, CL_VSM_new, atol=tol_llt_to_vsm_CL)
    assert np.allclose(CD_LLT_new, CD_VSM_new, atol=tol_llt_to_vsm_CD)

    polars_CFD = np.loadtxt("./CFD_DATA/CFD_data.csv", delimiter=",")
    alpha_CFD = polars_CFD[:, 0]
    CL_CFD = polars_CFD[:, 1]
    CD_CFD = polars_CFD[:, 2]
    # interpolating Maneia results to match the alphas
    CL_maneia_at_new_alphas = []
    CD_maneia_at_new_alphas = []
    for alpha in aoas:
        alpha = np.rad2deg(alpha)
        CL_maneia_at_new_alphas.append(np.interp(alpha, alpha_CFD, CL_CFD))
        CD_maneia_at_new_alphas.append(np.interp(alpha, alpha_CFD, CD_CFD))

    # checking new VSM close to Maneia
    assert np.allclose(CL_maneia_at_new_alphas, CL_VSM_new, atol=1e-1)
    assert np.allclose(CD_maneia_at_new_alphas, CD_VSM_new, atol=1e-2)


if __name__ == "__main__":

    case_params = get_v3_case_params()

    aoas = np.deg2rad(np.arange(0,1, 1))
    case_params[1] = aoas
    # comparing solution
    CL_struts = np.loadtxt("./CFD_data/RANS_CL_alpha_struts.csv", delimiter=",")
    CD_struts = np.loadtxt("./CFD_data/RANS_CD_alpha_struts.csv", delimiter=",")

    CL_CFD = CL_struts[:, 1]
    aoas_CFD = CL_struts[:, 0]
    CD_CFD = np.interp(aoas_CFD, CD_struts[:, 0], CD_struts[:, 1])
    polars_CFD = np.vstack((aoas_CFD,CL_CFD, CD_CFD)).T
    # OLD numerical
    CL_LLT, CD_LLT, CL_VSM, CD_VSM, gamma_LLT, gamma_VSM = (
        test_utils.calculate_old_for_alpha_range(case_params)
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
    ) = test_utils.calculate_new_for_alpha_range(
        case_params,
        is_plotting=False,
    )

    test_utils.plotting_CL_CD_gamma_LLT_VSM_old_new_comparison(
        panel_y=panel_y,
        AR=0,
        wing_type="LEI_kite",
        aoas=[polars_CFD[:, 0], aoas],
        CL_list=[polars_CFD[:, 1], CL_LLT, CL_LLT_new, CL_VSM, CL_VSM_new],
        CD_list=[polars_CFD[:, 2], CD_LLT, CD_LLT_new, CD_VSM, CD_VSM_new],
        gamma_list=[gamma_LLT, gamma_LLT_new, gamma_VSM, gamma_VSM_new],
        labels=["RANS [Lebesque]", "LLT", "LLT_new", "VSM", "VSM_new"],
    )
    labels = ["Polars CFD", "LLT", "LLT_new", "VSM", "VSM_new"]
    CL_list = [polars_CFD[:, 1], CL_LLT, CL_LLT_new, CL_VSM, CL_VSM_new]
    CD_list = [polars_CFD[:, 2], CD_LLT, CD_LLT_new, CD_VSM, CD_VSM_new]
    for i, aoa in enumerate(aoas):
        print(f"aoa = {np.rad2deg(aoa)}")
        for label, CD, CL in zip(labels, CD_list, CL_list):
            print(f"{label}: CL = {CL}, CD = {CD}")

    plt.show()
