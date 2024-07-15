# -*- coding: utf-8 -*-
import numpy as np
import logging
import sys
import os

# Go back to root folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, root_path)
import tests.utils as test_utils


def test_curved():
    (
        max_chord,
        span,
        dist,
        N,
        aoas,
        wing_type,
        data_airf,
        Umag,
        AR,
        max_iterations,
        allowed_error,
        relaxation_factor,
        core_radius_fraction,
        data_airf,
    ) = get_elliptical_case_params()

    # OLD numerical
    CL_LLT, CD_LLT, CL_VSM, CD_VSM, gamma_LLT, gamma_VSM = (
        test_utils.calculate_old_for_alpha_range(
            [max_chord, span, N, dist],
            Umag,
            aoas,
            wing_type,
            data_airf,
            max_iterations,
            allowed_error,
            relaxation_factor,
            core_radius_fraction,
        )
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
        [max_chord, span, N, dist],
        Umag,
        aoas,
        wing_type,
        data_airf,
        max_iterations,
        allowed_error,
        relaxation_factor,
        core_radius_fraction,
        is_plotting=False,
    )
    for aoa in aoas:
        aoa_deg = np.rad2deg(aoa)
        # checking all LLTs to be close
        assert np.allclose(CL_LLT, CL_LLT_new, atol=1e-2)
        assert np.allclose(CD_LLT, CD_LLT_new, atol=1e-4)
        assert np.allclose(gamma_LLT, gamma_LLT_new, atol=1e-2)

        # checking VSMs to be close to one another
        assert np.allclose(CL_VSM, CL_VSM_new, atol=1e-2)
        assert np.allclose(CD_VSM, CD_VSM_new, atol=1e-4)

        # checking the LLT to be close to the VSM, with HIGHER tolerance
        tol_llt_to_vsm_CL = 1e-1
        tol_llt_to_vsm_CD = 1e-3
        assert np.allclose(CL_LLT, CL_VSM, atol=tol_llt_to_vsm_CL)
        assert np.allclose(CD_LLT, CD_VSM, atol=tol_llt_to_vsm_CD)
        assert np.allclose(CL_LLT_new, CL_VSM_new, atol=tol_llt_to_vsm_CL)
        assert np.allclose(CD_LLT_new, CD_VSM_new, atol=tol_llt_to_vsm_CD)


def get_curved_case_params():

    max_chord = 2.18
    span = 6.969
    dist = "lin"
    N = 60
    aoas = np.arange(-4, 24, 1) / 180 * np.pi
    aoas = np.deg2rad([5, 10])
    wing_type = "curved"
    Umag = 20
    AR = span / max_chord
    Atot = 16
    R = 4.673
    theta = 45 * np.pi / 180

    coord_input_params = [max_chord, span, theta, R, N, dist]
    # convergence criteria
    max_iterations = 1500
    allowed_error = 1e-5
    relaxation_factor = 0.03
    core_radius_fraction = 1e-20

    data_airf = np.loadtxt(r"./polars/clarky_maneia.csv", delimiter=",")

    return (
        coord_input_params,
        aoas,
        wing_type,
        data_airf,
        Umag,
        AR,
        Atot,
        max_iterations,
        allowed_error,
        relaxation_factor,
        core_radius_fraction,
        data_airf,
    )


if __name__ == "__main__":

    ## params
    (
        coord_input_params,
        aoas,
        wing_type,
        data_airf,
        Umag,
        AR,
        Atot,
        max_iterations,
        allowed_error,
        relaxation_factor,
        core_radius_fraction,
        data_airf,
    ) = get_curved_case_params()

    # comparing solution
    polars_Maneia = np.loadtxt("./polars/curved_wing_polars_maneia.csv", delimiter=",")

    # OLD numerical
    CL_LLT, CD_LLT, CL_VSM, CD_VSM, gamma_LLT, gamma_VSM = (
        test_utils.calculate_old_for_alpha_range(
            coord_input_params,
            Umag,
            Atot,
            aoas,
            wing_type,
            data_airf,
            max_iterations,
            allowed_error,
            relaxation_factor,
            core_radius_fraction,
        )
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
        coord_input_params,
        Umag,
        Atot,
        aoas,
        wing_type,
        data_airf,
        max_iterations,
        allowed_error,
        relaxation_factor,
        core_radius_fraction,
        is_plotting=False,
    )
    test_utils.plotting_CL_CD_gamma_LLT_VSM_old_new_comparison(
        panel_y=panel_y,
        AR=AR,
        wing_type="curved",
        aoas=[polars_Maneia[:, 0], aoas],
        CL_list=[polars_Maneia[:, 1], CL_LLT, CL_LLT_new, CL_VSM, CL_VSM_new],
        CD_list=[polars_Maneia[:, 2], CD_LLT, CD_LLT_new, CD_VSM, CD_VSM_new],
        gamma_list=[gamma_LLT, gamma_LLT_new, gamma_VSM, gamma_VSM_new],
        labels=["Analytic LLT", "LLT", "LLT_new", "VSM", "VSM_new"],
    )
    # labels = ["Polars Maneia", "LLT", "LLT_new", "VSM", "VSM_new"]
    # CL_list = [polars_Maneia[:, 1], CL_LLT, CL_LLT_new, CL_VSM, CL_VSM_new]
    # CD_list = [polars_Maneia[:, 2], CD_LLT, CD_LLT_new, CD_VSM, CD_VSM_new]
    # for i, aoa in enumerate(aoas):
    #     print(f"aoa = {np.rad2deg(aoa)}")
    #     for label, CD, CL in zip(labels, CD_list, CL_list):
    #         print(f"{label}: CL = {CL[i]}, CD = {CD[i]}")
