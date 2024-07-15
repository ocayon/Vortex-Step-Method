# -*- coding: utf-8 -*-
import numpy as np
import logging
import sys
import os

# Go back to root folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, root_path)
import tests.utils as test_utils


def test_elliptical():
    wing_type = "elliptical"
    aoas = np.deg2rad([5, 10])
    N = 40
    max_chord = 1
    span = 15.709  # AR = 20
    # span = 2.36  # AR = 3
    Umag = 20
    AR = span**2 / (np.pi * span * max_chord / 4)
    dist = "cos"
    # convergence criteria
    max_iterations = 1500
    allowed_error = 1e-5
    relaxation_factor = 0.03
    core_radius_fraction = 1e-20

    # analytical solution
    CL_th = 2 * np.pi * aoas / (1 + 2 / AR)
    CDi_th = CL_th**2 / np.pi / AR
    # OLD numerical
    CL_LLT, CD_LLT, CL_VSM, CD_VSM, gamma_LLT, gamma_VSM = (
        test_utils.calculate_old_for_alpha_range(
            coord_input_params=[max_chord, span, N, dist],
            Umag=Umag,
            aoas=aoas,
            wing_type=wing_type,
            max_iterations=max_iterations,
            allowed_error=allowed_error,
            relaxation_factor=relaxation_factor,
            core_radius_fraction=core_radius_fraction,
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
        coord_input_params=[max_chord, span, N, dist],
        Umag=Umag,
        aoas=aoas,
        wing_type=wing_type,
        max_iterations=max_iterations,
        allowed_error=allowed_error,
        relaxation_factor=relaxation_factor,
        core_radius_fraction=core_radius_fraction,
        is_plotting=False,
    )
    for aoa in aoas:
        aoa_deg = np.rad2deg(aoa)
        # checking all LLTs to be close
        assert np.allclose(CL_th, CL_LLT, atol=1e-2)
        assert np.allclose(CDi_th, CD_LLT, atol=1e-4)
        assert np.allclose(CL_th, CL_LLT_new, atol=1e-2)
        assert np.allclose(CDi_th, CD_LLT_new, atol=1e-4)
        assert np.allclose(gamma_LLT, gamma_LLT_new, atol=1e-2)

        # checking VSMs to be close to one another
        assert np.allclose(CL_VSM, CL_VSM_new, atol=1e-2)
        assert np.allclose(CD_VSM, CD_VSM_new, atol=1e-4)

        # checking the LLT to be close to the VSM, with HIGHER tolerance
        tol_llt_to_vsm_CL = 1e-1
        tol_llt_to_vsm_CD = 1e-3
        assert np.allclose(CL_th, CL_VSM, atol=tol_llt_to_vsm_CL)
        assert np.allclose(CDi_th, CD_VSM, atol=tol_llt_to_vsm_CD)
        assert np.allclose(CL_th, CL_VSM_new, atol=tol_llt_to_vsm_CL)
        assert np.allclose(CDi_th, CD_VSM_new, atol=tol_llt_to_vsm_CD)
        assert np.allclose(CL_LLT, CL_VSM, atol=tol_llt_to_vsm_CL)
        assert np.allclose(CD_LLT, CD_VSM, atol=tol_llt_to_vsm_CD)
        assert np.allclose(CL_LLT_new, CL_VSM_new, atol=tol_llt_to_vsm_CL)
        assert np.allclose(CD_LLT_new, CD_VSM_new, atol=tol_llt_to_vsm_CD)


if __name__ == "__main__":

    max_chord = 1
    span = 15.709  # AR = 20
    # span = 2.36  # AR = 3
    dist = "cos"
    N = 40
    aoas = np.arange(0, 20, 1) / 180 * np.pi
    aoas = np.deg2rad([5, 10])
    wing_type = "elliptical"
    Umag = 20
    AR = span**2 / (np.pi * span * max_chord / 4)
    # convergence criteria
    max_iterations = 1500
    allowed_error = 1e-5
    relaxation_factor = 0.05
    core_radius_fraction = 1e-20

    # analytical solution
    CL_th = 2 * np.pi * aoas / (1 + 2 / AR)
    CDi_th = CL_th**2 / np.pi / AR
    # OLD numerical
    CL_LLT, CD_LLT, CL_VSM, CD_VSM, gamma_LLT, gamma_VSM = (
        test_utils.calculate_old_for_alpha_range(
            coord_input_params=[max_chord, span, N, dist],
            Umag=Umag,
            aoas=aoas,
            wing_type=wing_type,
            max_iterations=max_iterations,
            allowed_error=allowed_error,
            relaxation_factor=relaxation_factor,
            core_radius_fraction=core_radius_fraction,
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
        coord_input_params=[max_chord, span, N, dist],
        Umag=Umag,
        aoas=aoas,
        wing_type=wing_type,
        max_iterations=max_iterations,
        allowed_error=allowed_error,
        relaxation_factor=relaxation_factor,
        core_radius_fraction=core_radius_fraction,
        is_plotting=False,
    )

    logging.debug(f"CD_LLT_new = {CD_LLT_new}")
    logging.debug(f"CD_VSM_new = {CD_VSM_new}")

    for i, aoa in enumerate(aoas):
        print(f"aoa = {np.rad2deg(aoa)}")
        print(
            f"CL_th: {CL_th[i]}, CL_LLT: {CL_LLT[i]}, CL_VSM: {CL_LLT[i]}, CL_LLT_new: {CL_LLT_new[i]}, CL_VSM_new: {CL_VSM_new[i]}"
        )
        print(
            f"CDi_th: {CDi_th[i]}, CD_LLT: {CL_LLT[i]}, CD_VSM: {CL_VSM[i]}, CD_LLT_new: {CD_LLT_new[i]}, CD_VSM_new: {CD_VSM_new[i]}"
        )

    aoas_deg = np.rad2deg(aoas)
    test_utils.plotting(
        x_axis_list=[aoas_deg, aoas_deg, aoas_deg],
        y_axis_list=[CL_th, CL_LLT, CL_LLT_new],
        labels=["Analytic LLT", "LLT", "LLT_new"],
        x_label=r"$\alpha$ ($^\circ$)",
        y_label=r"$C_L$ ()",
        title="CL_alpha_LTT_elliptic_AR_" + str(round(AR, 1)),
        markers=None,
        alphas=None,
        colors=None,
        file_type=".pdf",
    )
    test_utils.plotting(
        x_axis_list=[aoas_deg, aoas_deg, aoas_deg],
        y_axis_list=[CDi_th, CL_LLT, CD_LLT_new],
        labels=["Analytic LLT", "LLT", "LLT_new"],
        x_label=r"$\alpha$ ($^\circ$)",
        y_label=r"$C_D$ ()",
        title="CD_alpha_LTT_elliptic_AR_" + str(round(AR, 1)),
        markers=None,
        alphas=None,
        colors=None,
        file_type=".pdf",
    )
    test_utils.plotting(
        x_axis_list=[aoas_deg, aoas_deg, aoas_deg, aoas_deg],
        y_axis_list=[CL_th, CL_VSM, CL_VSM_new],
        labels=["Analytic LLT", "VSM", "VSM_new"],
        x_label=r"$\alpha$ ($^\circ$)",
        y_label=r"$C_L$ ()",
        title="CL_alpha_VSM_elliptic_AR_" + str(round(AR, 1)),
        markers=None,
        alphas=None,
        colors=None,
        file_type=".pdf",
    )
    test_utils.plotting(
        x_axis_list=[aoas_deg, aoas_deg, aoas_deg, aoas_deg],
        y_axis_list=[CDi_th, CL_VSM, CD_VSM_new],
        labels=["Analytic LLT", "VSM", "VSM_new"],
        x_label=r"$\alpha$ ($^\circ$)",
        y_label=r"$C_D$ ()",
        title="CD_alpha_VSM_elliptic_AR_" + str(round(AR, 1)),
        markers=None,
        alphas=None,
        colors=None,
        file_type=".pdf",
    )
    # Plotting gamma
    idx = idx = int(len(aoas_deg) // 2)
    test_utils.plotting(
        x_axis_list=[panel_y, panel_y, panel_y, panel_y],
        y_axis_list=[
            gamma_LLT[idx],
            gamma_VSM[idx],
            gamma_LLT_new[idx],
            gamma_VSM_new[idx],
        ],
        labels=["LLT", "VSM", "LLT_new", "VSM_new"],
        x_label=r"$y$",
        y_label=r"$Gamma$",
        title="gamma_distribution_elliptic_AR_" + str(round(AR, 1)),
        markers=None,
        alphas=None,
        colors=None,
        file_type=".pdf",
    )
