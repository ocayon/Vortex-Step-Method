import pytest
import numpy as np
from VSM.Solver import Solver
from VSM.Panel import Panel
from VSM.WingAerodynamics import WingAerodynamics
from VSM.WingGeometry import Wing


def vec_norm(v):
    """
    Norm of a vector

    """
    return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def dot_product(r1, r2):
    """
    Dot product between r1 and r2

    """
    return r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]


def vector_projection(v, u):
    """
    Find the projection of a vector into a direction

    Parameters
    ----------
    v : vector to be projected
    u : direction

    Returns
    -------
    proj : projection of the vector v onto u

    """
    # Inputs:
    #     u = direction vector
    #     v = vector to be projected

    unit_u = u / np.linalg.norm(u)
    proj = np.dot(v, unit_u) * unit_u

    return proj


def output_results(Fmag, aero_coeffs, ringvec, Uinf, controlpoints, Atot, rho=1.225):
    """
    Post-process results to get global forces and aerodynamic coefficients

    Parameters
    ----------
    Fmag : Lift, Drag and Moment magnitudes
    aero_coeffs : alpha, cl, cd, cm
    ringvec : List of dictionaries containing the vectors that define each ring
    Uinf : Wind speed velocity vector
    controlpoints : List of dictionaries with the variables needed to define each wing section
    Atot : Planform area

    Returns
    -------
    F_rel : Lift and drag forces relative to the local angle of attack
    F_gl : Lift and drag forces relative to the wind direction
    Ltot : Total lift
    Dtot : Total drag
    CL : Global CL
    CD : Global CD

    """
    alpha = aero_coeffs[:, 0]
    F_rel = []
    F_gl = []
    Fmag_gl = []
    SideF = []
    Ltot = 0
    Dtot = 0
    SFtot = 0
    for i in range(len(alpha)):

        r0 = ringvec[i]["r0"]
        # Relative wind speed direction
        dir_urel = (
            np.cos(alpha[i]) * controlpoints[i]["tangential"]
            + np.sin(alpha[i]) * controlpoints[i]["normal"]
        )
        dir_urel = dir_urel / np.linalg.norm(dir_urel)
        # Lift direction relative to Urel
        dir_L = np.cross(dir_urel, r0)
        dir_L = dir_L / np.linalg.norm(dir_L)
        # Drag direction relative to Urel
        dir_D = np.cross([0, 1, 0], dir_L)
        dir_D = dir_D / np.linalg.norm(dir_D)
        # Lift and drag relative to Urel
        L_rel = dir_L * Fmag[i, 0]
        D_rel = dir_D * Fmag[i, 1]
        F_rel.append([L_rel, D_rel])
        # Lift direction relative to the wind speed
        dir_L_gl = np.cross(Uinf, [0, 1, 0])
        dir_L_gl = dir_L_gl / vec_norm(dir_L_gl)
        # Lift and drag relative to the windspeed
        L_gl = vector_projection(L_rel, dir_L_gl) + vector_projection(D_rel, dir_L_gl)
        D_gl = vector_projection(L_rel, Uinf) + vector_projection(D_rel, Uinf)
        F_gl.append([L_gl, D_gl])
        Fmag_gl.append(
            [
                dot_product(L_rel, dir_L_gl) + dot_product(D_rel, dir_L_gl),
                dot_product(L_rel, Uinf / vec_norm(Uinf))
                + dot_product(D_rel, Uinf / vec_norm(Uinf)),
            ]
        )
        SideF.append(dot_product(L_rel, [0, 1, 0]) + dot_product(D_rel, [0, 1, 0]))

    # Calculate total aerodynamic forces
    for i in range(len(Fmag_gl)):
        Ltot += Fmag_gl[i][0] * np.linalg.norm(ringvec[i]["r0"])
        Dtot += Fmag_gl[i][1] * np.linalg.norm(ringvec[i]["r0"])
        SFtot += SideF[i] * np.linalg.norm(ringvec[i]["r0"])

    Umag = np.linalg.norm(Uinf)
    CL = Ltot / (0.5 * Umag**2 * Atot * rho)
    CD = Dtot / (0.5 * Umag**2 * Atot * rho)
    CS = SFtot / (0.5 * Umag**2 * Atot * rho)

    return F_rel, F_gl, Ltot, Dtot, CL, CD, CS


def test_calculate_results():
    # Setup
    density = 1.225  # kg/m^3

    # Create a wing
    wing = Wing(n_panels=10)  # Using 3 panels for simplicity in this test

    # Add sections to the wing
    span = 20
    wing.add_section([0, -span / 2, 0], [-1, -span / 2, 0], ["inviscid"])
    wing.add_section([0, 0, 0], [-1, 0, 0], ["inviscid"])
    wing.add_section([0, span / 2, 0], [-1, span / 2, 0], ["inviscid"])

    # Initialize wing aerodynamics
    wing_aero = WingAerodynamics([wing])

    # Define inflow conditions
    Umag = 20
    aoa = 3 * np.pi / 180
    Uinf = np.array([np.cos(aoa), 0, -np.sin(aoa)]) * -Umag
    wing_aero.va = Uinf

    # Initialize solver
    VSM = Solver(aerodynamic_model_type="VSM")

    # Solve the aerodynamics
    results_VSM, wing_aero_VSM = VSM.solve(wing_aero)

    # Now test the calculate_results method
    calculate_results_output = wing_aero_VSM.calculate_results(density)

    # Check the type and structure of the output
    assert isinstance(
        calculate_results_output, dict
    ), "calculate_results should return a dictionary"

    # Extract the results from the dictionary
    results_dict = calculate_results_output

    # Prepare inputs for the reference function
    """
        Fmag: Lift, Drag, and Moment magnitudes.
        aero_coeffs: alpha, cl, cd, cm.
        ringvec: List of dictionaries containing the vectors that define each ring.
        Uinf: Wind speed velocity vector.
        controlpoints: List of dictionaries with the variables needed to define each wing section.
        Atot: Planform area.
    """

    # Calculating Fmag, using UNCORRECTED alpha
    alpha = results_VSM["alpha_uncorrected"]
    dyn_visc = 0.5 * density * np.linalg.norm(Uinf) ** 2
    n_panels = len(wing_aero_VSM.panels)
    lift, drag, moment = np.zeros(n_panels), np.zeros(n_panels), np.zeros(n_panels)
    for i, panel in enumerate(wing_aero_VSM.panels):
        lift[i] = dyn_visc * panel.calculate_cl(alpha[i]) * panel.chord
        drag[i] = dyn_visc * panel.calculate_cd_cm(alpha[i])[0] * panel.chord
        moment[i] = dyn_visc * panel.calculate_cd_cm(alpha[i])[1] * (panel.chord**2)
        print("lift:", lift, "drag:", drag, "moment:", moment)
    Fmag = np.column_stack([lift, drag, moment])

    # Calculating aero_coeffs, using CORRECTED alpha
    alpha = results_VSM["alpha_at_ac"]
    aero_coeffs = np.column_stack(
        (
            [alpha[i] for i, panel in enumerate(wing_aero_VSM.panels)],
            [
                panel.calculate_cl(alpha[i])
                for i, panel in enumerate(wing_aero_VSM.panels)
            ],
            [
                panel.calculate_cd_cm(alpha[i])[0]
                for i, panel in enumerate(wing_aero_VSM.panels)
            ],
            [
                panel.calculate_cd_cm(alpha[i])[1]
                for i, panel in enumerate(wing_aero_VSM.panels)
            ],
        )
    )
    ringvec = [{"r0": panel.z_airf} for panel in wing_aero_VSM.panels]
    controlpoints = [
        {"tangential": panel.y_airf, "normal": panel.x_airf}
        for panel in wing_aero_VSM.panels
    ]
    Atot = sum(
        panel.chord * np.linalg.norm(panel.z_airf) for panel in wing_aero_VSM.panels
    )

    # printing the inputs
    print("Fmag:", Fmag)
    print("aero_coeffs:", aero_coeffs)
    print("ringvec:", ringvec)
    print("Uinf:", Uinf)
    print("controlpoints:", controlpoints)
    print("Atot:", Atot)
    print("density:", density)

    # Calculate results using the reference function
    F_rel_ref, F_gl_ref, Ltot_ref, Dtot_ref, CL_ref, CD_ref, CS_ref = output_results(
        Fmag, aero_coeffs, ringvec, Uinf, controlpoints, Atot, density
    )
    # Debug: Print the compared results
    print("CL_ref:", CL_ref, "CL_wing:", results_dict["cl_wing"])
    print("CD_ref:", CD_ref, "CD_wing:", results_dict["cd_wing"])
    print("CS_ref:", CS_ref, "CS_wing:", results_dict["cs_wing"])
    print("Ltot_ref:", Ltot_ref, "Lift_wing:", results_dict["lift_wing"])
    print("Dtot_ref:", Dtot_ref, "Drag_wing:", results_dict["drag_wing"])

    # Assert that the results are close
    np.testing.assert_allclose(results_dict["cl_wing"], CL_ref, rtol=1e-5)
    np.testing.assert_allclose(results_dict["cd_wing"], CD_ref, rtol=1e-5)
    np.testing.assert_allclose(results_dict["cs_wing"], CS_ref, rtol=1e-5)
    np.testing.assert_allclose(results_dict["lift_wing"], Ltot_ref, rtol=1e-5)
    np.testing.assert_allclose(results_dict["drag_wing"], Dtot_ref, rtol=1e-5)

    # Check the structure of the results_dict
    expected_keys = {
        "cl",
        "cd",
        "cm",
        "cl_wing",
        "cd_wing",
        "cs_wing",
        "cmx_wing",
        "cmy_wing",
        "cmz_wing",
        "lift_wing",
        "drag_wing",
        "side_wing",
        "mx_wing",
        "my_wing",
        "mz_wing",
        "alpha_at_ac",
        "alpha_uncorrected",
        "gamma_distribution",
    }
    assert (
        set(results_dict.keys()) == expected_keys
    ), f"Mismatch in results_dict keys. Expected: {expected_keys}, Got: {set(results_dict.keys())}"

    # Check the shape of array outputs
    assert len(results_dict["cl"]) == len(wing_aero_VSM.panels)
    assert len(results_dict["cd"]) == len(wing_aero_VSM.panels)
    assert len(results_dict["cm"]) == len(wing_aero_VSM.panels)

    # You may want to add more detailed checks for the array outputs if needed

    # Check that the wing_aero_VSM is updated
    assert hasattr(wing_aero_VSM, "_alpha_aerodynamic_center")
    assert hasattr(wing_aero_VSM, "_alpha_uncorrected")
    assert hasattr(wing_aero_VSM, "_gamma_distribution")

    # Optional: Check that these attributes have the expected length
    assert len(wing_aero_VSM._alpha_aerodynamic_center) == len(wing_aero_VSM.panels)
    assert len(wing_aero_VSM._alpha_uncorrected) == len(wing_aero_VSM.panels)
    assert len(wing_aero_VSM._gamma_distribution) == len(wing_aero_VSM.panels)
