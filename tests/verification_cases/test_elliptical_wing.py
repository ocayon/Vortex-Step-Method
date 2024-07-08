import pytest
import numpy as np
import matplotlib.pyplot as plt
from VSM.Solver import Solver
from VSM.WingAerodynamics import WingAerodynamics
from VSM.WingGeometry import Wing


def cosspace(min, max, n_points):
    """
    Create an array with cosine spacing, from min to max values, with n points
    """
    mean = (max + min) / 2
    amp = (max - min) / 2
    return mean + amp * np.cos(np.linspace(np.pi, 0, n_points))


def x_coords(y_coords, chord_root, span):
    """
    Calculate chord lengths based on elliptical distribution.
    """
    return chord_root * np.sqrt(1 - (2 * y_coords / span) ** 2)


def calculate_elliptical_wing(n_panels=20, plot_wing=False):
    # Constants
    density = 1.225  # kg/m^3
    span = 5  # m
    chord_root = 1  # m
    AR = span**2 / (chord_root * span / 4)  # Aspect ratio
    Umag = 10  # m/s

    # Create the wing
    wing = Wing(n_panels)  # Adjust the number of panels as needed

    # Add sections to the wing using cosine distribution
    y_coords = cosspace(-span / 2, span / 2, 100)
    x_chords = x_coords(y_coords, chord_root, span)

    for y, x in zip(y_coords, x_chords):
        wing.add_section([0.25 * x, y, 0], [-0.75 * x, y, 0], ["inviscid"])

    # span = 20
    # wing.add_section([0, -span / 2, 0], [-1, -span / 2, 0], ["inviscid"])
    # wing.add_section([0, span / 2, 0], [-1, span / 2, 0], ["inviscid"])

    # Initialize wing aerodynamics
    wing_aero = WingAerodynamics([wing])
    aoa_deg = np.linspace(0, 15, 10)  # Angles of attack in degrees
    aoa_deg = [3, 6, 9]
    results = []

    # plotting the geometry wing
    if plot_wing:
        aoa_rad = np.deg2rad(3)
        wing_aero.va = np.array([np.cos(aoa_rad), 0, -np.sin(aoa_rad)]) * -Umag
        wing_aero.plot()

    for aoa in aoa_deg:
        aoa_rad = np.deg2rad(aoa)
        Uinf = np.array([np.cos(aoa_rad), 0, -np.sin(aoa_rad)]) * -Umag
        wing_aero.va = Uinf

        # Initialize solvers
        VSM = Solver(aerodynamic_model_type="VSM")
        LLT = Solver(aerodynamic_model_type="LLT")

        # Solve the aerodynamics
        results_VSM, wing_aero_VSM = VSM.solve(wing_aero)
        results_LLT, wing_aero_LLT = LLT.solve(wing_aero)

        # Calculate results
        results_VSM = wing_aero_VSM.calculate_results(density)
        results_LTT = wing_aero_LLT.calculate_results(density)

        # Analytical solutions
        CL_analytic = (2 * np.pi * aoa_rad) / (1 + 2 / AR)
        CDi_analytic = CL_analytic**2 / (np.pi * AR)

        results.append([aoa, results_VSM, results_LTT, CL_analytic, CDi_analytic])

    return results


# def test_elliptic_wing():
#     results = calculate_elliptical_wing(20)

#     for (
#         aoa,
#         result_VSM,
#         result_LTT,
#         CL_analytic,
#         CDi_analytic,
#     ) in results:
#         np.testing.assert_allclose(
#             result_VSM["cfz"],
#             CL_analytic,
#             rtol=1e-2,
#             err_msg=f"Failed at AoA = {aoa} for VSM",
#         )
#         np.testing.assert_allclose(
#             result_LTT["cfz"],
#             CL_analytic,
#             rtol=1e-2,
#             err_msg=f"Failed at AoA = {aoa} for LLT",
#         )
#         np.testing.assert_allclose(
#             result_VSM["cfx"],
#             CDi_analytic,
#             rtol=1e-2,
#             err_msg=f"Failed at AoA = {aoa} for VSM",
#         )
#         np.testing.assert_allclose(
#             result_LTT["cfx"],
#             CDi_analytic,
#             rtol=1e-2,
#             err_msg=f"Failed at AoA = {aoa} for LLT",
#         )


def plot_elliptic_wing(n_panels=20, plot_wing=False):
    results = calculate_elliptical_wing(n_panels, plot_wing)

    # Extract results for plotting
    aoa_list = []
    CL_wing_VSM_list = []
    CD_wing_VSM_list = []
    CL_wing_LLT_list = []
    CD_wing_LLT_list = []
    CL_analytic_list = []
    CDi_analytic_list = []

    for (
        aoa,
        result_VSM,
        result_LTT,
        CL_analytic,
        CDi_analytic,
    ) in results:
        aoa_list.append(aoa)
        CL_wing_VSM_list.append(result_VSM["cl"])
        CD_wing_VSM_list.append(result_VSM["cd"])
        CL_wing_LLT_list.append(result_LTT["cl"])
        CD_wing_LLT_list.append(result_LTT["cd"])
        CL_analytic_list.append(CL_analytic)
        CDi_analytic_list.append(CDi_analytic)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(aoa_list, CL_wing_VSM_list, "g-o", label="VSM CL")
    ax[0].plot(aoa_list, CL_wing_LLT_list, "r-o", label="LLT CL")
    ax[0].plot(aoa_list, CL_analytic_list, "b-x", label="Analytical LLT CL")
    ax[0].set_xlabel("Angle of Attack (deg)")
    ax[0].set_ylabel("CL")
    ax[0].legend()
    ax[0].set_title(f"Lift Coefficient Comparison (n_panels:{n_panels})")

    ax[1].plot(aoa_list, CD_wing_VSM_list, "g-o", label="VSM CD")
    ax[1].plot(aoa_list, CD_wing_LLT_list, "r-o", label="LLT CD")
    ax[1].plot(aoa_list, CDi_analytic_list, "b-x", label="Analytical LLT CD")
    ax[1].set_xlabel("Angle of Attack (deg)")
    ax[1].set_ylabel("CD")
    ax[1].legend()
    ax[1].set_title(f"Drag Coefficient Comparison(n_panels:{n_panels})")

    plt.tight_layout()
    plt.savefig(
        f"elliptic_wing_verification_n_panels_{str(int(n_panels))}.png"
    )  # Save the plot as a file
    plt.show()  # Display the plot


if __name__ == "__main__":
    # plot_elliptic_wing(3, True)
    plot_elliptic_wing(20, True)
