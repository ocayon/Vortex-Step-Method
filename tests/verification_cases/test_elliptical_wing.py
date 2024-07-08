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
    y_coords = cosspace(-span / 2, span / 2, 40)
    x_chords = x_coords(y_coords, chord_root, span)

    for y, x in zip(y_coords, x_chords):
        wing.add_section([0.25 * x, y, 0], [-0.75 * x, y, 0], ["inviscid"])

    # Initialize wing aerodynamics
    wing_aero = WingAerodynamics([wing])
    aoa_deg = np.linspace(0, 10, 2)  # (0, 15, 10)  # Angles of attack in degrees
    results = []

    # plotting the geometry wing
    if plot_wing:
        wing_aero.va = np.array([-Umag, 0, 0])
        wing_aero.plot()

    for aoa in aoa_deg:
        aoa_rad = np.deg2rad(aoa)
        Uinf = np.array([np.cos(aoa_rad), 0, -np.sin(aoa_rad)]) * -Umag
        wing_aero.va = Uinf

        wing_aero.plot()

        # Initialize solvers
        VSM = Solver(aerodynamic_model_type="VSM")
        LLT = Solver(aerodynamic_model_type="LLT")

        # Solve the aerodynamics
        results_VSM, wing_aero_VSM = VSM.solve(wing_aero)
        results_LLT, wing_aero_LLT = LLT.solve(wing_aero)

        # Calculate results
        calculate_results_output_VSM = wing_aero_VSM.calculate_results(density)
        calculate_results_output_LLT = wing_aero_LLT.calculate_results(density)

        # Analytical solutions
        CL_analytic = (2 * np.pi * aoa_rad) / (1 + 2 / AR)
        CDi_analytic = CL_analytic**2 / (np.pi * AR)

        # Extract the results
        CL_wing_VSM = calculate_results_output_VSM["cl_wing"]
        CD_wing_VSM = calculate_results_output_VSM["cd_wing"]
        CL_wing_LLT = calculate_results_output_LLT["cl_wing"]
        CD_wing_LLT = calculate_results_output_LLT["cd_wing"]

        # Print results
        print(f"Angle of Attack: {aoa} deg")
        print(
            f"Analytical CL: {CL_analytic}, VSM CL: {CL_wing_VSM}, LLT CL: {CL_wing_LLT}"
        )
        print(
            f"Analytical CDi: {CDi_analytic}, VSM CD: {CD_wing_VSM}, LLT CD: {CD_wing_LLT}"
        )

        # Store results for comparison
        results.append(
            (
                aoa,
                CL_wing_VSM,
                CL_wing_LLT,
                CL_analytic,
                CD_wing_VSM,
                CD_wing_LLT,
                CDi_analytic,
            )
        )

    return results


def test_elliptic_wing():
    results = calculate_elliptical_wing(20)

    # Compare results
    for (
        aoa,
        CL_wing_VSM,
        CL_wing_LLT,
        CL_analytic,
        CD_wing_VSM,
        CD_wing_LLT,
        CDi_analytic,
    ) in results:
        np.testing.assert_allclose(
            CL_wing_VSM,
            CL_analytic,
            rtol=1e-2,
            err_msg=f"Failed at AoA = {aoa} for VSM",
        )
        np.testing.assert_allclose(
            CL_wing_LLT,
            CL_analytic,
            rtol=1e-2,
            err_msg=f"Failed at AoA = {aoa} for LLT",
        )
        np.testing.assert_allclose(
            CD_wing_VSM,
            CDi_analytic,
            rtol=1e-2,
            err_msg=f"Failed at AoA = {aoa} for VSM",
        )
        np.testing.assert_allclose(
            CD_wing_LLT,
            CDi_analytic,
            rtol=1e-2,
            err_msg=f"Failed at AoA = {aoa} for LLT",
        )


def plot_elliptic_wing(n_panels=20, plot_wing=False):
    results = calculate_elliptical_wing(n_panels, plot_wing)

    # Extract results for plotting
    (
        aoa_list,
        CL_wing_VSM_list,
        CL_wing_LLT_list,
        CL_analytic_list,
        CD_wing_VSM_list,
        CD_wing_LLT_list,
        CDi_analytic_list,
    ) = zip(*results)

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
