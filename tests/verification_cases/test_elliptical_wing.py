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


def calculate_elliptical_wing(
    n_panels, AR, plot_wing=False, spacing="linear", aoa_deg=[3]
):
    chord_root = 1  # m
    # Aspect Ratio Equation Inverted: AR = span**2 / (chord_root * span / 4) p.49 eq (7.1)
    span = (np.pi * AR * chord_root) / 4
    print(f"Aspect ratio: {AR}, span: {span} m, chord: {chord_root} m")
    Umag = 10  # m/s

    # Create the wing
    wing = Wing(n_panels, spacing)  # Adjust the number of panels as needed

    # Add sections to the wing using cosine distribution
    y_coords = cosspace(-span / 2, span / 2, 10)
    x_chords = x_coords(y_coords, chord_root, span)

    for y, x in zip(y_coords, x_chords):
        wing.add_section([0.25 * x, y, 0], [-0.75 * x, y, 0], ["inviscid"])

    # # Initialize wing aerodynamics
    wing_aero = WingAerodynamics([wing])

    # plotting the geometry wing
    if plot_wing:
        wing_aero.plot()

    # Initialize gamma distributions, to speed up the solver
    VSM_gamma_distribution = None
    LLT_gamma_distribution = None

    results = []
    for aoa in aoa_deg:

        aoa_rad = np.deg2rad(aoa)
        Uinf = np.array([np.cos(aoa_rad), 0, -np.sin(aoa_rad)]) * -Umag
        wing_aero.va = Uinf

        # Initialize solvers
        VSM = Solver(
            aerodynamic_model_type="VSM",
            relaxation_factor=0.03,
            max_iterations=int(5e3),
        )
        LLT = Solver(
            aerodynamic_model_type="LLT",
            relaxation_factor=0.03,
            max_iterations=int(5e3),
        )

        # Solve the aerodynamics
        results_VSM, wing_aero_VSM = VSM.solve(wing_aero, VSM_gamma_distribution)
        results_LLT, wing_aero_LLT = LLT.solve(wing_aero, LLT_gamma_distribution)

        # Populate gamma_distributions for next iteration
        VSM_gamma_distribution = wing_aero_VSM.gamma_distribution
        LLT_gamma_distribution = wing_aero_LLT.gamma_distribution

        # Analytical solutions
        CL_analytic = (2 * np.pi * aoa_rad) / (1 + 2 / AR)
        CDi_analytic = CL_analytic**2 / (np.pi * AR)

        results.append([aoa, results_VSM, results_LLT, CL_analytic, CDi_analytic])

    return results


def test_elliptic_wing():
    aoa_deg = np.linspace(0, 15, 3)
    results = calculate_elliptical_wing(
        20, AR=20, plot_wing=False, spacing="cosine", aoa_deg=aoa_deg
    )

    for (
        aoa,
        result_VSM,
        result_LTT,
        CL_analytic,
        CDi_analytic,
    ) in results:

        print(f"Analytical AoA: {aoa}")
        print(f"Analytical CL: {CL_analytic}")
        print(f"VSM CL: {result_VSM['cl']}")
        print(f"LLT CL: {result_LTT['cl']}")
        print(f"VSM CDi: {result_VSM['cd']}")
        print(f"Analytical CDi: {CDi_analytic}")
        print(f"LLT CDi: {result_LTT['cd']}")

        np.testing.assert_allclose(
            result_VSM["cl"],
            CL_analytic,
            atol=1e-2,
            err_msg=f"Failed at AoA = {aoa} for VSM",
        )
        np.testing.assert_allclose(
            result_LTT["cl"],
            CL_analytic,
            atol=1e-2,
            err_msg=f"Failed at AoA = {aoa} for LLT",
        )
        np.testing.assert_allclose(
            result_VSM["cd"],
            CDi_analytic,
            atol=1e-3,
            err_msg=f"Failed at AoA = {aoa} for VSM",
        )
        np.testing.assert_allclose(
            result_LTT["cd"],
            CDi_analytic,
            atol=1e-3,
            err_msg=f"Failed at AoA = {aoa} for LLT",
        )


def plot_elliptic_wing(n_panels, AR, plot_wing=False, spacing="linear", aoa_deg=[3]):
    results = calculate_elliptical_wing(n_panels, AR, plot_wing, spacing, aoa_deg)

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
    ax[0].plot(
        aoa_list, CL_wing_LLT_list, "r-o", label="LLT CL", linewidth=2, alpha=0.3
    )
    ax[0].plot(aoa_list, CL_analytic_list, "b-x", label="Analytical LLT CL")
    ax[0].set_xlabel("Angle of Attack (deg)")
    ax[0].set_ylabel("CL")
    ax[0].legend()
    ax[0].grid()
    ax[0].set_title(f"Elliptic Wing Verification, AR:{AR}, n_panels:{n_panels}")

    ax[1].plot(aoa_list, CD_wing_VSM_list, "g-o", label="VSM CD")
    ax[1].plot(aoa_list, CD_wing_LLT_list, "r-o", label="LLT CD", alpha=0.3)
    ax[1].plot(aoa_list, CDi_analytic_list, "b-x", label="Analytical LLT CD")
    ax[1].set_xlabel("Angle of Attack (deg)")
    ax[1].set_ylabel("CD")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.savefig(
        f"elliptic_wing_verification_AR_{str(AR)}_n_panels_{str(int(n_panels))}.png"
    )
    # Save the plot as a file
    plt.show()  # Display the plot


if __name__ == "__main__":

    aoa_deg = np.linspace(0, 19, 19)
    aoa_deg = [3]
    # plot_elliptic_wing(60, AR=3, plot_wing=True, spacing="cosine", aoa_deg=aoa_deg)
    plot_elliptic_wing(40, AR=20, plot_wing=True, spacing="cosine", aoa_deg=aoa_deg)
