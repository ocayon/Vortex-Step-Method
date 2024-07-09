import numpy as np
import pytest
from VSM.WingGeometry import Wing, Section  # Replace with your actual module names


def test_wing_initialization():
    example_wing = Wing(n_panels=10)
    assert example_wing.n_panels == 10
    assert example_wing.spanwise_panel_distribution == "linear"
    np.testing.assert_array_equal(example_wing.spanwise_direction, np.array([0, 1, 0]))
    assert isinstance(example_wing.sections, list)
    assert len(example_wing.sections) == 0  # Initially, sections list should be empty


def test_add_section():
    example_wing = Wing(n_panels=10)
    # Test adding a section
    example_wing.add_section(np.array([0, 0, 0]), np.array([-1, 0, 0]), "inviscid")
    assert len(example_wing.sections) == 1

    # Test if the section was added correctly
    section = example_wing.sections[0]
    np.testing.assert_array_equal(section.LE_point, np.array([0, 0, 0]))
    np.testing.assert_array_equal(section.TE_point, np.array([-1, 0, 0]))
    assert section.aero_input == "inviscid"


def test_refine_aerodynamic_mesh():
    n_panels = 4
    span = 20

    ## Test linear distribution
    wing = Wing(n_panels, spanwise_panel_distribution="linear")
    wing.add_section([0, -span / 2, 0], [-1, -span / 2, 0], ["inviscid"])
    wing.add_section([0, span / 2, 0], [-1, span / 2, 0], ["inviscid"])
    sections = wing.refine_aerodynamic_mesh()

    # Each panel has 2 corresponding sections, so we expect n_panels + 1 sections
    assert len(sections) == wing.n_panels + 1

    # Numerical tests for analytical calculations
    # Assuming linear interpolation, check specific points
    for i in range(len(sections)):
        # Calculate expected points for linear interpolation
        expected_LE = np.array([0, -span / 2 + i * span / n_panels, 0])
        expected_TE = np.array([-1, -span / 2 + i * span / n_panels, 0])

        # Check interpolated points with adjusted tolerance
        np.testing.assert_allclose(sections[i].LE_point, expected_LE, rtol=1e-5)
        np.testing.assert_allclose(sections[i].TE_point, expected_TE, rtol=1e-4)

    ## Test cosine distribution
    wing = Wing(n_panels, spanwise_panel_distribution="cosine")
    wing.add_section([0, -span / 2, 0], [-1, -span / 2, 0], ["inviscid"])
    wing.add_section([0, span / 2, 0], [-1, span / 2, 0], ["inviscid"])
    sections = wing.refine_aerodynamic_mesh()

    # Each panel has 2 corresponding sections, so we expect n_panels + 1 sections
    assert len(sections) == wing.n_panels + 1

    # Numerical tests for cosine interpolation
    for i in range(len(sections)):
        # Cosine distribution formula for linear span
        theta = np.linspace(0, np.pi, n_panels + 1)
        expected_LE_y = -span / 2 * np.cos(theta)
        expected_LE = np.array([0, expected_LE_y[i], 0])
        expected_TE = np.array([-1, expected_LE_y[i], 0])

        # Check interpolated points with adjusted tolerance
        np.testing.assert_allclose(sections[i].LE_point, expected_LE, atol=1e-8)
        np.testing.assert_allclose(sections[i].TE_point, expected_TE, atol=1e-8)


def test_refine_aerodynamic_mesh_1_panel():
    n_panels = 1
    span = 20

    wing = Wing(n_panels, spanwise_panel_distribution="linear")
    wing.add_section([0, -span / 2, 0], [-1, -span / 2, 0], ["inviscid"])
    wing.add_section([0, span / 2, 0], [-1, span / 2, 0], ["inviscid"])

    sections = wing.refine_aerodynamic_mesh()

    # Each panel has 2 corresponding sections, so we expect n_panels + 1 sections
    assert len(sections) == wing.n_panels + 1

    np.testing.assert_array_equal(sections[0].LE_point, np.array([0, -span / 2, 0]))
    np.testing.assert_array_equal(sections[0].TE_point, np.array([-1, -span / 2, 0]))


def test_refine_aeordynamic_mesh_2_panel():
    n_panels = 2
    span = 20

    wing = Wing(n_panels, spanwise_panel_distribution="linear")
    wing.add_section([0, -span / 2, 0], [-1, -span / 2, 0], ["inviscid"])
    wing.add_section([0, span / 2, 0], [-1, span / 2, 0], ["inviscid"])

    sections = wing.refine_aerodynamic_mesh()

    # Each panel has 2 corresponding sections, so we expect n_panels + 1 sections
    assert len(sections) == wing.n_panels + 1

    np.testing.assert_array_equal(sections[0].LE_point, np.array([0, -span / 2, 0]))
    np.testing.assert_array_equal(sections[1].LE_point, np.array([0, 0, 0]))
    np.testing.assert_array_equal(sections[2].LE_point, np.array([0, span / 2, 0]))


def test_refine_aeordynamic_mesh_more_sections_than_panels():
    n_panels = 2
    span = 20

    wing = Wing(n_panels, spanwise_panel_distribution="linear")
    wing.add_section([0, -span / 2, 0], [-1, -span / 2, 0], ["inviscid"])
    wing.add_section([0, -span / 3, 0], [-1, -span / 3, 0], ["inviscid"])
    wing.add_section([0, -span / 4, 0], [-1, -span / 4, 0], ["inviscid"])
    wing.add_section([0, 0, 0], [-1, 0, 0], ["inviscid"])
    wing.add_section([0, span / 4, 0], [-1, span / 4, 0], ["inviscid"])
    wing.add_section([0, span / 2, 0], [-1, span / 2, 0], ["inviscid"])

    sections = wing.refine_aerodynamic_mesh()

    # Each panel has 2 corresponding sections, so we expect n_panels + 1 sections
    assert len(sections) == wing.n_panels + 1

    for i in range(len(sections)):
        # Calculate expected points for linear interpolation
        expected_LE = np.array([0, -span / 2 + i * span / n_panels, 0])
        expected_TE = np.array([-1, -span / 2 + i * span / n_panels, 0])

        # Check interpolated points with adjusted tolerance
        np.testing.assert_allclose(sections[i].LE_point, expected_LE, rtol=1e-5)
        np.testing.assert_allclose(sections[i].TE_point, expected_TE, rtol=1e-4)


def test_refine_aerodynamic_mesh_for_different_LE_and_TE_distances():
    n_panels = 2
    span = 20

    wing = Wing(n_panels, spanwise_panel_distribution="linear")
    wing.add_section([0, -10, 0], [-1, -10, 0], ["inviscid"])
    wing.add_section([0, 5, 0], [-1, 10, 0], ["inviscid"])

    sections = wing.refine_aerodynamic_mesh()

    expected_LE_y = [-10, -2.5, 5]

    for i, section in enumerate(sections):
        # Calculate expected points for linear interpolation
        expected_LE = np.array([0, expected_LE_y[i], 0])
        expected_TE = np.array([-1, -span / 2 + i * span / n_panels, 0])

        # Check interpolated points with adjusted tolerance
        np.testing.assert_allclose(section.LE_point, expected_LE, rtol=1e-5)
        np.testing.assert_allclose(section.TE_point, expected_TE, rtol=1e-4)


def test_refine_aeordynamic_mesh_lei_airfoil_interpolation():
    n_panels = 4
    span = 20

    wing = Wing(n_panels, spanwise_panel_distribution="linear")
    wing.add_section(
        [0, -span / 2, 0], [-1, -span / 2, 0], ["lei_airfoil_breukels", [0, 0]]
    )
    wing.add_section([0, 0, 0], [-1, 0, 0], ["lei_airfoil_breukels", [2, 0.5]])
    wing.add_section(
        [0, span / 2, 0], [-1, span / 2, 0], ["lei_airfoil_breukels", [4, 1]]
    )

    sections = wing.refine_aerodynamic_mesh()

    # Each panel has 2 corresponding sections, so we expect n_panels + 1 sections
    assert len(sections) == wing.n_panels + 1

    expected_tube_diameter = np.linspace(0, 4, n_panels + 1)
    expected_chamber_height = np.linspace(0, 1, n_panels + 1)
    print(f"expected_tube_diameter: {expected_tube_diameter}")
    print(f"expected_chamber_height: {expected_chamber_height}")

    for i, section in enumerate(sections):
        # Calculate expected points for linear interpolation
        expected_LE = np.array([0, -span / 2 + i * span / n_panels, 0])
        expected_TE = np.array([-1, -span / 2 + i * span / n_panels, 0])

        # Check interpolated points with adjusted tolerance
        np.testing.assert_allclose(section.LE_point, expected_LE, rtol=1e-5)
        np.testing.assert_allclose(section.TE_point, expected_TE, rtol=1e-4)

        # Check if the airfoil data is correctly interpolated
        aero_input_i = section.aero_input
        assert np.array_equal(str(aero_input_i[0]), "lei_airfoil_breukels")

        tube_diam = aero_input_i[1][0]
        chamber_height = aero_input_i[1][1]
        assert np.isclose(tube_diam, expected_tube_diameter[i])
        assert np.isclose(chamber_height, expected_chamber_height[i])


if __name__ == "__main__":
    pytest.main()
