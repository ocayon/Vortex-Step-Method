import pytest
import numpy as np
from VSM.Panel import Panel  # Assuming the Panel class is in a file named Panel.py


# Mock Section class for testing
class MockSection:
    def __init__(self, LE_point, TE_point):
        self.LE_point = np.array(LE_point)
        self.TE_point = np.array(TE_point)
        self.aero_input = ["inviscid"]


@pytest.fixture
def sample_panel():
    section1 = MockSection([0, 0, 0], [-1, 0, 0])
    section2 = MockSection([0, 10, 0], [-1, 10, 0])
    return Panel(
        section1,
        section2,
        aerodynamic_center_location=0.25,
        control_point_location=0.75,
    )


def test_panel_initialization(sample_panel):
    assert isinstance(sample_panel, Panel)


def test_panel_te_le_points(sample_panel):
    assert np.allclose(sample_panel.TE_point_1, [-1, 0, 0])
    assert np.allclose(sample_panel.TE_point_2, [-1, 10, 0])
    assert np.allclose(sample_panel.LE_point_1, [0, 0, 0])
    assert np.allclose(sample_panel.LE_point_2, [0, 10, 0])


def test_panel_corner_points(sample_panel):
    expected_corner_points = np.array(
        [
            [0, 0, 0],  # LE_point_1
            [-1, 0, 0],  # TE_point_1
            [-1, 10, 0],  # TE_point_2
            [0, 10, 0],  # LE_point_2
        ]
    )
    assert np.allclose(sample_panel.corner_points, expected_corner_points)


def test_panel_chord(sample_panel):
    rib_1 = sample_panel.corner_points[1] - sample_panel.corner_points[0]
    norm_rib_1 = np.linalg.norm(rib_1)
    rib_2 = sample_panel.corner_points[3] - sample_panel.corner_points[2]
    norm_rib_2 = np.linalg.norm(rib_2)
    chord_sample_panel = (norm_rib_1 + norm_rib_2) / 2
    assert np.isclose(sample_panel.chord, chord_sample_panel)


def test_va_initialization(sample_panel):
    assert sample_panel.va is None


def test_panel_aerodynamic_center(sample_panel):
    expected_ac = np.array([-0.25, 5, 0])
    assert np.allclose(sample_panel.aerodynamic_center, expected_ac)


def test_panel_control_point(sample_panel):
    expected_control_point = np.array([-0.75, 5, 0])
    assert np.allclose(sample_panel.control_point, expected_control_point)


def test_panel_reference_frame(sample_panel):
    # Calculate the local reference frame
    # x_airf defined upwards from the chord-line, perpendicular to the panel
    # y_airf defined parallel to the chord-line, from LE-to-TE
    # z_airf along the LE, in plane (towards left tip, along span) from the airfoil perspective

    # you can think about subtracting of vectors like:
    # to get from A to B you do C = B - A, so work from backwards.

    LE_point_1 = sample_panel.LE_point_1
    LE_point_2 = sample_panel.LE_point_2
    TE_point_1 = sample_panel.TE_point_1
    TE_point_2 = sample_panel.TE_point_2

    print(f"LE_point_1: {LE_point_1}")
    print(f"LE_point_2: {LE_point_2}")
    print(f"TE_point_1: {TE_point_1}")
    print(f"TE_point_2: {TE_point_2}")
    print(f"rib_1 (LE - TE): {LE_point_1 - TE_point_1}")
    print(f"rib_2 (LE - TE): {LE_point_2 - TE_point_2}")
    print(f"rib_1 (TE - LE): {TE_point_1 - LE_point_1}")
    print(f"rib_2 (TE - LE): {TE_point_2 - LE_point_2}")

    mid_LE_point = LE_point_1 + 0.5 * (LE_point_2 - LE_point_1)
    mid_TE_point = TE_point_1 + 0.5 * (TE_point_2 - TE_point_1)
    vec_LE_to_TE = mid_TE_point - mid_LE_point

    print(f"mid_LE_point: {mid_LE_point}")
    print(f"mid_TE_point: {mid_TE_point}")
    print(f"vec_LE_to_TE: {vec_LE_to_TE}")

    y_airf = vec_LE_to_TE / np.linalg.norm(vec_LE_to_TE)
    bound_point_1 = LE_point_1 + 0.25 * (TE_point_1 - LE_point_1)
    bound_point_2 = LE_point_2 + 0.25 * (TE_point_2 - LE_point_2)

    print(f"section_1_aerodynamic_center/bound_point_1: {bound_point_1}")
    print(f"section_2_aerodynamic_center/bound_point_2: {bound_point_2}")
    z_airf = (bound_point_2 - bound_point_1) / np.linalg.norm(
        bound_point_2 - bound_point_1
    )
    x_airf = np.cross(y_airf, z_airf)

    print(f"vec_LE_to_TE: {vec_LE_to_TE}")
    print(f"x_airf: {x_airf}")
    print(f"y_airf: {y_airf}")
    print(f"z_airf: {z_airf}")

    # testing against expected values
    assert np.allclose(sample_panel.x_airf, [0, 0, -1])
    assert np.allclose(sample_panel.y_airf, [-1, 0, 0])
    assert np.allclose(sample_panel.z_airf, [0, 1, 0])

    # testing against algorithm
    assert np.allclose(sample_panel.x_airf, x_airf)
    assert np.allclose(sample_panel.y_airf, y_airf)
    assert np.allclose(sample_panel.z_airf, z_airf)

    # testing the bound_points
    assert np.allclose(sample_panel.bound_point_1, bound_point_1)
    assert np.allclose(sample_panel.bound_point_2, bound_point_2)


def test_panel_custom_initialization():
    section1 = MockSection([1, 1, 1], [2, 1, 1])
    section2 = MockSection([1, 2, 1], [2, 2, 1])
    custom_panel = Panel(
        section1, section2, aerodynamic_center_location=0.3, control_point_location=0.7
    )

    assert np.allclose(custom_panel.aerodynamic_center, [1.3, 1.5, 1])
    assert np.allclose(custom_panel.control_point, [1.7, 1.5, 1])


def test_va_setter(sample_panel):
    test_va = np.array([1, 2, 3])
    sample_panel.va = test_va
    assert np.array_equal(sample_panel.va, test_va)


def test_calculate_relative_alpha_and_relative_velocity(sample_panel):
    sample_panel.va = np.array([10, 0, 0])
    induced_velocity = np.array([1, 1, 1])

    alpha_calc, relative_velocity_calc = (
        sample_panel.calculate_relative_alpha_and_relative_velocity(induced_velocity)
    )

    # Calculate terms of induced corresponding to the airfoil directions
    norm_airf = sample_panel.x_airf
    tan_airf = sample_panel.y_airf

    # Calculate relative velocity and angle of attack
    relative_velocity = sample_panel.va + induced_velocity
    vn = np.dot(norm_airf, relative_velocity)
    vtan = np.dot(tan_airf, relative_velocity)
    alpha = np.arctan(vn / vtan)

    print(f"vn: {vn}")
    print(f"vtan: {vtan}")
    print(f"alpha_calc: {alpha_calc}")
    print(f"alpha: {alpha}")
    print(f"relative_velocity_calc: {relative_velocity_calc}")
    print(f"relative_velocity: {relative_velocity}")
    assert np.isclose(alpha, alpha_calc)
    assert np.allclose(relative_velocity, relative_velocity_calc)


def test_calculate_inviscid_polar_data(sample_panel):
    airfoil_data = sample_panel.calculate_inviscid_polar_data()

    # The output should be a numpy array with shape (41, 4)
    assert airfoil_data.shape == (41, 4)  # -20 to 20 degrees, 4 columns

    # the alpha values should be ranging from -20 to 20 degrees, but in radians
    assert np.isclose(airfoil_data[0, 0], -20 * np.pi / 180)

    # the cl values should be 2 pi alpha
    assert np.isclose(airfoil_data[0, 1], 2 * np.pi * np.sin(-20 * np.pi / 180))

    # the cd values should be 0.05
    assert np.isclose(airfoil_data[0, 2], 0.05)

    # the cm values should be 0.01
    assert np.isclose(airfoil_data[0, 3], 0.01)

    # When inviscid, the airfoil_data should be equal to the inviscid data
    aero_input_1 = ("inviscid",)
    aero_input_2 = ("inviscid",)
    airfoil_data_new = sample_panel.calculate_airfoil_data(aero_input_1, aero_input_2)
    assert np.allclose(airfoil_data, airfoil_data_new)


def test_calculate_airfoil_data_lei_airfoil_breukels(sample_panel):
    pass


def test_calculate_cl(sample_panel):
    alpha = np.pi / 4  # 45 degrees
    cl = sample_panel.calculate_cl(alpha)
    expected_cl = 2 * np.pi * np.sin(alpha)

    assert np.isclose(cl, expected_cl)


def test_calculate_velocity_induced_bound_2D(sample_panel):
    control_point = np.array([0.5, 5, 0])
    gamma = 1.0
    induced_velocity = sample_panel.calculate_velocity_induced_bound_2D(
        control_point, gamma
    )

    assert isinstance(induced_velocity, np.ndarray)
    assert induced_velocity.shape == (3,)  # 2D velocity


def test_calculate_velocity_induced_horseshoe(sample_panel):
    control_point = np.array([0.5, 5, 0])
    gamma = 1.0
    induced_velocity = sample_panel.calculate_velocity_induced_horseshoe(
        control_point, gamma
    )

    assert isinstance(induced_velocity, np.ndarray)
    assert induced_velocity.shape == (3,)  # 3D velocity


def test_calculate_filaments_for_plotting(sample_panel):
    filaments_for_plotting = sample_panel.calculate_filaments_for_plotting()
    for filament in filaments_for_plotting:
        assert filament[0].shape == (3,)
        assert filament[1].shape == (3,)
        assert isinstance(filament[2], str)
