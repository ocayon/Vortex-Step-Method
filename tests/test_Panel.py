import pytest
import numpy as np
from VSM.Panel import Panel  # Assuming the Panel class is in a file named Panel.py


# Mock Section class for testing
class MockSection:
    def __init__(self, LE_point, TE_point, aero_input):
        self.LE_point = np.array(LE_point)
        self.TE_point = np.array(TE_point)
        self.aero_input = aero_input


@pytest.fixture
def sample_panel():
    section1 = MockSection([0, 0, 0], [-1, 0, 0], ["inviscid"])
    section2 = MockSection([0, 10, 0], [-1, 10, 0], ["inviscid"])
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


# testing polar_data_input_option
def test_polar_data_input():
    # Generate mock polar data, using inviscid standards
    aoa = np.arange(-180, 180, 1)
    airfoil_data = np.empty((len(aoa), 4))
    for j, alpha in enumerate(aoa):
        cl, cd, cm = 2 * np.pi * np.sin(np.deg2rad(alpha)), 0.05, 0.01
        airfoil_data[j, 0] = np.deg2rad(alpha)
        airfoil_data[j, 1] = cl
        airfoil_data[j, 2] = cd
        airfoil_data[j, 3] = cm

    polar_data_test1 = ["polar_data", airfoil_data]
    polar_data_test2 = ["polar_data", airfoil_data * 1.1]  # Slightly different data

    # Create two sections with slightly different polar data
    section1 = MockSection([0, 0, 0], [-1, 0, 0], polar_data_test1)
    section2 = MockSection([0, 10, 0], [-1, 10, 0], polar_data_test2)

    # Create panel
    panel = Panel(
        section1,
        section2,
        aerodynamic_center_location=0.25,
        control_point_location=0.75,
    )

    assert hasattr(panel, "_panel_polar_data")
    assert panel._panel_polar_data is not None
    assert panel._panel_polar_data.shape == airfoil_data.shape

    # Check if panel_polar_data is correctly averaged
    expected_data = (airfoil_data + airfoil_data * 1.1) / 2
    assert np.allclose(panel._panel_polar_data, expected_data, atol=1e-6)


# taken from Uri thesis code
def LEI_airf_coeff(t, k, alpha):
    """
    ----------
    t : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.

    Returns
    -------
    Cl : TYPE
        DESCRIPTION.
    Cd : TYPE
        DESCRIPTION.
    Cm : TYPE
        DESCRIPTION.

    """
    C20 = -0.008011
    C21 = -0.000336
    C22 = 0.000992
    C23 = 0.013936
    C24 = -0.003838
    C25 = -0.000161
    C26 = 0.001243
    C27 = -0.009288
    C28 = -0.002124
    C29 = 0.012267
    C30 = -0.002398
    C31 = -0.000274
    C32 = 0
    C33 = 0
    C34 = 0
    C35 = -3.371000
    C36 = 0.858039
    C37 = 0.141600
    C38 = 7.201140
    C39 = -0.676007
    C40 = 0.806629
    C41 = 0.170454
    C42 = -0.390563
    C43 = 0.101966
    C44 = 0.546094
    C45 = 0.022247
    C46 = -0.071462
    C47 = -0.006527
    C48 = 0.002733
    C49 = 0.000686
    C50 = 0.123685
    C51 = 0.143755
    C52 = 0.495159
    C53 = -0.105362
    C54 = 0.033468
    C55 = -0.284793
    C56 = -0.026199
    C57 = -0.024060
    C58 = 0.000559
    C59 = -1.787703
    C60 = 0.352443
    C61 = -0.839323
    C62 = 0.137932

    S9 = C20 * t**2 + C21 * t + C22
    S10 = C23 * t**2 + C24 * t + C25
    S11 = C26 * t**2 + C27 * t + C28
    S12 = C29 * t**2 + C30 * t + C31
    S13 = C32 * t**2 + C33 * t + C34
    S14 = C35 * t**2 + C36 * t + C37
    S15 = C38 * t**2 + C39 * t + C40
    S16 = C41 * t**2 + C42 * t + C43

    lambda5 = S9 * k + S10
    lambda6 = S11 * k + S12
    lambda7 = S13 * k + S14
    lambda8 = S15 * k + S16

    Cl = lambda5 * alpha**3 + lambda6 * alpha**2 + lambda7 * alpha + lambda8
    Cd = (
        ((C44 * t + C45) * k**2 + (C46 * t + C47) * k + (C48 * t + C49)) * alpha**2
        + (C50 * t + C51) * k
        + (C52 * t**2 + C53 * t + C54)
    )
    Cm = (
        ((C55 * t + C56) * k + (C57 * t + C58)) * alpha**2
        + (C59 * t + C60) * k
        + (C61 * t + C62)
    )

    if alpha > 20 or alpha < -20:
        Cl = 2 * np.cos(alpha * np.pi / 180) * np.sin(alpha * np.pi / 180) ** 2
        Cd = 2 * np.sin(alpha * np.pi / 180) ** 3

    return Cl, Cd, Cm


def test_lei_airfoil_breukels_input():
    # Create two sections with LEI airfoil parameters
    t1, k1 = 0.12, 0.8
    t2, k2 = 0.15, 0.7
    section1 = MockSection([0, 0, 0], [-1, 0, 0], ["lei_airfoil_breukels", [t1, k1]])
    section2 = MockSection([0, 10, 0], [-1, 10, 0], ["lei_airfoil_breukels", [t2, k2]])

    # Create panel
    panel = Panel(
        section1,
        section2,
        aerodynamic_center_location=0.25,
        control_point_location=0.75,
    )

    # Check if LEI airfoil coefficients are correctly calculated
    assert hasattr(panel, "_cl_coefficients")
    assert hasattr(panel, "_cd_coefficients")
    assert hasattr(panel, "_cm_coefficients")

    # Test a few angles to ensure coefficients are calculated correctly
    test_angles = [-10, 0, 10]
    for alpha in test_angles:
        t = (t1 + t2) / 2
        k = (k1 + k2) / 2
        cl, cd, cm = LEI_airf_coeff(t, k, alpha)

        calculated_cl = np.polyval(panel._cl_coefficients, alpha)
        calculated_cd = np.polyval(panel._cd_coefficients, alpha)
        calculated_cm = np.polyval(panel._cm_coefficients, alpha)

        assert np.isclose(calculated_cl, cl, atol=1e-6)
        assert np.isclose(calculated_cd, cd, atol=1e-6)
        assert np.isclose(calculated_cm, cm, atol=1e-6)


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
    section1 = MockSection([1, 1, 1], [2, 1, 1], ["inviscid"])
    section2 = MockSection([1, 2, 1], [2, 2, 1], ["inviscid"])
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


def test_calculate_cl_and_cd_cm(sample_panel):
    # Generate mock polar data, using inviscid standards
    aoa = np.arange(-180, 180, 1)
    airfoil_data = np.empty((len(aoa), 4))
    for j, alpha in enumerate(aoa):
        cl, cd, cm = 2 * np.pi * np.sin(np.deg2rad(alpha)), 0.05, 0.01
        airfoil_data[j, 0] = np.deg2rad(alpha)
        airfoil_data[j, 1] = cl
        airfoil_data[j, 2] = cd
        airfoil_data[j, 3] = cm

    polar_data_test1 = ["polar_data", airfoil_data]

    # Create two sections with slightly different polar data
    inviscid_section1 = MockSection([0, 0, 0], [-1, 0, 0], ["inviscid"])
    inviscid_section2 = MockSection([0, 10, 0], [-1, 10, 0], ["inviscid"])
    polar_data_section1 = MockSection([0, 0, 0], [-1, 0, 0], polar_data_test1)
    polar_data_section2 = MockSection([0, 10, 0], [-1, 10, 0], polar_data_test1)
    lei_airfoil_section1 = MockSection(
        [0, 0, 0], [-3, 0, 0], ["lei_airfoil_breukels", [0.12, 0.8]]
    )
    lei_airfoil_section2 = MockSection(
        [0, 10, 0], [-3, 10, 0], ["lei_airfoil_breukels", [0.15, 0.7]]
    )

    # Create panels
    inviscid_panel_instance = Panel(
        inviscid_section1,
        inviscid_section2,
        aerodynamic_center_location=0.25,
        control_point_location=0.75,
    )
    polar_data_panel_instance = Panel(
        polar_data_section1,
        polar_data_section2,
        aerodynamic_center_location=0.25,
        control_point_location=0.75,
    )
    lei_airfoil_panel_instance = Panel(
        lei_airfoil_section1,
        lei_airfoil_section2,
        aerodynamic_center_location=0.25,
        control_point_location=0.75,
    )
    # calculating average t and k
    t_avg = (0.12 + 0.15) / 2
    k_avg = (0.8 + 0.7) / 2
    t = t_avg / lei_airfoil_panel_instance._chord
    k = k_avg / lei_airfoil_panel_instance._chord

    # testing several angles
    test_alphas = [-10, 0, 10, 40]
    for alpha in test_alphas:
        alpha_rad = np.deg2rad(alpha)

        # inviscid panel
        cl_inviscid = inviscid_panel_instance.calculate_cl(alpha_rad)
        expected_cl_inviscid = 2 * np.pi * np.sin(alpha_rad)
        assert np.isclose(cl_inviscid, expected_cl_inviscid)

        cd_cm_inviscid = inviscid_panel_instance.calculate_cd_cm(alpha_rad)
        expected_cm_cd_inviscid = [0.0, 0.0]
        assert np.isclose(cd_cm_inviscid[0], expected_cm_cd_inviscid[1])

        # polar data panel
        cl_polar_data = polar_data_panel_instance.calculate_cl(alpha_rad)
        expected_cl_polar_data = 2 * np.pi * np.sin(alpha_rad)
        assert np.isclose(cl_polar_data, expected_cl_polar_data)

        cd_cm_polar_data = polar_data_panel_instance.calculate_cd_cm(alpha_rad)
        expected_cm_cd_polar_data = [0.01, 0.05]
        assert np.isclose(cd_cm_polar_data[0], expected_cm_cd_polar_data[1])

        # LEI airfoil panel
        cl_lei_airfoil = lei_airfoil_panel_instance.calculate_cl(alpha_rad)
        expected_cl_lei_airfoil = LEI_airf_coeff(t, k, alpha)[0]
        print(f" ")
        print(f"alpha: {alpha}")
        print(f"expected_cl={expected_cl_lei_airfoil},calculating with t={t}, k={k}")
        print(f"cl_lei_airfoil: {cl_lei_airfoil}")
        assert np.isclose(cl_lei_airfoil, expected_cl_lei_airfoil)

        cd_cm_lei_airfoil = lei_airfoil_panel_instance.calculate_cd_cm(alpha_rad)
        expected_cd_cm_lei_airfoil = LEI_airf_coeff(t, k, alpha)[1:]
        print(
            f"LEI Airfoil Cd/Cm: alpha={alpha}, cd={cd_cm_lei_airfoil[0]}, expected_cd={expected_cd_cm_lei_airfoil[0]}, cm={cd_cm_lei_airfoil[1]}, expected_cm={expected_cd_cm_lei_airfoil[1]}"
        )
        assert np.isclose(cd_cm_lei_airfoil[0], expected_cd_cm_lei_airfoil[0])
        assert np.isclose(cd_cm_lei_airfoil[1], expected_cd_cm_lei_airfoil[1])


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
