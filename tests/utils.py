import numpy as np
import logging


def cosspace(min, max, n_points):
    """
    Create an array with cosine spacing, from min to max values, with n points
    """
    mean = (max + min) / 2
    amp = (max - min) / 2
    return mean + amp * np.cos(np.linspace(np.pi, 0, n_points))


def generate_coordinates_rect_wing(chord, span, twist, beta, N, dist):
    """
    Generates the 3D coordinates of a rectangular wing with twist and dihedral.

    Parameters:
    chord: array-like
        The chord lengths of the wing panels.
    span: float
        The total span of the wing.
    twist: array-like
        The twist angles of the wing panels, in radians.
    beta: array-like
        The dihedral angles of the wing panels, in radians.
    N: int
        The number of panels along the span of the wing.
    dist: str
        The distribution type of the spanwise positions ('cos' for cosine-spaced, 'lin' for linear).

    Returns:
    coord: np.ndarray
        A 2N x 3 array of 3D coordinates for the leading and trailing edges of the wing panels.
    """
    coord = np.empty((2 * N, 3))
    if dist == "cos":
        span = cosspace(-span / 2, span / 2, N)
    elif dist == "lin":
        span = np.linspace(-span / 2, span / 2, N)

    for i in range(N):
        coord[2 * i, :] = np.array(
            [
                -0 * chord[i] * np.cos(twist[i]),
                span[i],
                0 * chord[i] * np.sin(twist[i]) - abs(span[i] * np.sin(beta[i])),
            ]
        )
        coord[2 * i + 1, :] = np.array(
            [
                1 * chord[i] * np.cos(twist[i]),
                span[i],
                -1 * chord[i] * np.sin(twist[i]) - abs(span[i] * np.sin(beta[i])),
            ]
        )

    return coord


def generate_coordinates_curved_wing(chord, span, theta, R, N, dist):
    """
    theta:
        This represents the angular extent of the curvature of the wing, in radians.
        It defines the range of angles over which the wing coordinates are distributed.
        A larger theta means a more pronounced curvature.

    R:
        This represents the radius of curvature of the wing.
        It defines the distance from the center of curvature (the origin in this case)
        to the points on the curved surface of the wing.
        A larger R means a larger radius, resulting in a more gradual curve.
    """
    coord = np.empty((2 * N, 3))
    if dist == "cos":
        theta = cosspace(-theta, theta, N)
    elif dist == "lin":
        theta = np.linspace(-theta, theta, N)
    elif dist == "cos2":
        theta1 = cosspace(-theta, -theta / N / 10, int(N / 2))
        theta2 = cosspace(theta / N / 10, theta, int(N / 2))
        theta = np.concatenate((theta1, theta2))

    for i in range(N):
        coord[2 * i, :] = np.array([0, R * np.sin(theta[i]), R * np.cos(theta[i])])
        coord[2 * i + 1, :] = np.array(
            [chord, R * np.sin(theta[i]), R * np.cos(theta[i])]
        )

    return coord


def generate_coordinates_el_wing(max_chord, span, N, dist):
    coord = np.empty((2 * N, 3))
    start = span * 1e-5
    if dist == "cos":
        y_arr = cosspace(-span / 2 + start, span / 2 - start, N)
    elif dist == "lin":
        y_arr = np.linspace(-span / 2 + start, span / 2 - start, N)

    c_arr = 2 * np.sqrt(1 - (y_arr / (span / 2)) ** 2) * max_chord / 2

    for i in range(N):
        coord[2 * i, :] = [-0.25 * c_arr[i], y_arr[i], 0]
        coord[2 * i + 1, :] = [0.75 * c_arr[i], y_arr[i], 0]

    return coord


# Check if the results are the same in list of dictionaries
def asserting_all_elements_in_list_dict(variable1, variable_expected):
    for i, (variable1_i, variable_expected_i) in enumerate(
        zip(variable1, variable_expected)
    ):
        for j, (key1, key2) in enumerate(
            zip(variable1_i.keys(), variable_expected_i.keys())
        ):
            logging.debug(f"key1 {key1}, key2 {key2}")
            logging.debug(f"variable1_i {variable1_i[key1]}")
            logging.debug(f"variable_expected_i {variable_expected_i[key1]}")
            # assert key1 == key2
            assert np.allclose(variable1_i[key1], variable_expected_i[key1], atol=1e-5)


# Check if the results are the same in list of list of dictionaries
def asserting_all_elements_in_list_list_dict(variable1, variable_expected):
    for i, (list1, list_expected) in enumerate(zip(variable1, variable_expected)):
        logging.info(f"list1 {list1}")
        logging.info(f"list_expected {list_expected}")
        for j, (dict1, dict_expected) in enumerate(zip(list1, list_expected)):
            logging.info(f"dict1 {dict1}")
            logging.info(f"dict_expected {dict_expected}")
            assert dict1.keys() == dict_expected.keys()
            for key1, key2 in zip(dict1.keys(), dict_expected.keys()):
                logging.info(f"key1 {key1}, key2 {key2}")
                logging.info(f"dict1[key1] {dict1[key1]}")
                logging.info(f"dict_expected[key1] {dict_expected[key1]}")
                assert key1 == key2
                # check breaks when entry is a string
                if isinstance(dict1[key1], str):
                    assert dict1[key1] == dict_expected[key1]
                else:
                    assert np.allclose(dict1[key1], dict_expected[key1], atol=1e-5)


def create_controlpoints_from_wing_object(wing, model):
    result = []

    for panel in wing.panels:
        if model == "VSM":
            cp = {
                "coordinates": panel.control_point,
                "chord": panel.chord,
                "normal": panel.x_airf,
                "tangential": panel.y_airf,
                "arf_coord": np.column_stack(
                    [panel.x_airf, panel.y_airf, panel.z_airf]
                ),
                "coordinates_aoa": panel.aerodynamic_center,
            }
        elif model == "LLT":
            cp = {
                "coordinates": panel.aerodynamic_center,
                "chord": panel.chord,
                "normal": panel.x_airf,
                "tangential": panel.y_airf,
                "airf_coord": np.column_stack(
                    [panel.x_airf, panel.y_airf, panel.z_airf]
                ),
            }
        else:
            raise ValueError(f"Model {model} not recognized")

        result.append(cp)
    return result


def create_ring_from_wing_object(wing, gamma_data=None):
    result = []
    va_norm = wing.va / np.linalg.norm(wing.va)
    filaments_list = [panel.filaments for panel in wing.panels]
    if gamma_data is None:
        gamma_data = [0 for _ in filaments_list]
    else:
        gamma_data = [gamma_data for _ in filaments_list]
    for filaments, gamma in zip(filaments_list, gamma_data):
        ring_filaments = []

        # bound starts
        filament_bound = filaments[0]
        filament_trailing_1 = filaments[1]
        filament_trailing_2 = filaments[2]
        filament_semi_1 = filaments[3]
        filament_semi_2 = filaments[4]

        # appending them in the correct order
        ring_filaments.append(
            {
                "id": "bound",
                "x1": filament_bound.x1,
                "x2": filament_bound.x2,
                "Gamma": gamma,
            }
        )
        ring_filaments.append(
            {
                "x1": filament_trailing_1.x1,
                "x2": filament_trailing_1.x2,
                "Gamma": gamma,
                "id": "trailing1",
            }
        )
        ring_filaments.append(
            {
                "dir": va_norm,
                "id": "trailing_inf1",
                "x1": filament_semi_1.x1,
                "Gamma": gamma,
            }
        )
        # be wary of incorrect naming convention here.
        ring_filaments.append(
            {
                "x2": filament_trailing_2.x2,
                "x1": filament_trailing_2.x1,
                "Gamma": gamma,
                "id": "trailing1",
            }
        )
        ring_filaments.append(
            {
                "x1": filament_semi_2.x1,
                "dir": va_norm,
                "id": "trailing_inf2",
                "Gamma": gamma,
            }
        )

        result.append(ring_filaments)

    return result


def create_wingpanels_from_wing_object(wing):
    wingpanels = []

    coordinates = np.zeros((2 * (len(wing.panels) + 1), 3))
    n_panels = len(wing.panels)
    for i in range(n_panels):
        coordinates[2 * i] = wing.panels[i].LE_point_1
        coordinates[2 * i + 1] = wing.panels[i].TE_point_1
        coordinates[2 * i + 2] = wing.panels[i].LE_point_2
        coordinates[2 * i + 3] = wing.panels[i].TE_point_2

    # Go through all wing panels
    for i, panel in enumerate(wing.panels):

        # Identify points defining the panel
        section = {
            "p1": coordinates[2 * i, :],
            "p2": coordinates[2 * i + 2, :],
            "p3": coordinates[2 * i + 3, :],
            "p4": coordinates[2 * i + 1, :],
        }
        wingpanels.append(section)
    return wingpanels


def create_ring_vec_from_wing_object(wing, model):
    result = []

    for panel in wing.panels:
        bound_1 = panel.bound_point_1
        bound_2 = panel.bound_point_2
        if model == "VSM":
            evaluation_point = panel.control_point
        elif model == "LLT":
            evaluation_point = panel.aerodynamic_center

        # Calculate the required vectors
        logging.debug(f"bound_1 {bound_1}")
        logging.debug(f"bound_2 {bound_2}")
        r0 = bound_2 - bound_1
        r1 = evaluation_point - bound_1
        r2 = evaluation_point - bound_2
        r3 = evaluation_point - (bound_2 + bound_1) / 2

        # Add the calculated vectors to the result
        result.append({"r0": r0, "r1": r1, "r2": r2, "r3": r3})

    return result


def create_coord_L_from_wing_object(wing):
    coord_L = []
    for panel in wing.panels:
        coord_L.append(panel.aerodynamic_center)
    return coord_L


def create_geometry_from_wing_object(wing, model):
    controlpoints = create_controlpoints_from_wing_object(wing, model)
    rings = create_ring_from_wing_object(wing)
    wingpanels = create_wingpanels_from_wing_object(wing)
    ringvec = create_ring_vec_from_wing_object(wing, model)
    coord_L = create_coord_L_from_wing_object(wing)

    return controlpoints, rings, wingpanels, ringvec, coord_L


def flip_created_coord_in_pairs(coord):
    # Reshape the array into pairs
    reshaped = coord.reshape(-1, 2, 3)

    # Reverse the order of the pairs
    flipped = np.flip(reshaped, axis=0)

    # Flatten back to the original shape
    return flipped.reshape(-1, 3)
