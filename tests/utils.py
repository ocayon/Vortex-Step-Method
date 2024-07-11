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
        logging.debug(f"list1 {list1}")
        logging.debug(f"list_expected {list_expected}")
        for j, (dict1, dict_expected) in enumerate(zip(list1, list_expected)):
            logging.debug(f"dict1 {dict1}")
            logging.debug(f"dict_expected {dict_expected}")
            assert dict1.keys() == dict_expected.keys()
            for key1, key2 in zip(dict1.keys(), dict_expected.keys()):
                logging.debug(f"key1 {key1}, key2 {key2}")
                logging.debug(f"dict1[key1] {dict1[key1]}")
                logging.debug(f"dict_expected[key1] {dict_expected[key1]}")
                assert key1 == key2
                # check breaks when entry is a string
                if isinstance(dict1[key1], str):
                    assert dict1[key1] == dict_expected[key1]
                else:
                    assert np.allclose(dict1[key1], dict_expected[key1], atol=1e-5)
