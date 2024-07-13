import numpy as np
import logging
from copy import deepcopy
from VSM.Filament import BoundFilament, SemiInfiniteFilament
from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics

import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)
from tests.utils import (
    generate_coordinates_el_wing,
    generate_coordinates_rect_wing,
    generate_coordinates_curved_wing,
    asserting_all_elements_in_list_list_dict,
    asserting_all_elements_in_list_dict,
    create_controlpoints_from_wing_object,
    create_ring_from_wing_object,
    create_wingpanels_from_wing_object,
    create_ring_vec_from_wing_object,
    create_coord_L_from_wing_object,
    flip_created_coord_in_pairs,
)
from tests.thesis_functions_oriol_cayon import create_geometry_general


def create_geometry_from_wing_object(wing, model):
    controlpoints = create_controlpoints_from_wing_object(wing, model)
    rings = create_ring_from_wing_object(wing)
    wingpanels = create_wingpanels_from_wing_object(wing)
    ringvec = create_ring_vec_from_wing_object(wing, model)
    coord_L = create_coord_L_from_wing_object(wing)

    return controlpoints, rings, wingpanels, ringvec, coord_L


# Example usage:
# Assuming you have a wing object
# vectors = calculate_wing_panel_vectors(wing)

# To print the result (for testing purposes):
# import pprint
# pprint.pprint(vectors)


def test_create_geometry_general():

    N = 3
    max_chord = 1
    span = 2.36
    AR = span**2 / (np.pi * span * max_chord / 4)
    Umag = 20
    aoa = 5.7106 * np.pi / 180
    Uinf = np.array([np.cos(aoa), 0, np.sin(aoa)]) * Umag

    ### Elliptical Wing
    # coord = generate_coordinates_el_wing(max_chord, span, N, "cos")
    coord = generate_coordinates_rect_wing(
        max_chord * np.ones(N),
        span,
        twist=np.zeros(N),
        beta=np.zeros(N),
        N=N,
        dist="lin",
    )
    logging.debug(f"coord {coord}")
    coord_left_to_right = flip_created_coord_in_pairs(deepcopy(coord))
    logging.debug(f"coord_left_to_right {coord_left_to_right}")

    wing = Wing(N, "unchanged")
    for i in range(int(len(coord_left_to_right) / 2)):
        wing.add_section(
            coord_left_to_right[2 * i], coord_left_to_right[2 * i + 1], ["inviscid"]
        )
    wing_aero = WingAerodynamics([wing])
    wing_aero.va = Uinf
    # wing_aero.plot()

    model = "VSM"
    # Generate geometry
    (
        expected_controlpoints,
        expected_rings,
        expected_bladepanels,
        expected_ringvec,
        expected_coord_L,
    ) = create_geometry_general(coord, Uinf, int(len(coord) / 2), "5fil", model)

    logging.info(f"expected_controlpoints {expected_controlpoints}")
    logging.debug(f"expected_rings {expected_rings}")
    logging.debug(f"expected_bladepanels {expected_bladepanels}")
    logging.debug(f"expected_ringvec {expected_ringvec}")
    logging.debug(f"expected_coord_L {expected_coord_L}")

    # Generate geometry from wing object
    controlpoints, rings, wingpanels, ringvec, coord_L = (
        create_geometry_from_wing_object(wing_aero, model)
    )
    logging.info(f"---controlpoints--- type: {type(controlpoints)}, {controlpoints}")
    asserting_all_elements_in_list_dict(controlpoints, expected_controlpoints)

    logging.info(f"---rings--- type: {type(rings)}, {rings}")
    logging.info(f"---expected_rings--- type: {type(expected_rings)}, {expected_rings}")
    asserting_all_elements_in_list_list_dict(rings, expected_rings)
    asserting_all_elements_in_list_dict(wingpanels, expected_bladepanels)
    asserting_all_elements_in_list_dict(ringvec, expected_ringvec)
    logging.debug(f"---coord_L--- type: {type(coord_L)}, {coord_L}")
    assert np.allclose(coord_L, expected_coord_L, atol=1e-5)

    model = "LLT"
    # Generate geometry
    (
        expected_controlpoints,
        expected_rings,
        expected_bladepanels,
        expected_ringvec,
        expected_coord_L,
    ) = create_geometry_general(coord, Uinf, int(len(coord) / 2), "5fil", model)
    # Generate geometry from wing object
    controlpoints, rings, wingpanels, ringvec, coord_L = (
        create_geometry_from_wing_object(wing_aero, model)
    )

    # Check if the results are the same
    logging.debug(f"---controlpoints--- type: {type(controlpoints)}, {controlpoints}")
    asserting_all_elements_in_list_dict(controlpoints, expected_controlpoints)
    logging.debug(f"---rings--- type: {type(rings)}, {rings}")
    logging.debug(
        f"---expected_rings--- type: {type(expected_rings)}, {expected_rings}"
    )
    asserting_all_elements_in_list_list_dict(rings, expected_rings)
    asserting_all_elements_in_list_dict(wingpanels, expected_bladepanels)
    asserting_all_elements_in_list_dict(ringvec, expected_ringvec)
    logging.debug(f"---coord_L--- type: {type(coord_L)}, {coord_L}")
    assert np.allclose(coord_L, expected_coord_L, atol=1e-5)


#     ### Curved Wing
#     theta = np.pi / 4
#     R = 5
#     coord = generate_coordinates_curved_wing(max_chord, span, theta, R, N, "cos")
#     wing = Wing(N, "unchanged")
#     for i in range(int(len(coord) / 2)):
#         wing.add_section(coord[2 * i], coord[2 * i + 1], ["inviscid"])
#     wing_aero = WingAerodynamics([wing])
#     wing_aero.va = Uinf

#     model = "VSM"
#     # Generate geometry
#     (
#         expected_controlpoints,
#         expected_rings,
#         expected_bladepanels,
#         expected_ringvec,
#         expected_coord_L,
#     ) = create_geometry_general(coord, Uinf, int(len(coord) / 2), "5fil", model)
#     # Generate geometry from wing object
#     controlpoints, rings, wingpanels, ringvec, coord_L = (
#         create_geometry_from_wing_object(wing_aero, model)
#     )

#     # Check if the results are the same
#     logging.debug(f"---controlpoints--- type: {type(controlpoints)}, {controlpoints}")
#     asserting_all_elements_in_list_dict(controlpoints, expected_controlpoints)
#     logging.debug(f"---rings--- type: {type(rings)}, {rings}")
#     logging.debug(
#         f"---expected_rings--- type: {type(expected_rings)}, {expected_rings}"
#     )
#     asserting_all_elements_in_list_list_dict(rings, expected_rings)
#     asserting_all_elements_in_list_dict(wingpanels, expected_bladepanels)
#     asserting_all_elements_in_list_dict(ringvec, expected_ringvec)
#     logging.debug(f"---coord_L--- type: {type(coord_L)}, {coord_L}")
#     assert np.allclose(coord_L, expected_coord_L, atol=1e-5)

#     model = "LLT"
#     # Generate geometry
#     (
#         expected_controlpoints,
#         expected_rings,
#         expected_bladepanels,
#         expected_ringvec,
#         expected_coord_L,
#     ) = create_geometry_general(coord, Uinf, int(len(coord) / 2), "5fil", model)
#     # Generate geometry from wing object
#     controlpoints, rings, wingpanels, ringvec, coord_L = (
#         create_geometry_from_wing_object(wing_aero, model)
#     )

#     # Check if the results are the same
#     logging.debug(f"---controlpoints--- type: {type(controlpoints)}, {controlpoints}")
#     asserting_all_elements_in_list_dict(controlpoints, expected_controlpoints)
#     logging.debug(f"---rings--- type: {type(rings)}, {rings}")
#     logging.debug(
#         f"---expected_rings--- type: {type(expected_rings)}, {expected_rings}"
#     )
#     asserting_all_elements_in_list_list_dict(rings, expected_rings)
#     asserting_all_elements_in_list_dict(wingpanels, expected_bladepanels)
#     asserting_all_elements_in_list_dict(ringvec, expected_ringvec)
#     logging.debug(f"---coord_L--- type: {type(coord_L)}, {coord_L}")
#     assert np.allclose(coord_L, expected_coord_L, atol=1e-5)


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     test_create_geometry_general()
