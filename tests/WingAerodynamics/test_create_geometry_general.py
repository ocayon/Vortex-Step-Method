import numpy as np
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
)


def vec_norm(v):
    """
    Norm of a vector

    """
    return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def create_geometry_general(coordinates, Uinf, N, ring_geo, model):
    """
    Create geometry structures necessary for solving the system of circualtion

    Parameters
    ----------
    coordinates : coordinates the nodes (each section is defined by two nodes,
                                         the first is the LE, so each section
                                         defined by a pair of coordinates)
    Uinf : Wind speed vector
    N : Number of sections
    ring_geo :  - '3fil': Each horsehoe is defined by 3 filaments
                - '5fil': Each horseshoe is defined by 5 filaments
    model : VSM: Vortex Step method/ LLT: Lifting Line Theory

    Returns
    -------
    controlpoints :  List of dictionaries with the variables needed to define each wing section
    rings : List of list with the definition of each vortex filament
    wingpanels : List with the points defining each wing pannel
    ringvec : List of dictionaries containing the vectors that define each ring
    coord_L : coordinates of the aerodynamic centers of each wing panel

    """

    filaments = []
    controlpoints = []
    rings = []
    wingpanels = []
    ringvec = []
    coord_L = []

    # Go through all wing panels
    for i in range(N - 1):

        # Identify points defining the panel
        section = {
            "p1": coordinates[2 * i, :],
            "p2": coordinates[2 * i + 2, :],
            "p3": coordinates[2 * i + 3, :],
            "p4": coordinates[2 * i + 1, :],
        }
        wingpanels.append(section)

        di = vec_norm(
            coordinates[2 * i, :] * 0.75
            + coordinates[2 * i + 1, :] * 0.25
            - (coordinates[2 * i + 2, :] * 0.75 + coordinates[2 * i + 3, :] * 0.25)
        )
        if i == 0:
            diplus = vec_norm(
                coordinates[2 * (i + 1), :] * 0.75
                + coordinates[2 * (i + 1) + 1, :] * 0.25
                - (
                    coordinates[2 * (i + 1) + 2, :] * 0.75
                    + coordinates[2 * (i + 1) + 3, :] * 0.25
                )
            )
            ncp = di / (di + diplus)
        elif i == N - 2:
            dimin = vec_norm(
                coordinates[2 * (i - 1), :] * 0.75
                + coordinates[2 * (i - 1) + 1, :] * 0.25
                - (
                    coordinates[2 * (i - 1) + 2, :] * 0.75
                    + coordinates[2 * (i - 1) + 3, :] * 0.25
                )
            )
            ncp = dimin / (dimin + di)
        else:
            dimin = vec_norm(
                coordinates[2 * (i - 1), :] * 0.75
                + coordinates[2 * (i - 1) + 1, :] * 0.25
                - (
                    coordinates[2 * (i - 1) + 2, :] * 0.75
                    + coordinates[2 * (i - 1) + 3, :] * 0.25
                )
            )
            diplus = vec_norm(
                coordinates[2 * (i + 1), :] * 0.75
                + coordinates[2 * (i + 1) + 1, :] * 0.25
                - (
                    coordinates[2 * (i + 1) + 2, :] * 0.75
                    + coordinates[2 * (i + 1) + 3, :] * 0.25
                )
            )
            ncp = 0.25 * (dimin / (dimin + di) + di / (di + diplus) + 1)

        ncp = 1 - ncp
        chord = np.linalg.norm(
            (section["p2"] + section["p1"]) / 2 - (section["p3"] + section["p4"]) / 2
        )
        LLpoint = (section["p2"] * (1 - ncp) + section["p1"] * ncp) * 3 / 4 + (
            section["p3"] * (1 - ncp) + section["p4"] * ncp
        ) * 1 / 4
        VSMpoint = (section["p2"] * (1 - ncp) + section["p1"] * ncp) * 1 / 4 + (
            section["p3"] * (1 - ncp) + section["p4"] * ncp
        ) * 3 / 4
        coord_L.append(LLpoint)

        # Define bound vortex filament
        bound = {
            "id": "bound",
            "x1": section["p1"] * 3 / 4 + section["p4"] * 1 / 4,
            "x2": section["p2"] * 3 / 4 + section["p3"] * 1 / 4,
            "Gamma": 0,
        }
        filaments.append(bound)

        x_airf = np.cross(VSMpoint - LLpoint, section["p2"] - section["p1"])
        x_airf = x_airf / np.linalg.norm(x_airf)
        y_airf = VSMpoint - LLpoint
        y_airf = y_airf / np.linalg.norm(y_airf)
        z_airf = bound["x2"] - bound["x1"]
        # z_airf[0] = 0
        z_airf = z_airf / np.linalg.norm(z_airf)
        airf_coord = np.column_stack([x_airf, y_airf, z_airf])

        normal = x_airf
        tangential = y_airf
        if model == "VSM":
            cp = {
                "coordinates": VSMpoint,
                "chord": chord,
                "normal": normal,
                "tangential": tangential,
                "airf_coord": airf_coord,
                "coordinates_aoa": LLpoint,
            }
            controlpoints.append(cp)
        elif model == "LLT":

            cp = {
                "coordinates": LLpoint,
                "chord": chord,
                "normal": normal,
                "tangential": tangential,
                "airf_coord": airf_coord,
            }
            controlpoints.append(cp)

        temp = {
            "r0": bound["x2"] - bound["x1"],
            "r1": cp["coordinates"] - bound["x1"],
            "r2": cp["coordinates"] - bound["x2"],
            "r3": cp["coordinates"] - (bound["x2"] + bound["x1"]) / 2,
        }
        ringvec.append(temp)

        temp = Uinf / np.linalg.norm(Uinf)
        if ring_geo == "3fil":
            # create trailing filaments, at x1 of bound filament
            temp1 = {"dir": temp, "id": "trailing_inf1", "x1": bound["x1"], "Gamma": 0}
            filaments.append(temp1)

            # create trailing filaments, at x2 of bound filament
            temp1 = {"x1": bound["x2"], "dir": temp, "id": "trailing_inf2", "Gamma": 0}
            filaments.append(temp1)
        elif ring_geo == "5fil":
            temp1 = {
                "x1": section["p4"],
                "x2": bound["x1"],
                "Gamma": 0,
                "id": "trailing1",
            }
            filaments.append(temp1)

            temp1 = {
                "dir": temp,
                "id": "trailing_inf1",
                "x1": section["p4"],
                "Gamma": 0,
            }
            filaments.append(temp1)

            # create trailing filaments, at x2 of bound filament
            temp1 = {
                "x2": section["p3"],
                "x1": bound["x2"],
                "Gamma": 0,
                "id": "trailing1",
            }
            filaments.append(temp1)

            temp1 = {
                "x1": section["p3"],
                "dir": temp,
                "id": "trailing_inf2",
                "Gamma": 0,
            }
            filaments.append(temp1)

        #

        rings.append(filaments)
        filaments = []

    coord_L = np.array(coord_L)
    return controlpoints, rings, wingpanels, ringvec, coord_L


def create_controlpoints_from_wing_object(wing, model):
    result = []

    for panel in wing.panels:
        if model == "VSM":
            cp = {
                "coordinates": panel.aerodynamic_center,
                "chord": panel.chord,
                "normal": panel.x_airf,
                "tangential": panel.y_airf,
                "airf_coord": np.column_stack(
                    [panel.x_airf, panel.y_airf, panel.z_airf]
                ),
                "coordinates_aoa": panel.control_point,
            }
        elif model == "LLT":
            cp = {
                "coordinates": panel.control_point,
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
    filaments = [panel.filaments for panel in wing.panels]
    gamma_data = np.ones(len(filaments))
    for ring, gamma in zip(filaments, gamma_data):
        ring_filaments = []

        for i, filament in enumerate(ring):
            if isinstance(filament, BoundFilament):
                if i == 0:  # First BoundFilament in the ring
                    ring_filaments.append(
                        {
                            "id": "bound",
                            "x1": filament.x1,
                            "x2": filament.x2,
                            "Gamma": gamma,
                        }
                    )
                else:  # Other BoundFilaments are treated as 'trailing1'
                    ring_filaments.append(
                        {
                            "id": "trailing1",
                            "x1": filament.x1,
                            "x2": filament.x2,
                            "Gamma": gamma,
                        }
                    )
            elif isinstance(filament, SemiInfiniteFilament):
                ring_filaments.append(
                    {
                        "id": f"trailing_inf{len(ring_filaments) - 2}",  # -2 because we start counting after the bound filament
                        "x1": filament.x1,
                        "dir": filament._direction,  # Assuming _direction is the correct attribute
                        "Gamma": gamma,
                    }
                )

        result.append(ring_filaments)

    return result


def create_wingpanels_from_wing_object(wing):
    wingpanels = []
    # Go through all wing panels
    for i, panel in enumerate(wing.panels):

        coordinates = np.array(
            [panel.LE_point_1, panel.TE_point_1, panel.LE_point_2, panel.TE_point_2]
        )

        # Identify points defining the panel
        section = {
            "p1": coordinates[2 * i, :],
            "p2": coordinates[2 * i + 2, :],
            "p3": coordinates[2 * i + 3, :],
            "p4": coordinates[2 * i + 1, :],
        }
        wingpanels.append(section)
    return wingpanels


def create_ring_vec_from_wing_object(wing):
    result = []

    for panel in wing.panels:

        # Calculate the required vectors
        r0 = panel.bound_point_2 - panel.bound_point_1
        r1 = panel.control_point - panel.bound_point_1
        r2 = panel.control_point - panel.bound_point_2
        r3 = panel.control_point - (panel.bound_point_2 + panel.bound_point_1) / 2

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
    ringvec = create_ring_vec_from_wing_object(wing)
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
    coord = generate_coordinates_el_wing(max_chord, span, N, "cos")
    wing = Wing(N, "unchanged")
    for i in range(int(len(coord) / 2)):
        wing.add_section(coord[2 * i], coord[2 * i + 1], ["inviscid"])
    wing_aero = WingAerodynamics([wing])
    wing_aero.va = Uinf

    model = "VSM"
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
    assert np.allclose(controlpoints, expected_controlpoints)
    assert np.allclose(rings, expected_rings)
    assert np.allclose(wingpanels, expected_bladepanels)
    assert np.allclose(ringvec, expected_ringvec)
    assert np.allclose(coord_L, expected_coord_L)

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
    assert np.allclose(controlpoints, expected_controlpoints)
    assert np.allclose(rings, expected_rings)
    assert np.allclose(wingpanels, expected_bladepanels)
    assert np.allclose(ringvec, expected_ringvec)
    assert np.allclose(coord_L, expected_coord_L)

    ### Curved Wing
    theta = np.pi / 4
    R = 5
    coord = generate_coordinates_curved_wing(max_chord, span, theta, R, N, "cos")
    wing = Wing(N, "unchanged")
    for i in range(int(len(coord) / 2)):
        wing.add_section(coord[2 * i], coord[2 * i + 1], ["inviscid"])
    wing_aero = WingAerodynamics([wing])
    wing_aero.va = Uinf

    model = "VSM"
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
    assert np.allclose(controlpoints, expected_controlpoints)
    assert np.allclose(rings, expected_rings)
    assert np.allclose(wingpanels, expected_bladepanels)
    assert np.allclose(ringvec, expected_ringvec)
    assert np.allclose(coord_L, expected_coord_L)

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
    assert np.allclose(controlpoints, expected_controlpoints)
    assert np.allclose(rings, expected_rings)
    assert np.allclose(wingpanels, expected_bladepanels)
    assert np.allclose(ringvec, expected_ringvec)
    assert np.allclose(coord_L, expected_coord_L)
