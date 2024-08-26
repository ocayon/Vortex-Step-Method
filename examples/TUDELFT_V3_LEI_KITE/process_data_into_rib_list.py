import numpy as np
import csv
import logging
import pickle
import os
from pathlib import Path
from VSM.WingGeometry import Wing, flip_created_coord_in_pairs_if_needed


def vec_norm(v):
    """
    Norm of a vector

    """
    return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def cross_product(r1, r2):
    """
    Cross product between r1 and r2

    """

    return np.array(
        [
            r1[1] * r2[2] - r1[2] * r2[1],
            r1[2] * r2[0] - r1[0] * r2[2],
            r1[0] * r2[1] - r1[1] * r2[0],
        ]
    )


def struct2aero_geometry(coord_struc):

    coord = np.empty((20, 3))

    coord[0, :] = coord_struc[20, :]
    coord[1, :] = coord_struc[10, :]

    coord[2, :] = coord_struc[9, :]
    coord[3, :] = coord_struc[11, :]

    coord[4, :] = coord_struc[8, :]
    coord[5, :] = coord_struc[12, :]

    coord[6, :] = coord_struc[7, :]
    coord[7, :] = coord_struc[13, :]

    coord[8, :] = coord_struc[6, :]
    coord[9, :] = coord_struc[14, :]

    coord[10, :] = coord_struc[5, :]
    coord[11, :] = coord_struc[15, :]

    coord[12, :] = coord_struc[4, :]
    coord[13, :] = coord_struc[16, :]

    coord[14, :] = coord_struc[3, :]
    coord[15, :] = coord_struc[17, :]

    coord[16, :] = coord_struc[2, :]
    coord[17, :] = coord_struc[18, :]

    coord[18, :] = coord_struc[19, :]
    coord[19, :] = coord_struc[1, :]

    return coord


def create_geometry_LEI(coordinates, Uinf, N, ring_geo, model):

    filaments = []
    controlpoints = []
    rings = []
    wingpanels = []
    ringvec = []
    coord_L = []
    for i in range(N - 1):

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


def refine_LEI_mesh_ballooning(wingpanels, ball_angle, N_split):
    refined_coord = []
    for i_sec in range(len(wingpanels)):
        angle = ball_angle[i_sec] * np.pi / 180
        L_sec1 = vec_norm(wingpanels[i_sec]["p2"] - wingpanels[i_sec]["p1"])
        R1 = L_sec1 / 2 / np.sin(angle)
        L_sec2 = vec_norm(wingpanels[i_sec]["p3"] - wingpanels[i_sec]["p4"])
        R2 = L_sec2 / 2 / np.sin(angle)
        zvec = (wingpanels[i_sec]["p2"] + wingpanels[i_sec]["p1"]) / 2 - (
            wingpanels[i_sec]["p4"] + wingpanels[i_sec]["p3"]
        ) / 2
        zvec = zvec / vec_norm(zvec)

        xvec1 = wingpanels[i_sec]["p2"] - wingpanels[i_sec]["p1"]
        xvec1 = xvec1 / vec_norm(xvec1)
        yvec1 = cross_product(zvec, xvec1)
        yvec1 = yvec1 / vec_norm(yvec1)

        xvec2 = wingpanels[i_sec]["p3"] - wingpanels[i_sec]["p4"]
        xvec2 = xvec2 / vec_norm(xvec2)
        yvec2 = cross_product(zvec, xvec2)
        yvec2 = yvec2 / vec_norm(yvec2)

        if i_sec > 4:
            xvec1 = wingpanels[i_sec]["p1"] - wingpanels[i_sec]["p2"]
            xvec1 = xvec1 / vec_norm(xvec1)
            xvec2 = wingpanels[i_sec]["p4"] - wingpanels[i_sec]["p3"]
            xvec2 = xvec2 / vec_norm(xvec2)

        xloc1 = np.linspace(-L_sec1 / 2, L_sec1 / 2, N_split)
        yloc01 = np.sqrt(R1**2 - (L_sec1 / 2) ** 2)
        yloc1 = -np.sqrt(R1**2 - xloc1**2) + yloc01
        zloc1 = np.zeros(N_split)

        xloc2 = np.linspace(-L_sec2 / 2, L_sec2 / 2, N_split)
        yloc02 = np.sqrt(R2**2 - (L_sec2 / 2) ** 2)
        yloc2 = -np.sqrt(R2**2 - xloc2**2) + yloc02
        zloc2 = np.zeros(N_split)

        vec1 = np.array([xvec1, yvec1, zvec]).T
        vec2 = np.array([xvec2, yvec2, zvec]).T
        ax_pos1 = (wingpanels[i_sec]["p2"] + wingpanels[i_sec]["p1"]) / 2
        ax_pos2 = (wingpanels[i_sec]["p3"] + wingpanels[i_sec]["p4"]) / 2
        temp_coord = np.empty((int(N_split * 2), 3))
        for i_spl in range(N_split):
            coord_loc1 = np.array([xloc1[i_spl], yloc1[i_spl], zloc1[i_spl]])
            coord_loc2 = np.array([xloc2[i_spl], yloc2[i_spl], zloc2[i_spl]])
            coord1 = np.matmul(vec1, coord_loc1) + ax_pos1
            coord2 = np.matmul(vec2, coord_loc2) + ax_pos2

            if i_sec > 4:
                ind = 2 * N_split - 1 - (2 * i_spl + 1)
                temp_coord[ind] = coord1
                ind = 2 * N_split - 1 - 2 * i_spl
                temp_coord[ind] = coord2
            else:
                temp_coord[2 * i_spl] = coord1
                temp_coord[2 * i_spl + 1] = coord2

        if i_sec == 0:
            refined_coord = temp_coord
        else:
            refined_coord = np.append(refined_coord, temp_coord[2::, :], axis=0)

    return refined_coord


############################################


def save_rib_input_list(
    LE_array, TE_array, d_tube_array, camber_array, file_name, root_dir
):

    logging.info(f"shape LE: {len(LE_array)} | shape TE: {len(TE_array)}")

    if len(d_tube_array) != len(LE_array):
        d_tube_array = np.interp(
            np.linspace(0, len(d_tube_array) - 1, len(LE_array)),
            np.arange(len(d_tube_array)),
            d_tube_array,
        )
        camber_array = np.interp(
            np.linspace(0, len(camber_array) - 1, len(LE_array)),
            np.arange(len(camber_array)),
            camber_array,
        )
    logging.info(
        f"shape d_tube: {len(d_tube_array)} | shape camber: {len(camber_array)}"
    )

    # Create wing geometry
    input_rib_list = []
    for LE, TE, d_tube, camber in zip(LE_array, TE_array, d_tube_array, camber_array):
        airfoil_input = ["lei_airfoil_breukels", [d_tube, camber]]
        input_rib_list.append([LE, TE, airfoil_input])

    # save_path = (
    #     Path(root_dir) / "processed_data" / "TUDELFT_V3_LEI_KITE" / f"{file_name}.pkl"
    # )
    logging.info(
        f"file_name: {file_name} | shape: ({len(input_rib_list)},{len(input_rib_list[0])})"
    )

    # with open(save_path, "wb") as file:
    #     pickle.dump(input_rib_list, file)

    # Save wing geometry in a csv file
    csv_file_path = (
        Path(root_dir) / "processed_data" / "TUDELFT_V3_LEI_KITE" / f"{file_name}.csv"
    )
    if csv_file_path is None:
        raise ValueError("You must provide a csv_file_path.")
    if not os.path.exists(csv_file_path):
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    with open(csv_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Write the header
        writer.writerow(
            ["LE_x", "LE_y", "LE_z", "TE_x", "TE_y", "TE_z", "d_tube", "camber"]
        )

        # Write the data for each rib
        for LE, TE, d_tube, camber in zip(
            LE_array, TE_array, d_tube_array, camber_array
        ):
            le_x, le_y, le_z = LE
            te_x, te_y, te_z = TE
            writer.writerow([le_x, le_y, le_z, te_x, te_y, te_z, d_tube, camber])


def extract_d_tube_and_camber_from_surfplan(root_dir):
    surfplan_d_tube_camber_path = (
        Path(root_dir)
        / "data"
        / "TUDELFT_V3_LEI_KITE"
        / "geometry"
        / "surfplan_d_tube_camber.csv"
    )
    d_tube, camber = np.loadtxt(
        surfplan_d_tube_camber_path, delimiter=",", skiprows=1, unpack=True
    )
    # print(f"d_tube: {d_tube}")
    # print(f"camber: {camber}")
    return d_tube, camber


def extract_LE_and_TE_from_CAD_data_csv(root_dir):
    # Getting the geometry of the points
    CAD_path = (
        Path(root_dir)
        / "data"
        / "TUDELFT_V3_LEI_KITE"
        / "geometry"
        # / "Geometry_modified_kcu.csv"
        / "CAD_extracted_coords_V3_kite.csv"
    )
    CAD = np.loadtxt(CAD_path, delimiter=",")

    coord = struct2aero_geometry(CAD) / 1000
    coord = flip_created_coord_in_pairs_if_needed(coord)
    LE, TE = [], []
    for i in range(len(coord)):
        if i % 2 == 0:
            LE.append(coord[i, :])
        else:
            TE.append(coord[i, :])

    logging.info(f"shape LE: {len(LE)} | shape TE: {len(TE)}")

    # make arrays from things
    LE_array = np.array(LE)
    TE_array = np.array(TE)

    # arange LE_array and TE_array from left-to-right
    LE_array = LE_array[np.argsort(-LE_array[:, 1])]
    TE_array = TE_array[np.argsort(-TE_array[:, 1])]

    return LE_array, TE_array


def extract_LE_and_TE_from_CAD_pickle(root_dir):
    CAD_path_OLD = (
        Path(root_dir)
        / "processed_data"
        / "TUDELFT_V3_LEI_KITE"
        / "CAD_extracted_input_rib_list.pkl"
    )
    with open(CAD_path_OLD, "rb") as file:
        CAD_input_rib_list_OLD = pickle.load(file)
    LE_array, TE_array, d_tube_array, camber_array = [], [], [], []
    for i, CAD_rib_i in enumerate(CAD_input_rib_list_OLD):
        LE_array.append(CAD_rib_i[0])
        TE_array.append(CAD_rib_i[1])
        d_tube_array.append(CAD_rib_i[2][1][0])
        camber_array.append(CAD_rib_i[2][1][1])

    logging.info(f"shape LE: {len(LE_array)} | shape TE: {len(TE_array)}")

    return np.array(LE_array), np.array(TE_array), d_tube_array, camber_array


# def refine_LE_and_TE_using_ballooning(LE_array, TE_array, N_split):

#     coord = []
#     for LE, TE in zip(LE_array, TE_array):
#         coord.append(LE)
#         coord.append(TE)
#     coord = np.array(coord)
#     Uinf = np.array([1, 0, 0])

#     N = int(len(coord / 2))
#     ring_geo = "5fil"
#     model = "VSM"
#     controlpoints, rings, wingpanels, ringvec, coord_L = create_geometry_LEI(
#         coord, Uinf, N, ring_geo, model
#     )

#     ball_angle = [1, 10, 15, 18, 20, 18, 15, 10, 1]
#     # Refine mesh, include ballooning
#     coord = refine_LEI_mesh_ballooning(wingpanels, ball_angle, N_split)
#     LE, TE = [], []
#     for i in range(len(coord)):
#         if i % 2 == 0:
#             LE.append(coord[i, :])
#         else:
#             TE.append(coord[i, :])
#     return np.array(LE), np.array(TE)


if __name__ == "__main__":

    # Find the root directory of the repository
    root_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
        root_dir = os.path.abspath(os.path.join(root_dir, ".."))
        if root_dir == "/":
            raise FileNotFoundError(
                "Could not find the root directory of the repository."
            )

    # LE_array, TE_array, d_tube_array, camber_array = extract_LE_and_TE_from_CAD_pickle(
    #     root_dir
    # )
    LE_array, TE_array = extract_LE_and_TE_from_CAD_data_csv(root_dir)
    d_tube_array, camber_array = extract_d_tube_and_camber_from_surfplan(root_dir)
    save_rib_input_list(
        LE_array,
        TE_array,
        d_tube_array,
        camber_array,
        "rib_list_new_run_need_to_specify_the_right_name",
        root_dir,
    )

    # ## Refine the LE and TE using ballooning
    # LE_array_refined, TE_array_refined = refine_LE_and_TE_using_ballooning(
    #     LE_array, TE_array, 2
    # )
    # save_rib_input_list(
    #     LE_array_refined,
    #     TE_array_refined,
    #     d_tube_array,
    #     camber_array,
    #     "rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_refined",
    # )
