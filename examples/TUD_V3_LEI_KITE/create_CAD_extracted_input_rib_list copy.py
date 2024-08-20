import numpy as np

import logging
import pickle
import os
from pathlib import Path
from VSM.WingGeometry import Wing, flip_created_coord_in_pairs_if_needed

# Find the root directory of the repository
root_dir = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
    root_dir = os.path.abspath(os.path.join(root_dir, ".."))
    if root_dir == "/":
        raise FileNotFoundError("Could not find the root directory of the repository.")


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


def airfoil_coeffs(alpha, coeffs):

    cl = np.interp(alpha * 180 / np.pi, coeffs[:, 0], coeffs[:, 1])
    cd = np.interp(alpha * 180 / np.pi, coeffs[:, 0], coeffs[:, 2])
    cm = np.interp(alpha * 180 / np.pi, coeffs[:, 0], coeffs[:, 3])

    return cl, cd, cm


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


def airf_data_from_uri_thesis_untouched(CAD, N_split):

    #   Model and program specifics
    ring_geo = "5fil"
    model = "VSM"

    # Convergence criteria
    conv_crit = {"Niterations": 1500, "error": 1e-5, "Relax_factor": 0.01}

    # Wind speed mag and direction
    Umag = 22
    aoa = 12 * np.pi / 180
    sideslip = 0 / 180 * np.pi
    Uinf = (
        np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)]) * Umag
    )

    coord = struct2aero_geometry(CAD) / 1000
    Atot = 19.753
    N = int(len(coord) / 2)

    controlpoints, rings, wingpanels, ringvec, coord_L = create_geometry_LEI(
        coord, Uinf, N, ring_geo, model
    )
    # Correct angle of attack
    aoa0 = np.arctan(
        (wingpanels[4]["p3"][2] - wingpanels[4]["p2"][2])
        / (wingpanels[4]["p3"][0] - wingpanels[4]["p2"][0])
    )

    # Camber for each section
    k = [0.018, 0.028, 0.038, 0.048, 0.057, 0.057, 0.048, 0.038, 0.028, 0.018]
    k = [0.02, 0.03, 0.04, 0.05, 0.06, 0.06, 0.05, 0.04, 0.03, 0.02]
    # k = 0.09*np.ones(10)
    # Thickness in each section
    t = [
        0.118753,
        0.151561,
        0.178254,
        0.19406,
        0.202418,
        0.202418,
        0.19406,
        0.178254,
        0.151561,
        0.118753,
    ]
    # t = [0.118753,0.17,0.18,0.19406,0.202418,0.202418,0.19406,0.178254,0.151561,0.118753]

    # N_split = 16
    # Ballooning angle
    ball_angle = [1, 10, 15, 18, 20, 18, 15, 10, 1]
    # Refine mesh, include ballooning
    coord = refine_LEI_mesh_ballooning(wingpanels, ball_angle, N_split)
    # Define system of vorticity
    controlpoints, rings, wingpanels, ringvec, coord_L = create_geometry_LEI(
        coord, Uinf, int((N - 1) * (N_split - 1) + 1), ring_geo, model
    )
    N = int(len(coord) / 2)  # Number of section after refining the mesh

    Gamma0 = np.zeros(len(controlpoints))

    aoas = np.arange(-20, 21, 1)

    thicc = np.array([])
    camb = np.array([])
    for i in range(9):
        temp_t = np.linspace(t[i], t[i + 1], N_split)
        temp_k = np.linspace(k[i], k[i + 1], N_split)
        temp1 = []
        temp2 = []
        for a in range(len(temp_t) - 1):
            temp1.append((temp_t[a] + temp_t[a + 1]) / 2)
            temp2.append((temp_k[a] + temp_k[a + 1]) / 2)
        thicc = np.append(thicc, temp1)
        camb = np.append(camb, temp2)

    data_airf_br = np.empty((len(aoas), 4, N - 1))
    t_c = np.empty(N - 1)
    for i in range(N - 1):
        for j in range(len(aoas)):
            chord = controlpoints[i]["chord"]
            t_c[i] = thicc[i] / chord
            alpha = aoas[j]

            Cl, Cd, Cm = LEI_airf_coeff(t_c[i], camb[i], alpha)
            data_airf_br[j, 0, i] = alpha
            data_airf_br[j, 1, i] = Cl
            data_airf_br[j, 2, i] = Cd
            data_airf_br[j, 3, i] = Cm

    return coord, thicc, camb


def airf_data_from_uri_thesis_adjusted(CAD, N_split):

    coord = struct2aero_geometry(CAD) / 1000

    # Camber for each section
    # k = [0.018, 0.028, 0.038, 0.048, 0.057, 0.057, 0.048, 0.038, 0.028, 0.018]
    k = [0.02, 0.03, 0.04, 0.05, 0.06, 0.06, 0.05, 0.04, 0.03, 0.02]
    # k = 0.09*np.ones(10)
    # Thickness in each section
    t = [
        0.118753,
        0.151561,
        0.178254,
        0.19406,
        0.202418,
        0.202418,
        0.19406,
        0.178254,
        0.151561,
        0.118753,
    ]
    # t = [0.118753,0.17,0.18,0.19406,0.202418,0.202418,0.19406,0.178254,0.151561,0.118753]

    if N_split > 1:
        N = int(len(coord) / 2)

        #   Model and program specifics
        ring_geo = "5fil"
        model = "VSM"
        # Wind speed mag and direction
        Umag = 22
        aoa = 12 * np.pi / 180
        sideslip = 0 / 180 * np.pi
        Uinf = (
            np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)])
            * Umag
        )
        controlpoints, rings, wingpanels, ringvec, coord_L = create_geometry_LEI(
            coord, Uinf, N, ring_geo, model
        )
        # Ballooning angle
        ball_angle = [1, 10, 15, 18, 20, 18, 15, 10, 1]
        # Refine mesh, include ballooning
        coord = refine_LEI_mesh_ballooning(wingpanels, ball_angle, N_split)

    return coord, t, k


def create_and_save_rib_list(
    coord, LE_thickness, camber_dimensionless, file_name, root_dir
):

    # Create wing geometry
    input_rib_list = []
    for idx, idx2 in enumerate(range(0, len(coord) - 2, 2)):

        logging.info(f"idx: {idx} | coord[{idx2}] = {coord[idx2]}")

        ## Create the rib input list_le_te
        coord_length = np.linalg.norm(coord[idx2] - coord[idx2 + 1])
        LE = coord[idx2]
        TE = coord[idx2 + 1]
        camber_i = camber_dimensionless[idx]
        LE_thickness_i = LE_thickness[idx] / coord_length
        airfoil_input = ["lei_airfoil_breukels", [LE_thickness_i, camber_i]]
        input_rib_list.append([LE, TE, airfoil_input])

    save_path = (
        Path(root_dir) / "processed_data" / "TUD_V3_LEI_KITE" / f"{file_name}.pkl"
    )
    with open(save_path, "wb") as file:
        pickle.dump(input_rib_list, file)


if __name__ == "__main__":

    # Getting the geometry of the points
    CAD_path = (
        Path(root_dir)
        / "data"
        / "TUD_V3_LEI_KITE"
        / "geometry"
        / "Geometry_modified_kcu.csv"
    )
    CAD = np.loadtxt(CAD_path, delimiter=",")

    # Getting the original, thesis defined geometry
    coord, thicc, camb = airf_data_from_uri_thesis_untouched(CAD, 8)
    coord = flip_created_coord_in_pairs_if_needed(coord)

    create_and_save_rib_list(
        coord, thicc, camb, "CAD_rib_input_list_untouched_thesis_n_split_8", root_dir
    )

    # Getting just the V3 input shape
    coord, thicc, camb = airf_data_from_uri_thesis_adjusted(CAD, 1)
    coord = flip_created_coord_in_pairs_if_needed(coord)
    create_and_save_rib_list(
        coord, thicc, camb, "CAD_rib_input_list_adjusted_thesis_n_split_1", root_dir
    )
