import numpy as np
from VSM.WingGeometry import Wing, flip_created_coord_in_pairs_if_needed
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver
import logging
#TODO: Convert into a Kite class

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

def refine_LEI_mesh(coord, N_sect, N_split):
    refined_coord = []

    for i_sec in range(N_sect):
        temp_coord = np.empty((int(N_split * 2), 3))
        for i_spl in range(N_split):
            temp_coord[2 * i_spl] = (
                coord[2 * i_sec, :] * (N_split - i_spl) / N_split
                + coord[2 * (i_sec + 1), :] * (i_spl) / N_split
            )
            temp_coord[2 * i_spl + 1] = (
                coord[2 * i_sec + 1, :] * (N_split - i_spl) / N_split
                + coord[2 * (i_sec + 1) + 1, :] * (i_spl) / N_split
            )
        if i_sec == 0:
            refined_coord = temp_coord
        else:
            refined_coord = np.append(refined_coord, temp_coord, axis=0)

    refined_coord = np.append(
        refined_coord, [coord[2 * N_sect, :], coord[2 * N_sect + 1, :]], axis=0
    )

    return refined_coord

#%% Read the coordinates from the CAD file
coord_struc = np.loadtxt("./data/coordinates/coords_v3_kite.csv", delimiter=",")

## Convert the coordinates to the aero coordinates
coord_aero = struct2aero_geometry(coord_struc)/1000
n_aero = len(coord_aero) // 2
coord = refine_LEI_mesh(coord_aero, n_aero-1, 5)
coord = flip_created_coord_in_pairs_if_needed(coord)

n_sections = len(coord) // 2
n_panels = n_sections - 1
# thickness of the leading edge tube
LE_thicc = np.ones(n_sections) * 0.1

# camber of the leading edge airfoil
camber = np.ones(n_sections) * 0.095


#Create wing geometry
wing = Wing(n_panels, "unchanged")
for idx,idx2 in enumerate(range(0, len(coord), 2)):
    logging.debug(f"coord[{idx2}] = {coord[idx2]}")
    wing.add_section(
        coord[idx2],
        coord[idx2 + 1],
        ["lei_airfoil_breukels", [LE_thicc[idx], camber[idx]]],
    )

wing_aero = WingAerodynamics([wing])
# VSM
VSM = Solver(
    aerodynamic_model_type="VSM",
)

aoas = np.arange(0, 21, 1)
cl_straight = np.zeros(len(aoas))
cd_straight = np.zeros(len(aoas))
cs_straight = np.zeros(len(aoas))
gamma_straight = np.zeros((len(aoas), len(wing_aero.panels)))
cl_turn = np.zeros(len(aoas))
cd_turn = np.zeros(len(aoas))
cs_turn = np.zeros(len(aoas))
gamma_turn = np.zeros((len(aoas), len(wing_aero.panels)))
yaw_rate = 1.5
for i, aoa in enumerate(aoas):
    aoa = np.deg2rad(aoa)
    sideslip = 0
    Umag = 20


    wing_aero.va = np.array([np.cos(aoa)*np.cos(sideslip), np.sin(sideslip), np.sin(aoa)]) * Umag, 0
    results, wing_aero= VSM.solve(wing_aero)
    cl_straight[i] = results["cl"]
    cd_straight[i] = results["cd"]
    cs_straight[i] = results["cs"]
    gamma_straight[i] = results["gamma_distribution"]
    print(f"Straight: aoa: {aoa}, CL: {cl_straight[i]}, CD: {cd_straight[i]}, CS: {cs_straight[i]}")

    wing_aero.va = np.array([np.cos(aoa)*np.cos(sideslip), np.sin(sideslip), np.sin(aoa)]) * Umag, yaw_rate
    results, wing_aero= VSM.solve(wing_aero)
    cl_turn[i] = results["cl"]
    cd_turn[i] = results["cd"]
    cs_turn[i] = results["cs"]
    gamma_turn[i] = results["gamma_distribution"]
    print(f"Turn: aoa: {aoa}, CL: {cl_turn[i]}, CD: {cd_turn[i]}, CS: {cs_turn[i]}")
    

# Plot the results
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, figsize=(10, 15))
axs[0].plot(aoas, cl_straight, label="$C_L$ straight")
axs[0].plot(aoas, cl_turn, label="$C_L$ turn")
axs[0].legend()
axs[0].grid()
axs[0].set_ylabel("CL")
axs[0].set_xlabel("AOA [deg]")
axs[1].plot(aoas, cd_straight, label="$C_D$ straight")
axs[1].plot(aoas, cd_turn, label="$C_D$ turn")
axs[1].legend()
axs[1].set_ylabel("CD")
axs[1].set_xlabel("AOA [deg]")
axs[1].grid()
axs[2].plot(aoas, cs_straight, label="$C_S$ straight")
axs[2].plot(aoas, cs_turn, label="$C_S$ turn")
axs[2].legend()
axs[2].set_ylabel("CS")
axs[2].set_xlabel("AOA [deg]")
axs[2].grid()
plt.show()


# plot two gamma distributions
fig, axs = plt.subplots(2)
for i in range(0, len(gamma_straight), 5):
    axs[0].plot(gamma_straight[i], label=f"AOA: {aoas[i]}")
    axs[1].plot(gamma_turn[i], label=f"AOA: {aoas[i]}")
axs[0].legend()
axs[0].set_title("Straight flight")
axs[0].set_ylabel("Gamma")
axs[1].legend()
axs[1].set_title("Turning flight")
axs[1].set_ylabel("Gamma")


plt.show()