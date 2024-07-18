import numpy as np
from VSM.WingGeometry import Wing, flip_created_coord_in_pairs_if_needed
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver
import logging

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
#%% Read the coordinates from the CAD file
coord_struc = np.loadtxt("./data/coordinates/coords_v3_kite.csv", delimiter=",")

## Convert the coordinates to the aero coordinates
coord = struct2aero_geometry(coord_struc)/1000
coord = flip_created_coord_in_pairs_if_needed(coord)
N = len(coord) // 2

# thickness of the leading edge tube
LE_thicc = np.ones(N) * 0.1

# camber of the leading edge airfoil
camber = np.ones(N) * 0.095

#Create wing geometry
wing = Wing(N, "unchanged")
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
CL = np.zeros(len(aoas))
CD = np.zeros(len(aoas))
gamma = np.zeros((len(aoas), len(wing_aero.panels)))
for i, aoa in enumerate(aoas):
    aoa = np.deg2rad(aoa)
    sideslip = 0
    Umag = 20

    wing_aero.va = np.array([np.cos(aoa)*np.cos(sideslip), np.sin(sideslip), np.sin(aoa)]) * Umag

    # Calculate the aerodynamic forces
    

    results, wing_aero= VSM.solve(wing_aero)
    CL[i] = results["cl"]
    CD[i] = results["cd"]
    gamma[i] = results["gamma_distribution"]

# Plot the results