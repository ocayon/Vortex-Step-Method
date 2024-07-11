# Go back to root folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)
from Aerostructural_model_LEI.functions import functions_VSM_LLT as VSM_thesis
from tests.WingAerodynamics.test_wing_aero_object_against_create_geometry_general import (
    create_geometry_general,
    create_geometry_from_wing_object,
)

############################################
### Old script
############################################
idx_aoa = 0
N = 3
max_chord = 1
span = 2.36
AR = span**2 / (np.pi * span * max_chord / 4)
dist = "cos"
coord = VSM.generate_coordinates_el_wing(max_chord, span, N, dist)
Atot = max_chord / 2 * span / 2 * np.pi

Umag = 20
aoas = [np.deg2rad(3)]
Uinf = np.array([np.cos(aoas[idx_aoa]), 0, np.sin(aoas[idx_aoa])]) * Umag
model = "LLT"
# Define system of vorticity
controlpoints, rings, bladepanels, ringvec, coord_L = create_geometry_general(
    coord, Uinf, N, ring_geo, model
)

#####################################
### Running the test_elliptical_wing.py script

from tests.utils import (
    asserting_all_elements_in_list_dict,
    asserting_all_elements_in_list_list_dict,
)
from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver


### Elliptical Wing
wing = Wing(N, "unchanged")
for i in range(int(len(coord) / 2)):
    wing.add_section(coord[2 * i], coord[2 * i + 1], ["inviscid"])
wing_aero = WingAerodynamics([wing])
wing_aero.va = Uinf
# Generate geometry from wing object
new_controlpoints, new_rings, new_wingpanels, new_ringvec, new_coord_L = (
    create_geometry_from_wing_object(wing_aero, model)
)
# Check if geometry input is the same
asserting_all_elements_in_list_dict(controlpoints, new_controlpoints)
asserting_all_elements_in_list_list_dict(rings, new_rings)
asserting_all_elements_in_list_dict(bladepanels, new_wingpanels)
asserting_all_elements_in_list_dict(ringvec, new_ringvec)
assert np.allclose(coord_L, new_coord_L, atol=1e-5)


#############################
## Solving OLD METHOD
#############################
# Solve for Gamma
Fmag, Gamma, aero_coeffs = VSM.solve_lifting_line_system_matrix_approach_semiinfinite(
    ringvec, controlpoints, rings, Uinf, Gamma0, data_airf, conv_crit, model
)
# Output forces
F_rel, F_gl, Ltot, Dtot, CL_old, CD_old, CS = VSM.output_results(
    Fmag, aero_coeffs, ringvec, Uinf, controlpoints, Atot
)

#############################
# Solving NEW METHOD
#############################
LLT = Solver(aerodynamic_model_type="LLT", core_radius_fraction=1e-5)
results_LLT, wing_aero_LLT = LLT.solve(wing_aero)

# from tests.verification_cases import test_elliptical_wing
# aoa_deg = [3]
# results = test_elliptical_wing.calculate_elliptical_wing(
#     N, AR=3, plot_wing=False, spacing="unchanged", aoa_deg=aoa_deg
# )
# results_LTT = [results_aoa[2] for results_aoa in results]
# wing_aero_LTT = [results_aoa[-1] for results_aoa in results][0]

#############################
## ANALYTICAL
#############################
CL_th = 2 * np.pi * aoas[idx_aoa] / (1 + 2 / AR)
CDi_th = CL_th**2 / np.pi / AR


logging.info("--------------")
logging.info(f'CL_old = {CL_old}, CL_new = {results_LLT["cl"]}, CL_ana = {CL_th}')
logging.info(f'CD_old = {CD_old}, CD_new = {results_LLT["cd"]}, CD_ana = {CDi_th}')
logging.info(f'gamma_old = {Gamma}, gamma_new = {results_LLT["gamma_distribution"]}')
