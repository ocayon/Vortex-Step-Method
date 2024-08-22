import numpy as np
import logging
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
import time as time
from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver
from VSM.plotting import plot_distribution, plot_polars


def run_speed_test_cprofile():
    # Find the root directory of the repository
    root_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
        root_dir = os.path.abspath(os.path.join(root_dir, ".."))
        if root_dir == "/":
            raise FileNotFoundError(
                "Could not find the root directory of the repository."
            )

    # Load from Pickle file
    CAD_path = (
        Path(root_dir)
        / "processed_data"
        / "TUDELFT_V3_LEI_KITE"
        / "rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber.pkl"
    )
    with open(CAD_path, "rb") as file:
        CAD_input_rib_list = pickle.load(file)

    # Create wing geometry
    n_panels = 36
    spanwise_panel_distribution = "split_provided"
    CAD_wing = Wing(n_panels, spanwise_panel_distribution)

    # Settign the solver type
    VSM = Solver(aerodynamic_model_type="VSM")
    VSM_with_stall_correction = Solver(
        aerodynamic_model_type="VSM",
        is_with_artificial_damping=True,
        relaxation_factor=0.04,
        is_only_f_and_gamma_output=True,
    )

    # Defining va
    aoa_rad = np.deg2rad(20)
    side_slip = 0
    yaw_rate = 0
    Umag = 15
    va_definition = (
        np.array(
            [
                np.cos(aoa_rad) * np.cos(side_slip),
                np.sin(side_slip),
                np.sin(aoa_rad),
            ]
        )
        * Umag,
        yaw_rate,
    )

    logging.info("Starting the simulation")

    time_start = time.time()
    # Populate the wing geometry
    for i, CAD_rib_i in enumerate(CAD_input_rib_list):
        CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
    print(f"Time : {time.time() - time_start:.2f} s")

    # Create wing aerodynamics objects
    CAD_wing_aero = WingAerodynamics([CAD_wing])
    print(f"Time : {time.time() - time_start:.2f} s")

    # Setting va
    CAD_wing_aero.va = va_definition
    print(f"Time : {time.time() - time_start:.2f} s")

    # Solving
    results = VSM_with_stall_correction.solve(CAD_wing_aero)

    time_end = time.time()
    print(f"Time taken for the simulation: {time_end - time_start:.2f}  seconds")
    return


def run_speed_test(gamma):
    # Find the root directory of the repository
    root_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
        root_dir = os.path.abspath(os.path.join(root_dir, ".."))
        if root_dir == "/":
            raise FileNotFoundError(
                "Could not find the root directory of the repository."
            )

    # Load from Pickle file
    CAD_path = (
        Path(root_dir)
        / "processed_data"
        / "TUDELFT_V3_LEI_KITE"
        / "rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber.pkl"
    )
    with open(CAD_path, "rb") as file:
        CAD_input_rib_list = pickle.load(file)

    # Create wing geometry
    n_panels = 36
    spanwise_panel_distribution = "split_provided"
    CAD_wing = Wing(n_panels, spanwise_panel_distribution)

    # Settign the solver type
    VSM = Solver(aerodynamic_model_type="VSM")
    VSM_with_stall_correction = Solver(
        aerodynamic_model_type="VSM",
        is_with_artificial_damping=True,
        relaxation_factor=0.04,
        is_only_f_and_gamma_output=True,
    )

    # Defining va
    aoa_rad = np.deg2rad(20)
    side_slip = 0
    yaw_rate = 0
    Umag = 15
    va_definition = (
        np.array(
            [
                np.cos(aoa_rad) * np.cos(side_slip),
                np.sin(side_slip),
                np.sin(aoa_rad),
            ]
        )
        * Umag,
        yaw_rate,
    )

    logging.info("Starting the simulation")

    time_start = time.time()
    # Populate the wing geometry
    for i, CAD_rib_i in enumerate(CAD_input_rib_list):
        CAD_wing.add_section(CAD_rib_i[0], CAD_rib_i[1], CAD_rib_i[2])
    # print(f"Time : {time.time() - time_start:.2f} s")

    # Create wing aerodynamics objects
    CAD_wing_aero = WingAerodynamics([CAD_wing])
    # print(f"Time : {time.time() - time_start:.2f} s")

    # Setting va
    CAD_wing_aero.va = va_definition
    # print(f"Time : {time.time() - time_start:.2f} s")

    # Solving
    results = VSM_with_stall_correction.solve(CAD_wing_aero, gamma_distribution=gamma)

    time_end = time.time()
    # print(f"Time taken for the simulation: {time_end - time_start:.2f}  seconds")
    return results["gamma_distribution"]


if __name__ == "__main__":
    # from line_profiler import LineProfiler
    import cProfile
    from numba import jit
    import time as time

    # @jit
    # def test_numba():
    #     return 3 + 4

    # test_numba_output = test_numba()
    # print(f"test_numba_output: {test_numba_output}")

    gamma = None
    print("Starting the speed test")
    time_before = time.time()
    for i in range(10):
        time_before_this_loop = time.time()
        gamma = run_speed_test(gamma)
        print(f"Time taken: {time.time() - time_before_this_loop:.2f} s")
        gamma = None
    # cProfile.run("run_speed_test_cprofile()", sort="tottime")
    # lp = LineProfiler()
    # VSM_with_stall_correction = Solver(
    #     aerodynamic_model_type="VSM",
    #     is_with_artificial_damping=True,
    # )
    # lp_wrapper = lp(VSM_with_stall_correction.calculate_gamma_new_iteratively())
    # lp_wrapper()
    # lp.print_stats()

# # Testing the results
# y_coords = [panels.aerodynamic_center[1] for panels in CAD_wing_aero.panels]
# plot_distribution(
#     [y_coords],
#     [results],
#     ["VSM speed run"],
#     title="spanwise_distribution",
#     data_type=".pdf",
#     save_path=None,
#     is_save=False,
#     is_show=True,
# )
# ### plotting polar
# save_folder = Path(root_dir) / "results" / "TUD_V3_LEI_KITE"
# angle_of_attack_range = np.linspace(-5, 25, 15)
# path_cfd_lebesque_3e6 = (
#     Path(root_dir)
#     / "data"
#     / "TUD_V3_LEI_KITE"
#     / "literature_results"
#     / "V3_CL_CD_RANS_CFD_lebesque_2020_Rey_3e6.csv"
# )
# path_cfd_lebesque_100e4 = (
#     Path(root_dir)
#     / "data"
#     / "TUD_V3_LEI_KITE"
#     / "literature_results"
#     / "V3_CL_CD_RANS_Lebesque_2024_Rey_100e4.csv"
# )
# path_cfd_lebesque_300e4 = (
#     Path(root_dir)
#     / "data"
#     / "TUD_V3_LEI_KITE"
#     / "literature_results"
#     / "V3_CL_CD_RANS_Lebesque_2024_Rey_300e4.csv"
# )
# path_wind_tunnel_poland_56e4 = (
#     Path(root_dir)
#     / "data"
#     / "TUD_V3_LEI_KITE"
#     / "literature_results"
#     / "V3_CL_CD_Wind_Tunnel_Poland_2024_Rey_56e4.csv"
# )
# plot_polars(
#     solver_list=[
#         VSM_with_stall_correction,
#     ],
#     wing_aero_list=[CAD_wing_aero],
#     label_list=[
#         f"CAD with stall correction",
#         f"RANS CFD Lebesque Rey = 10e5 (2020)",
#         f"RANS CFD Lebesque Rey = 30e5 (2020)",
#         f"Wind Tunnel Poland Rey = 5.6e5 (2024)",
#     ],
#     literature_path_list=[
#         path_cfd_lebesque_100e4,
#         path_cfd_lebesque_300e4,
#         path_wind_tunnel_poland_56e4,
#     ],
#     angle_range=angle_of_attack_range,
#     angle_type="angle_of_attack",
#     angle_of_attack=0,
#     side_slip=side_slip,
#     yaw_rate=yaw_rate,
#     Umag=Umag,
#     title="polars_CAD_vs_surfplan",
#     data_type=".pdf",
#     save_path=Path(save_folder) / "polars",
#     is_save=False,
#     is_show=True,
# )
