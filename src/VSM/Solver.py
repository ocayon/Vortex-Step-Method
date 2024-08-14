import numpy as np
import logging

# Maurits-tips :)
# call the methods of child-classes, inhereted or composed of
# do not call the attributes of child-classes, call them through getter methods
# only pass the attributes that you need to pass, not the whole object
# only use the methods of level higher/lower, not grabbing methods from higher/lower
# class solve_VSM(Solver)
# class solve_LLM(Solver)


# make abstract class
class Solver:
    """Solver class is used to solve the aerodynamic model

    It is used to solve the circulation distribution of the wing,
    and calculate the aerodynamic forces

    init input:
        aerodynamic_model_type (str): Type of aerodynamic model, VSM or LLT
        density (float): = 1.225 | Air density
        max_iterations (int): = 1000 | Maximum number of iterations
        allowed_error (float): = 1e-5 | Allowed error
        tol_reference_error (float): = 0.001 | Reference error
        relaxation_factor (float): = 0.03 | Relaxation factor
        artificial_damping (dict): = {"k2": 0.0, "k4": 0.0} | Artificial damping
        type_initial_gamma_distribution (str): = "elliptic" | Type of initial gamma distribution
        core_radius_fraction (float): = 1e-20 | Core radius fraction

    Methods:
        solve: Solve the circulation distribution
        solve_iterative_loop: Solve the circulation distribution using an iterative loop
        calculate_artificial_damping: Calculate artificial damping
    """

    def __init__(
        self,
        ### Below are all settings, with a default value, that can but don't have to be changed
        aerodynamic_model_type: str = "VSM",
        density: float = 1.225,
        max_iterations: int = 1000,
        allowed_error: float = 1e-5,
        tol_reference_error: float = 0.001,
        relaxation_factor: float = 0.03,
        artificial_damping: dict = {"k2": 0.0, "k4": 0.0},
        type_initial_gamma_distribution: str = "elliptic",
        core_radius_fraction: float = 1e-20,
        ## TODO: would be nice to having these defined here instead of inside the panel class?
        # aerodynamic_center_location: float = 0.25,
        # control_point_location: float = 0.75,
        ## TODO: these are hardcoded in the Filament, should be defined here
        # alpha_0 = 1.25643
        # nu = 1.48e-5
    ):
        self.aerodynamic_model_type = aerodynamic_model_type
        self.density = density
        self.max_iterations = max_iterations
        self.allowed_error = allowed_error
        self.tol_reference_error = tol_reference_error
        self.relaxation_factor = relaxation_factor
        self.artificial_damping = artificial_damping
        self.type_initial_gamma_distribution = type_initial_gamma_distribution
        self.core_radius_fraction = core_radius_fraction

    def solve(self, wing_aero, gamma_distribution=None):

        if wing_aero.va is None:
            raise ValueError("Inflow conditions are not set")

        # Calculate the new circulation distribution iteratively
        gamma_new = self.calculate_gamma_new_iteratively(wing_aero, gamma_distribution)

        # Calculating results (incl. updating angle of attack for VSM)
        results = wing_aero.calculate_results(
            gamma_new,
            self.density,
            self.aerodynamic_model_type,
            self.core_radius_fraction,
        )

        return results, wing_aero

    def calculate_gamma_new_iteratively(self, wing_aero, gamma_distribution):
        AIC_x, AIC_y, AIC_z = wing_aero.calculate_AIC_matrices(
            self.aerodynamic_model_type, self.core_radius_fraction
        )

        logging.debug(f"AIC_x: {AIC_x}")
        logging.debug(f"AIC_y: {AIC_y}")
        logging.debug(f"AIC_z: {AIC_z}")

        # TODO: CPU optimization: instantiate non-changing (geometric dependent) attributes here
        # TODO: Further optimization: is to populate only in the loop, not create new arrays
        panels = wing_aero.panels
        n_panels = wing_aero.n_panels
        z_airf_array = np.array([panel.z_airf for panel in panels])
        va_array = np.array([panel.va for panel in panels])
        chord_array = np.array([panel.chord for panel in panels])
        alpha = np.zeros(len(panels))
        gamma = np.zeros(len(panels))
        # Adjust relaxation factor based on the number of panels
        relaxation_factor = self.relaxation_factor * 20 / n_panels

        # initialize gamma distribution inside
        if (
            gamma_distribution is None
            and self.type_initial_gamma_distribution == "elliptic"
        ):
            gamma_new = wing_aero.calculate_circulation_distribution_elliptical_wing()
        elif len(gamma_distribution) == n_panels:
            gamma_new = gamma_distribution

        # TODO: for now use ZERO, remove this to speed things up
        gamma_new = np.zeros(len(gamma_new))
        # logging.info("Initial gamma_new: %s", gamma_new)

        converged = False
        for i in range(self.max_iterations):

            gamma = np.array(gamma_new)

            for icp, panel in enumerate(panels):
                # Initialize induced velocity to 0
                u = 0
                v = 0
                w = 0
                # Compute induced velocities with previous gamma distribution
                for jring, gamma_jring in enumerate(gamma):
                    u = u + AIC_x[icp][jring] * gamma_jring
                    # x-component of velocity
                    v = v + AIC_y[icp][jring] * gamma_jring
                    # y-component of velocity
                    w = w + AIC_z[icp][jring] * gamma_jring
                    # z-component of velocity

                # TODO: shouldn't grab from different classes inside the solver for CPU-efficiency
                induced_velocity = np.array([u, v, w])

                # This is double checked
                alpha[icp], relative_velocity = (
                    panel.calculate_relative_alpha_and_relative_velocity(
                        induced_velocity
                    )
                )
                relative_velocity_crossz = np.cross(relative_velocity, panel.z_airf)
                Umag = np.linalg.norm(relative_velocity_crossz)
                Uinfcrossz = np.cross(panel.va, panel.z_airf)
                Umagw = np.linalg.norm(Uinfcrossz)

                # TODO: CPU this should ideally be instantiated upfront, from the wing_aero object
                # Lookup cl for this specific alpha
                cl = panel.calculate_cl(alpha[icp])

                # Find the new gamma using Kutta-Joukouski law
                gamma_new[icp] = 0.5 * Umag**2 / Umagw * cl * chord_array[icp]

                # logging.info("--------- icp: %d", icp)
                # logging.info("induced_velocity: %s", induced_velocity)
                # logging.info("alpha: %f", alpha)
                # logging.info("relative_velocity: %s", relative_velocity)
                # logging.info("cl: %f", cl)
                # logging.info("Umag: %f", Umag)
                # logging.info("Umagw: %f", Umagw)
                # logging.info("chord: %f", chord_array[icp])

            # Dealing with stalled cases
            stall = False
            # TODO: aoa_stall should be input and not fixed, and also adjusted
            aoa_stall = np.deg2rad(20.0)
            for ia, alpha_i in enumerate(alpha):
                if alpha_i > aoa_stall:
                    # TODO: set to True
                    stall = False
                    logging.debug(
                        "Stall detected, alpha[i]: %f, it: %d, panel_number: %d"
                        % (np.rad2deg(alpha_i), i, ia)
                    )
                    break
            if not stall:
                damp = 0
            else:
                damp = self.calculate_artificial_damping(gamma)
            gamma_new = (
                (1 - relaxation_factor) * gamma + relaxation_factor * gamma_new + damp
            )

            # Checking Convergence
            reference_error = np.amax(np.abs(gamma_new))
            reference_error = max(reference_error, self.tol_reference_error)
            error = np.amax(np.abs(gamma_new - gamma))
            normalized_error = error / reference_error

            # logging.info("Iteration: %d, normalized_error: %f", _, normalized_error)
            # logging.info("Iteration: %d, reference_error: %f", _, reference_error)
            # logging.info("Iteration: %d", i)
            # logging.info("gamma: %s", gamma)
            # logging.info("gamma_new: %s", gamma_new)

            # relative error
            if normalized_error < self.allowed_error:
                # if error smaller than limit, stop iteration cycle
                converged = True
                break

        if converged:
            logging.info("------------------------------------")
            logging.info(
                f"{self.aerodynamic_model_type} Converged after {i} iterations"
            )
            logging.info("------------------------------------")
        elif not converged:
            logging.info("------------------------------------")
            logging.info(
                f"{self.aerodynamic_model_type} Not converged after {str(self.max_iterations)} iterations"
            )
            logging.info("------------------------------------")

        return gamma_new

    def calculate_artificial_damping(self, gamma):
        n_gamma = len(gamma)
        damp = np.zeros(n_gamma)
        for ig, gamma_ig in enumerate(gamma):
            if ig == 0:
                gim2 = gamma[0]
                gim1 = gamma[0]
                gi = gamma[0]
                gip1 = gamma[1]
                gip2 = gamma[2]
            elif ig == 1:
                gim2 = gamma[0]
                gim1 = gamma[0]
                gi = gamma[1]
                gip1 = gamma[2]
                gip2 = gamma[3]
            elif ig == n_gamma - 2:
                gim2 = gamma[n_gamma - 4]
                gim1 = gamma[n_gamma - 3]
                gi = gamma[n_gamma - 2]
                gip1 = gamma[n_gamma - 1]
                gip2 = gamma[n_gamma - 1]
            elif ig == n_gamma - 1:
                gim2 = gamma[n_gamma - 3]
                gim1 = gamma[n_gamma - 2]
                gi = gamma[n_gamma - 1]
                gip1 = gamma[n_gamma - 1]
                gip2 = gamma[n_gamma - 1]
            else:
                gim2 = gamma[ig - 2]
                gim1 = gamma[ig - 1]
                gi = gamma[ig]
                gip1 = gamma[ig + 1]
                gip2 = gamma[ig + 2]

            dif2 = (gip1 - gi) - (gi - gim1)
            dif4 = (gip2 - 3.0 * gip1 + 3.0 * gi - gim1) - (
                gip1 - 3.0 * gi + 3.0 * gim1 - gim2
            )
            damp[ig] = (
                self.artificial_damping["k2"] * dif2
                - self.artificial_damping["k4"] * dif4
            )
        return damp


# make abstract class
class Solver_with_stall_correction:
    """Solver class is used to solve the aerodynamic model

    It is used to solve the circulation distribution of the wing,
    and calculate the aerodynamic forces

    init input:
        aerodynamic_model_type (str): Type of aerodynamic model, VSM or LLT
        density (float): = 1.225 | Air density
        max_iterations (int): = 1000 | Maximum number of iterations
        allowed_error (float): = 1e-5 | Allowed error
        tol_reference_error (float): = 0.001 | Reference error
        relaxation_factor (float): = 0.03 | Relaxation factor
        artificial_damping (dict): = {"k2": 0.0, "k4": 0.0} | Artificial damping
        type_initial_gamma_distribution (str): = "elliptic" | Type of initial gamma distribution
        core_radius_fraction (float): = 1e-20 | Core radius fraction

    Methods:
        solve: Solve the circulation distribution
        solve_iterative_loop: Solve the circulation distribution using an iterative loop
        calculate_artificial_damping: Calculate artificial damping
    """

    def __init__(
        self,
        ### Below are all settings, with a default value, that can but don't have to be changed
        aerodynamic_model_type: str = "VSM",
        density: float = 1.225,
        max_iterations: int = 1000,
        allowed_error: float = 1e-5,
        tol_reference_error: float = 0.001,
        relaxation_factor: float = 0.03,
        artificial_damping: dict = {"k2": 0.0, "k4": 0.0},
        type_initial_gamma_distribution: str = "elliptic",
        core_radius_fraction: float = 1e-20,
        ## TODO: would be nice to having these defined here instead of inside the panel class?
        # aerodynamic_center_location: float = 0.25,
        # control_point_location: float = 0.75,
        ## TODO: these are hardcoded in the Filament, should be defined here
        # alpha_0 = 1.25643
        # nu = 1.48e-5
    ):
        self.aerodynamic_model_type = aerodynamic_model_type
        self.density = density
        self.max_iterations = max_iterations
        self.allowed_error = allowed_error
        self.tol_reference_error = tol_reference_error
        self.relaxation_factor = relaxation_factor
        self.artificial_damping = artificial_damping
        self.type_initial_gamma_distribution = type_initial_gamma_distribution
        self.core_radius_fraction = core_radius_fraction

    def solve(self, wing_aero, gamma_distribution=None):

        if wing_aero.va is None:
            raise ValueError("Inflow conditions are not set")

        # Calculate the new circulation distribution iteratively
        gamma_new = self.calculate_gamma_new_iteratively(wing_aero, gamma_distribution)

        # Calculating results (incl. updating angle of attack for VSM)
        results = wing_aero.calculate_results(
            gamma_new,
            self.density,
            self.aerodynamic_model_type,
            self.core_radius_fraction,
        )

        return results, wing_aero

    def calculate_gamma_new_iteratively(self, wing_aero, gamma_distribution):
        AIC_x, AIC_y, AIC_z = wing_aero.calculate_AIC_matrices(
            self.aerodynamic_model_type, self.core_radius_fraction
        )

        logging.debug(f"AIC_x: {AIC_x}")
        logging.debug(f"AIC_y: {AIC_y}")
        logging.debug(f"AIC_z: {AIC_z}")

        # TODO: CPU optimization: instantiate non-changing (geometric dependent) attributes here
        # TODO: Further optimization: is to populate only in the loop, not create new arrays
        panels = wing_aero.panels
        n_panels = wing_aero.n_panels
        z_airf_array = np.array([panel.z_airf for panel in panels])
        va_array = np.array([panel.va for panel in panels])
        chord_array = np.array([panel.chord for panel in panels])
        alpha = np.zeros(len(panels))
        gamma = np.zeros(len(panels))
        # Adjust relaxation factor based on the number of panels
        relaxation_factor = self.relaxation_factor * 20 / n_panels

        # initialize gamma distribution inside
        if (
            gamma_distribution is None
            and self.type_initial_gamma_distribution == "elliptic"
        ):
            gamma_new = wing_aero.calculate_circulation_distribution_elliptical_wing()
        elif len(gamma_distribution) == n_panels:
            gamma_new = gamma_distribution

        # TODO: for now use ZERO, remove this to speed things up
        gamma_new = np.zeros(len(gamma_new))
        # logging.info("Initial gamma_new: %s", gamma_new)

        converged = False
        for i in range(self.max_iterations):

            gamma = np.array(gamma_new)

            for icp, panel in enumerate(panels):
                # Initialize induced velocity to 0
                u = 0
                v = 0
                w = 0
                # Compute induced velocities with previous gamma distribution
                for jring, gamma_jring in enumerate(gamma):
                    u = u + AIC_x[icp][jring] * gamma_jring
                    # x-component of velocity
                    v = v + AIC_y[icp][jring] * gamma_jring
                    # y-component of velocity
                    w = w + AIC_z[icp][jring] * gamma_jring
                    # z-component of velocity

                # TODO: shouldn't grab from different classes inside the solver for CPU-efficiency
                induced_velocity = np.array([u, v, w])

                # This is double checked
                alpha[icp], relative_velocity = (
                    panel.calculate_relative_alpha_and_relative_velocity(
                        induced_velocity
                    )
                )
                relative_velocity_crossz = np.cross(relative_velocity, panel.z_airf)
                Umag = np.linalg.norm(relative_velocity_crossz)
                Uinfcrossz = np.cross(panel.va, panel.z_airf)
                Umagw = np.linalg.norm(Uinfcrossz)

                # TODO: CPU this should ideally be instantiated upfront, from the wing_aero object
                # Lookup cl for this specific alpha
                cl = panel.calculate_cl(alpha[icp])

                # Find the new gamma using Kutta-Joukouski law
                gamma_new[icp] = 0.5 * Umag**2 / Umagw * cl * chord_array[icp]

                # logging.info("--------- icp: %d", icp)
                # logging.info("induced_velocity: %s", induced_velocity)
                # logging.info("alpha: %f", alpha)
                # logging.info("relative_velocity: %s", relative_velocity)
                # logging.info("cl: %f", cl)
                # logging.info("Umag: %f", Umag)
                # logging.info("Umagw: %f", Umagw)
                # logging.info("chord: %f", chord_array[icp])

            # Dealing with stalled cases
            stall = False
            # TODO: aoa_stall should be input and not fixed, and also adjusted
            aoa_stall = np.deg2rad(15.0)
            for ia, alpha_i in enumerate(alpha):
                if alpha_i > aoa_stall:
                    stall = True
                    logging.debug(
                        "Stall detected, alpha[i]: %f, it: %d, panel_number: %d"
                        % (np.rad2deg(alpha_i), i, ia)
                    )
                    break
            if not stall:
                damp = 0
            else:
                logging.debug(f"applying artificial damping for stall correction")
                damp = self.calculate_artificial_damping(gamma)
            gamma_new = (
                (1 - relaxation_factor) * gamma + relaxation_factor * gamma_new + damp
            )

            # Checking Convergence
            reference_error = np.amax(np.abs(gamma_new))
            reference_error = max(reference_error, self.tol_reference_error)
            error = np.amax(np.abs(gamma_new - gamma))
            normalized_error = error / reference_error

            # logging.info("Iteration: %d, normalized_error: %f", _, normalized_error)
            # logging.info("Iteration: %d, reference_error: %f", _, reference_error)
            # logging.info("Iteration: %d", i)
            # logging.info("gamma: %s", gamma)
            # logging.info("gamma_new: %s", gamma_new)

            # relative error
            if normalized_error < self.allowed_error:
                # if error smaller than limit, stop iteration cycle
                converged = True
                break

        if converged:
            logging.info("------------------------------------")
            logging.info(
                f"{self.aerodynamic_model_type} Converged after {i} iterations"
            )
            logging.info("------------------------------------")
        elif not converged:
            logging.info("------------------------------------")
            logging.info(
                f"{self.aerodynamic_model_type} Not converged after {str(self.max_iterations)} iterations"
            )
            logging.info("------------------------------------")

        return gamma_new

    def calculate_artificial_damping(self, gamma):
        n_gamma = len(gamma)
        damp = np.zeros(n_gamma)
        for ig, gamma_ig in enumerate(gamma):
            if ig == 0:
                gim2 = gamma[0]
                gim1 = gamma[0]
                gi = gamma[0]
                gip1 = gamma[1]
                gip2 = gamma[2]
            elif ig == 1:
                gim2 = gamma[0]
                gim1 = gamma[0]
                gi = gamma[1]
                gip1 = gamma[2]
                gip2 = gamma[3]
            elif ig == n_gamma - 2:
                gim2 = gamma[n_gamma - 4]
                gim1 = gamma[n_gamma - 3]
                gi = gamma[n_gamma - 2]
                gip1 = gamma[n_gamma - 1]
                gip2 = gamma[n_gamma - 1]
            elif ig == n_gamma - 1:
                gim2 = gamma[n_gamma - 3]
                gim1 = gamma[n_gamma - 2]
                gi = gamma[n_gamma - 1]
                gip1 = gamma[n_gamma - 1]
                gip2 = gamma[n_gamma - 1]
            else:
                gim2 = gamma[ig - 2]
                gim1 = gamma[ig - 1]
                gi = gamma[ig]
                gip1 = gamma[ig + 1]
                gip2 = gamma[ig + 2]

            dif2 = (gip1 - gi) - (gi - gim1)
            dif4 = (gip2 - 3.0 * gip1 + 3.0 * gi - gim1) - (
                gip1 - 3.0 * gi + 3.0 * gim1 - gim2
            )
            damp[ig] = (
                self.artificial_damping["k2"] * dif2
                - self.artificial_damping["k4"] * dif4
            )
        return damp
