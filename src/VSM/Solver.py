import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
# Maurits-tips :)
# call the methods of child-classes, inhereted or composed of
# do not call the attributes of child-classes, call them through getter methods
# only pass the attributes that you need to pass, not the whole object
# only use the methods of level higher/lower, not grabbing methods from higher/lower
# class solve_VSM(Solver)

# class solve_LLM(Solver)


# make abstract class
class Solver:
    def __init__(
        self,
        # Below are all settings, with a default value, that can but don't have to be changed
        aerodynamic_model_type: str = "VSM",
        density: float = 1.225,
        max_iterations: int = 1000,
        allowed_error: float = 1e-5,
        tol_reference_error: float = 0.001,
        relaxation_factor: float = 0.03,
        artificial_damping: dict = {"k2": 0.0, "k4": 0.0},
    ):
        self.aerodynamic_model_type = aerodynamic_model_type
        self.density = density
        self.max_iterations = max_iterations
        self.allowed_error = allowed_error
        self.tol_reference_error = tol_reference_error
        self.relaxation_factor = relaxation_factor
        self.artificial_damping = artificial_damping

    def solve(self, wing_aero):

        if wing_aero.va is None:
            raise ValueError("Inflow conditions are not set")

        # Solve the circulation distribution
        wing_aero = self.solve_iterative_loop(wing_aero)

        results, wing_aero = wing_aero.calculate_results()

        return results, wing_aero

    def solve_iterative_loop(self, wing_aero):
        AIC_x, AIC_y, AIC_z = wing_aero.calculate_AIC_matrices(
            self.aerodynamic_model_type
        )

        gamma_new = wing_aero.calculate_gamma_distribution()
        # logging.info("Initial gamma_new: %s", gamma_new)

        # TODO: CPU optimization: instantiate non-changing (geometric dependent) attributes here
        # TODO: Further optimization: is to populate only in the loop, not create new arrays
        panels = wing_aero.panels
        z_airf_array = np.array([panel.z_airf for panel in panels])
        va_array = np.array([panel.va for panel in panels])
        chord_array = np.array([panel.chord for panel in panels])
        gamma = np.zeros(len(panels))

        converged = False
        for i in range(self.max_iterations):
            
            for ig, gamma_n in enumerate(gamma_new):
                gamma[ig] = gamma_n

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
                alpha, relative_velocity = (
                    panel.calculate_relative_alpha_and_relative_velocity(
                        induced_velocity
                    )
                )

                relative_velocity_crossz = np.cross(
                    relative_velocity, z_airf_array[icp]
                )
                Umag = np.linalg.norm(relative_velocity_crossz)
                Uinfcrossz = np.cross(va_array[icp], z_airf_array[icp])
                Umagw = np.linalg.norm(Uinfcrossz)

                # TODO: CPU this should ideally be instantiated upfront, from the wing_aero object
                # Lookup cl for this specific alpha
                cl = panel.calculate_cl(alpha)

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

            reference_error = np.amax(np.abs(gamma_new))
            reference_error = max(reference_error, self.tol_reference_error)
            error = np.amax(np.abs(gamma_new - gamma))
            normalized_error = error / reference_error

            # logging.info("Iteration: %d, normalized_error: %f", _, normalized_error)
            # logging.info("Iteration: %d, reference_error: %f", _, reference_error)
            logging.info("Iteration: %d", i)
            # logging.info("gamma: %s", gamma)
            logging.info("gamma_new: %s", gamma_new)

            # relative error
            if normalized_error < self.allowed_error:
                # if error smaller than limit, stop iteration cycle
                converged = True
                break

            gamma_new = (
                1 - self.relaxation_factor
            ) * gamma + self.relaxation_factor * gamma_new

            # if self.artificial_damping is not None:
            #     gamma_new = self.apply_artificial_damping(gamma_new)

        if converged:
            print(" ")
            print("Converged after " + str(i) + " iterations")
            print("------------------------------------")
        if not converged:
            print("Not converged after " + str(self.max_iterations) + " iterations")

        wing_aero.calculate_gamma_distribution(gamma)

        return wing_aero

    def apply_artificial_damping(self, gamma):
        n_gamma = len(gamma)
        gamma_damped = np.zeros(n_gamma)
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

            k2, k4 = self.artificial_damping["k2"], self.artificial_damping["k4"]
            gamma_damped[ig] = k2 * dif2 - k4 * dif4

        gamma_new = gamma + gamma_damped

        return gamma_new
