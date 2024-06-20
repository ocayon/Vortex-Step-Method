import numpy as np

from VSM.functions_needed_for_solve import *
from VSM.WingGeometry import Wing, Section
from VSM.WingAerodynamicModel import WingAerodynamics

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
        relaxation_factor: float = 0.03,
        artificial_damping: dict = {"k2": 0.0, "k4": 0.0},
    ):
        self.aerodynamic_model_type = aerodynamic_model_type
        self.density = density
        self.max_iterations = max_iterations
        self.allowed_error = allowed_error
        self.relaxation_factor = relaxation_factor
        self.artificial_damping = artificial_damping

    def solve(self, wing_aero):

        if wing_aero.va is None:
            raise ValueError("Inflow conditions are not set")

        # Solve the circulation distribution
        wing_aero = self.solve_iterative_loop(wing_aero)

        # Calculate effective angle of attack at the aerodynamic center
        #TODO: This can go inside update_aerodynamic but to remember to correct it (It is always at the aerodynamic center)
        wing_aero.update_effective_angle_of_attack()
        wing_aero.update_aerodynamic_coefficients_and_alpha()

        # Calculate aerodynamic coefficients in the panel reference frame and store them in the Panel object
        wing_aero.update_aerodynamics()

        return wing_aero

    def solve_iterative_loop(self, wing_aero):
        if self.aerodynamic_model_type == "VSM":
            AIC_x, AIC_y, AIC_z, U_2D = wing_aero.calculate_AIC_matrices(
                control_point="three_quarter_chord"
            )
        elif self.aerodynamic_model_type == "LLT":
            AIC_x, AIC_y, AIC_z, U_2D = wing_aero.calculate_AIC_matrices(
                control_point="aerodynamic_center"
            )
        else:
            raise ValueError("Invalid aerodynamic model type")

        gamma_new = wing_aero.get_gamma_distribution()
        #TODO: instantiate non-chnagning atrributes here 
        for _ in self.max_iterations:

            gamma = gamma_new  # I used to do this in a loop, not sure if

            for icp, panel in enumerate(wing_aero.get_panels()):
                # Initialize induced velocity to 0
                u = 0
                v = 0
                w = 0
<<<<<<< HEAD
                # Compute induced velocities with previous Gamma distribution
                for jring, gamma in enumerate(Gamma):
                    u = u + AIC_x[icp][jring] * gamma
                    # x-component of velocity
                    v = v + AIC_y[icp][jring] * gamma
                    # y-component of velocity
                    w = w + AIC_z[icp][jring] * gamma
=======
                # Compute induced velocities with previous gamma distribution
                for jring, gamma_jring in enumerate(gamma):
                    u = u + AIC_x[icp][jring] * gamma_jring
                    # x-component of velocity
                    v = v + AIC_y[icp][jring] * gamma_jring
                    # y-component of velocity
                    w = w + AIC_z[icp][jring] * gamma_jring
>>>>>>> 94b2a9061ccf341860d75ff4c39a4b01fb3f442e
                    # z-component of velocity

                u = u - U_2D[icp, 0] * gamma[icp]
                v = v - U_2D[icp, 1] * gamma[icp]
                w = w - U_2D[icp, 2] * gamma[icp]

                induced_velocity = np.array([u, v, w])
                alpha, relative_velocity = (
                    panel.get_relative_alpha_and_relative_velocity(induced_velocity)
                )

<<<<<<< HEAD
                # Calculate relative velocity and angle of attack
                Uinf = panel.get_apparent_velocity
                Urel = Uinf + np.array([u, v, w])
                vn = np.dot(norm_airf, Urel)
                vtan = np.dot(tan_airf, Urel)
                alpha = np.arctan(vn / vtan)
                if alpha > panel.stall_aoa:
                    stall = True

                Urelcrossz = np.cross(Urel, z_airf)
                Umag = np.linalg.norm(Urelcrossz)
                Uinfcrossz = np.cross(Uinf, z_airf)
=======
                # TODO: shouldn't grab from different classes inside the solver for CPU-efficiency
                z_airf = panel.get_z_airf()
                relative_velocity_crossz = np.cross(relative_velocity, z_airf)
                Umag = np.linalg.norm(relative_velocity_crossz)
                Uinfcrossz = np.cross(panel.get_va, z_airf)
>>>>>>> 94b2a9061ccf341860d75ff4c39a4b01fb3f442e
                Umagw = np.linalg.norm(Uinfcrossz)

                # Look-up airfoil 2D coefficients
                cl = panel.get_cl(alpha)
                chord = panel.get_chord()

                # Find the new gamma using Kutta-Joukouski law
                gamma_new[icp] = 0.5 * Umag**2 / Umagw * cl * chord

            reference_error = np.amax(np.abs(gamma_new))
            reference_error = max(reference_error, 0.001)
            error = np.amax(np.abs(gamma_new - gamma))
            normalized_error = error / reference_error
            # relative error
            if normalized_error < self.allowed_error:
                # if error smaller than limit, stop iteration cycle
                converged = True
                break

            gamma_new = (
                1 - self.relaxation_factor
            ) * gamma + self.relaxation_factor * gamma_new

            if self.artificial_damping is not None:
<<<<<<< HEAD
                if stall:
                    GammaNew = self.apply_artificial_damping(GammaNew)
                    stall = False
=======
                gamma_new = self.apply_artificial_damping(gamma_new)
>>>>>>> 94b2a9061ccf341860d75ff4c39a4b01fb3f442e

        if not converged:
            print("Not converged after " + str(self.max_iterations) + " iterations")

        wing_aero.set_gamma_distribution(gamma)

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
