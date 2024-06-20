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
        # This can go inside update_aerodynamics but to remember to correct it (It is always at the aerodynamic center)
        wing_aero.update_effective_angle_of_attack()

        # Calculate aerodynamic coefficients in the panel reference frame and store them in the Panel object
        wing_aero.update_aerodynamics()

        return wing_aero

    # TODO: Why are the AIC matrices and U2D input here? To allow for an update method?
    def solve_iterative_loop(self, wing_aero, AIC_x, AIC_y, AIC_z, U_2D):
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

        GammaNew = wing_aero.get_gamma_distribution()
        for _ in self.max_iterations:

            Gamma = GammaNew  # I used to do this in a loop, not sure if

            for icp, panel in enumerate(wing_aero.get_panels()):
                # Initialize induced velocity to 0
                u = 0
                v = 0
                w = 0
                # Compute induced velocities with previous Gamma distribution
                for jring in range(len(Gamma)):
                    u = u + AIC_x[icp][jring] * Gamma[jring]
                    # x-component of velocity
                    v = v + AIC_y[icp][jring] * Gamma[jring]
                    # y-component of velocity
                    w = w + AIC_z[icp][jring] * Gamma[jring]
                    # z-component of velocity

                u = u - U_2D[icp, 0] * Gamma[icp]
                v = v - U_2D[icp, 1] * Gamma[icp]
                w = w - U_2D[icp, 2] * Gamma[icp]

                # Calculate terms of induced corresponding to the airfoil directions
                dcm = panel.get_reference_frame()
                norm_airf = dcm[:, 0]
                tan_airf = dcm[:, 1]
                z_airf = dcm[:, 2]

                # Calculate relative velocity and angle of attack
                Uinf = panel.get_apparent_velocity
                Urel = Uinf + np.array([u, v, w])
                vn = dot_product(norm_airf, Urel)
                vtan = dot_product(tan_airf, Urel)
                alpha = np.arctan(vn / vtan)

                Urelcrossz = np.cross(Urel, z_airf)
                Umag = np.linalg.norm(Urelcrossz)
                Uinfcrossz = np.cross(Uinf, z_airf)
                Umagw = np.linalg.norm(Uinfcrossz)

                # Look-up airfoil 2D coefficients
                cl = panel.get_cl(alpha)

                chord = panel.get_chord()

                # Find the new gamma using Kutta-Joukouski law
                GammaNew[icp] = 0.5 * Umag**2 / Umagw * cl * chord

            reference_error = np.amax(np.abs(GammaNew))
            reference_error = max(reference_error, 0.001)
            error = np.amax(np.abs(GammaNew - Gamma))
            normalized_error = error / reference_error
            # relative error
            if normalized_error < self.allowed_error:
                # if error smaller than limit, stop iteration cycle
                converged = True
                break

            GammaNew = (
                1 - self.relaxation_factor
            ) * Gamma + self.relaxation_factor * GammaNew

            if self.artificial_damping is not None:
                GammaNew = self.apply_artificial_damping(GammaNew, alpha)

        if converged == False:
            print("Not converged after " + str(self.max_iterations) + " iterations")

        wing_aero.set_gamma_distribution(Gamma)

        return wing_aero

    def apply_artificial_damping(self, Gamma):
        N = len(Gamma)
        Gamma_damped = np.zeros(N)
        for ig in range(N):
            if ig == 0:
                Gim2 = Gamma[0]
                Gim1 = Gamma[0]
                Gi = Gamma[0]
                Gip1 = Gamma[1]
                Gip2 = Gamma[2]
            elif ig == 1:
                Gim2 = Gamma[0]
                Gim1 = Gamma[0]
                Gi = Gamma[1]
                Gip1 = Gamma[2]
                Gip2 = Gamma[3]
            elif ig == N - 2:
                Gim2 = Gamma[N - 4]
                Gim1 = Gamma[N - 3]
                Gi = Gamma[N - 2]
                Gip1 = Gamma[N - 1]
                Gip2 = Gamma[N - 1]
            elif ig == N - 1:
                Gim2 = Gamma[N - 3]
                Gim1 = Gamma[N - 2]
                Gi = Gamma[N - 1]
                Gip1 = Gamma[N - 1]
                Gip2 = Gamma[N - 1]
            else:
                Gim2 = Gamma[ig - 2]
                Gim1 = Gamma[ig - 1]
                Gi = Gamma[ig]
                Gip1 = Gamma[ig + 1]
                Gip2 = Gamma[ig + 2]

            dif2 = (Gip1 - Gi) - (Gi - Gim1)
            dif4 = (Gip2 - 3.0 * Gip1 + 3.0 * Gi - Gim1) - (
                Gip1 - 3.0 * Gi + 3.0 * Gim1 - Gim2
            )

            k2, k4 = self.artificial_damping["k2"], self.artificial_damping["k4"]
            Gamma_damped[ig] = k2 * dif2 - k4 * dif4

        GammaNew = Gamma + Gamma_damped

        return GammaNew
