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

        # Calculate aerodynamic coefficients in the panel reference frame and store them in the panel object
        wing_aero.update_aerodynamics()


        return wing_aero

    def solve_iterative_loop(self, wing_aero, AIC_x, AIC_y, AIC_z, U_2D, panels):
        N = len(panels)
        alpha = np.zeros(N)
        Lift = np.zeros(N)
        Drag = np.zeros(N)
        Ma = np.zeros(N)
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

        panels = wing_aero.get_panels()
        GammaNew = wing_aero.get_gamma_distribution()
        Gamma = np.zeros(len(GammaNew))
        for _ in self.max_iterations:

            for ig in range(len(Gamma)):
                Gamma[ig] = GammaNew[ig]

            for icp in range(N):
                # Initialize induced velocity to 0
                u = 0
                v = 0
                w = 0
                # Compute induced velocities with previous Gamma distribution
                for jring in range(N):
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
                dcm = panels[icp].get_reference_frame()
                norm_airf = dcm[:, 0]
                tan_airf = dcm[:, 1]
                z_airf = dcm[:, 2]

                # Calculate relative velocity and angle of attack
                Uinf = wing_aero.get_apparent_velocity(panel = icp)
                Urel = Uinf + np.array([u, v, w])
                vn = dot_product(norm_airf, Urel)
                vtan = dot_product(tan_airf, Urel)
                alpha[icp] = np.arctan(vn / vtan)

                Urelcrossz = np.cross(Urel, z_airf)
                Umag = np.linalg.norm(Urelcrossz)
                Uinfcrossz = np.cross(Uinf, z_airf)
                Umagw = np.linalg.norm(Uinfcrossz)

                # Look-up airfoil 2D coefficients
                cl, cd, cm = panels[icp].get_aerodynamic_properties(alpha[icp])

                chord = panels[icp].get_chord()
                # Retrieve forces and moments
                Lift[icp] = 0.5 * self.rho * Umag**2 * cl * chord
                Drag[icp] = 0.5 * self.rho * Umag**2 * cd * chord
                Ma[icp] = 0.5 * self.rho * Umag**2 * cm * chord**2

                # Find the new gamma using Kutta-Joukouski law
                GammaNew[icp] = 0.5 * Umag**2 / Umagw * cl[icp] * chord

            # check convergence of solution
            refererror = np.amax(np.abs(GammaNew))
            refererror = np.amax([refererror, 0.001])
            # define scale of bound circulation
            error = np.amax(np.abs(GammaNew - Gamma))
            # difference betweeen iterations
            error = error / refererror
            # relative error
            if error < self.allowed_error:
                # if error smaller than limit, stop iteration cycle
                converged = True
                break
            for ig in range(len(Gamma)):
                GammaNew[ig] = (1 - self.relaxation_factor) * Gamma[
                    ig
                ] + self.relaxation_factor * GammaNew[ig]

        if converged == False:
            print("Not converged after " + str(self.max_iterations) + " iterations")

        wing_aero.set_gamma_distribution(Gamma)
        return wing_aero



