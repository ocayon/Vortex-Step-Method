from abc import ABC
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class HorshoeVortex:
    """
    A class to represent a horshoe vortex.

    input:
    a single panel object
    containing all the corner points and aerodynamic properties

    output:
    a horshoe vortex object

    """

    def __init__(
        self,
        LE_point_1,
        TE_point_1,
        LE_point_2,
        TE_point_2,
        aerodynamic_center_location,
        control_point_location,
    ):

        self.filaments = []
        self.bound_point_1 = (
            LE_point_1 * (1 - aerodynamic_center_location)
            + TE_point_1 * aerodynamic_center_location
        )
        self.bound_point_2 = (
            LE_point_2 * (1 - aerodynamic_center_location)
            + TE_point_2 * aerodynamic_center_location
        )
        self.filaments.append(
            BoundFilament(x1=self.bound_point_1, x2=self.bound_point_2)
        )  # bound vortex
        self.filaments.append(
            BoundFilament(x1=TE_point_1, x2=self.bound_point_1)
        )  # trailing edge vortex
        self.filaments.append(
            BoundFilament(x1=self.bound_point_2, x2=TE_point_2)
        )  # trailing edge vortex

        # appending a initial wake filament
        self.filaments.append(None)
        self.filaments.append(None)

        self._gamma = None  # Initialize the gamma attribute

        logging.info("Horshoe vortex created")
        logging.info("Bound point 1: %s", self.bound_point_1)
        logging.info("TE point 1: %s", TE_point_1)
        logging.info("Bound point 2: %s", self.bound_point_2)
        logging.info("TE point 2: %s", TE_point_2)

    @property
    def filaments_for_plotting(self):
        return self._filaments_for_plotting

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if value < 0:
            raise ValueError("Gamma must be a non-negative value")
        self._gamma = value

    def calculate_velocity_induced_bound_2D(self, control_point, gamma=None):
        """ "
        This function calculates the 2D induced velocity at the control point due to the bound vortex filaments
        """
        if gamma is None:
            gamma = self.gamma
        A = self.filaments[1].x1
        B = self.filaments[1].x2
        r0 = B - A
        AP = control_point - A

        # Projection of AP onto AB
        r0_unit = r0 / np.linalg.norm(r0)
        projection_length = np.dot(AP, r0_unit)
        projection_point = +projection_length * r0_unit

        # Vector r3 from the projection point to the control point P
        r3 = control_point - projection_point

        # Calculate the cross product of r0 and r3
        cross = np.cross(r0, r3)

        # Magnitude squared of the cross product vector
        cross_norm_sq = np.dot(cross, cross)

        # Induced velocity calculation
        ind_vel = (gamma / (2 * np.pi)) * (cross / cross_norm_sq) * np.linalg.norm(r0)

        return ind_vel

    def calculate_velocity_induced_horseshoe(self, control_point, gamma=None):
        """ "
        This function calculates the induced velocity at the control point due to the bound vortex filaments
        """
        if gamma is None:
            gamma = self.gamma

        ind_vel = np.zeros(3)
        for filament in self.filaments:
            ind_vel += filament.calculate_induced_velocity(control_point, gamma)

        return ind_vel

    def update_filaments_for_wake(self, point1, dir_1, point2, dir_2):
        self.filaments[3] = SemiInfiniteFilament(point1, dir_1)
        self.filaments[4] = SemiInfiniteFilament(point2, dir_2)
        print("Wake filaments updated")
        print(self.filaments[3].x2)
        print(self.filaments[4].x2)

    def calculate_filaments_for_plotting(self):
        filaments = []
        for filament in self.filaments:
            filaments.append([filament.x1, filament.x2])
        return filaments


class Filament(ABC):
    """
    A class to represent a filament.

    input:
    two points defining the filament

    output:
    a filament object

    """

    def __init__(self):
        pass

    def calculate_induced_velocity(self, point, gamma):
        pass


class BoundFilament:
    """
    A class to represent a filament.

    input:
    two points defining the filament

    output:
    a filament object

    """

    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.length = np.linalg.norm(x2 - x1)

    def calculate_induced_velocity(self, point, gamma=1.0):
        r0 = self.x2 - self.x1  # Vortex filament
        r1 = point - self.x1  # Controlpoint to one end of the vortex filament
        r2 = point - self.x2  # Controlpoint to one end of the vortex filament
        # Cross products used for later computations
        r1Xr0 = np.cross(r1, r0)
        r2Xr0 = np.cross(r2, r0)

        epsilon = 0.05 * self.length  # Cut-off radius

        if (
            np.linalg.norm(r1Xr0) / self.length > epsilon
        ):  # Perpendicular distance from XVP to vortex filament (r0)
            r1Xr2 = np.cross(r1, r2)
            vel_ind = (
                gamma
                / (4 * np.pi)
                * r1Xr2
                / (np.linalg.norm(r1Xr2) ** 2)
                * np.dot(r0, r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2))
            )
        else:
            # The control point is placed on the edge of the radius core
            # proj stands for the vectors respect to the new controlpoint
            r1_proj = np.dot(r1, r0) * r0 / (
                self.length**2
            ) + epsilon * r1Xr0 / np.linalg.norm(r1Xr0)
            r2_proj = np.dot(r2, r0) * r0 / (
                self.length**2
            ) + epsilon * r2Xr0 / np.linalg.norm(r2Xr0)
            r1Xr2_proj = np.cross(r1_proj, r2_proj)
            vel_ind_proj = (
                gamma
                / (4 * np.pi)
                * r1Xr2_proj
                / (np.linalg.norm(r1Xr2_proj) ** 2)
                * np.dot(
                    r0,
                    r1_proj / np.linalg.norm(r1_proj)
                    - r2_proj / np.linalg.norm(r2_proj),
                )
            )
            vel_ind = np.linalg.norm(r1Xr0) / (self.length * epsilon) * vel_ind_proj

        return vel_ind


class SemiInfiniteFilament:
    """
    A class to represent a filament.

    input:
    two points defining the filament

    output:
    a filament object

    """

    def __init__(self, x1, direction, Uinf=1.0, alpha0=1.25643, nu=1.48e-5):
        self.x1 = x1
        # x2 is a point far away from the filament, defined here for plotting purposes
        self.x2 = x1 + direction * 0.5
        self.direction = direction
        self.alpha0 = alpha0  # Oseen parameter
        self.nu = nu  # Kinematic viscosity of air

    # TODO: what is the purpose of Uinf here? It does NOT take correct value
    def calculate_induced_velocity(self, point, gamma, Uinf=1):
        r1 = point - self.x1
        r1XVf = np.cross(r1, self.direction)

        r_perp = (
            np.dot(r1, self.direction) * self.direction
        )  # Vector from XV1 to XVP perpendicular to the core radius
        epsilon = np.sqrt(
            4 * self.alpha0 * self.nu * np.linalg.norm(r_perp) / Uinf
        )  # Cut-off radius

        if np.linalg.norm(r1XVf) / np.linalg.norm(self.direction) > epsilon:
            # determine scalar
            K = (
                gamma
                / 4
                / np.pi
                / np.linalg.norm(r1XVf) ** 2
                * (1 + np.dot(r1, self.direction) / np.linalg.norm(r1))
            )
            # determine the three velocity components
            vel_ind = K * r1XVf
        else:
            r1_proj = np.dot(r1, self.direction) * self.direction + epsilon * (
                r1 / np.linalg.norm(r1) - self.direction
            ) / np.linalg.norm(r1 / np.linalg.norm(r1) - self.direction)
            r1XVf_proj = np.cross(r1_proj, self.direction)
            K = (
                gamma
                / 4
                / np.pi
                / np.linalg.norm(r1XVf_proj) ** 2
                * (1 + np.dot(r1_proj, self.direction) / np.linalg.norm(r1_proj))
            )
            # determine the three velocity components
            vel_ind = K * r1XVf_proj
        # output results, vector with the three velocity components
        return vel_ind
