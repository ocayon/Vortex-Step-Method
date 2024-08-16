from abc import ABC, abstractmethod
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class Filament(ABC):
    """
    A class to represent a filament.

    Input:
    two points defining the filament

    Output:
    a filament object
    """

    @abstractmethod
    def __init__(self):
        pass

    # TODO: could bring this back again, by renaming all the functions, eh?
    # @abstractmethod
    # def calculate_induced_velocity(self, point, gamma, core_radius_fraction):
    #     pass


class BoundFilament(Filament):
    """
    A class to represent a bound vortex filament.

    Input:
    two points defining the filament

    Output:
    a filament object
    """

    # TODO: --CPU-- could initialize more attributes here, such that a new calculation with an existing object will go faster?
    def __init__(self, x1, x2):
        self._x1 = np.array(x1)
        self._x2 = np.array(x2)
        self._length = np.linalg.norm(self._x2 - self._x1)
        self._r0 = self._x2 - self._x1

    @property
    def x1(self):
        return self._x1

    @property
    def x2(self):
        return self._x2

    def velocity_3D_bound_vortex(self, XVP, gamma, core_radius_fraction):
        """
            Calculate the velocity induced by a bound vortex filament in a point in space ()

            Vortex core correction from:
                Rick Damiani et al. “A vortex step method for nonlinear airfoil polar data as implemented in
        KiteAeroDyn”.

            ----------
            XV1 : Point A of Bound vortex (array)
            XV2 : Point B of Bound vortex (array)
            XVP : Control point (array)
            gamma : Strength of the vortex (scalar)

            Returns
            -------
            vel_ind : Induced velocity (array)

        """
        XV1 = self.x1
        XV2 = self.x2

        r0 = XV2 - XV1  # Vortex filament
        r1 = XVP - XV1  # Controlpoint to one end of the vortex filament
        r2 = XVP - XV2  # Controlpoint to one end of the vortex filament

        # Cross products used for later computations
        r1Xr0 = np.cross(r1, r0)
        r2Xr0 = np.cross(r2, r0)

        epsilon = core_radius_fraction * np.linalg.norm(r0)  # Cut-off radius
        # If point is outside the core radius of filament
        if np.linalg.norm(r1Xr0) / np.linalg.norm(r0) > epsilon:
            # Perpendicular distance from XVP to vortex filament (r0)
            r1Xr2 = np.cross(r1, r2)
            return (
                gamma
                / (4 * np.pi)
                * r1Xr2
                / (np.linalg.norm(r1Xr2) ** 2)
                * np.dot(r0, r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2))
            )
        # If point is on the filament
        elif np.linalg.norm(r1Xr0) / np.linalg.norm(r0) == 0:
            return np.zeros(3)
        # If point is inside the core radius of filament
        else:
            logging.info(f"inside core radius")
            # logging.info(f"epsilon: {epsilon}")
            logging.info(
                f"distance from control point to filament: {np.linalg.norm(r1Xr0) / np.linalg.norm(r0)}"
            )
            # The control point is placed on the edge of the radius core
            # proj stands for the vectors respect to the new controlpoint
            r1_proj = np.dot(r1, r0) * r0 / (
                np.linalg.norm(r0) ** 2
            ) + epsilon * r1Xr0 / np.linalg.norm(r1Xr0)
            r2_proj = np.dot(r2, r0) * r0 / (
                np.linalg.norm(r0) ** 2
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
            return np.linalg.norm(r1Xr0) / (np.linalg.norm(r0) * epsilon) * vel_ind_proj

    def velocity_3D_trailing_vortex(self, XVP, gamma, Uinf):
        """
            Calculate the velocity induced by a trailing vortex filament in a point in space

            Vortex core correction from:
                Rick Damiani et al. “A vortex step method for nonlinear airfoil polar data as implemented in
        KiteAeroDyn”.
            ----------
            XV1 : Point A of the vortex filament (array)
            XV2 : Point B of the vortex filament (array)
            XVP : Controlpoint (array)
            gamma : Strength of the vortex (scalar)
            Uinf : Inflow velocity modulus (scalar)

            Returns
            -------
            vel_ind : induced velocity by the trailing fil. (array)

        """
        XV1 = self.x1
        XV2 = self.x2

        r0 = XV2 - XV1  # Vortex filament
        r1 = XVP - XV1  # Controlpoint to one end of the vortex filament
        r2 = XVP - XV2  # Controlpoint to one end of the vortex filament

        alpha0 = 1.25643  # Oseen parameter
        nu = 1.48e-5  # Kinematic viscosity of air
        r_perp = (
            np.dot(r1, r0) * r0 / (np.linalg.norm(r0) ** 2)
        )  # Vector from XV1 to XVP perpendicular to the core radius
        epsilon = np.sqrt(
            4 * alpha0 * nu * np.linalg.norm(r_perp) / Uinf
        )  # Cut-off radius

        # Cross products used for later computations
        r1Xr0 = np.cross(r1, r0)
        r2Xr0 = np.cross(r2, r0)

        # if point is outside the core radius of filament
        if (
            np.linalg.norm(r1Xr0) / np.linalg.norm(r0) > epsilon
        ):  # Perpendicular distance from XVP to vortex filament (r0)
            r1Xr2 = np.cross(r1, r2)
            return (
                gamma
                / (4 * np.pi)
                * r1Xr2
                / (np.linalg.norm(r1Xr2) ** 2)
                * np.dot(r0, r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2))
            )
        # if point is on the filament
        elif np.linalg.norm(r1Xr0) / np.linalg.norm(r0) == 0:
            return np.zeros(3)
        # if point is inside the core radius of filament
        else:
            # The control point is placed on the edge of the radius core
            # proj stands for the vectors respect to the new controlpoint
            r1_proj = np.dot(r1, r0) * r0 / (
                np.linalg.norm(r0) ** 2
            ) + epsilon * r1Xr0 / np.linalg.norm(r1Xr0)
            r2_proj = np.dot(r2, r0) * r0 / (
                np.linalg.norm(r0) ** 2
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
            return np.linalg.norm(r1Xr0) / (np.linalg.norm(r0) * epsilon) * vel_ind_proj


class SemiInfiniteFilament(Filament):
    """
    A class to represent a filament.

    input:
    two points defining the filament

    output:
    a filament object

    """

    def __init__(
        self, x1, direction, vel_mag, filament_direction, alpha0=1.25643, nu=1.48e-5
    ):
        self._x1 = x1  # the trailing edge point, of which the trailing vortex starts
        # x2 is a point far away from the filament, defined here for plotting purposes
        # self._x2 = x1 + filament_direction * direction * 0.5
        self._direction = direction  # unit vector of apparent wind speed
        self._alpha0 = alpha0  # Oseen parameter
        self._nu = nu  # Kinematic viscosity of air
        self._vel_mag = vel_mag  # the magnitude of the apparent wind speed
        self._filament_direction = filament_direction  # -1 or 1, indicating if its with or against the direction of the apparent wind speed

    @property
    def x1(self):
        return self._x1

    @property
    def filament_direction(self):
        return self._filament_direction

    def velocity_3D_trailing_vortex_semiinfinite(self, Vf, XVP, GAMMA, Uinf):
        """
            Calculate the velocity induced by a semiinfinite trailing vortex filament in a point in space

            Vortex core correction from:
                Rick Damiani et al. “A vortex step method for nonlinear airfoil polar data as implemented in
        KiteAeroDyn”.
            ----------
            XV1 : Point A of the vortex filament (array)
            XV2 : Point B of the vortex filament (array)
            XVP : Controlpoint (array)
            gamma : Strength of the vortex (scalar)
            Uinf : Inflow velocity modulus (scalar)

            Returns
            -------
            vel_ind : induced velocity by the trailing fil. (array)

        """
        XV1 = self.x1
        GAMMA = -GAMMA * self.filament_direction

        r1 = XVP - XV1  # Vector from XV1 to XVP
        r1XVf = np.cross(r1, Vf)

        alpha0 = 1.25643  # Oseen parameter
        nu = 1.48e-5  # Kinematic viscosity of air
        r_perp = (
            np.dot(r1, Vf) * Vf
        )  # Vector from XV1 to XVP perpendicular to the core radius
        epsilon = np.sqrt(
            4 * alpha0 * nu * np.linalg.norm(r_perp) / Uinf
        )  # Cut-off radius

        # if point is outside the core radius of filament
        if np.linalg.norm(r1XVf) / np.linalg.norm(Vf) > epsilon:
            # determine scalar
            K = (
                GAMMA
                / 4
                / np.pi
                / np.linalg.norm(r1XVf) ** 2
                * (1 + np.dot(r1, Vf) / np.linalg.norm(r1))
            )
            # determine the three velocity components
            return K * r1XVf
        # if point is on the filament
        elif np.linalg.norm(r1XVf) / np.linalg.norm(Vf) == 0:
            return np.zeros(3)
        # else, if point within core
        else:
            r1_proj = np.dot(r1, Vf) * Vf + epsilon * (
                r1 / np.linalg.norm(r1) - Vf
            ) / np.linalg.norm(r1 / np.linalg.norm(r1) - Vf)
            r1XVf_proj = np.cross(r1_proj, Vf)
            K = (
                GAMMA
                / 4
                / np.pi
                / np.linalg.norm(r1XVf_proj) ** 2
                * (1 + np.dot(r1_proj, Vf) / np.linalg.norm(r1_proj))
            )
            # determine the three velocity components
            return K * r1XVf_proj
