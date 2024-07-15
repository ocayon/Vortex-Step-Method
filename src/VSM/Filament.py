from abc import ABC, abstractmethod
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def cross_product(r1, r2):
    """
    Cross product between r1 and r2

    """

    return np.array(
        [
            r1[1] * r2[2] - r1[2] * r2[1],
            r1[2] * r2[0] - r1[0] * r2[2],
            r1[0] * r2[1] - r1[1] * r2[0],
        ]
    )


def vec_norm(v):
    """
    Norm of a vector

    """
    return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def dot_product(r1, r2):
    """
    Dot product between r1 and r2

    """
    return r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]


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
        r1Xr0 = cross_product(r1, r0)
        r2Xr0 = cross_product(r2, r0)

        epsilon = core_radius_fraction * vec_norm(r0)  # Cut-off radius
        # If point is outside the core radius of filament
        if vec_norm(r1Xr0) / vec_norm(r0) > epsilon:
            # Perpendicular distance from XVP to vortex filament (r0)
            r1Xr2 = cross_product(r1, r2)
            return (
                gamma
                / (4 * np.pi)
                * r1Xr2
                / (vec_norm(r1Xr2) ** 2)
                * dot_product(r0, r1 / vec_norm(r1) - r2 / vec_norm(r2))
            )
        # If point is on the filament
        elif vec_norm(r1Xr0) / vec_norm(r0) == 0:
            return np.zeros(3)
        # If point is inside the core radius of filament
        else:
            logging.info(f"inside core radius")
            # logging.info(f"epsilon: {epsilon}")
            logging.info(
                f"distance from control point to filament: {vec_norm(r1Xr0) / vec_norm(r0)}"
            )
            # The control point is placed on the edge of the radius core
            # proj stands for the vectors respect to the new controlpoint
            r1_proj = dot_product(r1, r0) * r0 / (
                vec_norm(r0) ** 2
            ) + epsilon * r1Xr0 / vec_norm(r1Xr0)
            r2_proj = dot_product(r2, r0) * r0 / (
                vec_norm(r0) ** 2
            ) + epsilon * r2Xr0 / vec_norm(r2Xr0)
            r1Xr2_proj = cross_product(r1_proj, r2_proj)
            vel_ind_proj = (
                gamma
                / (4 * np.pi)
                * r1Xr2_proj
                / (vec_norm(r1Xr2_proj) ** 2)
                * dot_product(
                    r0, r1_proj / vec_norm(r1_proj) - r2_proj / vec_norm(r2_proj)
                )
            )
            return vec_norm(r1Xr0) / (vec_norm(r0) * epsilon) * vel_ind_proj

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
            dot_product(r1, r0) * r0 / (vec_norm(r0) ** 2)
        )  # Vector from XV1 to XVP perpendicular to the core radius
        epsilon = np.sqrt(4 * alpha0 * nu * vec_norm(r_perp) / Uinf)  # Cut-off radius

        # Cross products used for later computations
        r1Xr0 = cross_product(r1, r0)
        r2Xr0 = cross_product(r2, r0)

        # if point is outside the core radius of filament
        if (
            vec_norm(r1Xr0) / vec_norm(r0) > epsilon
        ):  # Perpendicular distance from XVP to vortex filament (r0)
            r1Xr2 = cross_product(r1, r2)
            return (
                gamma
                / (4 * np.pi)
                * r1Xr2
                / (vec_norm(r1Xr2) ** 2)
                * dot_product(r0, r1 / vec_norm(r1) - r2 / vec_norm(r2))
            )
        # if point is on the filament
        elif vec_norm(r1Xr0) / vec_norm(r0) == 0:
            return np.zeros(3)
        # if point is inside the core radius of filament
        else:
            # The control point is placed on the edge of the radius core
            # proj stands for the vectors respect to the new controlpoint
            r1_proj = dot_product(r1, r0) * r0 / (
                vec_norm(r0) ** 2
            ) + epsilon * r1Xr0 / vec_norm(r1Xr0)
            r2_proj = dot_product(r2, r0) * r0 / (
                vec_norm(r0) ** 2
            ) + epsilon * r2Xr0 / vec_norm(r2Xr0)
            r1Xr2_proj = cross_product(r1_proj, r2_proj)
            vel_ind_proj = (
                gamma
                / (4 * np.pi)
                * r1Xr2_proj
                / (vec_norm(r1Xr2_proj) ** 2)
                * dot_product(
                    r0, r1_proj / vec_norm(r1_proj) - r2_proj / vec_norm(r2_proj)
                )
            )
            return vec_norm(r1Xr0) / (vec_norm(r0) * epsilon) * vel_ind_proj


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
        r1XVf = cross_product(r1, Vf)

        alpha0 = 1.25643  # Oseen parameter
        nu = 1.48e-5  # Kinematic viscosity of air
        r_perp = (
            dot_product(r1, Vf) * Vf
        )  # Vector from XV1 to XVP perpendicular to the core radius
        epsilon = np.sqrt(4 * alpha0 * nu * vec_norm(r_perp) / Uinf)  # Cut-off radius

        # if point is outside the core radius of filament
        if vec_norm(r1XVf) / vec_norm(Vf) > epsilon:
            # determine scalar
            K = (
                GAMMA
                / 4
                / np.pi
                / vec_norm(r1XVf) ** 2
                * (1 + dot_product(r1, Vf) / vec_norm(r1))
            )
            # determine the three velocity components
            return K * r1XVf
        # if point is on the filament
        elif vec_norm(r1XVf) / vec_norm(Vf) == 0:
            return np.zeros(3)
        # else, if point within core
        else:
            r1_proj = dot_product(r1, Vf) * Vf + epsilon * (
                r1 / vec_norm(r1) - Vf
            ) / vec_norm(r1 / vec_norm(r1) - Vf)
            r1XVf_proj = cross_product(r1_proj, Vf)
            K = (
                GAMMA
                / 4
                / np.pi
                / vec_norm(r1XVf_proj) ** 2
                * (1 + dot_product(r1_proj, Vf) / vec_norm(r1_proj))
            )
            # determine the three velocity components
            return K * r1XVf_proj


class Infinite2DFilament(Filament):
    """
    A class to represent an infinite 2D vortex filament.

    Input:
    two points defining the filament

    Output:
    a filament object
    """

    def __init__(self, x1, x2):
        self._x1 = np.array(x1)
        self._x2 = np.array(x2)
        self._length = np.linalg.norm(self._x2 - self._x1)
        self._AB = self._x2 - self._x1
        self._midpoint = self._x1 + 0.5 * self._AB

    @property
    def x1(self):
        return self._x1

    @property
    def x2(self):
        return self._x2
