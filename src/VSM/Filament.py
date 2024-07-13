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

    # def calculate_induced_velocity(self, point, gamma, core_radius_fraction):
    #     """Calculate the induced velocity at a control point due to the vortex filament.

    #         Checks if the evaluation point is on the filament and returns zero velocity to avoid singularity.
    #         Uses the Biot-Savart law to calculate the induced velocity at the control point.
    #         If evaluation point is within the core radius, applies smoothing to avoid singularities.

    #     Args:
    #         point (np.ndarray): The control point at which the induced velocity is calculated.
    #         gamma (float): The circulation strength of the vortex filament.
    #         core_radius_fraction (float): The fraction of the filament length that defines the core radius.

    #     Returns:
    #         np.ndarray: The induced velocity at the control point.
    #     """
    #     r1 = (
    #         np.array(point) - self._x1
    #     )  # Control point to one end of the vortex filament
    #     r2 = (
    #         np.array(point) - self._x2
    #     )  # Control point to the other end of the filament

    #     # Copmute distance from the control point to the filament
    #     dist = np.linalg.norm(np.cross(r1, self._r0)) / max(self._length, 1e-12)

    #     # Determine the core radius
    #     epsilon_bound = core_radius_fraction * self._length

    #     # Check if the control point is on the filament
    #     r1_cross_r2 = np.cross(r1, r2)
    #     r1_cross_r2_norm = np.linalg.norm(r1_cross_r2)
    #     if r1_cross_r2_norm < 1e-12:
    #         #  logging.warning(
    #         #     "Control point is on the filament. Returning zero induced velocity to avoid singularity."
    #         # )
    #         return np.zeros(3)

    #     # Outside of Core Radius
    #     elif dist > epsilon_bound:
    #         # Calculate the velocity using the Biot-Savart law
    #         return (
    #             gamma
    #             / (4 * np.pi)
    #             * r1_cross_r2
    #             / (r1_cross_r2_norm**2)
    #             * np.dot(self._r0, r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2))
    #         )

    #     # Within Core Radius apply smoothing
    #     else:
    #         # calculating smoothing factor
    #         smoothing_factor = (dist / epsilon_bound) ** 2
    #         return (
    #             smoothing_factor
    #             * gamma
    #             / (4 * np.pi)
    #             * r1_cross_r2
    #             / (r1_cross_r2_norm**2)
    #             * np.dot(self._r0, r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2))
    #         )

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

    # def calculate_induced_velocity(self, point, gamma, core_radius_fraction):

    #     # vector from the evaluation point to the start of the filament
    #     r1 = np.array(point) - self._x1
    #     # Vector perpendicular to both r1 and the filament direction (used to determine the direction of induced velocity)
    #     r1_cross_dir = np.cross(r1, self._direction)
    #     # Magnitude of the cross product vector
    #     norm_r1_cross_dir = np.linalg.norm(r1_cross_dir)
    #     # Component of r1 that is parallel to the filament direction (projection of r1 onto the direction)
    #     r1_dot_dir = np.dot(r1, self._direction)
    #     # # r1 vector projected onto the direction (r1's part that's parallel to filament)
    #     # r_perp = r1_dot_dir * self._direction
    #     # Defining the Core Radius of Semi Infinite Trailing Vortex Filament (eq 7, p.24, O.Cayon thesis)
    #     epsilon_semi_infinite = np.sqrt(
    #         4
    #         * self._alpha0
    #         * self._nu
    #         * np.linalg.norm(r1_dot_dir * self._direction)
    #         / self._vel_mag
    #     )  # Cut-off radius

    #     # if zero's
    #     if norm_r1_cross_dir == 0:
    #         logging.warning(
    #             "Control point is on the filament. Returning zero induced velocity to avoid singularity."
    #         )
    #         return np.zeros(3)

    #     # If point is outside core-radius
    #     elif norm_r1_cross_dir > epsilon_semi_infinite:
    #         # Calculate vel_ind using adapted Biot-Savart law (eq. 4.5, p.24, O.Cayon thesis)
    #         vel_ind = (
    #             (gamma / (4 * np.pi))
    #             * (1 + (r1_dot_dir / np.linalg.norm(r1)))
    #             * (1 / (norm_r1_cross_dir**2))
    #             * r1_cross_dir
    #         )
    #         return vel_ind * self._filament_direction

    #     # TODO: write proper test, for when inside the core radius
    #     # If point is inside core-radius
    #     else:
    #         # calculate projection Pj onto the vortex-center-line
    #         proj_r1 = r1_dot_dir * self._direction + epsilon_semi_infinite * (
    #             r1 / np.linalg.norm(r1) - self._direction
    #         ) / np.linalg.norm(r1 / np.linalg.norm(r1) - self._direction)
    #         cross_proj_r1 = np.cross(proj_r1, self._direction)

    #         # Calculate vel_ind using PROJECTED input through adapted Biot-Savart law (eq. 4.5, p.24, O.Cayon thesis)
    #         proj_vel_ind = (
    #             (gamma / (4 * np.pi))
    #             * (1 + (np.dot(proj_r1, self._direction) / np.linalg.norm(proj_r1)))
    #             * (1 / (np.linalg.norm(np.cross(proj_r1, self._direction) ** 2)))
    #             * cross_proj_r1
    #         )
    #         vel_ind = (cross_proj_r1 / epsilon_semi_infinite) * proj_vel_ind
    #         return vel_ind * self._filament_direction

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

    # def calculate_induced_velocity(self, point, gamma, core_radius_fraction):

    #     MP = np.array(point) - self._midpoint

    #     # calc. perpendicular distance
    #     P_to_AB = np.linalg.norm(
    #         np.cross(self._x2 - self._x1, point - self._x1)
    #     ) / np.linalg.norm(self._x2 - self._x1)

    #     # define the core radius
    #     epsilon_infinite = core_radius_fraction * self._length

    #     # outside of core radius
    #     if P_to_AB > epsilon_infinite:
    #         AB_cross_MP = np.cross(self._AB, MP)
    #         AB_cross_MP_dot_AB_cross_MP = np.dot(AB_cross_MP, AB_cross_MP)
    #         return (
    #             (gamma / (2 * np.pi))
    #             * (AB_cross_MP / AB_cross_MP_dot_AB_cross_MP)
    #             * np.linalg.norm(self._AB)
    #         )

    #     # if point on the filament
    #     elif P_to_AB == 0:
    #         logging.warning(
    #             "Control point is on the filament. Returning zero induced velocity to avoid singularity."
    #         )
    #         return np.zeros(3)

    #     # inside the core radius
    #     else:
    #         AB_cross_MP = np.cross(self._AB, MP)
    #         AB_cross_MP_dot_AB_cross_MP = np.dot(AB_cross_MP, AB_cross_MP)
    #         smoothing_factor = (P_to_AB / epsilon_infinite) ** 2
    #         return (
    #             smoothing_factor
    #             * (gamma / (2 * np.pi))
    #             * (AB_cross_MP / AB_cross_MP_dot_AB_cross_MP)
    #             * np.linalg.norm(self._AB)
    #         )
