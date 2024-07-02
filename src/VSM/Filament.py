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

    @abstractmethod
    def calculate_induced_velocity(self, point, gamma):
        pass


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

    @property
    def x1(self):
        return self._x1

    @property
    def x2(self):
        return self._x2

    def calculate_induced_velocity(self, point, gamma=1.0, core_radius_fraction=0.01):
        """Calculate the induced velocity at a control point due to the vortex filament.

            Checks if the evaluation point is on the filament and returns zero velocity to avoid singularity.
            Uses the Biot-Savart law to calculate the induced velocity at the control point.
            If evaluation point is within the core radius, applies smoothing to avoid singularities.

        Args:
            point (np.ndarray): The control point at which the induced velocity is calculated.
            gamma (float): The circulation strength of the vortex filament.
            core_radius_fraction (float): The fraction of the filament length that defines the core radius.

        Returns:
            np.ndarray: The induced velocity at the control point.
        """
        r0 = self._x2 - self._x1  # Vortex filament
        r1 = (
            np.array(point) - self._x1
        )  # Control point to one end of the vortex filament
        r2 = (
            np.array(point) - self._x2
        )  # Control point to the other end of the filament

        # Check if the control point is on the filament
        r1_cross_r2 = np.cross(r1, r2)
        r1_cross_r2_norm = np.linalg.norm(r1_cross_r2)
        if r1_cross_r2_norm < 1e-12:
            logging.warning(
                "Control point is on the filament. Returning zero induced velocity to avoid singularity."
            )
            return np.zeros(3)

        # Calculate the velocity using the Biot-Savart law
        vel_ind = (
            gamma
            / (4 * np.pi)
            * r1_cross_r2
            / (r1_cross_r2_norm**2)
            * np.dot(r0, r1 / np.linalg.norm(r1) - r2 / np.linalg.norm(r2))
        )

        # evaluation point within the core distance, apply smoothing
        epsilon_bound = core_radius_fraction * self._length
        dist = np.linalg.norm(np.cross(r1, r0)) / self._length
        # TODO: change equation back to original shown on p.24 of thesis
        if dist <= epsilon_bound:
            smoothing_factor = (dist / epsilon_bound) ** 2
            vel_ind *= smoothing_factor

        return vel_ind


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
        self._x2 = x1 + filament_direction * direction * 0.5
        self._direction = direction  # unit vector of apparent wind speed
        self._alpha0 = alpha0  # Oseen parameter
        self._nu = nu  # Kinematic viscosity of air
        self._vel_mag = vel_mag  # the magnitude of the apparent wind speed
        self._filament_direction = filament_direction  # -1 or 1, indicating if its with or against the direction of the apparent wind speed

    @property
    def x1(self):
        return self._x1

    @property
    def x2(self):
        return self._x2

    # TODO: Uinf scales epsilon, why does this make sense?
    def calculate_induced_velocity(self, point, gamma=1):

        # vector from the evaluation point to the start of the filament
        r1 = np.array(point) - self._x1
        # Vector perpendicular to both r1 and the filament direction (used to determine the direction of induced velocity)
        r1_cross_dir = np.cross(r1, self._direction)
        # Magnitude of the cross product vector
        norm_r1_cross_dir = np.linalg.norm(r1_cross_dir)
        # Component of r1 that is parallel to the filament direction (projection of r1 onto the direction)
        r1_dot_dir = np.dot(r1, self._direction)
        # # r1 vector projected onto the direction (r1's part that's parallel to filament)
        # r_perp = r1_dot_dir * self._direction
        # Defining the Core Radius of Semi Infinite Trailing Vortex Filament (eq 7, p.24, O.Cayon thesis)
        epsilon_semi_infinite = np.sqrt(
            4
            * self._alpha0
            * self._nu
            * np.linalg.norm(r1_dot_dir * self._direction)
            / self._vel_mag
        )  # Cut-off radius

        # if zero's
        if norm_r1_cross_dir == 0:
            logging.warning(
                "Control point is on the filament. Returning zero induced velocity to avoid singularity."
            )
            return np.zeros(3)

        # If point is outside core-radius
        elif norm_r1_cross_dir > epsilon_semi_infinite:
            # Calculate vel_ind using adapted Biot-Savart law (eq. 4.5, p.24, O.Cayon thesis)
            vel_ind = (
                (gamma / (4 * np.pi))
                * (1 + (r1_dot_dir / np.linalg.norm(r1)))
                * (1 / (norm_r1_cross_dir**2))
                * r1_cross_dir
            )
            return vel_ind * self._filament_direction

        # TODO: write proper test, for when inside the core radius
        # If point is inside core-radius
        else:
            # calculate projection Pj onto the vortex-center-line
            proj_r1 = r1_dot_dir * self._direction + epsilon_semi_infinite * (
                r1 / np.linalg.norm(r1) - self._direction
            ) / np.linalg.norm(r1 / np.linalg.norm(r1) - self._direction)
            cross_proj_r1 = np.cross(proj_r1, self._direction)

            # Calculate vel_ind using PROJECTED input through adapted Biot-Savart law (eq. 4.5, p.24, O.Cayon thesis)
            proj_vel_ind = (
                (gamma / (4 * np.pi))
                * (1 + (np.dot(proj_r1, self._direction) / np.linalg.norm(proj_r1)))
                * (1 / (np.linalg.norm(np.cross(proj_r1, self._direction) ** 2)))
                * cross_proj_r1
            )
            vel_ind = (cross_proj_r1 / epsilon_semi_infinite) * proj_vel_ind
            return vel_ind * self._filament_direction


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

    @property
    def x1(self):
        return self._x1

    @property
    def x2(self):
        return self._x2

    def calculate_induced_velocity(self, point, gamma=1.0):
        A = self._x1
        B = self._x2
        r0 = B - A
        AP = point - A

        # Projection of AP onto AB
        r0_unit = r0 / np.linalg.norm(r0)
        projection_length = np.dot(AP, r0_unit)
        projection_point = projection_length * r0_unit

        # Vector r3 from the projection point to the control point P
        r3 = point - projection_point

        # Calculate the cross product of r0 and r3
        cross = np.cross(r0, r3)

        # Magnitude squared of the cross product vector
        cross_norm_sq = np.dot(cross, cross)

        # Induced velocity calculation
        ind_vel = (gamma / (2 * np.pi)) * (cross / cross_norm_sq) * np.linalg.norm(r0)

        return ind_vel
