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
    
    def __init__(self, x1, x2):
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.length = np.linalg.norm(self.x2 - self.x1)

    def calculate_induced_velocity(self, point, gamma=1.0):
        point = np.array(point)
        r0 = self.x2 - self.x1  # Vortex filament
        r1 = point - self.x1  # Control point to one end of the vortex filament
        r2 = point - self.x2  # Control point to the other end of the vortex filament
        # Cross products used for later computations
        r1Xr0 = np.cross(r1, r0)
        r2Xr0 = np.cross(r2, r0)

        epsilon = 0.05 * self.length  # Cut-off radius

        if np.linalg.norm(r1Xr0) / self.length > epsilon:  # Perpendicular distance from XVP to vortex filament (r0)
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
            # proj stands for the vectors respect to the new control point
            r1_proj = (
                np.dot(r1, r0) * r0 / (self.length**2)
                + epsilon * r1Xr0 / np.linalg.norm(r1Xr0)
            )
            r2_proj = (
                np.dot(r2, r0) * r0 / (self.length**2)
                + epsilon * r2Xr0 / np.linalg.norm(r2Xr0)
            )
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

class SemiInfiniteFilament(Filament):
    """
    A class to represent a filament.

    input:
    two points defining the filament

    output:
    a filament object

    """

    def __init__(self, x1, direction, filament_direction, alpha0=1.25643, nu=1.48e-5):
        self.x1 = x1
        # x2 is a point far away from the filament, defined here for plotting purposes
        self.x2 = x1 + filament_direction*direction * 0.5
        self.direction = direction
        self.alpha0 = alpha0  # Oseen parameter
        self.nu = nu  # Kinematic viscosity of air
        self._filament_direction = filament_direction

    # TODO: what is the purpose of Uinf here? It does NOT take correct value
    def calculate_induced_velocity(self, point, gamma, Uinf=10):
        r1 = self.x1 - point  # Vector from control point to XV1
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
        return vel_ind*self._filament_direction

class Infinite2DFilament(Filament):
    """
    A class to represent an infinite 2D vortex filament.
    
    Input:
    two points defining the filament
    
    Output:
    a filament object
    """

    def __init__(self, x1, x2):
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.length = np.linalg.norm(self.x2 - self.x1)

    def calculate_induced_velocity(self, control_point, gamma=1.0):
        A = self.x1
        B = self.x2
        r0 = B - A
        AP = control_point - A

        # Projection of AP onto AB
        r0_unit = r0 / np.linalg.norm(r0)
        projection_length = np.dot(AP, r0_unit)
        projection_point = projection_length * r0_unit

        # Vector r3 from the projection point to the control point P
        r3 = control_point - projection_point

        # Calculate the cross product of r0 and r3
        cross = np.cross(r0, r3)

        # Magnitude squared of the cross product vector
        cross_norm_sq = np.dot(cross, cross)

        # Induced velocity calculation
        ind_vel = (gamma / (2 * np.pi)) * (cross / cross_norm_sq) * np.linalg.norm(r0)

        return ind_vel