import numpy as np
from VSM.HorshoeVortex import HorshoeVortex

class Panel:
    def __init__(self, section_1, section_2):
        self.corner_points
        self.aerodynamic_properties
        self.horshoe_vortex = HorshoeVortex(self.corner_points, self.aerodynamic_center)
        self._va = None

    ###########################
    ## GETTER FUNCTIONS
    ###########################
    @property
    def local_reference_frame(self):
        # Calculate reference frame
        return self._local_reference_frame


    @property
    def control_point(self):
        # Calculate here
        return self._control_point

    @property
    def z_airf(self):
        '''Returns the z vector of the airfoil frame of reference
        
        This is the spanwise/out of plane direction of the airfoil'''
        return self.local_reference_frame()[:, 2]

    @property
    def va(self):
        return self._va

    @property
    def aerodynamic_center(self):
        return self._aerodynamic_center

    ###########################
    ## SETTER FUNCTIONS
    ###########################
    #TODO: remove this comment. To set va: panel.va = value (array)
    @va.setter
    def va(self, value)
        self._va = value

    ###########################
    ## CALCULATE FUNCTIONS      # All this return smthing
    ###########################

    def calculate_velocity_induced_bound_2D(self):
        """Calculates the induced velocity inside HorshoeVortex Class"""
        return self.horshoe_vortex.get_velocity_induced_bound_2D(self.control_point)

    def calculate_velocity_induced(self, control_point: np.array, strength: float):
        pass

    def calculate_aerodynamic_coefficients(self, alpha: float):
        pass

    def calculate_relative_alpha_and_relative_velocity(self, induced_velocity: np.array):
        # Calculate terms of induced corresponding to the airfoil directions
        norm_airf = self.local_reference_frame()[:, 0]
        tan_airf = self.local_reference_frame()[:, 1]

        # Calculate relative velocity and angle of attack
        relative_velocity = self.va + induced_velocity
        vn = np.dot(norm_airf, relative_velocity)
        vtan = np.dot(tan_airf, relative_velocity)
        alpha = np.arctan(vn / vtan)
        return alpha, relative_velocity
