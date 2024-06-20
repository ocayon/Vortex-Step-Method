import numpy as np
from VSM.HorshoeVortex import HorshoeVortex


class Panel:
    def __init__(self, section_1, section_2):
        self.corner_points
        self.aerodynamic_properties
        self.local_reference_frame
        self.horshoe_vortex = HorshoeVortex(self.corner_points, self.aerodynamic_center)
        self.control_point
        self.aerodynamic_center


    
    def get_relative_alpha_and_relative_velocity(self,induced_velocity):
        # Calculate terms of induced corresponding to the airfoil directions
        norm_airf = self.local_reference_frame()[:, 0]
        tan_airf = self.local_reference_frame()[:, 1]

        # Calculate relative velocity and angle of attack
        relative_velocity = self.va + induced_velocity
        vn = np.dot(norm_airf, relative_velocity)
        vtan = np.dot(tan_airf, relative_velocity)
        alpha = np.arctan(vn / vtan)
        return alpha, relative_velocity


    def get_control_point(self):
        return self.control_point

    def velocity_induced_bound_2D(self):
        """Calculates the induced velocity inside HorshoeVortex Class"""
        return self.horshoe_vortex.get_velocity_induced_bound_2D(self.control_point)

    def update_va(va)
        self.va = va
    
    def get_z_airf():
        '''Returns the z vector of the airfoil frame of reference
            
            This is the spanwise/out of plane direction of the airfoil'''
        return self.local_reference_frame()[:,2]

    def get_va(self):
        return self.va