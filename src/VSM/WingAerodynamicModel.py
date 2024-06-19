import numpy as np
from VSM.Panel import Panel

class WingAerodynamics:
    def __init__(
        self,
        wings: list, # List of Wing object instances
        initial_gamma_distribution: str = "elliptic",
        ring_geometry: str = "5fil",
    ):
        """
        A class to represent a vortex system.
        """
    
        for wing_instance in wings:
            sections = wing_instance.refine_aerodynamic_mesh()
            for i in range(len(sections)-1):
                self.panels.append(Panel(sections[i],sections[i+1]))
        self.va = None

    def calculate_AIC_matrices(self):
        pass
    def calculate_U2D_matrix(self):
        pass

    def get_panels(self):
        pass

    def get_gamma_distribution(self):
        pass

    def set_gamma_distribution(self, gamma):
        pass

    def calculate_effective_angle_of_attack(self, aerodynamic_center):
        # Placeholder for actual implementation
        pass

    def calculate_aerodynamic_coefficients(self, alpha):
        # Placeholder for actual implementation
        pass

    def calculate_aerodynamic_forces(self, alpha, cl, cd, cm):
        # Placeholder for actual implementation
        pass

    def get_wing_coefficients(self):
        # Placeholder for actual implementation
        return {"CL": 0.0, "CD": 0.0, "CS": 0.0, "CM": 0.0}
    
    def update_wake(self, va_matrix):
        # Placeholder for actual implementation
        pass

    def set_va(self, va, yaw_rate):
        self.va = va
        self.yaw_rate = yaw_rate
        # Make n panels, 3 array of the list va
        va_matrix = np.repeat([va], len(self.panels), axis=0)
        self.update_wake(va_matrix) # Define the trailing wake filaments
        
        


