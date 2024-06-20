from  abc import ABC

class HorshoeVortex:
    """
    A class to represent a horshoe vortex.

    input:
    a single panel object
    containing all the corner points and aerodynamic properties

    output:
    a horshoe vortex object

    """

    def __init__(self, LE_point_1, TE_point_1, LE_point_2, TE_point_2, aerodynamic_center = 0.25):
        
        self.filaments = []
        bound_point_1 = LE_point_1*(1-aerodynamic_center)+TE_point_1*aerodynamic_center
        bound_point_2 = LE_point_2*(1-aerodynamic_center)+TE_point_2*aerodynamic_center      
        self.filaments.append(BoundFilament(x1 = bound_point_1, x2 = bound_point_2))
        self.filaments.append(BoundFilament(x1 = TE_point_1, x2 = bound_point_1))
        self.filaments.append(BoundFilament(x1 = bound_point_2, x2 = TE_point_2))

    

    def get_velocity_induced_bound_2D(self, control_point):
        """"
        This function calculates the 2D induced velocity at the control point due to the bound vortex filaments
        """
        pass
    

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

    def calculate_induced_velocity(self, point):
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
    
    def calculate_induced_velocity(self, point):
        pass

class InfiniteFilament:
    """
    A class to represent a filament.

    input:
    two points defining the filament

    output:
    a filament object

    """

    def __init__(self, x1, direction):
        self.x1 = x1
        self.direction = direction
    
    def calculate_induced_velocity(self, point):
        pass