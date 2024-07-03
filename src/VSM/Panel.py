import numpy as np
from VSM.Filament import BoundFilament, Infinite2DFilament


class Panel:
    def __init__(
        self,
        section_1,
        section_2,
        aerodynamic_center_location=0.25,
        control_point_location=0.75,
    ):

        self._TE_point_1 = section_1.TE_point
        self._LE_point_1 = section_1.LE_point
        self._TE_point_2 = section_2.TE_point
        self._LE_point_2 = section_2.LE_point
        self._airfoil_aero_model = section_1.aero_input[0]
        self._airfoil_data = self.calculate_airfoil_data(
            section_1.aero_input, section_2.aero_input
        )
        self._chord = np.average(
            [
                np.linalg.norm(section_1.TE_point - section_1.LE_point),
                np.linalg.norm(section_2.TE_point - section_2.LE_point),
            ]
        )
        self._va = None

        # Calculate the control point and aerodynamic center
        # Be wary that x is defined positive into the wind, so from TE to LE
        mid_LE_point = self.LE_point_1 + 0.5 * (self._LE_point_2 - self._LE_point_1)
        mid_TE_point = self.TE_point_1 + 0.5 * (self._TE_point_2 - self._TE_point_1)
        vec_LE_to_TE = mid_TE_point - mid_LE_point
        self._aerodynamic_center = (
            mid_LE_point + aerodynamic_center_location * vec_LE_to_TE
        )
        self._control_point = mid_LE_point + control_point_location * vec_LE_to_TE

        # Setting up the filaments
        self.filaments = []
        self.bound_point_1 = self._LE_point_1 + aerodynamic_center_location * (
            self._TE_point_1 - self._LE_point_1
        )
        self.bound_point_2 = self._LE_point_2 + aerodynamic_center_location * (
            self._TE_point_2 - self._LE_point_2
        )
        self.filaments.append(
            BoundFilament(x1=self.bound_point_1, x2=self.bound_point_2)
        )
        self.filaments.append(BoundFilament(x1=self.TE_point_1, x2=self.bound_point_1))
        self.filaments.append(BoundFilament(x1=self.bound_point_2, x2=self.TE_point_2))
        self._filament_2d = Infinite2DFilament(self.bound_point_1, self.bound_point_2)
        self._gamma = None  # Initialize the gamma attribute

        # Calculate the local reference frame, below are all unit_vectors
        # x_airf defined upwards from the chord-line, perpendicular to the panel
        # y_airf defined parallel to the chord-line, from LE-to-TE
        # z_airf along the LE, in plane (towards left tip, along span) from the airfoil perspective
        self._y_airf = vec_LE_to_TE / np.linalg.norm(vec_LE_to_TE)
        self._z_airf = (self.bound_point_2 - self.bound_point_1) / np.linalg.norm(
            self.bound_point_2 - self.bound_point_1
        )
        self._x_airf = np.cross(self._y_airf, self._z_airf)

    ###########################
    ## GETTER FUNCTIONS
    ###########################
    @property
    def control_point(self):
        return self._control_point

    @property
    def z_airf(self):
        """Unit vector pointing in the airfoil plane, so that is towards left-tip in spanwise direction"""
        return self._z_airf

    @property
    def x_airf(self):
        """Unit vector pointing upwards from the chord-line, perpendicular to the panel"""
        return self._x_airf

    @property
    def y_airf(self):
        """Unit vector pointing parallel to the chord-line, from LE-to-TE"""
        return self._y_airf

    @property
    def va(self):
        return self._va

    @property
    def aerodynamic_center(self):
        return self._aerodynamic_center

    @property
    def corner_points(self):
        return np.array(
            [self._LE_point_1, self._TE_point_1, self._TE_point_2, self._LE_point_2]
        )

    @property
    def chord(self):
        return self._chord

    @property
    def TE_point_1(self):
        return self._TE_point_1

    @property
    def TE_point_2(self):
        return self._TE_point_2

    @property
    def LE_point_1(self):
        return self._LE_point_1

    @property
    def LE_point_2(self):
        return self._LE_point_2

    ###########################
    ## SETTER FUNCTIONS
    ###########################
    @va.setter
    def va(self, value):
        self._va = value

    ###########################
    ## CALCULATE FUNCTIONS      # All this return smthing
    ###########################

    def calculate_relative_alpha_and_relative_velocity(
        self, induced_velocity: np.array
    ):
        """Calculates the relative angle of attack and relative velocity of the panel

        Args:
            induced_velocity (np.array): Induced velocity at the control point

        Returns:
            alpha (float): Relative angle of attack of the panel
            relative_velocity (np.array): Relative velocity of the panel
        """
        # Calculate relative velocity and angle of attack
        relative_velocity = self.va + induced_velocity
        v_normal = np.dot(self._x_airf, relative_velocity)
        v_tangential = np.dot(self._y_airf, relative_velocity)
        alpha = np.arctan(v_normal / v_tangential)
        return alpha, relative_velocity

    def calculate_inviscid_polar_data(self):
        """Calculates the lift, drag and moment coefficients of the panel

        Args:
            None

        Returns:
            airfoil_data (np.array): Array containing the lift, drag and moment coefficients of the panel
        """
        aoa = np.arange(-20, 21, 1)
        airfoil_data = np.empty(
            (
                len(aoa),
                4,
            )
        )
        for j, alpha in enumerate(aoa):
            cl, cd, cm = 2 * np.pi * np.sin(np.deg2rad(alpha)), 0.05, 0.01
            airfoil_data[j, 0] = np.deg2rad(alpha)
            airfoil_data[j, 1] = cl
            airfoil_data[j, 2] = cd
            airfoil_data[j, 3] = cm

        return airfoil_data

    def calculate_airfoil_data(self, aero_input_1, aero_input_2):
        """Calculates the aerodynamic properties of the panel

        Args:
            aero_input_1 (tuple): Aerodynamic properties of the first section
            aero_input_2 (tuple): Aerodynamic properties of the second section

        Returns:
            airfoil_data (np.array): [alpha,cl,cd,cm] of size (n_alpha, 4)
        """
        if (aero_input_1[0] and aero_input_2[0]) == "inviscid":
            return self.calculate_inviscid_polar_data()

        elif (aero_input_1[0] and aero_input_2[0]) == "lei_airfoil_breukels":
            # TODO: 1. Average the Geometry, to find the mid-panel airfoil
            # TODO: 2. Calculate the aerodynamic properties of the mid-panel airfoil
            # TODO: 3. Write corresponding pytest in test_Panel.py
            return NotImplementedError
        else:
            raise NotImplementedError

    def calculate_cl(self, alpha: float):
        """Calculates the lift coefficient of the panel

        Args:
            alpha (float): Angle of attack of the panel

        Returns:
            float: Lift coefficient of the panel
        """

        if self._airfoil_aero_model == "inviscid":
            return 2 * np.pi * np.sin(alpha)
        else:
            # TODO: once implemented add pytest
            raise NotImplementedError

    def calculate_velocity_induced_bound_2D(self, control_point, gamma=None):
        """ "
        This function calculates the 2D induced velocity at the control point due to the bound vortex filaments
        """
        if gamma is None:
            gamma = self._gamma

        return self._filament_2d.calculate_induced_velocity(control_point, gamma)

    def calculate_velocity_induced_horseshoe(self, control_point, gamma=None):
        """ "
        This function calculates the induced velocity at the control point due to the bound vortex filaments
        """
        if gamma is None:
            gamma = self._gamma

        ind_vel = np.zeros(3)
        for filament in self.filaments:
            ind_vel += filament.calculate_induced_velocity(control_point, gamma)

        return ind_vel

    def calculate_filaments_for_plotting(self):
        filaments = []
        for filament in self.filaments:
            x1 = filament.x1
            if hasattr(filament, "x2") and filament.x2 is not None:
                x2 = filament.x2
                color = "magenta"
            else:
                # For semi-infinite filaments
                x2 = x1 + 5 * self.chord * (self.va / np.linalg.norm(self.va))
                color = "orange"
                if filament.filament_direction == -1:
                    x1, x2 = x2, x1
                    color = "red"

            filaments.append([x1, x2, color])
        return filaments
