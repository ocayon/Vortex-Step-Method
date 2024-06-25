from dataclasses import dataclass, field
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class Wing:
    n_panels: int
    spanwise_panel_distribution: str = "linear"
    spanwise_direction: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    sections: List["Section"] = field(default_factory=list)  # child-class

    def add_section(self, LE_point: np.array, TE_point: np.array, aero_input: str):
        self.sections.append(Section(LE_point, TE_point, aero_input))

    ## TODO: must be tested
    def refine_aerodynamic_mesh(self):
        LE = np.array([section.LE_point for section in self.sections])
        TE = np.array([section.TE_point for section in self.sections])
        aero_input = np.array([section.aero_input for section in self.sections])

        # Calculate the total length along the leading edge (LE) points
        lengths = np.linalg.norm(LE[1:] - LE[:-1], axis=1)
        n_provided = np.sum(lengths)

        # Compute the cumulative length along the LE points
        cum_length = np.concatenate(([0], np.cumsum(lengths)))

        # Compute the target lengths for each of the n_panels points
        target_lengths = np.linspace(0, n_provided, self.n_panels + 1)

        # Initialize arrays to hold the new LE and TE points
        new_LE = np.zeros((self.n_panels + 1, 3))
        new_TE = np.zeros((self.n_panels + 1, 3))
        new_aero_input = np.empty((self.n_panels + 1,), dtype=object)
        new_sections = []

        # Handle the inviscid case at once
        if aero_input[0][0] == "inviscid":
            airfoil_data = self._calculate_inviscid_polar_data()
        else:
            raise NotImplementedError

        # Interpolate new LE and TE points
        for i, target_length in enumerate(target_lengths):
            # Find the segment in which the target_length falls
            section_index = np.searchsorted(cum_length, target_length) - 1
            section_index = min(
                max(section_index, 0), len(cum_length) - 2
            )  # Ensure index bounds

            # Calculate interpolation factor t
            segment_start_length = cum_length[section_index]
            segment_end_length = cum_length[section_index + 1]
            t = (target_length - segment_start_length) / (
                segment_end_length - segment_start_length
            )

            # Interpolate points
            new_LE[i] = LE[section_index] + t * (
                LE[section_index + 1] - LE[section_index]
            )
            new_TE[i] = TE[section_index] + t * (
                TE[section_index + 1] - TE[section_index]
            )

            # Interpolate aero_input
            if aero_input[section_index][0] != aero_input[section_index + 1][0]:
                # this entails different aero model over the span
                raise NotImplementedError

            if aero_input[section_index][0] == "inviscid":
                new_aero_input[i] = ["polars", airfoil_data[i]]

            elif aero_input[section_index][0] == "polars":
                # TODO: perform a polar interpolation
                raise NotImplementedError

            elif aero_input[section_index][0] == "lei_airfoil_breukels":
                # TODO: perform geometric interpolation
                raise NotImplementedError

            # Appending to the new_section
            new_sections.append(Section(new_LE[i], new_TE[i], new_aero_input[i]))

        return new_sections

    def _calculate_inviscid_polar_data(self):
        aoa = np.arange(-20, 21, 1)
        airfoil_data = np.empty((len(aoa), 4, 1))
        for i in range(self.n_panels - 1):
            for j, alpha in enumerate(aoa):
                cl, cd, cm = 2 * np.pi * np.sin(alpha), 0.05, 0.01
                airfoil_data[j, 0, i] = alpha
                airfoil_data[j, 1, i] = cl
                airfoil_data[j, 2, i] = cd
                airfoil_data[j, 3, i] = cm
        return airfoil_data

    # @property
    # def n_panels(self):
    #     return self.n_panels

    # TODO: add test here, assessing for example the types of the inputs
    def calculate_wing_span(self):
        """Calculates the span of the wing along a given vector axis

        Args:
            None

        Returns:
            span (float): The span of the wing along the given vector axis"""
        # Normalize the vector_axis to ensure it's a unit vector
        vector_axis = self.spanwise_direction / np.linalg.norm(self.spanwise_direction)

        # Concatenate the leading and trailing edge points for all sections
        all_points = np.concatenate(
            [[section.LE_point, section.TE_point] for section in self.sections]
        )

        # Project all points onto the vector axis
        projections = np.dot(all_points, vector_axis)

        # Calculate the span of the wing along the given vector axis
        span = np.max(projections) - np.min(projections)
        return span


@dataclass
class Section:
    LE_point: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    TE_point: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    aero_input: list = field(default_factory=list)

    # TODO: Ideas on what the other aero_input could be populated with:
    # ['polars', [CL_alpha, CD_alpha, CM_alpha]]
    # ['lei_airfoil_breukels', [tube_diameter, chamber_height]]

    # CL_alpha: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    # CD_alpha: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    # CM_alpha: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))


######################
#### IDEATION ON FUNCTIONS THAT WILL BE USEFULL FOR THE REFINE_AERODYNAMIC_MESH FUNCTION
######################


def calculate_panel_distribution(self):
    """
    Input is the wing_instance of the Wing object
    1. transform the wing geometry into n_panels,
        using the defined spanwise_panel_distribution
        and finds the corner points of each.
    2. Using the provided aerodynamic properties
        for each section, calculates the aerodynamic
        properties for each panel by integrating
    """
    wing_panel_corner_points = self.__compute_wing_panel_distribution()
    return self.__compute_panel_aerodynamic_coefficients(wing_panel_corner_points)


def __compute_wing_panel_corner_points(self):
    """
    Computes the chordwise section of the wing
    Using
    - self.n_panels
    - self.spanwise_panel_distribution
    - self.sections
    """
    wing_panel_corner_points = ["panel_corner_points_1", "panel_corner_points_2"]
    return wing_panel_corner_points


def __compute_panel_aerodynamic_coefficients(self, sections):
    """
    Computes the aerodynamic properties of each panel
    Using
    - self.sections
    """
    wing_panel_corner_points = np.concatenate(
        [[section.LE_point, section.TE_point] for section in sections]
    )
    panel_aerodynamic_properties = ["CL-alpha", "CD-alpha", "CM-alpha"]
    return np.append(wing_panel_corner_points, panel_aerodynamic_properties, axis=1)
