import dataclasses
import numpy as np
from typing import List


@dataclasses
class Wing:
    n_panels: int
    spanwise_panel_distribution: str = "linear"
    spanwise_direction: np.array = np.array([0, 1, 0])
    Section: List[Section]  # child-class

    ## TODO: must be tested
    def refine_aerodynamic_mesh(self):
        LE = np.array([section.LE_point for section in self.Section])
        TE = np.array([section.TE_point for section in self.Section])

        # Calculate the total length along the leading edge (LE) points
        lengths = np.linalg.norm(LE[1:] - LE[:-1], axis=1)
        n_provided = np.sum(lengths)
        sections_length = n_provided / self.n_panels

        # Compute the cumulative length along the LE points
        cum_length = np.concatenate(([0], np.cumsum(lengths)))

        # Compute the target lengths for each of the n_panels points
        target_lengths = np.linspace(0, n_provided, self.n_panels)

        # Initialize arrays to hold the new LE and TE points
        new_LE = np.zeros((self.n_panels, 3))
        new_TE = np.zeros((self.n_panels, 3))

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

        # TODO: define the aerodynamic interpolation as well
        # TODO: define the right-output (See what panel expects)

        return new_LE, new_TE

    def get_n_panels(self):
        return self.n_panels

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
            [[section.LE_point, section.TE_point] for section in self.Section]
        )

        # Project all points onto the vector axis
        projections = np.dot(all_points, vector_axis)

        # Calculate the span of the wing along the given vector axis
        span = np.max(projections) - np.min(projections)
        return span


@dataclasses
class Section:
    LE_point: np.array
    TE_point: np.array
    CL_alpha: np.array
    CD_alpha: np.array
    CM_alpha: np.array


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
