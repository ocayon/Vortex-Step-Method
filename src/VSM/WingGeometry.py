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

    def refine_aerodynamic_mesh(self):

        # Extract LE, TE, and aero_input from the sections
        LE, TE, aero_input = (
            np.zeros((len(self.sections), 3)),
            np.zeros((len(self.sections), 3)),
            [],
        )
        for i, section in enumerate(self.sections):
            LE[i] = section.LE_point
            TE[i] = section.TE_point
            aero_input.append(section.aero_input)

        # Edge cases
        if len(LE) != len(TE) or len(LE) != len(aero_input):
            raise ValueError("LE, TE, and aero_input must have the same length")
        if self.n_panels == 1:
            new_sections = [
                Section(LE[0], TE[0], aero_input[0]),
                Section(LE[-1], TE[-1], aero_input[-1]),
            ]
            return new_sections

        # Calculate the length of each segment for LE and TE
        LE_lengths = np.linalg.norm(LE[1:] - LE[:-1], axis=1)
        TE_lengths = np.linalg.norm(TE[1:] - TE[:-1], axis=1)

        LE_total_length = np.sum(LE_lengths)
        TE_total_length = np.sum(TE_lengths)

        # Make cumulative arrays from 0 to the total length
        LE_cum_length = np.concatenate(([0], np.cumsum(LE_lengths)))
        TE_cum_length = np.concatenate(([0], np.cumsum(TE_lengths)))

        # Defining target_lengths array for LE and TE based on desired spacing
        if self.spanwise_panel_distribution == "linear":
            LE_target_lengths = np.linspace(0, LE_total_length, self.n_panels + 1)
            TE_target_lengths = np.linspace(0, TE_total_length, self.n_panels + 1)
        elif self.spanwise_panel_distribution == "cosine":
            theta = np.linspace(0, np.pi, self.n_panels + 1)
            LE_target_lengths = LE_total_length * (1 - np.cos(theta)) / 2
            TE_target_lengths = TE_total_length * (1 - np.cos(theta)) / 2
        else:
            raise ValueError("Unsupported spanwise panel distribution")

        new_LE = np.zeros((self.n_panels + 1, 3))
        new_TE = np.zeros((self.n_panels + 1, 3))
        new_aero_input = np.empty((self.n_panels + 1,), dtype=object)
        new_sections = []

        # loop over each new section, and populate it
        for i in range(self.n_panels + 1):
            LE_target_length = LE_target_lengths[i]
            TE_target_length = TE_target_lengths[i]

            # Find which segment the target length falls into for LE
            LE_section_index = np.searchsorted(LE_cum_length, LE_target_length) - 1
            LE_section_index = min(max(LE_section_index, 0), len(LE_cum_length) - 2)

            # Find which segment the target length falls into for TE
            TE_section_index = np.searchsorted(TE_cum_length, TE_target_length) - 1
            TE_section_index = min(max(TE_section_index, 0), len(TE_cum_length) - 2)

            # Interpolation for LE
            LE_segment_start_length = LE_cum_length[LE_section_index]
            LE_segment_end_length = LE_cum_length[LE_section_index + 1]
            LE_t = (LE_target_length - LE_segment_start_length) / (
                LE_segment_end_length - LE_segment_start_length
            )
            # Interpolation for TE
            TE_segment_start_length = TE_cum_length[TE_section_index]
            TE_segment_end_length = TE_cum_length[TE_section_index + 1]
            TE_t = (TE_target_length - TE_segment_start_length) / (
                TE_segment_end_length - TE_segment_start_length
            )

            # Keep the corner points fixed
            if i == 0:
                new_LE[i] = LE[0]
                new_TE[i] = TE[0]
                new_aero_input[i] = aero_input[0]
            elif i == self.n_panels:
                new_LE[i] = LE[-1]
                new_TE[i] = TE[-1]
                new_aero_input[i] = aero_input[-1]
            else:
                new_LE[i] = LE[LE_section_index] + LE_t * (
                    LE[LE_section_index + 1] - LE[LE_section_index]
                )
                new_TE[i] = TE[TE_section_index] + TE_t * (
                    TE[TE_section_index + 1] - TE[TE_section_index]
                )

            print(f"new_LE[i]: {new_LE[i]}, new_TE[i]: {new_TE[i]}")

            # Edge case: different aero models over the span
            if aero_input[LE_section_index][0] != aero_input[LE_section_index + 1][0]:
                raise NotImplementedError(
                    "Different aero models over the span are not supported"
                )

            if aero_input[LE_section_index][0] == "inviscid":
                new_aero_input[i] = ["inviscid"]
            elif aero_input[LE_section_index][0] == "polars":
                raise NotImplementedError("Polar interpolation not implemented")
            elif aero_input[LE_section_index][0] == "lei_airfoil_breukels":
                # Calculate how close we are to the provided sections left and right
                left_distance = LE_target_length - LE_cum_length[LE_section_index]
                right_distance = LE_cum_length[LE_section_index + 1] - LE_target_length
                total_distance = left_distance + right_distance
                left_weight = right_distance / total_distance
                right_weight = left_distance / total_distance
                # Interpolate the aero_input values
                tube_diameter_left = aero_input[LE_section_index][1][0]
                tube_diameter_right = aero_input[LE_section_index + 1][1][0]
                tube_diameter_i = (
                    tube_diameter_left * left_weight
                    + tube_diameter_right * right_weight
                )
                chamber_height_left = aero_input[LE_section_index][1][1]
                chamber_height_right = aero_input[LE_section_index + 1][1][1]
                chamber_height_i = (
                    chamber_height_left * left_weight
                    + chamber_height_right * right_weight
                )
                new_aero_input[i] = [
                    "lei_airfoil_breukels",
                    [tube_diameter_i, chamber_height_i],
                ]

                print(f"left_distance: {left_distance}")
                print(f"right_distance: {right_distance}")
                print(f"left_weight: {left_weight}")
                print(f"right_weight: {right_weight}")
                print(f"tube_diameter_left: {tube_diameter_left}")
                print(f"tube_diameter_right: {tube_diameter_right}")
                print(f"tube_diameter_i: {tube_diameter_i}")

            new_sections.append(Section(new_LE[i], new_TE[i], new_aero_input[i]))

        return new_sections

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
