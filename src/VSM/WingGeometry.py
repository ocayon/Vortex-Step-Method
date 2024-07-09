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

        # 1. Compute the 1/4 chord line
        quarter_chord = LE + 0.25 * (TE - LE)

        # Calculate the length of each segment for the quarter chord line
        qc_lengths = np.linalg.norm(quarter_chord[1:] - quarter_chord[:-1], axis=1)
        qc_total_length = np.sum(qc_lengths)

        # Make cumulative array from 0 to the total length
        qc_cum_length = np.concatenate(([0], np.cumsum(qc_lengths)))

        # 2. Define target lengths based on desired spacing
        if self.spanwise_panel_distribution == "linear":
            target_lengths = np.linspace(0, qc_total_length, self.n_panels + 1)
        elif self.spanwise_panel_distribution == "cosine":
            theta = np.linspace(0, np.pi, self.n_panels + 1)
            target_lengths = qc_total_length * (1 - np.cos(theta)) / 2
        else:
            raise ValueError("Unsupported spanwise panel distribution")

        new_quarter_chord = np.zeros((self.n_panels + 1, 3))
        new_LE = np.zeros((self.n_panels + 1, 3))
        new_TE = np.zeros((self.n_panels + 1, 3))
        new_aero_input = np.empty((self.n_panels + 1,), dtype=object)
        new_sections = []

        # 3. Calculate new quarter chord points and interpolate aero inputs
        for i in range(self.n_panels + 1):
            target_length = target_lengths[i]

            # Find which segment the target length falls into
            section_index = np.searchsorted(qc_cum_length, target_length) - 1
            section_index = min(max(section_index, 0), len(qc_cum_length) - 2)

            # 4. Determine weights
            segment_start_length = qc_cum_length[section_index]
            segment_end_length = qc_cum_length[section_index + 1]
            t = (target_length - segment_start_length) / (
                segment_end_length - segment_start_length
            )
            left_weight = 1 - t
            right_weight = t

            # 3. Calculate new quarter chord point
            new_quarter_chord[i] = quarter_chord[section_index] + t * (
                quarter_chord[section_index + 1] - quarter_chord[section_index]
            )

            # 5. Compute average chord vector (corrected method)
            left_chord = TE[section_index] - LE[section_index]
            right_chord = TE[section_index + 1] - LE[section_index + 1]

            # Normalize the chord vectors
            left_chord_norm = left_chord / max(np.linalg.norm(left_chord), 1e-12)
            right_chord_norm = right_chord / max(np.linalg.norm(right_chord), 1e-12)

            # Interpolate the direction
            avg_direction = (
                left_weight * left_chord_norm + right_weight * right_chord_norm
            )
            avg_direction = avg_direction / max(np.linalg.norm(avg_direction), 1e-12)

            # Interpolate the length
            left_length = np.linalg.norm(left_chord)
            right_length = np.linalg.norm(right_chord)
            avg_length = left_weight * left_length + right_weight * right_length

            # Compute the final average chord vector
            avg_chord = avg_direction * avg_length

            # 6. Calculate new LE and TE points
            new_LE[i] = new_quarter_chord[i] - 0.25 * avg_chord
            new_TE[i] = new_quarter_chord[i] + 0.75 * avg_chord

            # Interpolate aero_input
            if aero_input[section_index][0] != aero_input[section_index + 1][0]:
                raise NotImplementedError(
                    "Different aero models over the span are not supported"
                )

            if aero_input[section_index][0] == "inviscid":
                new_aero_input[i] = ["inviscid"]
            elif aero_input[section_index][0] == "polars":
                raise NotImplementedError("Polar interpolation not implemented")
            elif aero_input[section_index][0] == "lei_airfoil_breukels":
                tube_diameter_left = aero_input[section_index][1][0]
                tube_diameter_right = aero_input[section_index + 1][1][0]
                tube_diameter_i = (
                    tube_diameter_left * left_weight
                    + tube_diameter_right * right_weight
                )

                chamber_height_left = aero_input[section_index][1][1]
                chamber_height_right = aero_input[section_index + 1][1][1]
                chamber_height_i = (
                    chamber_height_left * left_weight
                    + chamber_height_right * right_weight
                )

                new_aero_input[i] = [
                    "lei_airfoil_breukels",
                    [tube_diameter_i, chamber_height_i],
                ]

                logging.debug(f"left_weight: {left_weight}")
                logging.debug(f"right_weight: {right_weight}")
                logging.debug(f"tube_diameter_i: {tube_diameter_i}")
                logging.debug(f"chamber_height_i: {chamber_height_i}")

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
