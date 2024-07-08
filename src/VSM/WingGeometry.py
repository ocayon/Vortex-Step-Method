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
        LE = np.array([section.LE_point for section in self.sections])
        TE = np.array([section.TE_point for section in self.sections])
        aero_input = np.array([section.aero_input for section in self.sections])

        lengths = np.linalg.norm(LE[1:] - LE[:-1], axis=1)
        n_provided = np.sum(lengths)
        cum_length = np.concatenate(([0], np.cumsum(lengths)))

        if self.spanwise_panel_distribution == "linear":
            target_lengths = np.linspace(0, n_provided, self.n_panels + 1)
        elif self.spanwise_panel_distribution == "cosine":
            # Cosine spacing formula
            theta = np.linspace(0, np.pi, self.n_panels + 1)
            target_lengths = n_provided * (1 - np.cos(theta)) / 2
        else:
            raise ValueError("Unsupported spanwise panel distribution")

        new_LE = np.zeros((self.n_panels + 1, 3))
        new_TE = np.zeros((self.n_panels + 1, 3))
        new_aero_input = np.empty((self.n_panels + 1,), dtype=object)
        new_sections = []

        for i, target_length in enumerate(target_lengths):
            section_index = np.searchsorted(cum_length, target_length) - 1
            section_index = min(max(section_index, 0), len(cum_length) - 2)

            # Keep the corner points fixed
            if i == 0:
                print(f"first section index: {section_index}")
                new_LE[i] = LE[0]
                new_TE[i] = TE[0]
                new_aero_input[i] = aero_input[0]
            elif i == self.n_panels:
                print(f"last section index: {section_index}")
                new_LE[i] = LE[-1]
                new_TE[i] = TE[-1]
                new_aero_input[i] = aero_input[-1]
            else:
                segment_start_length = cum_length[section_index]
                segment_end_length = cum_length[section_index + 1]
                t = (target_length - segment_start_length) / (
                    segment_end_length - segment_start_length
                )

                new_LE[i] = LE[section_index] + t * (
                    LE[section_index + 1] - LE[section_index]
                )
                new_TE[i] = TE[section_index] + t * (
                    TE[section_index + 1] - TE[section_index]
                )

            print(f"new_LE[i]: {new_LE[i]}, new_TE[i]: {new_TE[i]}")

            if aero_input[section_index][0] != aero_input[section_index + 1][0]:
                raise NotImplementedError(
                    "Different aero models over the span are not supported"
                )

            if aero_input[section_index][0] == "inviscid":
                new_aero_input[i] = ["inviscid"]
            elif aero_input[section_index][0] == "polars":
                raise NotImplementedError("Polar interpolation not implemented")
            elif aero_input[section_index][0] == "lei_airfoil_breukels":
                raise NotImplementedError("Geometric interpolation not implemented")

            new_sections.append(Section(new_LE[i], new_TE[i], new_aero_input[i]))

        return new_sections

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

    # Ideas on what the other aero_input could be populated with:
    # ['polars', [CL_alpha, CD_alpha, CM_alpha]]
    # ['lei_airfoil_breukels', [tube_diameter, chamber_height]]

    # CL_alpha: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    # CD_alpha: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    # CM_alpha: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
