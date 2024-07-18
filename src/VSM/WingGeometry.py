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
        """Refine the aerodynamic mesh of the wing

            Based on user input sections and desired spanwise panel distribution
            refines the aerodynamic mesh of the wing, giving out new_sections
            that can be used to make panels out of.

        Args:
            None

        Returns:
            new_sections (list): List of Section objects with refined aerodynamic mesh
        """

        # Ensure we get 1 section more than the desired number of panels
        n_sections = self.n_panels + 1
        logging.debug(f"n_panels: {self.n_panels}")
        logging.debug(f"n_sections: {n_sections}")

        ### Defining variables extracting from the object
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

        # Handling wrong input
        if len(LE) != len(TE) or len(LE) != len(aero_input):
            raise ValueError("LE, TE, and aero_input must have the same length")

        # If "unchanged" OR the provided section is equal to number of desired
        if (self.spanwise_panel_distribution == "unchanged") or (
            len(self.sections) == n_sections
        ):
            return self.sections

        # If only two sections are DESIRED, return them directly
        if n_sections == 2:
            new_sections = [
                Section(LE[0], TE[0], aero_input[0]),
                Section(LE[-1], TE[-1], aero_input[-1]),
            ]
            return new_sections

        # spacing based on the provided splits
        if self.spanwise_panel_distribution == "split_provided":
            return self.refine_mesh_by_splitting_provided_sections()

        # Linear or cosine spacing
        if (
            self.spanwise_panel_distribution == "linear"
            or "cosine"
            or "cosine_van_Garrel"
        ):
            return self.refine_mesh_for_linear_cosine_distribution(
                self.spanwise_panel_distribution, n_sections, LE, TE, aero_input
            )

    def refine_mesh_for_linear_cosine_distribution(
        self, spanwise_panel_distribution, n_sections, LE, TE, aero_input
    ):

        # 1. Compute the 1/4 chord line
        quarter_chord = LE + 0.25 * (TE - LE)

        # Calculate the length of each segment for the quarter chord line
        qc_lengths = np.linalg.norm(quarter_chord[1:] - quarter_chord[:-1], axis=1)
        qc_total_length = np.sum(qc_lengths)

        # Make cumulative array from 0 to the total length
        qc_cum_length = np.concatenate(([0], np.cumsum(qc_lengths)))

        # 2. Define target lengths based on desired spacing
        if spanwise_panel_distribution == "linear":
            target_lengths = np.linspace(0, qc_total_length, n_sections)
        elif spanwise_panel_distribution == "cosine" or "cosine_van_Garrel":
            theta = np.linspace(0, np.pi, n_sections)
            target_lengths = qc_total_length * (1 - np.cos(theta)) / 2
        else:
            raise ValueError("Unsupported spanwise panel distribution")

        new_quarter_chord = np.zeros((n_sections, 3))
        new_LE = np.zeros((n_sections, 3))
        new_TE = np.zeros((n_sections, 3))
        new_aero_input = np.empty((n_sections,), dtype=object)
        new_sections = []

        # 3. Calculate new quarter chord points and interpolate aero inputs
        for i in range(n_sections):
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
            new_aero_input[i] = self.calculate_new_aero_input(
                aero_input, section_index, left_weight, right_weight
            )

            new_sections.append(Section(new_LE[i], new_TE[i], new_aero_input[i]))

            if self.spanwise_panel_distribution == "cosine_van_Garrel":
                new_sections = self.calculate_cosine_van_Garrel(new_sections)

        return new_sections

    def calculate_new_aero_input(
        self, aero_input, section_index, left_weight, right_weight
    ):
        """Interpolates the aero_input of two sections

        Args:
            aero_input (list): List of aero_input for each section
            section_index (int): Index of the current LEFT section,
                                 assuming that next RIGHT has section_index+1
            left_weight (float): Weight of the left section
            right_weight (float): Weight of the right section

        Returns:
        """
        if aero_input[section_index][0] != aero_input[section_index + 1][0]:
            raise NotImplementedError(
                "Different aero models over the span are not supported"
            )
        if aero_input[section_index][0] == "inviscid":
            return ["inviscid"]
        elif aero_input[section_index][0] == "polar_data":
            polar_left = aero_input[section_index][1]
            polar_right = aero_input[section_index + 1][1]
            if polar_left == polar_right:
                return ["polar_data", polar_left]
            # TODO: Implement polar interpolation
            else:
                raise NotImplementedError("Polar interpolation not implemented")
        elif aero_input[section_index][0] == "lei_airfoil_breukels":
            tube_diameter_left = aero_input[section_index][1][0]
            tube_diameter_right = aero_input[section_index + 1][1][0]
            tube_diameter_i = np.array(
                tube_diameter_left * left_weight + tube_diameter_right * right_weight
            )

            chamber_height_left = aero_input[section_index][1][1]
            chamber_height_right = aero_input[section_index + 1][1][1]
            chamber_height_i = np.array(
                chamber_height_left * left_weight + chamber_height_right * right_weight
            )
            logging.debug(f"left_weight: {left_weight}")
            logging.debug(f"right_weight: {right_weight}")
            logging.debug(f"tube_diameter_i: {tube_diameter_i}")
            logging.debug(f"chamber_height_i: {chamber_height_i}")
            return [
                "lei_airfoil_breukels",
                np.array([tube_diameter_i, chamber_height_i]),
            ]

    def refine_mesh_by_splitting_provided_sections(self):
        n_provided_sections = len(self.sections)
        n_new_sections = self.n_panels + 1 - n_provided_sections
        n_section_pairs = n_provided_sections - 1
        new_sections_per_pair, remaining = divmod(n_new_sections, n_section_pairs)
        n_new_panels = n_section_pairs * (new_sections_per_pair + 1)
        indices_provided, indices_new = [], []
        current_index = 0
        for i in range(n_section_pairs):
            indices_provided.append(current_index)
            current_index += 1
            new_sections = new_sections_per_pair + (i < remaining)
            indices_new.extend(range(current_index, current_index + new_sections))
            current_index += new_sections
        indices_provided.append(current_index)

        logging.debug(f"n_provided_sections: {n_provided_sections}")
        logging.debug(f"n_new_sections: {n_new_sections}")
        logging.debug(f"n_section_pairs: {n_section_pairs}")
        logging.debug(f"new_sections_per_pair: {new_sections_per_pair}")
        logging.debug(f"indices_of_provided_sections: {indices_provided}")
        logging.debug(f"indices_of_new_sections: {indices_new}")

        # Keep this
        if n_new_panels != self.n_panels:
            logging.info(
                f" Number of new panels {n_new_panels} is NOT equal to the {self.n_panels} desired number of panels "
                f" This because of the even split of the provided {n_section_pairs} section pairs"
                f" Each getting {new_sections_per_pair} new sections, leading to {new_sections_per_pair+1} panels for each provided section."
            )

        ### Extracting the provided LE,TE, and aero_inputs
        LE, TE, aero_input = [], [], []
        for _, section in enumerate(self.sections):
            LE.append(np.array(section.LE_point))
            TE.append(np.array(section.TE_point))
            aero_input.append(section.aero_input)

        is_iterating_in_between_provided_sections = True
        left_section_index = 0
        new_sections = []
        # looping over all the new sections
        for i in range(0, n_new_panels + 1):
            logging.debug(
                f"---- i: {i}, is_iterating_in_between_provided_sections: {is_iterating_in_between_provided_sections}"
            )
            # if a provided section
            if i in indices_provided:
                # appending the first provided section, as the first new section
                new_sections.append(self.sections[left_section_index])
                # setting the boolean to true, to indicate that we are iterating in between provided sections
                is_iterating_in_between_provided_sections = True
                logging.debug(
                    f"i in indices_provided: {i}, left_section_index: {left_section_index}"
                )

            # if i is not a provided section, but a split section
            elif i in indices_new and is_iterating_in_between_provided_sections:
                logging.debug(
                    f"i in indices_new: {i}, left_section_index: {left_section_index}"
                )
                ### Defining a new section list to feed
                LE_pair_list = np.array(
                    [LE[left_section_index], LE[left_section_index + 1]]
                )
                TE_pair_list = np.array(
                    [TE[left_section_index], TE[left_section_index + 1]]
                )
                aero_input_pair_list = np.array(
                    [
                        aero_input[left_section_index],
                        aero_input[left_section_index + 1],
                    ]
                )
                new_splitted_sections = self.refine_mesh_for_linear_cosine_distribution(
                    "linear",
                    new_sections_per_pair + 2,
                    LE_pair_list,
                    TE_pair_list,
                    aero_input_pair_list,
                )
                # loop over the new sections, except the first and last one, those are provided and already appended
                for i in range(1, len(new_splitted_sections) - 1):
                    logging.debug(
                        f"i:{i}, appending new_section: {new_splitted_sections[i]}"
                    )
                    new_sections.append(new_splitted_sections[i])

                # setting the boolean to false, untill the next provided section is hit
                is_iterating_in_between_provided_sections = False
                left_section_index += 1

        return new_sections

    def calculate_cosine_van_Garrel(self, new_sections):
        """Calculate the van Garrel cosine distribution of sections
        URL: http://dx.doi.org/10.13140/RG.2.1.2773.8000

        Args:
            new_sections (list): List of Section objects

        Returns:
            new_sections_van_Garrel (list): List of Section objects with van Garrel cosine distribution
        """
        n = len(new_sections)
        control_points = np.zeros((n, 3))

        # Calculate chords and quarter chords
        chords = []
        quarter_chords = []
        for section in new_sections:
            chord = section.TE_point - section.LE_point
            chords.append(chord)
            quarter_chords.append(section.LE_point + 0.25 * chord)

        # Calculate widths
        widths = np.zeros(n - 1)
        for i in range(n - 1):
            widths[i] = np.linalg.norm(quarter_chords[i + 1] - quarter_chords[i])

        # Calculate correction eta_cp
        eta_cp = np.zeros(n - 1)

        # First panel
        eta_cp[0] = widths[0] / (widths[0] + widths[1])

        # Internal panels
        for j in range(1, n - 2):
            eta_cp[j] = 0.25 * (
                widths[j - 1] / (widths[j - 1] + widths[j])
                + widths[j] / (widths[j] + widths[j + 1])
                + 1
            )
            control_points[j] = quarter_chords[j] + eta_cp[j] * (
                quarter_chords[j + 1] - quarter_chords[j]
            )
        # Last panel
        eta_cp[-1] = widths[-2] / (widths[-2] + widths[-1])

        logging.debug(f"eta_cp: {eta_cp}")

        # Calculate control points
        control_points = []
        for i, eta_cp_i in enumerate(eta_cp):
            control_points.append(
                quarter_chords[i]
                + eta_cp_i * (quarter_chords[i + 1] - quarter_chords[i])
            )

        # Calculate new_sections_van_Garrel
        new_sections_van_Garrel = []

        for i, control_point_i in enumerate(control_points):
            # Use the original chord length
            chord = chords[i]
            new_LE_point = control_point_i - 0.25 * chord
            new_TE_point = control_point_i + 0.75 * chord

            # Keep the original aero_input
            aero_input_i = new_sections[i].aero_input

            new_sections_van_Garrel.append(
                Section(new_LE_point, new_TE_point, aero_input_i)
            )

        return new_sections_van_Garrel

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

    def calculate_projected_area(self, z_plane_vector=np.array([0, 0, 1])):
        """Calculates the projected area of the wing onto a specified plane.

        The projected area is calculated based on the leading and trailing edge points of each section
        projected onto a plane defined by a normal vector (default is z-plane).

        Args:
            z_plane_vector (np.ndarray): Normal vector defining the projection plane (default is [0, 0, 1]).

        Returns:
            projected_area (float): The projected area of the wing.
        """
        # Normalize the z_plane_vector
        z_plane_vector = z_plane_vector / np.linalg.norm(z_plane_vector)

        # Helper function to project a point onto the plane
        def project_onto_plane(point, normal):
            return point - np.dot(point, normal) * normal

        projected_area = 0.0
        for i in range(len(self.sections) - 1):
            # Get the points for the current and next section
            LE_current = self.sections[i].LE_point
            TE_current = self.sections[i].TE_point
            LE_next = self.sections[i + 1].LE_point
            TE_next = self.sections[i + 1].TE_point

            # Project the points onto the plane
            LE_current_proj = project_onto_plane(LE_current, z_plane_vector)
            TE_current_proj = project_onto_plane(TE_current, z_plane_vector)
            LE_next_proj = project_onto_plane(LE_next, z_plane_vector)
            TE_next_proj = project_onto_plane(TE_next, z_plane_vector)

            # Calculate the lengths of the projected edges
            chord_current_proj = np.linalg.norm(TE_current_proj - LE_current_proj)
            chord_next_proj = np.linalg.norm(TE_next_proj - LE_next_proj)

            # Calculate the spanwise distance between the projected sections
            spanwise_distance_proj = np.linalg.norm(LE_next_proj - LE_current_proj)

            # Calculate the projected area of the trapezoid formed by these points
            area = 0.5 * (chord_current_proj + chord_next_proj) * spanwise_distance_proj
            projected_area += area

        return projected_area


# TODO: remove this, no need for inherance here, can be just a function :)
@dataclass
class Section:
    LE_point: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    TE_point: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    aero_input: list = field(default_factory=list)


def flip_created_coord_in_pairs_if_needed(coord):
    """
    Ensure the coordinates are ordered from positive to negative along the y-axis.
    """
    # Reshape the array into pairs
    reshaped = coord.reshape(-1, 2, coord.shape[1])

    # Check the overall y-axis order
    overall_y = reshaped[:, 0, 1]  # Take the y values of the leading edge coordinates
    if not np.all(
        overall_y[:-1] >= overall_y[1:]
    ):  # Check if y values are in descending order
        reshaped = np.flip(reshaped, axis=0)

    # Flatten back to the original shape
    return reshaped.reshape(-1, coord.shape[1])
