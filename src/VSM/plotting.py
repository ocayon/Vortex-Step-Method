import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from VSM.color_palette import set_plot_style, get_color


def plot_line_segment(ax, segment, color, label, width: float = 3):
    ax.plot(
        [segment[0][0], segment[1][0]],
        [segment[0][1], segment[1][1]],
        [segment[0][2], segment[1][2]],
        color=color,
        label=label,
        linewidth=width,
    )
    dir = segment[1] - segment[0]
    ax.quiver(
        segment[0][0],
        segment[0][1],
        segment[0][2],
        dir[0],
        dir[1],
        dir[2],
        color=color,
    )


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])


def plot_geometry(
    wing_aero,
    title="wing_geometry",
    data_type=".pdf",
    save_path="./",
    is_save=False,
    is_show=True,
    view_elevation=30,
    view_azimuth=30,
):
    """
    Plots the wing panels and filaments in 3D.

    """

    # Set the plot style
    set_plot_style()

    # defining variables
    panels = wing_aero.panels
    va = wing_aero.va

    # Extract corner points, control points, and aerodynamic centers from panels
    corner_points = np.array([panel.corner_points for panel in panels])
    control_points = np.array([panel.control_point for panel in panels])
    aerodynamic_centers = np.array([panel.aerodynamic_center for panel in panels])

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot each panel
    for i, panel in enumerate(panels):
        # Get the corner points of the current panel and close the loop by adding the first point again
        x_corners = np.append(corner_points[i, :, 0], corner_points[i, 0, 0])
        y_corners = np.append(corner_points[i, :, 1], corner_points[i, 0, 1])
        z_corners = np.append(corner_points[i, :, 2], corner_points[i, 0, 2])

        # Plot the panel edges
        ax.plot(
            x_corners,
            y_corners,
            z_corners,
            color="grey",
            label="Panel Edges" if i == 0 else "",
            linewidth=1,
        )

        # Create a list of tuples representing the vertices of the polygon
        verts = [list(zip(x_corners, y_corners, z_corners))]
        poly = Poly3DCollection(verts, color="grey", alpha=0.1)
        ax.add_collection3d(poly)

        # Plot the control point
        ax.scatter(
            control_points[i, 0],
            control_points[i, 1],
            control_points[i, 2],
            color="green",
            label="Control Points" if i == 0 else "",
        )

        # Plot the aerodynamic center
        ax.scatter(
            aerodynamic_centers[i, 0],
            aerodynamic_centers[i, 1],
            aerodynamic_centers[i, 2],
            color="b",
            label="Aerodynamic Centers" if i == 0 else "",
        )

        # Plot the filaments
        filaments = panel.calculate_filaments_for_plotting()
        legends = ["Bound Vortex", "side1", "side2", "wake_1", "wake_2"]

        for filament, legend in zip(filaments, legends):
            x1, x2, color = filament
            logging.debug("Legend: %s", legend)
            plot_line_segment(ax, [x1, x2], color, legend)

    # Plot the va_vector using the plot_segment
    max_chord = np.max([panel.chord for panel in panels])
    va_vector_begin = -2 * max_chord * va / np.linalg.norm(va)
    va_vector_end = va_vector_begin + 1.5 * va / np.linalg.norm(va)
    plot_line_segment(ax, [va_vector_begin, va_vector_end], "lightblue", "va")

    # Add legends for the first occurrence of each label
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Add axis labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Set equal axis limits
    set_axes_equal(ax)

    # Flip the z-axis (to stick to body reference frame)
    # ax.invert_zaxis()

    # Set the initial view
    ax.view_init(elev=view_elevation, azim=view_azimuth)

    # Display the plot
    if is_show:
        plt.show()

    # Save the plot
    if is_save:
        plt.savefig(Path(save_path) / (title + data_type))
        plt.close()


def plot_distribution(
    results_list,
    label_list,
    title="spanwise_distribution",
    data_type=".pdf",
    save_path="./",
    is_save=True,
    is_show=True,
):
    if len(results_list) != len(label_list):
        raise ValueError(
            "The number of results and labels should be the same. Got {} results and {} labels".format(
                len(results_list), len(label_list)
            )
        )

    # Set the plot style
    set_plot_style()

    # Initializing plot
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Spanwise distributions", fontsize=16)

    # CL plot
    for result_i, label_i in zip(results_list, label_list):
        axs[0, 0].plot(result_i["cl_distribution"], label=label_i)
    axs[0, 0].set_title(r"$C_L$ Distribution")
    axs[0, 0].set_xlabel(r"Spanwise Position $y/b$")
    axs[0, 0].set_ylabel(r"Lift Coefficient $C_L$")
    axs[0, 0].legend()

    # CD plot
    for result_i, label_i in zip(results_list, label_list):
        axs[0, 1].plot(result_i["cd_distribution"], label=label_i)
    axs[0, 1].set_title(r"$C_D$ Distribution")
    axs[0, 1].set_xlabel(r"Spanwise Position $y/b$")
    axs[0, 1].set_ylabel(r"Drag Coefficient $C_D$")
    axs[0, 1].legend()

    # Gamma plot
    for result_i, label_i in zip(results_list, label_list):
        axs[1, 0].plot(result_i["gamma_distribution"], label=label_i)
    axs[1, 0].set_title(r"$\Gamma$ Distribution")
    axs[1, 0].set_xlabel(r"Spanwise Position $y/b$")
    axs[1, 0].set_ylabel(r"Circulation $\Gamma$")
    axs[1, 0].legend()

    # Calculated Alpha plot
    for result_i, label_i in zip(results_list, label_list):
        axs[1, 1].plot(result_i["alpha_at_ac"], label=label_i)
    axs[1, 1].set_title(r"$\alpha$ Comparison (from VSM)")
    axs[1, 1].set_xlabel(r"Spanwise Position $y/b$")
    axs[1, 1].set_ylabel(r"Angle of Attack $\alpha$ (rad)")
    axs[1, 1].legend()

    # Total Force Plot
    for result_i, label_i in zip(results_list, label_list):
        axs[2, 0].plot(result_i["Ftotal_distribution"], label=label_i)
    axs[2, 0].set_title(r"Total Force Distribution")
    axs[2, 0].set_xlabel(r"Spanwise Position $y/b$")
    axs[2, 0].set_ylabel(r"Total Force (N)")
    axs[2, 0].legend()

    # Geometric Alpha plot
    for result_i, label_i in zip(results_list, label_list):
        axs[2, 1].plot(result_i["alpha_geometric"], label=label_i)
    axs[2, 1].set_title(r"$\alpha$ Geometric")
    axs[2, 1].set_xlabel(r"Spanwise Position $y/b$")
    axs[2, 1].set_ylabel(r"Angle of Attack $\alpha$ (rad)")
    axs[2, 1].legend()

    plt.tight_layout()

    if is_show:
        plt.show()

    if is_save:
        plt.savefig(Path(save_path) / (title + data_type))
        plt.close()
