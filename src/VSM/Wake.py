import numpy as np
from VSM.Filament import SemiInfiniteFilament

# wake adds horshoe filaments to the horshoe class
# based on a WingAero input


# class Wake:
#     def __init__(self, va_distribution, panels, TE_point_1, TE_point_2):

#         for i, panel in enumerate(panels):
#             va_i = va_distribution[i]
#             # adding horshoe filament behind TE_poin_1
#             panel.horshoe_vortex.update_filaments_for_wake(TE_point_1, va_i)
#             # adding horshoe filament behind TE_point_2
#             panel.horshoe_vortex.update_filaments_for_wake(TE_point_2, va_i)


def frozen_wake(
    va_distribution,
    panels,
):
    for i, panel in enumerate(panels):
        va_i = va_distribution[i]
        vel_mag = np.linalg.norm(va_i)
        direction = va_i / np.linalg.norm(va_i)
        panel.filaments.append(
            SemiInfiniteFilament(
                panel.TE_point_1, direction, vel_mag, filament_direction=-1
            )
        )
        panel.filaments.append(
            SemiInfiniteFilament(
                panel.TE_point_2, direction, vel_mag, filament_direction=1
            )
        )
    return panels


def remove_frozen_wake(panels):
    """Remove the wake filaments from the panels

    Arguments:
        panels {List[Panel]} -- List of Panel Objects

    Returns:
        List[Panel] -- List of Panel Objects without the wake filaments
    """
    panels_without_wake = []
    # looping through each panel
    for i, panel in enumerate(panels):
        filament_without_wake = []
        # looping through each filament in the panel
        for filament in panel.filaments:
            # only if the filament is not a SemiInfiniteFilament, it may stay
            if not isinstance(filament, SemiInfiniteFilament):
                filament_without_wake.append(filament)
        # Updating the panel definition
        panel.filaments = filament_without_wake
        panels_without_wake.append(panel)
    return panels_without_wake
