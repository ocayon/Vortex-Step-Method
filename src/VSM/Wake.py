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
        dir = va_i / np.linalg.norm(va_i)
        panel.filaments.append(SemiInfiniteFilament(panel.TE_point_1, dir,filament_direction=-1))
        panel.filaments.append(SemiInfiniteFilament(panel.TE_point_2, dir,filament_direction=1))
        print("Wake filaments updated")
    return panels
