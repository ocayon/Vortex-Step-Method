# Test a horseshoe vortex induced velocities
import pytest
from VSM.HorshoeVortex import HorshoeVortex
from VSM.HorshoeVortex import BoundFilament

def test_bound_filament():
    x1 = [0, 0, 0]
    x2 = [1, 0, 0]
    filament = BoundFilament(x1, x2)
    point = [0.5, 0.5, 0]
    gamma = 1.0

    induced_velocity = filament.calculate_induced_velocity(point, gamma)
    print("Induced velocity at point {}: {}".format(point, induced_velocity))