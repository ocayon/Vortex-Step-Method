"""The numba-compiled AIC assembly (utils.assemble_AIC_matrices, used by
BodyAerodynamics.compute_AIC_matrices) must reproduce the pure-Python
per-panel reference loop (_compute_AIC_matrices_reference) to floating-point
round-off, for both VSM and LLT evaluation-point conventions, on a
non-planar swept/tapered geometry."""

import os
import sys

import numpy as np
import pytest

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, "src"))

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.core.WingGeometry import Wing


def _curved_swept_body(n_panels=12):
    """Non-planar wing: dihedral arc, sweep, and taper, so the packed filament
    geometry has no accidental symmetry or zero components."""
    alpha_rad = np.deg2rad(np.arange(-10.0, 31.0, 1.0))
    polar_data = np.column_stack(
        (
            alpha_rad,
            2 * np.pi * alpha_rad,
            np.full_like(alpha_rad, 0.01),
            np.zeros_like(alpha_rad),
        )
    )
    wing = Wing(n_panels=n_panels, spanwise_panel_distribution="uniform")
    span = 8.0
    for y in np.linspace(-span / 2, span / 2, n_panels + 1):
        eta = 2 * y / span
        chord = 1.5 - 0.6 * abs(eta)  # taper
        x_le = 0.4 * abs(eta)  # sweep
        z = 1.2 * (1 - np.sqrt(max(0.0, 1 - eta**2)))  # dihedral arc
        wing.add_section(
            np.array([x_le, y, z]),
            np.array([x_le + chord, y, z]),
            polar_data,
        )
    return BodyAerodynamics([wing])


@pytest.mark.parametrize("model_type", ["VSM", "LLT"])
def test_jit_AIC_matches_python_reference(model_type):
    body_aero = _curved_swept_body()
    body_aero.va_initialize(Umag=12.0, angle_of_attack=8.0, side_slip=4.0)

    solver = Solver(aerodynamic_model_type=model_type)
    n = body_aero.n_panels
    va_array = np.array([panel.va for panel in body_aero.panels])
    va_norm_array = np.linalg.norm(va_array, axis=1)
    va_unit_array = va_array / va_norm_array[:, None]

    fast = body_aero.compute_AIC_matrices(
        model_type, solver.core_radius_fraction, va_norm_array, va_unit_array
    )
    reference = body_aero._compute_AIC_matrices_reference(
        model_type, solver.core_radius_fraction, va_norm_array, va_unit_array
    )

    for fast_component, ref_component in zip(fast, reference):
        assert fast_component.shape == (n, n)
        np.testing.assert_allclose(
            fast_component, ref_component, rtol=1e-12, atol=1e-15
        )


def test_jit_AIC_end_to_end_solution_unchanged():
    """A full solve through the jit AIC path matches one forced through the
    Python reference path."""
    body_aero = _curved_swept_body()

    body_aero.va_initialize(Umag=12.0, angle_of_attack=8.0, side_slip=0.0)
    res_fast = Solver().solve(body_aero)

    body_aero.va_initialize(Umag=12.0, angle_of_attack=8.0, side_slip=0.0)
    solver_ref = Solver()
    body_aero.compute_AIC_matrices = body_aero._compute_AIC_matrices_reference
    res_ref = solver_ref.solve(body_aero)

    np.testing.assert_allclose(
        res_fast["gamma_distribution"],
        res_ref["gamma_distribution"],
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.isclose(res_fast["cl"], res_ref["cl"], rtol=1e-10)
    assert np.isclose(res_fast["cd"], res_ref["cd"], rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
