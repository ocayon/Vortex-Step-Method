"""Bridle segments are charged at their OWN station.

``compute_results`` used to hand every bridle line ``va_ref_vector``. That is
built from ``self._va`` -- the inflow as handed to the ``va`` setter, BEFORE
the rotational term is added -- so for the usual uniform case it is exactly the
freestream, and the bridle carried no ``-omega x (r - r0)`` at all. Every
segment was charged the inflow at the reference point no matter where it sat,
which on a rotating body is the wrong dynamic pressure in the force AND in its
moment about that point. With ``omega = 0`` the two coincide, which is why only
turning cases move.
"""

import numpy as np
import pytest

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.core.WingGeometry import Wing


def _body(with_bridle=True):
    alpha_rad = np.deg2rad(np.arange(-10.0, 31.0, 1.0))
    polar_data = np.column_stack(
        (
            alpha_rad,
            2 * np.pi * alpha_rad,
            np.full_like(alpha_rad, 0.01),
            np.zeros_like(alpha_rad),
        )
    )
    n_panels, span = 12, 8.0
    wing = Wing(n_panels=n_panels, spanwise_panel_distribution="uniform")
    for y in np.linspace(-span / 2, span / 2, n_panels + 1):
        wing.add_section(
            np.array([0.0, y, 6.0]), np.array([1.2, y, 6.0]), polar_data
        )
    # A bridle fanning from the origin (the confluence point) up to the wing,
    # i.e. spread along the rotation axis, where the wing's own inflow is a
    # poor proxy for any individual segment's.
    bridle = None
    if with_bridle:
        bridle = [
            [np.array([0.0, 0.0, 0.0]), np.array([0.0, y, 6.0]), 0.004]
            for y in (-3.0, -1.0, 1.0, 3.0)
        ]
    return BodyAerodynamics([wing], bridle_line_system=bridle)


def _solved(omega, reference_point=np.zeros(3)):
    body = _body()
    magnitude = float(np.linalg.norm(omega))
    body.va_initialize(
        Umag=15.0,
        angle_of_attack=6.0,
        side_slip=0.0,
        body_rates=magnitude,
        body_axis=(omega / magnitude) if magnitude > 0 else np.array([0.0, 0.0, 1.0]),
        reference_point=reference_point,
        rates_in_body_frame=True,
    )
    return body, Solver(reference_point=reference_point).solve(body)


def test_reference_point_is_stored_for_consumers():
    """Needed to evaluate the inflow anywhere else; used to live on the stack."""
    body = _body()
    assert np.allclose(body.reference_point, 0.0)
    body.va_initialize(Umag=10.0, angle_of_attack=4.0, reference_point=[0.0, 1.0, 2.0])
    assert np.allclose(body.reference_point, [0.0, 1.0, 2.0])


def test_reference_point_is_stored_even_without_body_rates():
    """It is not a property of the rotating branch, so it is set unconditionally."""
    body = _body()
    body.va_initialize(Umag=10.0, angle_of_attack=4.0, reference_point=[3.0, 0.0, 0.0])
    assert np.allclose(body.body_rates, 0.0)
    assert np.allclose(body.reference_point, [3.0, 0.0, 0.0])


def test_published_forces_are_one_row_per_segment():
    body, results = _solved(np.array([0.0, 0.1, -0.6]))
    forces = np.asarray(results["bridle_line_forces"], dtype=float)
    midpoints = np.asarray(results["bridle_line_midpoints"], dtype=float)
    assert forces.shape == (len(body._bridle_line_system), 3)
    assert midpoints.shape == forces.shape
    assert np.all(np.isfinite(forces))
    for line, midpoint in zip(body._bridle_line_system, midpoints):
        assert np.allclose(midpoint, 0.5 * (line[0] + line[1]))


def test_each_segment_is_charged_at_its_own_midpoint_inflow():
    omega = np.array([0.0, 0.1, -0.6])
    body, results = _solved(omega)
    va_free = np.asarray(body.va, dtype=float)
    expected = [
        body.compute_line_aerodynamic_force(
            va_free - np.cross(omega, 0.5 * (line[0] + line[1]) - body.reference_point),
            line,
            rho=1.225,
        )
        for line in body._bridle_line_system
    ]
    assert np.allclose(results["bridle_line_forces"], expected)


def test_va_ref_vector_is_the_freestream_not_the_panel_mean():
    """The premise of the fix, pinned so it cannot be misread again.

    ``va_ref_vector`` is computed from ``self._va``, which never has the
    rotational term subtracted into it -- it is NOT the area-weighted mean of
    the PANEL inflows, which does carry ``-omega x (r - r0)``.
    """
    body, _ = _solved(np.array([0.0, 0.1, -0.6]))
    areas = np.array([p.chord * p.width for p in body.panels], dtype=float)
    as_coded = body._compute_reference_velocity_from_distribution(
        body._va, len(body.panels), areas
    )
    panel_mean = body._compute_reference_velocity_from_distribution(
        np.array([p.va for p in body.panels], dtype=float), len(body.panels), areas
    )
    assert np.allclose(as_coded, np.asarray(body.va, dtype=float))
    assert not np.allclose(as_coded, panel_mean, rtol=1e-3)


def test_the_old_single_freestream_would_give_a_different_answer():
    """Guards the regression rather than just asserting the new formula."""
    omega = np.array([0.0, 0.1, -0.6])
    body, results = _solved(omega)
    va_free = np.asarray(body.va, dtype=float)
    old = np.array(
        [
            body.compute_line_aerodynamic_force(va_free, line, rho=1.225)
            for line in body._bridle_line_system
        ]
    )
    assert not np.allclose(results["bridle_line_forces"], old, rtol=1e-3)


def test_without_body_rates_the_two_conventions_coincide():
    body, results = _solved(np.zeros(3))
    va_free = np.asarray(body.va, dtype=float)
    same = [
        body.compute_line_aerodynamic_force(va_free, line, rho=1.225)
        for line in body._bridle_line_system
    ]
    assert np.allclose(results["bridle_line_forces"], same)


def test_a_body_without_bridles_publishes_empty_arrays():
    body = _body(with_bridle=False)
    body.va_initialize(Umag=15.0, angle_of_attack=6.0)
    results = Solver().solve(body)
    assert len(np.asarray(results["bridle_line_forces"])) == 0


def test_density_reaches_the_bridle_law():
    """The call used to drop ``rho`` and silently take the 1.225 default."""
    omega = np.array([0.0, 0.1, -0.6])
    body = _body()
    body.va_initialize(
        Umag=15.0,
        angle_of_attack=6.0,
        body_rates=float(np.linalg.norm(omega)),
        body_axis=omega / np.linalg.norm(omega),
        reference_point=np.zeros(3),
        rates_in_body_frame=True,
    )
    heavy = Solver(rho=2.450).solve(body)
    body.va_initialize(
        Umag=15.0,
        angle_of_attack=6.0,
        body_rates=float(np.linalg.norm(omega)),
        body_axis=omega / np.linalg.norm(omega),
        reference_point=np.zeros(3),
        rates_in_body_frame=True,
    )
    light = Solver(rho=1.225).solve(body)
    assert np.allclose(
        np.asarray(heavy["bridle_line_forces"]),
        2.0 * np.asarray(light["bridle_line_forces"]),
        rtol=1e-9,
    )
