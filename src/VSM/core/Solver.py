import numpy as np
import logging
from scipy.linalg import solve_banded
from . import jit_cross


class Solver:
    """Solver for aerodynamic circulation distribution and force computation.

    Implements iterative algorithms to determine circulation distribution that satisfies
    boundary conditions for VSM and LLT aerodynamic models.

    Attributes:
        aerodynamic_model_type (str): Aerodynamic model type ('VSM' or 'LLT').
        max_iterations (int): Maximum number of iterations for convergence.
        allowed_error (float): Convergence tolerance for normalized error.
        relaxation_factor (float): Under-relaxation factor for stability.
        core_radius_fraction (float): Vortex core radius fraction.
        gamma_loop_type (str): Iterative algorithm type.
        gamma_initial_distribution_type (str): Initial circulation distribution method.
        is_only_f_and_gamma_output (bool): Return only forces and circulation if True.
        is_with_viscous_drag_correction (bool): Enable viscous drag correction.
        reference_point (np.ndarray): Reference point for moment calculations.
        mu (float): Dynamic viscosity of fluid.
        rho (float): Fluid density.
        is_smooth_circulation (bool): Apply circulation smoothing.
        smoothness_factor (float): Smoothing strength parameter.
        is_artificial_damping (bool): Enable artificial damping for stall.
        artificial_damping (dict): Artificial damping parameters.
        is_with_simonet_artificial_viscosity (bool): Enable Simonet artificial viscosity.
        _simonet_artificial_viscosity_fva (float): Simonet model parameter.
        is_with_artificial_viscosity (bool): Enable Li/Gaunaa spanwise artificial
            viscosity (TORQUE 2026) for post-stall stabilization in gamma_loop.
        artificial_viscosity_factor (float): Coefficient k in the viscosity scaling
            (default 0.035, the conservative envelope from the paper).
    """

    def __init__(
        self,
        aerodynamic_model_type: str = "VSM",
        max_iterations: int = 5000,
        allowed_error: float = 1e-6,
        relaxation_factor: float = 0.01,
        core_radius_fraction: float = 0.05,  # Following Damiani et al. (2019) https://docs.nrel.gov/docs/fy19osti/72777.pdf
        gamma_loop_type: str = "base",
        gamma_initial_distribution_type: str = "zero",
        is_only_f_and_gamma_output: bool = False,
        is_with_viscous_drag_correction: bool = False,
        reference_point: np.ndarray | list | tuple | None = None,
        mu: float = 1.81e-5,
        rho: float = 1.225,
        is_smooth_circulation: bool = False,
        smoothness_factor: float = 0.08,
        is_artificial_damping: bool = False,
        artificial_damping: dict = {"k2": 0.1, "k4": 0.0},
        is_with_simonet_artificial_viscosity: bool = False,
        simonet_artificial_viscosity_fva: float = None,
        is_aoa_corrected: bool = False,
        is_with_artificial_viscosity: bool = False,
        artificial_viscosity_factor: float = 0.035,
        anderson_depth: int = 5,
        anderson_beta: float = 1.0,
        anderson_max_iterations: int = 1000,
        anderson_fallback_to_base: bool = False,
        stagnation_patience: int = 0,
        stagnation_rtol: float = 0.05,
    ):
        """Initialize solver with configuration parameters.

        Args:
            aerodynamic_model_type (str): Type of aerodynamic model ('VSM' or 'LLT').
            max_iterations (int): Maximum solver iterations.
            allowed_error (float): Convergence tolerance.
            relaxation_factor (float): Under-relaxation factor.
            core_radius_fraction (float): Vortex core radius fraction.
            gamma_loop_type (str): Iterative algorithm type.
            gamma_initial_distribution_type (str): Initial circulation distribution.
            is_only_f_and_gamma_output (bool): Return minimal output if True.
            is_with_viscous_drag_correction (bool): Enable viscous corrections.
            reference_point (array-like, optional): Reference point for moments.
                Must be shape (3,). Defaults to [0, 0, 0].
            mu (float): Dynamic viscosity.
            rho (float): Fluid density.
            is_smooth_circulation (bool): Apply circulation smoothing.
            smoothness_factor (float): Smoothing factor.
            is_artificial_damping (bool): Enable artificial damping.
            artificial_damping (dict): Damping parameters.
            is_with_simonet_artificial_viscosity (bool): Enable Simonet model.
            simonet_artificial_viscosity_fva (float): Simonet parameter.
        """
        self.aerodynamic_model_type = aerodynamic_model_type
        self.max_iterations = int(max_iterations)
        self.allowed_error = allowed_error
        self.relaxation_factor = relaxation_factor
        self.core_radius_fraction = core_radius_fraction
        self.gamma_loop_type = gamma_loop_type
        self.gamma_initial_distribution_type = gamma_initial_distribution_type
        self.is_only_f_and_gamma_output = is_only_f_and_gamma_output
        self.is_with_viscous_drag_correction = is_with_viscous_drag_correction
        self.reference_point = self._check_and_force_shape(reference_point)
        self.is_aoa_corrected = is_aoa_corrected
        # === athmospheric properties ===
        self.mu = mu
        self.rho = rho
        # ===============================
        #       STALL MODELS
        # ===============================
        # === STALL: smooth_circulation ===
        self.is_smooth_circulation = is_smooth_circulation
        self.smoothness_factor = smoothness_factor
        # === STALL: artificial damping ===
        self.is_artificial_damping = is_artificial_damping
        self.artificial_damping = artificial_damping
        # === STALL: simonet_aritificial_viscosity ===
        self.is_with_simonet_artificial_viscosity = is_with_simonet_artificial_viscosity
        self._simonet_artificial_viscosity_fva = simonet_artificial_viscosity_fva
        # === STALL: Li/Gaunaa spanwise artificial viscosity (TORQUE 2026) ===
        # Parameter-free post-stall regularization; see gamma_loop.
        self.is_with_artificial_viscosity = is_with_artificial_viscosity
        self.artificial_viscosity_factor = artificial_viscosity_factor
        # === Anderson-accelerated fixed-point loop (gamma_loop_type="anderson") ===
        # Depth m = number of past residuals mixed per step; beta = mixing/damping.
        # anderson_max_iterations bounds the accelerated attempt before solve()
        # falls back to the base relaxed-Picard loop (deep post-stall / stall-knee
        # safety net). Healthy Anderson converges in O(10s) of iterations, so a
        # small cap keeps the wasted work minimal on the rare limit-cycling state
        # (e.g. the stall knee) before the base loop takes over.
        self.anderson_depth = int(anderson_depth)
        self.anderson_beta = float(anderson_beta)
        self.anderson_max_iterations = int(anderson_max_iterations)
        # Whether a non-converged Anderson attempt retries with the base
        # relaxed-Picard loop. OFF by default since 2026-09-03, with the
        # iteration headroom raised to 1000 instead: measured on the AWETrim
        # 2019+2025 steering campaigns, the fallback rescued 99 of ~92,400
        # Anderson failures (0.1%) while costing up to two 1500-iteration
        # base loops per failure. Callers that want the old always-fall-back
        # robustness pass True (and may lower anderson_max_iterations).
        self.anderson_fallback_to_base = bool(anderson_fallback_to_base)

        # Give up on a circulation solve that has stopped improving, rather
        # than grinding out ``max_iterations``. OFF by default (patience 0):
        # a plain solve should keep its full budget, since a slow solve and a
        # hopeless one are only distinguishable by PROGRESS, never by an
        # iteration count -- shrinking a cap to bound the hopeless case kills
        # the slow-but-converging one too.
        #
        # The caller that wants this is a two-stage scheme whose first stage is
        # a PREDICTOR it may throw away (AWETrim's attached-branch finder): on a
        # genuinely stalled state that predictor exhausts the cap and is then
        # rejected regardless, so the whole budget is waste. ``patience``
        # iterations with no improvement better than ``rtol`` ends it.
        self.stagnation_patience = int(stagnation_patience)
        self.stagnation_rtol = float(stagnation_rtol)
        #: Diagnostic: did the last circulation solve stop on stagnation?
        self.last_stagnated = False

        ## Initializing some empty properties
        self.panels = None
        self.n_panels = None
        self.x_airf_array = None
        self.y_airf_array = None
        self.z_airf_array = None
        self.va_array = None
        self.chord_array = None
        self.width_array = None
        self.y_coords = None

    @staticmethod
    def _check_and_force_shape(
        reference_point: np.ndarray | list | tuple | None,
    ) -> np.ndarray:
        """Return reference_point as a float array with shape (3,)."""
        rp = (
            np.zeros(3, dtype=float)
            if reference_point is None
            else np.asarray(reference_point, dtype=float)
        )
        if rp.shape != (3,):
            raise ValueError(f"reference_point must be shape (3,), got {rp.shape}")
        return rp

    def solve(self, body_aero, gamma_distribution: np.ndarray = None) -> dict:
        """Solve aerodynamic model for circulation distribution and forces.

        Args:
            body_aero: BodyAerodynamics object with configured geometry and flow conditions.
            gamma_distribution (np.ndarray, optional): Initial circulation guess.

        Returns:
            dict: Comprehensive results dictionary with forces, moments, and distributions.

        Raises:
            ValueError: If inflow conditions are not set.
        """

        if body_aero.va is None:
            raise ValueError("Inflow conditions are not set")

        # Initialize variables here, outside the loop
        self.panels = body_aero.panels
        self.n_panels = body_aero.n_panels
        alpha_array = np.zeros(self.n_panels)
        (
            self.x_airf_array,
            self.y_airf_array,
            self.z_airf_array,
            self.va_array,
            self.chord_array,
            self.width_array,
            self.y_coords,
        ) = (
            np.zeros((self.n_panels, 3)),
            np.zeros((self.n_panels, 3)),
            np.zeros((self.n_panels, 3)),
            np.zeros((self.n_panels, 3)),
            np.zeros(self.n_panels),
            np.zeros(self.n_panels),
            np.zeros(self.n_panels),
        )
        for i, panel in enumerate(self.panels):
            self.x_airf_array[i] = panel.x_airf
            self.y_airf_array[i] = panel.y_airf
            self.z_airf_array[i] = panel.z_airf
            self.va_array[i] = panel.va
            self.chord_array[i] = panel.chord
            self.width_array[i] = panel.width
            self.y_coords[i] = panel.control_point[1]

        va_norm_array = np.linalg.norm(self.va_array, axis=1)
        va_unit_array = self.va_array / va_norm_array[:, None]

        # Calculate the new circulation distribution iteratively
        self.AIC_x, self.AIC_y, self.AIC_z = body_aero.compute_AIC_matrices(
            self.aerodynamic_model_type,
            self.core_radius_fraction,
            va_norm_array,
            va_unit_array,
        )

        if gamma_distribution is not None:
            gamma_initial = np.asarray(gamma_distribution, dtype=float)
            if gamma_initial.shape != (self.n_panels,):
                raise ValueError(
                    "gamma_distribution must match number of panels in solve()."
                )
        elif self.gamma_initial_distribution_type == "previous":
            gamma_initial = np.zeros(self.n_panels)
        elif self.gamma_initial_distribution_type == "elliptical":
            gamma_initial = body_aero.compute_circulation_distribution_elliptical_wing()
        elif self.gamma_initial_distribution_type == "cosine":
            gamma_initial = body_aero.compute_circulation_distribution_cosine()
        elif self.gamma_initial_distribution_type == "zero":
            gamma_initial = np.zeros(self.n_panels)
        else:
            raise ValueError(
                "Invalid gamma_initial_distribution_type, should be either: 'previous', 'elliptical', 'cosine' or 'zero'"
            )

        # === run one of the iterative loops ===
        if self.gamma_loop_type == "base":
            converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
                gamma_initial
            )
            # run again with half the relaxation factor if not converged
            if not converged:
                logging.info(
                    f" ---> Running again with half the relaxation_factor = {self.relaxation_factor / 2}"
                )
                converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
                    gamma_initial, extra_relaxation_factor=0.5
                )

        elif self.gamma_loop_type == "non_linear":
            converged, gamma_new, alpha_array, Umag_array = self.gamma_loop_non_linear(
                gamma_initial
            )

        elif self.gamma_loop_type == "anderson":
            converged, gamma_new, alpha_array, Umag_array = self.gamma_loop_anderson(
                gamma_initial
            )
            # Deep post-stall can trap Anderson in a limit cycle (the regime where
            # VSM is unreliable anyway and only the base loop's viscosity /
            # heavy relaxation converges). Fall back to the base loop — same
            # fixed point, same two-stage half-relaxation retry — so the
            # accelerated path is never less robust than ``base``. Optional
            # (anderson_fallback_to_base): measured rescue rate 0.1%.
            if not converged and self.anderson_fallback_to_base:
                logging.info(
                    " ---> Anderson did not converge; falling back to base "
                    "relaxed-Picard loop"
                )
                converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
                    gamma_initial
                )
                if not converged:
                    converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
                        gamma_initial, extra_relaxation_factor=0.5
                    )

        else:
            # Instiate the stall_solvers class
            import VSM.StallSolvers as StallSolvers

            stall_solvers = StallSolvers.StallSolvers(self)

            if self.gamma_loop_type == "simonet_stall":
                converged, gamma_new, alpha_array, Umag_array = (
                    stall_solvers.gamma_loop_simonet_stall(gamma_initial)
                )
                # run again with half the relaxation factor if not converged
                if not converged:
                    logging.info(
                        f" ---> Running again with half the relaxation_factor = {self.relaxation_factor / 2}"
                    )
                    converged, gamma_new, alpha_array, Umag_array = (
                        stall_solvers.gamma_loop_simonet_stall(
                            gamma_initial, extra_relaxation_factor=0.5
                        )
                    )
            elif self.gamma_loop_type == "non_linear_simonet_stall":
                converged, gamma_new, alpha_array, Umag_array = (
                    stall_solvers.gamma_loop_non_linear_simonet_stall(gamma_initial)
                )
            elif self.gamma_loop_type == "non_linear_simonet_stall_newton_raphson":
                converged, gamma_new, alpha_array, Umag_array = (
                    stall_solvers.gamma_loop_non_linear_simonet_stall_newton_raphson(
                        gamma_initial
                    )
                )
            else:
                raise ValueError(f"Invalid gamma_loop_type")
        # Calculating results (incl. updating angle of attack for VSM)
        results = body_aero.compute_results(
            gamma_new,
            self.rho,
            self.aerodynamic_model_type,
            self.core_radius_fraction,
            self.mu,
            alpha_array,
            Umag_array,
            self.chord_array,
            self.x_airf_array,
            self.y_airf_array,
            self.z_airf_array,
            self.va_array,
            va_norm_array,
            va_unit_array,
            self.panels,
            self.is_only_f_and_gamma_output,
            self.is_with_viscous_drag_correction,
            self.reference_point,
            self.is_aoa_corrected,
        )
        results["gamma_converged"] = bool(converged)
        return results

    def compute_aerodynamic_quantities(self, gamma: np.ndarray) -> tuple:
        """Compute aerodynamic quantities from circulation distribution.

        Args:
            gamma (np.ndarray): Circulation distribution (n x 1).

        Returns:
            tuple: (alpha_array, Umag_array, cl_array, Umagw_array)
                - alpha_array (np.ndarray): Effective angles of attack.
                - Umag_array (np.ndarray): Effective velocity magnitudes.
                - cl_array (np.ndarray): Lift coefficients.
                - Umagw_array (np.ndarray): Reference velocity magnitudes.
        """
        induced_velocity_all = np.array(
            [
                np.matmul(self.AIC_x, gamma),
                np.matmul(self.AIC_y, gamma),
                np.matmul(self.AIC_z, gamma),
            ]
        ).T  # v_ind
        relative_velocity_array = (
            self.va_array + induced_velocity_all
        )  # v_eff = v_inf + v_ind
        relative_velocity_crossz_array = jit_cross(
            relative_velocity_array, self.z_airf_array
        )  # v_eff x z
        Uinfcrossz_array = jit_cross(self.va_array, self.z_airf_array)
        v_normal_array = np.sum(self.x_airf_array * relative_velocity_array, axis=1)
        v_tangential_array = np.sum(self.y_airf_array * relative_velocity_array, axis=1)
        alpha_array = np.arctan2(v_normal_array, v_tangential_array)  # alpha_eff
        Umag_array = np.linalg.norm(
            relative_velocity_crossz_array, axis=1
        )  # |v_eff x z|
        Umagw_array = np.linalg.norm(Uinfcrossz_array, axis=1)
        cl_array = np.array(
            [panel.compute_cl(alpha) for panel, alpha in zip(self.panels, alpha_array)]
        )  # cl(alpha_eff)
        return alpha_array, Umag_array, cl_array, Umagw_array

    def _build_spanwise_laplacian(self) -> np.ndarray:
        """Discrete spanwise Laplacian ``L`` with second-order tip closures.

        Interior rows use the standard three-point stencil
        ``(L gamma)_i = gamma_{i-1} - 2 gamma_i + gamma_{i+1}``. The tip rows use
        the closures of Li, Gaunaa, Pirrung & Lønbæk (TORQUE 2026, Eq. 15),
        derived from a quadratic variation of circulation near the tip, which
        enforce ``gamma -> 0`` at the wing tips to second order:
        ``(L gamma)_0 = -4 gamma_0 + (4/3) gamma_1`` and
        ``(L gamma)_{N-1} = (4/3) gamma_{N-2} - 4 gamma_{N-1}``.

        Panels are assumed to be ordered consecutively along the span (the
        standard VSM panel ordering) and approximately uniformly spaced. The
        per-panel viscosity coefficient carries the ``1/dz_i^2`` spacing factor.
        """
        n = self.n_panels
        laplacian = np.zeros((n, n))
        if n < 3:
            return laplacian
        for i in range(1, n - 1):
            laplacian[i, i - 1] = 1.0
            laplacian[i, i] = -2.0
            laplacian[i, i + 1] = 1.0
        laplacian[0, 0] = -4.0
        laplacian[0, 1] = 4.0 / 3.0
        laplacian[n - 1, n - 1] = -4.0
        laplacian[n - 1, n - 2] = 4.0 / 3.0
        return laplacian

    def _local_lift_slope(
        self, alpha_array: np.ndarray, delta: float = np.deg2rad(0.5)
    ) -> np.ndarray:
        """Local lift-curve slope ``dCl/dalpha`` per panel via central differences.

        Evaluated from each panel's own 2-D polar at the current effective angle
        of attack. The slope is negative in post-stall, which is what activates
        the artificial-viscosity regularization in :meth:`gamma_loop`.

        Readable reference implementation and test oracle; the iteration hot
        path uses the vectorized :meth:`_lift_slope_from_ctx`, which evaluates
        the identical central difference from tables prepared once per solve.
        """
        slopes = np.empty(self.n_panels)
        for i, (panel, alpha) in enumerate(zip(self.panels, alpha_array)):
            cl_plus = panel.compute_cl(alpha + delta)
            cl_minus = panel.compute_cl(alpha - delta)
            slopes[i] = (cl_plus - cl_minus) / (2.0 * delta)
        return slopes

    def _panel_stall_angles(self) -> np.ndarray:
        """Per-panel stall-onset AoA [rad]: the first local Cl maximum in the
        positive-Cl region of each panel polar (``inf`` if the polar shows no
        peak).

        Used only as a cheap, geometry-fixed gate for the post-stall
        artificial-viscosity branch: the regularization is a no-op while every
        panel is below its stall onset, so this lets ``gamma_loop`` skip both the
        lift-slope evaluation and the linear solve in attached conditions.
        """
        angles = np.full(self.n_panels, np.inf)
        for i, panel in enumerate(self.panels):
            polar = np.asarray(panel.panel_polar_data, dtype=float)
            alpha, cl = polar[:, 0], polar[:, 1]
            pos = np.where(cl > 0)[0]
            for k in pos[1:-1]:  # first interior Cl peak in the positive-Cl region
                if cl[k] > cl[k - 1] and cl[k] > cl[k + 1]:
                    angles[i] = float(alpha[k])
                    break
        return angles

    # should add smooth circulation back
    # could add dynamic relaxation back, although it didnt work
    def gamma_loop(
        self, gamma_initial: np.ndarray, extra_relaxation_factor: float = 1.0
    ) -> tuple:
        """Standard fixed-point iteration with under-relaxation.

        Args:
            gamma_initial (np.ndarray): Initial circulation distribution.
            extra_relaxation_factor (float): Additional relaxation multiplier.

        Returns:
            tuple: (converged, gamma_new, alpha_array, Umag_array)
                - converged (bool): True if converged within tolerance.
                - gamma_new (np.ndarray): Final circulation distribution.
                - alpha_array (np.ndarray): Final angle of attack array.
                - Umag_array (np.ndarray): Final velocity magnitude array.
        """

        # looping untill max_iterations
        converged = False
        gamma_new = np.copy(gamma_initial)
        error_history = []

        # Spanwise artificial-viscosity regularization (Li, Gaunaa, Pirrung &
        # Lønbæk, TORQUE 2026). Stabilizes post-stall (negative lift-slope)
        # circulation distributions that otherwise develop non-physical sawtooth
        # oscillations and never converge. The context (tridiagonal Laplacian
        # diagonals, polar slope tables, planform area, stall-onset gate) is
        # built once since geometry and polars are frozen during the iteration.
        viscosity_ctx = self._build_viscosity_ctx()
        use_viscosity = viscosity_ctx is not None

        relaxation = self.relaxation_factor * extra_relaxation_factor
        self.last_stagnated = False
        for i in range(self.max_iterations):
            gamma = gamma_new
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.compute_aerodynamic_quantities(gamma)
            )
            gamma_target = (
                0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
            )
            if use_viscosity:
                gamma_target = self._regularize_gamma_target(
                    gamma_target, alpha_array, viscosity_ctx
                )
            gamma_new = (1 - relaxation) * gamma + relaxation * gamma_target

            if not np.all(np.isfinite(gamma_new)):
                # A non-finite circulation never recovers: every later
                # iterate is NaN. Return the last finite one, not converged,
                # instead of grinding max_iterations on NaN and handing the
                # caller NaN forces (seen on a doubled-back lifting line,
                # WingGeometry._warn_if_sections_double_back).
                logging.warning(
                    "Circulation loop produced non-finite gamma at iteration "
                    "%s -- stopping (degenerate mesh or expansive map); "
                    "returning the last finite circulation, not converged.",
                    i,
                )
                gamma_new = gamma
                break

            # Checking convergence using normalized error
            reference_error = (
                np.amax(np.abs(gamma_new)) if np.amax(np.abs(gamma_new)) != 0 else 1e-4
            )
            normalized_error = np.amax(np.abs(gamma_new - gamma)) / reference_error
            if (normalized_error) < self.allowed_error:
                converged = True
                break

            logging.debug(f"Normalized error at iteration {i}: {normalized_error}")
            # Store error for oscillation detection
            error_history.append(normalized_error)

            if self._stagnated(error_history):
                logging.debug(
                    "Circulation loop stagnated at iteration %s "
                    "(no improvement in %s iterations); stopping.",
                    i,
                    self.stagnation_patience,
                )
                self.last_stagnated = True
                break

            # Simple oscillation detection and handling. Skipped when artificial
            # viscosity is active, since the regularization already suppresses the
            # sawtooth oscillations this heuristic targets.
            if not use_viscosity and i >= 5 and len(error_history) >= 3:
                if (
                    error_history[-1] > error_history[-2]
                    and error_history[-2] < error_history[-3]
                ):
                    # Oscillation detected, apply additional damping
                    gamma_new = 0.75 * gamma_new + 0.25 * gamma
                    logging.debug(
                        f"Oscillation detected at iteration {i}, applying additional damping"
                    )

        if not converged:
            logging.warning(f"NOT Converged after {self.max_iterations} iterations")
        self.last_iterations = i + 1  # diagnostic: iterations used this solve
        return converged, gamma_new, alpha_array, Umag_array

    def _stagnated(self, error_history: list) -> bool:
        """True when the normalized error has stopped improving.

        Compares the best error of the last ``stagnation_patience`` iterations
        against the best of everything before them: if the recent window has
        not beaten the earlier best by at least ``stagnation_rtol``, the
        iteration is not going anywhere. Uses running minima rather than the
        latest value so an oscillating-but-descending solve is not cut off.
        """
        patience = int(getattr(self, "stagnation_patience", 0) or 0)
        if patience <= 0 or len(error_history) <= patience:
            return False
        recent_best = min(error_history[-patience:])
        prior_best = min(error_history[:-patience])
        return recent_best > prior_best * (1.0 - float(self.stagnation_rtol))

    def _build_viscosity_ctx(self) -> dict | None:
        """Pre-build the frozen-geometry objects the post-stall regularization
        needs, or ``None`` when artificial viscosity is disabled. Shared by the
        base and accelerated loops so they target the same regularized fixed
        point.

        Contents: per-panel stall onset (the cheap gate), planform area, the
        three diagonals of the tridiagonal spanwise Laplacian (the dense matrix
        of :meth:`_build_spanwise_laplacian` is tridiagonal, so the implicit
        solve is done banded), and each panel's polar table for the vectorized
        lift-slope evaluation. When all panels share one alpha grid (the normal
        outcome of batch polar generation) the cl columns are stacked into a
        single matrix so the slope evaluation needs no per-panel Python loop.
        """
        if not self.is_with_artificial_viscosity:
            return None
        laplacian = self._build_spanwise_laplacian()
        alpha_tables = [
            np.asarray(panel.panel_polar_data, dtype=float)[:, 0]
            for panel in self.panels
        ]
        cl_tables = [
            np.asarray(panel.panel_polar_data, dtype=float)[:, 1]
            for panel in self.panels
        ]
        shared_grid = all(
            table.shape == alpha_tables[0].shape
            and np.array_equal(table, alpha_tables[0])
            for table in alpha_tables[1:]
        )
        return {
            "stall_angles": self._panel_stall_angles(),
            "planform_area": float(np.sum(self.width_array * self.chord_array)),
            "L_diag": np.diag(laplacian).copy(),
            "L_super": np.diag(laplacian, 1).copy(),
            "L_sub": np.diag(laplacian, -1).copy(),
            "alpha_grid": alpha_tables[0] if shared_grid else None,
            "cl_matrix": np.vstack(cl_tables) if shared_grid else None,
            "alpha_tables": alpha_tables,
            "cl_tables": cl_tables,
        }

    @staticmethod
    def _interp_rows(
        query: np.ndarray, grid: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """Linear interpolation of ``values[i, :]`` at ``query[i]`` on a shared
        ``grid``, matching ``np.interp`` semantics (clamped at both grid ends).
        """
        idx = np.clip(np.searchsorted(grid, query), 1, grid.size - 1)
        x0 = grid[idx - 1]
        x1 = grid[idx]
        weight = np.clip((query - x0) / (x1 - x0), 0.0, 1.0)
        rows = np.arange(values.shape[0])
        y0 = values[rows, idx - 1]
        y1 = values[rows, idx]
        return y0 + weight * (y1 - y0)

    def _lift_slope_from_ctx(
        self,
        alpha_array: np.ndarray,
        viscosity_ctx: dict,
        delta: float = np.deg2rad(0.5),
    ) -> np.ndarray:
        """Vectorized equivalent of :meth:`_local_lift_slope`, evaluating the
        same central difference of each panel's piecewise-linear polar from the
        tables prepared in :meth:`_build_viscosity_ctx` (shared-grid fast path,
        per-panel fallback when panels carry different alpha grids).
        """
        grid = viscosity_ctx["alpha_grid"]
        if grid is not None and grid.size >= 2:
            cl_matrix = viscosity_ctx["cl_matrix"]
            cl_plus = self._interp_rows(alpha_array + delta, grid, cl_matrix)
            cl_minus = self._interp_rows(alpha_array - delta, grid, cl_matrix)
            return (cl_plus - cl_minus) / (2.0 * delta)
        slopes = np.empty(self.n_panels)
        for i, (alpha_table, cl_table) in enumerate(
            zip(viscosity_ctx["alpha_tables"], viscosity_ctx["cl_tables"])
        ):
            cl_plus = np.interp(alpha_array[i] + delta, alpha_table, cl_table)
            cl_minus = np.interp(alpha_array[i] - delta, alpha_table, cl_table)
            slopes[i] = (cl_plus - cl_minus) / (2.0 * delta)
        return slopes

    def _regularize_gamma_target(
        self,
        gamma_target: np.ndarray,
        alpha_array: np.ndarray,
        viscosity_ctx: dict | None,
    ) -> np.ndarray:
        """Apply the Li/Gaunaa implicit spanwise viscosity to the fixed-point
        target: solve ``(I - diag(mu) L) gamma = gamma_target``.

        Implicit fixed point (I - diag(mu) L) gamma = F(gamma): same steady
        solution as the explicit scheme but stable at relaxation factors of
        order one, whereas the explicit stable step shrinks like N^-2 in
        post-stall. The coefficient ``mu_i = max(0, -k S Cl'_i / dz_i^2)`` with
        k = 0.035 reduces to ``mu = max(0, -k N^2/AR Cl')`` for a uniformly
        spaced wing (Eq. 16).

        Returns ``gamma_target`` unchanged (same object, no solve) while no
        panel is past its stall onset or every ``mu`` is zero — the exact no-op
        that keeps attached-flow iterations as cheap as the unregularized loop.
        The system is tridiagonal, so the solve is banded, not dense.
        """
        if viscosity_ctx is None or not np.any(
            alpha_array > viscosity_ctx["stall_angles"]
        ):
            return gamma_target
        lift_slope = self._lift_slope_from_ctx(alpha_array, viscosity_ctx)
        mu_array = np.maximum(
            0.0,
            -self.artificial_viscosity_factor
            * viscosity_ctx["planform_area"]
            * lift_slope
            / self.width_array**2,
        )
        if not np.any(mu_array > 0.0):
            return gamma_target
        n = gamma_target.size
        # Banded storage of (I - diag(mu) L): row i couples only i-1, i, i+1.
        ab = np.zeros((3, n))
        ab[1] = 1.0 - mu_array * viscosity_ctx["L_diag"]
        ab[0, 1:] = -mu_array[:-1] * viscosity_ctx["L_super"]
        ab[2, :-1] = -mu_array[1:] * viscosity_ctx["L_sub"]
        return solve_banded((1, 1), ab, gamma_target)

    def _fixed_point_target(
        self, gamma: np.ndarray, viscosity_ctx: dict | None = None
    ) -> tuple:
        """Single evaluation of the circulation fixed-point map ``G(gamma)``.

        Returns ``(gamma_target, alpha_array, Umag_array)``. The fixed point
        ``gamma*`` satisfies ``gamma* = G(gamma*)`` — the very quantity the base
        :meth:`gamma_loop` relaxes toward with ``gamma_new = (1-w) gamma + w
        G(gamma)``. Sharing this map lets the accelerated loops converge to the
        identical solution. When artificial viscosity is active and any panel is
        past stall, the post-stall regularization (Li, Gaunaa, Pirrung & Lønbæk,
        TORQUE 2026) is folded into the target so ``base`` and ``anderson`` share
        the same regularized fixed point.
        """
        alpha_array, Umag_array, cl_array, Umagw_array = (
            self.compute_aerodynamic_quantities(gamma)
        )
        gamma_target = (
            0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
        )
        gamma_target = self._regularize_gamma_target(
            gamma_target, alpha_array, viscosity_ctx
        )
        return gamma_target, alpha_array, Umag_array

    def gamma_loop_anderson(self, gamma_initial: np.ndarray) -> tuple:
        """Anderson-accelerated fixed-point iteration for the circulation.

        Anderson acceleration is applied to the *under-relaxed* Picard map

            g(gamma) = (1 - w) gamma + w G(gamma),   w = relaxation_factor,

        not to the raw ``G(gamma)``: the raw circulation map is expansive here
        (hence the base loop under-relaxes heavily), and accelerating it
        directly diverges. The relaxed map is a contraction with the *same*
        fixed point, and each step mixes the last ``anderson_depth`` relaxed
        residuals through a small (``m x m``) least-squares problem (Walker & Ni
        2011, SIAM J. Numer. Anal. 49, 1715). This decouples ``w`` from the
        convergence rate — a conservative, robust ``w`` still converges in
        O(10s) of iterations instead of O(100s), removing the fragile speed/
        stability trade-off of the bare relaxation factor. The stopping rule and
        optional post-stall regularization are identical to ``base``, so the
        returned circulation matches it to solver tolerance.

        .. warning::
            Anderson terminates on a *superlinear* (jumpy) residual, so near the
            tolerance boundary a tiny change in the inflow can flip the returned
            circulation by ~one convergence jump (e.g. from a 1e-3 to a 1e-8
            residual). The converged gamma is therefore a slightly *non-smooth*
            function of the inflow. This is invisible for a standalone solve, but
            it corrupts any *outer* finite-difference Jacobian that differentiates
            through this loop (e.g. the AWETrim quasi-steady trim solvers) unless
            ``allowed_error`` is tight (~1e-8), which pushes the jump below the FD
            step. The base loop's slow *linear* convergence keeps its
            loosely-converged gamma smooth, so ``base`` is the safe choice for
            FD-outer-loop use at loose tolerance.

        Args:
            gamma_initial (np.ndarray): Initial circulation distribution.

        Returns:
            tuple: ``(converged, gamma_new, alpha_array, Umag_array)`` matching
            :meth:`gamma_loop`.
        """
        m = max(1, int(self.anderson_depth))
        beta = float(self.anderson_beta)
        w = self.relaxation_factor
        # Relative Tikhonov regularization of the depth-m least-squares problem:
        # damps the extrapolation when the residual-difference columns are
        # near-linearly-dependent, biasing toward the safe relaxed-Picard step
        # rather than an over-large quasi-Newton stride.
        reg = 1e-10
        # Anderson converges superlinearly here (O(10s) of iterations); if it has
        # not converged within this budget it is in a limit cycle (deep post-stall,
        # where VSM itself is unreliable and only the base loop's viscosity /
        # relaxation tames it). The caller (:meth:`solve`) then falls back to the
        # base relaxed-Picard loop, so this just bounds the wasted work.
        max_it = min(self.max_iterations, int(self.anderson_max_iterations))
        viscosity_ctx = self._build_viscosity_ctx()

        def relaxed_step(x):
            # One evaluation of the relaxed fixed-point map g(x) and its residual
            # f(x) = g(x) - x = w (G(x) - x). Same fixed point as G, contractive.
            target, alpha, umag = self._fixed_point_target(x, viscosity_ctx)
            g = (1.0 - w) * x + w * target
            return g, g - x, alpha, umag

        x = np.array(gamma_initial, dtype=float)
        g, f, alpha_array, Umag_array = relaxed_step(x)

        x_hist: list[np.ndarray] = []  # window of iterates (current one included)
        f_hist: list[np.ndarray] = []  # window of relaxed residuals g(x)-x
        converged = False
        last_k = 0
        error_history: list[float] = []
        self.last_stagnated = False

        for k in range(max_it):
            last_k = k
            # Same normalized-error measure as the base loop: |g - gamma| over
            # the peak circulation (g is the relaxed update, matching base's
            # ``max|gamma_new - gamma| / max|gamma_new|``).
            reference_error = np.amax(np.abs(g))
            reference_error = reference_error if reference_error != 0 else 1e-4
            normalized_error = np.amax(np.abs(f)) / reference_error
            if normalized_error < self.allowed_error:
                converged = True
                break
            logging.debug(
                f"Anderson normalized error at iteration {k}: {normalized_error}"
            )

            error_history.append(normalized_error)
            if self._stagnated(error_history):
                logging.debug(
                    "Anderson loop stagnated at iteration %s "
                    "(no improvement in %s iterations); stopping.",
                    k,
                    self.stagnation_patience,
                )
                self.last_stagnated = True
                break

            x_hist.append(x)
            f_hist.append(f)
            if len(f_hist) > m + 1:
                x_hist.pop(0)
                f_hist.pop(0)

            mk = len(f_hist) - 1
            if mk == 0:
                # No history yet: a single (damped) relaxed Picard step to seed.
                x_new = x + beta * f
            else:
                dF = np.stack(
                    [f_hist[j] - f_hist[j - 1] for j in range(1, len(f_hist))], axis=1
                )  # (n_panels, mk)
                dX = np.stack(
                    [x_hist[j] - x_hist[j - 1] for j in range(1, len(x_hist))], axis=1
                )  # (n_panels, mk)
                gram = dF.T @ dF
                lam = reg * float(np.trace(gram)) / dF.shape[1]
                theta = np.linalg.solve(gram + lam * np.eye(dF.shape[1]), dF.T @ f)
                x_new = x + beta * f - (dX + beta * dF) @ theta

            x = x_new
            g, f, alpha_array, Umag_array = relaxed_step(x)
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(g))):
                # Same guard as the base loop: fall back to the last finite
                # iterate and let the caller's base-loop fallback / failure
                # handling take over.
                logging.warning(
                    "Anderson circulation loop produced non-finite gamma at "
                    "iteration %s -- stopping on the last finite iterate.",
                    k,
                )
                x = x_hist[-1] if x_hist else np.array(gamma_initial, dtype=float)
                g, f, alpha_array, Umag_array = relaxed_step(x)
                break

        self.last_iterations = last_k + 1  # diagnostic: iterations used
        if not converged:
            logging.info(
                f"Anderson did not converge in {max_it} iterations "
                "(deep post-stall limit cycle); caller falls back to base loop."
            )
        return converged, x, alpha_array, Umag_array

    def gamma_loop_non_linear(self, gamma_initial: np.ndarray) -> tuple:
        """Nonlinear solver using robust SciPy optimization methods.

        Solves F(gamma) = gamma_new(gamma) - gamma = 0 using Broyden methods.

        Args:
            gamma_initial (np.ndarray): Initial guess for circulation distribution.

        Returns:
            tuple: (converged, gamma_new, alpha_array, Umag_array)
                - converged (bool): True if converged within tolerance.
                - gamma_new (np.ndarray): Final circulation distribution.
                - alpha_array (np.ndarray): Final angle of attack array.
                - Umag_array (np.ndarray): Final velocity magnitude array.
        """

        def compute_gamma_residual(gamma):
            _, Umag_array, cl_array, Umagw_array = self.compute_aerodynamic_quantities(
                gamma
            )
            gamma_new = (
                0.5 * ((Umag_array**2) / Umagw_array) * cl_array * self.chord_array
            )
            # Residual: difference between the computed and current gamma.
            F_val = gamma - gamma_new
            return F_val

        success = False
        if not success:
            try:
                gamma_new = broyden1(
                    lambda x: compute_gamma_residual(x),
                    gamma_initial,
                    f_tol=self.allowed_error,
                    maxiter=self.max_iterations,
                )
                if (
                    np.linalg.norm(compute_gamma_residual(gamma_new), ord=np.inf)
                    < self.allowed_error
                ):
                    success = True
                    logging.info("Converged (non_linear: broyden1)")
                else:
                    logging.warning(
                        "--> broyden1 method did not converge to desired tolerance"
                    )
            except Exception as e:
                logging.warning(f"--> broyden1 failed, running base")
        if not success:
            try:
                gamma_new = broyden2(
                    lambda x: compute_gamma_residual(x),
                    gamma_initial,
                    f_tol=self.allowed_error,
                    maxiter=self.max_iterations,
                )
                if (
                    np.linalg.norm(compute_gamma_residual(gamma_new), ord=np.inf)
                    < self.allowed_error
                ):
                    success = True
                    logging.info("Converged (non_linear: broyden2)")
                else:
                    logging.warning(
                        "--> broyden2 method did not converge to desired tolerance"
                    )
            except Exception as e:
                logging.warning(f"--> broyden2 failed, running base")

        if not success:
            return self.gamma_loop(
                gamma_initial,
            )
        if success:
            alpha_array, Umag_array, cl_array, Umagw_array = (
                self.compute_aerodynamic_quantities(gamma_new)
            )
            return True, gamma_new, alpha_array, Umag_array
