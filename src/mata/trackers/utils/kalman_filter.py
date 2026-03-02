"""Kalman filter implementations for object tracking.

Ported from Ultralytics tracker utils (MIT-compatible).

Two variants:
- KalmanFilterXYAH: State (cx, cy, aspect_ratio, height, vx, vy, va, vh)
  Used by ByteTrack's STrack.
- KalmanFilterXYWH: State (cx, cy, width, height, vx, vy, vw, vh)
  Used by BotSort's BOTrack.

Both use a constant-velocity motion model with 8-dimensional state:
  x = [position(4), velocity(4)]
  F = [[I, dt*I], [0, I]]   (motion/transition matrix)
  H = [I, 0]                (measurement/update matrix)

Cholesky-based Kalman updates are used for numerical stability.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


class KalmanFilterXYAH:
    """Kalman filter with state: (cx, cy, a, h, vcx, vcy, va, vh).

    Position is represented as (center_x, center_y, aspect_ratio, height).
    The aspect ratio ``a = w / h`` is dimensionless; its noise is modelled
    with fixed (non-scaled) weights to avoid coupling with object scale.

    Attributes:
        ndim (int): Number of position dimensions (4).
        dt (float): Time step between frames (1.0 frame).

    Process noise weights:
        ``_std_weight_position = 1/20``: Position std is  1/20 of height.
        ``_std_weight_velocity = 1/160``: Velocity std is 1/160 of height.
    """

    ndim: int = 4
    dt: float = 1.0

    def __init__(self) -> None:
        # ------------------------------------------------------------------ #
        # Motion (state-transition) matrix  F  of shape (8, 8):              #
        #   new_pos = pos + dt * vel                                          #
        #   new_vel = vel           (constant-velocity model)                 #
        # ------------------------------------------------------------------ #
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt

        # Measurement (observation) matrix  H  of shape (4, 8):
        #   z = [cx, cy, a, h]  (no velocity observed)
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)

        # Process / measurement noise scaling factors
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a new track from an unassociated measurement.

        Args:
            measurement: Bounding-box in format ``(cx, cy, a, h)`` where
                ``a`` is the aspect ratio ``w / h`` and ``h`` is the height.

        Returns:
            mean: Initial state mean of shape ``(8,)``.
            covariance: Initial state covariance of shape ``(8, 8)``.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,  # cx
            2 * self._std_weight_position * h,  # cy
            1e-2,  # a  (aspect ratio — fixed)
            2 * self._std_weight_position * h,  # h
            10 * self._std_weight_velocity * h,  # vcx
            10 * self._std_weight_velocity * h,  # vcy
            1e-5,  # va (aspect-ratio velocity)
            10 * self._std_weight_velocity * h,  # vh
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the Kalman filter prediction step (prior / time update).

        Propagates the state distribution forward by one time step using the
        constant-velocity motion model, adding process noise.

        Args:
            mean: Current state mean of shape ``(8,)``.
            covariance: Current state covariance of shape ``(8, 8)``.

        Returns:
            mean: Predicted state mean of shape ``(8,)``.
            covariance: Predicted state covariance of shape ``(8, 8)``.
        """
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,  # cx
            self._std_weight_position * h,  # cy
            1e-2,  # a
            self._std_weight_position * h,  # h
        ]
        std_vel = [
            self._std_weight_velocity * h,  # vcx
            self._std_weight_velocity * h,  # vcy
            1e-5,  # va
            self._std_weight_velocity * h,  # vh
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project the state distribution to measurement space.

        Transforms the 8-D state into the 4-D measurement space and adds
        measurement noise.

        Args:
            mean: State mean of shape ``(8,)``.
            covariance: State covariance of shape ``(8, 8)``.

        Returns:
            mean: Projected mean in measurement space, shape ``(4,)``.
            covariance: Projected covariance + measurement noise, shape ``(4, 4)``.
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,  # cx
            self._std_weight_position * h,  # cy
            1e-1,  # a  (larger uncertainty in meas space)
            self._std_weight_position * h,  # h
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the Kalman filter correction step (posterior / measurement update).

        Uses Cholesky decomposition of the projected covariance for
        numerically stable computation of the Kalman gain.

        Args:
            mean: Prior state mean of shape ``(8,)``.
            covariance: Prior state covariance of shape ``(8, 8)``.
            measurement: Observed bounding box ``(cx, cy, a, h)``.

        Returns:
            mean: Posterior state mean of shape ``(8,)``.
            covariance: Posterior state covariance of shape ``(8, 8)``.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain:  K = P * H^T * (H * P * H^T + R)^{-1}
        # Solved via Cholesky:  K = (S \ (P * H^T)^T)^T
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """Compute Mahalanobis gating distance between a track and measurements.

        Used during data association to filter infeasible assignments. The
        returned squared Mahalanobis distances are compared against a chi-squared
        threshold (typically ``scipy.stats.chi2.ppf(0.95, df)``).

        Args:
            mean: Track state mean of shape ``(8,)``.
            covariance: Track state covariance of shape ``(8, 8)``.
            measurements: Array of candidate measurements, shape ``(N, 4)``.
            only_position: If ``True``, use only the (cx, cy) sub-space
                (2-D chi-squared threshold), otherwise use the full
                4-D measurement space.

        Returns:
            squared_maha: Squared Mahalanobis distances, shape ``(N,)``.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        # Lower-triangular Cholesky factor L  s.t.  S = L @ L^T
        cholesky_factor = np.linalg.cholesky(projected_cov)

        # Solve L @ z = (measurements - mean)^T  →  ||z||^2 = Mahalanobis^2
        d = measurements - projected_mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha


class KalmanFilterXYWH(KalmanFilterXYAH):
    """Kalman filter with state: (cx, cy, w, h, vcx, vcy, vw, vh).

    Position is represented as (center_x, center_y, width, height).
    Used by BotSort's ``BOTrack`` — more intuitive than the aspect-ratio
    parameterisation of :class:`KalmanFilterXYAH`.

    The key difference is that width noise scales proportionally to the
    object's current **width** (``mean[2]``), whereas in ``KalmanFilterXYAH``
    the aspect-ratio noise uses a small fixed constant.

    All other machinery (motion model, Cholesky update, gating) is inherited
    unchanged from :class:`KalmanFilterXYAH`.
    """

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a new track from an unassociated measurement.

        Args:
            measurement: Bounding-box in format ``(cx, cy, w, h)`` where
                ``w`` is the width and ``h`` is the height (absolute pixels).

        Returns:
            mean: Initial state mean of shape ``(8,)``.
            covariance: Initial state covariance of shape ``(8, 8)``.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        w, h = measurement[2], measurement[3]
        std = [
            2 * self._std_weight_position * h,  # cx
            2 * self._std_weight_position * h,  # cy
            2 * self._std_weight_position * w,  # w  ← scales with width
            2 * self._std_weight_position * h,  # h
            10 * self._std_weight_velocity * h,  # vcx
            10 * self._std_weight_velocity * h,  # vcy
            10 * self._std_weight_velocity * w,  # vw ← scales with width
            10 * self._std_weight_velocity * h,  # vh
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the Kalman filter prediction step.

        Args:
            mean: Current state mean of shape ``(8,)``.
            covariance: Current state covariance of shape ``(8, 8)``.

        Returns:
            mean: Predicted state mean of shape ``(8,)``.
            covariance: Predicted state covariance of shape ``(8, 8)``.
        """
        w, h = mean[2], mean[3]
        std_pos = [
            self._std_weight_position * h,  # cx
            self._std_weight_position * h,  # cy
            self._std_weight_position * w,  # w
            self._std_weight_position * h,  # h
        ]
        std_vel = [
            self._std_weight_velocity * h,  # vcx
            self._std_weight_velocity * h,  # vcy
            self._std_weight_velocity * w,  # vw
            self._std_weight_velocity * h,  # vh
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project the state distribution to measurement space.

        Args:
            mean: State mean of shape ``(8,)``.
            covariance: State covariance of shape ``(8, 8)``.

        Returns:
            mean: Projected mean, shape ``(4,)``.
            covariance: Projected covariance + measurement noise, shape ``(4, 4)``.
        """
        w, h = mean[2], mean[3]
        std = [
            self._std_weight_position * h,  # cx
            self._std_weight_position * h,  # cy
            self._std_weight_position * w,  # w
            self._std_weight_position * h,  # h
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the Kalman filter correction step.

        Delegates entirely to the parent class — the update algebra is
        identical between XYAH and XYWH; only the noise model differs.

        Args:
            mean: Prior state mean of shape ``(8,)``.
            covariance: Prior state covariance of shape ``(8, 8)``.
            measurement: Observed bounding box ``(cx, cy, w, h)``.

        Returns:
            mean: Posterior state mean of shape ``(8,)``.
            covariance: Posterior state covariance of shape ``(8, 8)``.
        """
        return super().update(mean, covariance, measurement)
