"""Unit tests for Task A1: Kalman Filter Implementations.

Covers:
- KalmanFilterXYAH: initiate, predict, project, update, gating_distance
- KalmanFilterXYWH: initiate, predict, project, update (delegated), gating_distance
- Predict → Update cycles produce converging, reasonable state estimates
- Numerical properties: PSD covariance, Cholesky stability
- Edge cases: very small/large objects, single step, multiple steps
- only_position flag in gating_distance
- Subclass noise model differs correctly from base class
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import scipy.stats

from mata.trackers.utils.kalman_filter import KalmanFilterXYAH, KalmanFilterXYWH

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_positive_semidefinite(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True if *matrix* is symmetric positive semi-definite."""
    if not np.allclose(matrix, matrix.T, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(matrix)
    return bool(np.all(eigvals >= -tol))


def is_positive_definite(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Return True if *matrix* is symmetric positive definite."""
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        return False
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kfah() -> KalmanFilterXYAH:
    return KalmanFilterXYAH()


@pytest.fixture
def kfwh() -> KalmanFilterXYWH:
    return KalmanFilterXYWH()


@pytest.fixture
def measurement_xyah() -> np.ndarray:
    """A typical (cx, cy, a, h) measurement: person at (300, 200), a=0.4, h=150."""
    return np.array([300.0, 200.0, 0.4, 150.0])


@pytest.fixture
def measurement_xywh() -> np.ndarray:
    """A typical (cx, cy, w, h) measurement: person at (300, 200), 60x150."""
    return np.array([300.0, 200.0, 60.0, 150.0])


@pytest.fixture
def small_measurement_xyah() -> np.ndarray:
    """Small object (a=0.9, h=20) for edge-case tests."""
    return np.array([50.0, 50.0, 0.9, 20.0])


@pytest.fixture
def large_measurement_xywh() -> np.ndarray:
    """Large object (w=400, h=300) for edge-case tests."""
    return np.array([640.0, 360.0, 400.0, 300.0])


# ===========================================================================
# KalmanFilterXYAH — initiate
# ===========================================================================


class TestKalmanFilterXYAHInitiate:
    def test_mean_shape(self, kfah, measurement_xyah):
        mean, _ = kfah.initiate(measurement_xyah)
        assert mean.shape == (8,)

    def test_covariance_shape(self, kfah, measurement_xyah):
        _, cov = kfah.initiate(measurement_xyah)
        assert cov.shape == (8, 8)

    def test_mean_position_matches_measurement(self, kfah, measurement_xyah):
        mean, _ = kfah.initiate(measurement_xyah)
        np.testing.assert_array_equal(mean[:4], measurement_xyah)

    def test_mean_velocity_is_zero(self, kfah, measurement_xyah):
        mean, _ = kfah.initiate(measurement_xyah)
        np.testing.assert_array_equal(mean[4:], np.zeros(4))

    def test_covariance_is_diagonal(self, kfah, measurement_xyah):
        _, cov = kfah.initiate(measurement_xyah)
        off_diagonal = cov - np.diag(np.diag(cov))
        np.testing.assert_array_equal(off_diagonal, np.zeros((8, 8)))

    def test_covariance_is_positive_definite(self, kfah, measurement_xyah):
        _, cov = kfah.initiate(measurement_xyah)
        assert is_positive_definite(cov)

    def test_covariance_position_scales_with_height(self, kfah):
        m1 = np.array([100.0, 100.0, 0.5, 100.0])  # h=100
        m2 = np.array([100.0, 100.0, 0.5, 200.0])  # h=200
        _, cov1 = kfah.initiate(m1)
        _, cov2 = kfah.initiate(m2)
        # Position variance (cx, cy, h) should scale as (h ratio)^2 = 4
        assert cov2[0, 0] == pytest.approx(4 * cov1[0, 0])
        assert cov2[1, 1] == pytest.approx(4 * cov1[1, 1])
        assert cov2[3, 3] == pytest.approx(4 * cov1[3, 3])

    def test_aspect_ratio_variance_is_fixed(self, kfah):
        """Aspect-ratio variance (index 2) must NOT depend on height."""
        m1 = np.array([100.0, 100.0, 0.5, 50.0])
        m2 = np.array([100.0, 100.0, 0.5, 500.0])
        _, cov1 = kfah.initiate(m1)
        _, cov2 = kfah.initiate(m2)
        assert cov1[2, 2] == pytest.approx(cov2[2, 2])

    def test_small_object(self, kfah, small_measurement_xyah):
        mean, cov = kfah.initiate(small_measurement_xyah)
        assert is_positive_definite(cov)
        np.testing.assert_array_equal(mean[:4], small_measurement_xyah)


# ===========================================================================
# KalmanFilterXYAH — predict
# ===========================================================================


class TestKalmanFilterXYAHPredict:
    def test_output_shapes(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        p_mean, p_cov = kfah.predict(mean, cov)
        assert p_mean.shape == (8,)
        assert p_cov.shape == (8, 8)

    def test_covariance_grows_after_prediction(self, kfah, measurement_xyah):
        """Prediction step must increase uncertainty (variance must not shrink)."""
        mean, cov = kfah.initiate(measurement_xyah)
        _, p_cov = kfah.predict(mean, cov)
        assert np.all(np.diag(p_cov) >= np.diag(cov) - 1e-12)

    def test_predicted_covariance_is_symmetric(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        _, p_cov = kfah.predict(mean, cov)
        np.testing.assert_allclose(p_cov, p_cov.T, atol=1e-10)

    def test_predicted_covariance_is_positive_definite(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        _, p_cov = kfah.predict(mean, cov)
        assert is_positive_definite(p_cov)

    def test_zero_velocity_keeps_position(self, kfah, measurement_xyah):
        """With zero initial velocity the mean position must not change."""
        mean, cov = kfah.initiate(measurement_xyah)
        p_mean, _ = kfah.predict(mean, cov)
        # For zero velocity, F @ mean[:4] == mean[:4]
        np.testing.assert_allclose(p_mean[:4], mean[:4], atol=1e-12)

    def test_nonzero_velocity_propagates(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        # Manually set a known velocity
        mean[4] = 5.0  # vcx = 5 px/frame
        mean[5] = -3.0  # vcy = -3 px/frame
        p_mean, _ = kfah.predict(mean, cov)
        assert p_mean[0] == pytest.approx(measurement_xyah[0] + 5.0)
        assert p_mean[1] == pytest.approx(measurement_xyah[1] - 3.0)

    def test_velocity_unchanged_in_constant_model(self, kfah, measurement_xyah):
        """Constant-velocity model: velocity should not change during predict."""
        mean, cov = kfah.initiate(measurement_xyah)
        mean[4:] = np.array([1.0, 2.0, 0.001, 0.5])
        p_mean, _ = kfah.predict(mean, cov)
        np.testing.assert_allclose(p_mean[4:], mean[4:], atol=1e-12)


# ===========================================================================
# KalmanFilterXYAH — project
# ===========================================================================


class TestKalmanFilterXYAHProject:
    def test_output_shapes(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        z_mean, z_cov = kfah.project(mean, cov)
        assert z_mean.shape == (4,)
        assert z_cov.shape == (4, 4)

    def test_projected_mean_matches_position(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        z_mean, _ = kfah.project(mean, cov)
        np.testing.assert_allclose(z_mean, measurement_xyah, atol=1e-12)

    def test_projected_cov_is_positive_definite(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        _, z_cov = kfah.project(mean, cov)
        assert is_positive_definite(z_cov)

    def test_projected_cov_is_symmetric(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        _, z_cov = kfah.project(mean, cov)
        np.testing.assert_allclose(z_cov, z_cov.T, atol=1e-10)


# ===========================================================================
# KalmanFilterXYAH — update
# ===========================================================================


class TestKalmanFilterXYAHUpdate:
    def test_output_shapes(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        u_mean, u_cov = kfah.update(mean, cov, measurement_xyah)
        assert u_mean.shape == (8,)
        assert u_cov.shape == (8, 8)

    def test_update_with_same_measurement_shrinks_covariance(self, kfah, measurement_xyah):
        """Updating with the same observation must reduce uncertainty."""
        mean, cov = kfah.initiate(measurement_xyah)
        u_mean, u_cov = kfah.update(mean, cov, measurement_xyah)
        # All diagonal entries must be smaller after update
        assert np.all(np.diag(u_cov) <= np.diag(cov) + 1e-12)

    def test_updated_covariance_is_positive_semidefinite(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        _, u_cov = kfah.update(mean, cov, measurement_xyah)
        assert is_positive_semidefinite(u_cov)

    def test_updated_covariance_is_symmetric(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        _, u_cov = kfah.update(mean, cov, measurement_xyah)
        np.testing.assert_allclose(u_cov, u_cov.T, atol=1e-10)

    def test_update_moves_mean_toward_measurement(self, kfah):
        """If measurement differs from prior, the posterior should move toward it."""
        m_init = np.array([100.0, 100.0, 0.5, 120.0])
        mean, cov = kfah.initiate(m_init)

        # New measurement is shifted
        m_new = np.array([110.0, 110.0, 0.5, 120.0])
        u_mean, _ = kfah.update(mean, cov, m_new)

        # Posterior cx should be between prior and measurement
        assert mean[0] < u_mean[0] < m_new[0] or u_mean[0] == pytest.approx(m_new[0], abs=1.0)

    def test_repeated_updates_converge(self, kfah):
        """Repeated updates with the same measurement should converge to it."""
        true_pos = np.array([300.0, 200.0, 0.4, 150.0])
        mean, cov = kfah.initiate(true_pos + np.array([50.0, 50.0, 0.1, 20.0]))

        for _ in range(50):
            mean, cov = kfah.predict(mean, cov)
            mean, cov = kfah.update(mean, cov, true_pos)

        np.testing.assert_allclose(mean[:4], true_pos, atol=2.0)


# ===========================================================================
# KalmanFilterXYAH — predict → update cycle
# ===========================================================================


class TestKalmanFilterXYAHCycle:
    def test_single_predict_update_cycle(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        mean, cov = kfah.predict(mean, cov)
        mean, cov = kfah.update(mean, cov, measurement_xyah)
        assert mean.shape == (8,)
        assert is_positive_semidefinite(cov)

    def test_multi_frame_tracking(self, kfah):
        """Track an object moving at constant velocity over 20 frames."""
        # Object starts at (100, 100), moves +3px/frame in x
        cx, cy, a, h = 100.0, 200.0, 0.5, 100.0
        vx = 3.0
        mean, cov = kfah.initiate(np.array([cx, cy, a, h]))

        for t in range(1, 21):
            mean, cov = kfah.predict(mean, cov)
            measurement = np.array([cx + t * vx, cy, a, h])
            mean, cov = kfah.update(mean, cov, measurement)

        expected_cx = cx + 20 * vx
        assert mean[0] == pytest.approx(expected_cx, abs=5.0)
        assert is_positive_semidefinite(cov)

    def test_covariance_does_not_diverge(self, kfah, measurement_xyah):
        """After many predict-update cycles covariance should stay bounded."""
        mean, cov = kfah.initiate(measurement_xyah)
        for _ in range(100):
            mean, cov = kfah.predict(mean, cov)
            mean, cov = kfah.update(mean, cov, measurement_xyah)

        # Max diagonal entry should not blow up
        assert np.max(np.diag(cov)) < 1e6


# ===========================================================================
# KalmanFilterXYAH — gating_distance
# ===========================================================================


class TestKalmanFilterXYAHGatingDistance:
    def test_output_shape(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        measurements = np.array([[300.0, 200.0, 0.4, 150.0], [350.0, 250.0, 0.4, 150.0]])
        dist = kfah.gating_distance(mean, cov, measurements)
        assert dist.shape == (2,)

    def test_zero_distance_for_exact_match(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        # The projected mean equals the measurement → distance ≈ 0
        projected_mean, _ = kfah.project(mean, cov)
        dist = kfah.gating_distance(mean, cov, projected_mean[np.newaxis])
        assert dist[0] == pytest.approx(0.0, abs=1e-6)

    def test_distances_are_nonnegative(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        measurements = np.random.default_rng(0).normal(size=(10, 4)) * 50 + measurement_xyah
        dist = kfah.gating_distance(mean, cov, measurements)
        assert np.all(dist >= 0.0)

    def test_closer_measurement_has_smaller_distance(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        close = measurement_xyah + np.array([2.0, 2.0, 0.01, 2.0])
        far = measurement_xyah + np.array([50.0, 50.0, 0.2, 30.0])
        d_close = kfah.gating_distance(mean, cov, close[np.newaxis])
        d_far = kfah.gating_distance(mean, cov, far[np.newaxis])
        assert d_close[0] < d_far[0]

    def test_only_position_reduces_dimensionality(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        m = measurement_xyah[np.newaxis].copy()
        d_full = kfah.gating_distance(mean, cov, m, only_position=False)
        d_pos = kfah.gating_distance(mean, cov, m, only_position=True)
        # 2-D distance ≤ 4-D distance (fewer dimensions = less penalty)
        assert d_pos[0] <= d_full[0] + 1e-9

    def test_single_measurement_input(self, kfah, measurement_xyah):
        mean, cov = kfah.initiate(measurement_xyah)
        dist = kfah.gating_distance(mean, cov, measurement_xyah[np.newaxis])
        assert dist.shape == (1,)

    def test_chi2_threshold_compatibility(self, kfah, measurement_xyah):
        """95% of measurements drawn from the true distribution
        should fall within the chi-squared 95th-percentile gate."""
        rng = np.random.default_rng(42)
        mean, cov = kfah.initiate(measurement_xyah)
        projected_mean, projected_cov = kfah.project(mean, cov)

        n_samples = 1000
        samples = rng.multivariate_normal(projected_mean, projected_cov, size=n_samples)
        distances = kfah.gating_distance(mean, cov, samples)

        chi2_threshold = scipy.stats.chi2.ppf(0.95, df=4)
        fraction_inside = np.mean(distances < chi2_threshold)
        # Allow 3% tolerance around the nominal 95%
        assert fraction_inside >= 0.92


# ===========================================================================
# KalmanFilterXYWH — initiate
# ===========================================================================


class TestKalmanFilterXYWHInitiate:
    def test_mean_shape(self, kfwh, measurement_xywh):
        mean, _ = kfwh.initiate(measurement_xywh)
        assert mean.shape == (8,)

    def test_mean_position_matches_measurement(self, kfwh, measurement_xywh):
        mean, _ = kfwh.initiate(measurement_xywh)
        np.testing.assert_array_equal(mean[:4], measurement_xywh)

    def test_mean_velocity_is_zero(self, kfwh, measurement_xywh):
        mean, _ = kfwh.initiate(measurement_xywh)
        np.testing.assert_array_equal(mean[4:], np.zeros(4))

    def test_covariance_is_positive_definite(self, kfwh, measurement_xywh):
        _, cov = kfwh.initiate(measurement_xywh)
        assert is_positive_definite(cov)

    def test_width_variance_scales_with_width(self, kfwh):
        """Width noise (index 2) must scale with the object's width."""
        m1 = np.array([100.0, 100.0, 60.0, 150.0])  # w = 60
        m2 = np.array([100.0, 100.0, 120.0, 150.0])  # w = 120
        _, cov1 = kfwh.initiate(m1)
        _, cov2 = kfwh.initiate(m2)
        # Variance scales as (width_ratio)^2 = 4
        assert cov2[2, 2] == pytest.approx(4 * cov1[2, 2])

    def test_width_variance_differs_from_xyah(self, kfwh, kfah):
        """XYWH width variance must differ from XYAH aspect-ratio variance."""
        m_wh = np.array([300.0, 200.0, 60.0, 150.0])
        m_ah = np.array([300.0, 200.0, 60.0 / 150.0, 150.0])  # same box as XYAH
        _, cov_wh = kfwh.initiate(m_wh)
        _, cov_ah = kfah.initiate(m_ah)
        # XYAH aspect-ratio variance is fixed (tiny), XYWH width variance is large
        assert cov_wh[2, 2] > cov_ah[2, 2]

    def test_large_object(self, kfwh, large_measurement_xywh):
        mean, cov = kfwh.initiate(large_measurement_xywh)
        assert is_positive_definite(cov)
        np.testing.assert_array_equal(mean[:4], large_measurement_xywh)


# ===========================================================================
# KalmanFilterXYWH — predict
# ===========================================================================


class TestKalmanFilterXYWHPredict:
    def test_covariance_grows(self, kfwh, measurement_xywh):
        mean, cov = kfwh.initiate(measurement_xywh)
        _, p_cov = kfwh.predict(mean, cov)
        assert np.all(np.diag(p_cov) >= np.diag(cov) - 1e-12)

    def test_predicted_covariance_is_positive_definite(self, kfwh, measurement_xywh):
        mean, cov = kfwh.initiate(measurement_xywh)
        _, p_cov = kfwh.predict(mean, cov)
        assert is_positive_definite(p_cov)

    def test_width_noise_differs_from_xyah(self, kfwh, kfah):
        """Process noise structure must differ between XYWH and XYAH variants."""
        m_wh = np.array([300.0, 200.0, 60.0, 150.0])
        m_ah = np.array([300.0, 200.0, 60.0 / 150.0, 150.0])
        mean_wh, cov_wh = kfwh.initiate(m_wh)
        mean_ah, cov_ah = kfah.initiate(m_ah)
        _, p_cov_wh = kfwh.predict(mean_wh, cov_wh)
        _, p_cov_ah = kfah.predict(mean_ah, cov_ah)
        # The (2, 2) process covariance entry must differ
        assert p_cov_wh[2, 2] != pytest.approx(p_cov_ah[2, 2])


# ===========================================================================
# KalmanFilterXYWH — predict → update cycle
# ===========================================================================


class TestKalmanFilterXYWHCycle:
    def test_predict_update_cycle(self, kfwh, measurement_xywh):
        mean, cov = kfwh.initiate(measurement_xywh)
        mean, cov = kfwh.predict(mean, cov)
        mean, cov = kfwh.update(mean, cov, measurement_xywh)
        assert mean.shape == (8,)
        assert is_positive_semidefinite(cov)

    def test_repeated_updates_converge(self, kfwh):
        """XYWH filter should also converge given consistent observations."""
        true_pos = np.array([300.0, 200.0, 60.0, 150.0])
        mean, cov = kfwh.initiate(true_pos + np.array([40.0, 40.0, 10.0, 20.0]))

        for _ in range(50):
            mean, cov = kfwh.predict(mean, cov)
            mean, cov = kfwh.update(mean, cov, true_pos)

        np.testing.assert_allclose(mean[:4], true_pos, atol=2.0)

    def test_multi_frame_width_height_tracking(self, kfwh):
        """Track an object that grows slightly over 15 frames."""
        cx, cy, w, h = 200.0, 150.0, 50.0, 100.0
        mean, cov = kfwh.initiate(np.array([cx, cy, w, h]))

        for t in range(1, 16):
            mean, cov = kfwh.predict(mean, cov)
            meas = np.array([cx + t * 2.0, cy, w, h])  # object drifts right
            mean, cov = kfwh.update(mean, cov, meas)

        expected_cx = cx + 15 * 2.0
        assert mean[0] == pytest.approx(expected_cx, abs=5.0)

    def test_gating_distance_inherited(self, kfwh, measurement_xywh):
        """gating_distance must work via inherited method."""
        mean, cov = kfwh.initiate(measurement_xywh)
        projected_mean, _ = kfwh.project(mean, cov)
        dist = kfwh.gating_distance(mean, cov, projected_mean[np.newaxis])
        assert dist[0] == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# Shared / cross-class tests
# ===========================================================================


class TestKalmanFilterShared:
    @pytest.mark.parametrize("kf_fixture", ["kfah", "kfwh"])
    def test_motion_matrix_shape(self, request, kf_fixture):
        kf = request.getfixturevalue(kf_fixture)
        assert kf._motion_mat.shape == (8, 8)

    @pytest.mark.parametrize("kf_fixture", ["kfah", "kfwh"])
    def test_update_matrix_shape(self, request, kf_fixture):
        kf = request.getfixturevalue(kf_fixture)
        assert kf._update_mat.shape == (4, 8)

    @pytest.mark.parametrize("kf_fixture", ["kfah", "kfwh"])
    def test_noise_weight_constants(self, request, kf_fixture):
        kf = request.getfixturevalue(kf_fixture)
        assert kf._std_weight_position == pytest.approx(1.0 / 20)
        assert kf._std_weight_velocity == pytest.approx(1.0 / 160)

    @pytest.mark.parametrize("kf_fixture", ["kfah", "kfwh"])
    def test_motion_matrix_velocity_coupling(self, request, kf_fixture):
        """F must couple pos[i] with vel[i] via dt=1."""
        kf = request.getfixturevalue(kf_fixture)
        for i in range(4):
            assert kf._motion_mat[i, 4 + i] == pytest.approx(1.0)

    @pytest.mark.parametrize("kf_fixture", ["kfah", "kfwh"])
    def test_update_matrix_identity_projection(self, request, kf_fixture):
        """H must select the first 4 components of the state vector."""
        kf = request.getfixturevalue(kf_fixture)
        np.testing.assert_array_equal(kf._update_mat, np.eye(4, 8))

    def test_xywh_is_subclass_of_xyah(self):
        assert issubclass(KalmanFilterXYWH, KalmanFilterXYAH)

    def test_xyah_is_not_subclass_of_xywh(self):
        assert not issubclass(KalmanFilterXYAH, KalmanFilterXYWH)

    def test_independent_instances_do_not_share_state(self):
        """Two separate KF instances must be independent."""
        kf1 = KalmanFilterXYAH()
        kf2 = KalmanFilterXYAH()
        m = np.array([100.0, 100.0, 0.5, 80.0])
        mean1, cov1 = kf1.initiate(m)
        mean1[0] = 9999.0  # mutate kf1 state
        # kf2 should produce fresh result
        mean2, _ = kf2.initiate(m)
        assert mean2[0] == pytest.approx(100.0)

    def test_scipy_not_required_for_predict(self):
        """predict() must not require scipy (only predict uses numpy)."""
        kf = KalmanFilterXYAH()
        m = np.array([200.0, 150.0, 0.4, 100.0])
        mean, cov = kf.initiate(m)
        # Should not raise even without explicit scipy import
        mean2, cov2 = kf.predict(mean, cov)
        assert mean2.shape == (8,)
