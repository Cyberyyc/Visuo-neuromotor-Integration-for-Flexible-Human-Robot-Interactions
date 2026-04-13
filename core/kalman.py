"""Feature-space Kalman decoding with sparse visual-anchor calibration."""
from dataclasses import dataclass
import numpy as np


def state_from_labels(labels, dt=0.02):
    """[angle(3), force(3)] -> [angle, angle_rate, force, force_rate]."""
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] != 6 or len(labels) < 2 or dt <= 0:
        raise ValueError("Expected an ordered (N, 6) label sequence and positive dt.")
    rate = np.vstack([np.zeros((1, 6)), np.diff(labels, axis=0) / dt])
    return np.column_stack([labels[:, :3], rate[:, :3], labels[:, 3:], rate[:, 3:]])


@dataclass
class KalmanParameters:
    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    def save(self, path):
        np.savez(path, A=self.A, C=self.C, Q=self.Q, R=self.R)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{name: data[name] for name in ('A', 'C', 'Q', 'R')})


def fit_parameters(feature_sequences, label_sequences, dt=0.02, variance_floor=1e-6):
    """Stage II: least-squares dynamics and observations from ordered training sequences."""
    if len(feature_sequences) != len(label_sequences) or not len(label_sequences):
        raise ValueError("Provide paired feature and label sequences.")
    states, features = [], []
    for y, labels in zip(feature_sequences, label_sequences):
        x = state_from_labels(labels, dt)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or len(y) != len(x) or not np.isfinite(y).all() or not np.isfinite(x).all():
            raise ValueError("Each feature sequence must align with finite physical states.")
        states.append(x)
        features.append(y)
    before = np.concatenate([x[:-1] for x in states])
    after = np.concatenate([x[1:] for x in states])
    A = np.zeros((12, 12))
    for block in (slice(0, 6), slice(6, 12)):
        A[block, block] = np.linalg.lstsq(before[:, block], after[:, block], rcond=None)[0].T
    X, Y = np.concatenate(states), np.concatenate(features)
    C = np.linalg.lstsq(X, Y, rcond=None)[0].T
    process_error = after - before @ A.T
    observation_error = Y - X @ C.T
    Q = np.diag(np.maximum(np.var(process_error, axis=0), variance_floor))
    R = np.diag(np.maximum(np.var(observation_error, axis=0), variance_floor))
    return KalmanParameters(A, C, Q, R)


class AdaptiveKalmanDecoder:
    """12-D latent state; output is three joint angles and three forces."""
    def __init__(self, parameters, alpha=0.05, initial_state=None):
        self.parameters = parameters
        if parameters.A.shape != (12, 12) or parameters.C.shape[1] != 12:
            raise ValueError("Expected 12-D kinematic/dynamic state matrices.")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self.reset(initial_state)

    def reset(self, state=None):
        self.state = np.zeros(12) if state is None else np.asarray(state, dtype=float).copy()
        if self.state.shape != (12,):
            raise ValueError("Initial state must have 12 entries.")
        self.covariance = np.eye(12)
        self.bias = np.zeros(self.parameters.C.shape[0])

    def update(self, observation, visual_anchor=None):
        p = self.parameters
        y = np.asarray(observation, dtype=float)
        if y.shape != self.bias.shape or not np.isfinite(y).all():
            raise ValueError("Observation must be a finite latent feature vector.")
        predicted = p.A @ self.state
        predicted_cov = p.A @ self.covariance @ p.A.T + p.Q
        if visual_anchor is not None:
            anchor = np.asarray(visual_anchor, dtype=float)
            if anchor.shape != (3,) or not np.isfinite(anchor).all():
                raise ValueError("Visual anchor must contain three normalized joint angles.")
            reference = predicted.copy()
            reference[:3], reference[3:6] = anchor, 0.0
            residual = y - p.C @ reference
            self.bias = (1-self.alpha) * self.bias + self.alpha * residual
        innovation_cov = p.C @ predicted_cov @ p.C.T + p.R
        gain = np.linalg.solve(innovation_cov, p.C @ predicted_cov).T
        self.state = predicted + gain @ (y - self.bias - p.C @ predicted)
        correction = np.eye(12) - gain @ p.C
        self.covariance = correction @ predicted_cov @ correction.T + gain @ p.R @ gain.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return np.concatenate([self.state[:3], self.state[6:9]])
