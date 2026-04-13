"""Dual-reference first-order variable admittance (paper Eqs. 13–17)."""
import numpy as np


class DualReferenceAdmittance:
    def __init__(self, dt=0.02, gamma_min=0.06, gamma_max=0.14, k_min=2.0, k_max=100.0):
        if not (dt > 0 and 0 < gamma_min <= gamma_max and 0 < k_min <= k_max):
            raise ValueError("Positive sampling, damping and stiffness parameters are required.")
        self.dt, self.gamma_min, self.gamma_max = dt, gamma_min, gamma_max
        self.k_min, self.k_max = k_min, k_max
        self.reset()

    def reset(self, initial_angle=None):
        self.command = np.zeros(3) if initial_angle is None else np.asarray(initial_angle, dtype=float).copy()

    def update(self, intended_angle, intended_force, external_force):
        angle, force, external = [np.asarray(x, dtype=float) for x in (intended_angle, intended_force, external_force)]
        if any(x.shape != (3,) or not np.isfinite(x).all() for x in (angle, force, external)):
            raise ValueError("Each input must be a finite three-dimensional vector.")
        force = np.clip(force, 0, 1)
        gamma = self.gamma_min + (self.gamma_max-self.gamma_min) * force
        stiffness = self.k_min + (self.k_max-self.k_min) * force
        beta = gamma / (gamma+self.dt)
        target = angle + external / stiffness
        self.command = beta * self.command + (1-beta) * target
        return self.command.copy()
