"""
Anomaly detection for Agencity.
"""

from __future__ import annotations
import numpy as np

EPS = 1e-12


def detect_amplitude_anomalies(b, threshold=3.0):
    """Z-score anomalies on |b|."""
    mag = np.abs(np.asarray(b))

    mean = np.mean(mag)
    std = np.std(mag)

    if std < EPS:
        return np.array([], dtype=int)

    z = np.abs((mag - mean) / std)
    return np.where(z > threshold)[0]


def detect_phase_jumps(b, threshold=np.pi/2):
    """Detect sudden phase jumps."""
    theta = np.unwrap(np.angle(b))
    dtheta = np.diff(theta)

    return np.where(np.abs(dtheta) > threshold)[0]


def detect_energy_spikes(b, factor=5.0):
    """Detect spikes in energy."""
    energy = np.abs(b)**2
    mean = np.mean(energy)

    return np.where(energy > factor * mean)[0]


def anomaly_summary(b):
    return {
        "amplitude_anomalies": detect_amplitude_anomalies(b).tolist(),
        "phase_jumps": detect_phase_jumps(b).tolist(),
        "energy_spikes": detect_energy_spikes(b).tolist(),
    }