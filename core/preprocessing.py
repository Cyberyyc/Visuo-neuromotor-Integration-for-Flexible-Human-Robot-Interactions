"""Causal sEMG preprocessing and end-aligned observation windows."""
import numpy as np
from scipy.signal import butter, iirnotch, lfilter, sosfilt


def mvc_normalize(signal, rest, mvc):
    signal, rest, mvc = map(lambda x: np.asarray(x, dtype=np.float64), (signal, rest, mvc))
    scale = mvc - rest
    if np.any(scale <= 0):
        raise ValueError("MVC calibration must exceed resting calibration for every channel.")
    return (signal - rest) / scale


def filter_emg(signal, fs=1000, low_hz=20, high_hz=499, notch_hz=50):
    """Input (samples, channels), with no future-sample filtering."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or not 0 < low_hz < high_hz < fs/2:
        raise ValueError("Expected a 2-D signal and a passband strictly below Nyquist.")
    filtered = sosfilt(butter(4, [low_hz, high_hz], fs=fs, btype='bandpass', output='sos'), signal, axis=0)
    b, a = iirnotch(notch_hz, 30, fs=fs)
    return lfilter(b, a, filtered, axis=0)


def aligned_windows(emg, labels, emg_fs=1000, label_fs=200, window=200, stride=20):
    """Return (N, 8, 200) windows and their last-time-point six-dimensional labels."""
    emg, labels = np.asarray(emg), np.asarray(labels)
    if emg.ndim != 2 or labels.ndim != 2 or emg.shape[1] != 8 or labels.shape[1] != 6:
        raise ValueError("Expected EMG (T, 8) and labels (U, 6).")
    if min(emg_fs, label_fs, window, stride) <= 0 or len(emg) < window:
        raise ValueError("Invalid sampling parameters or insufficient EMG samples.")
    ends = np.arange(window-1, len(emg), stride)
    label_indices = np.floor(ends * label_fs / emg_fs).astype(int)
    valid = label_indices < len(labels)
    ends, label_indices = ends[valid], label_indices[valid]
    if not len(ends):
        raise ValueError("No aligned windows are available.")
    windows = np.stack([emg[end-window+1:end+1].T for end in ends])
    return windows.astype(np.float32), labels[label_indices].astype(np.float32)
