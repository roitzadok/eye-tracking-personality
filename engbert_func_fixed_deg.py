#!/usr/bin/env python


from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
from overrides import override
from scipy.signal import savgol_filter  # kept for parity with original header

# -----------------------------------------------------------------------------
# Minimal stubs & constants (unchanged)
# -----------------------------------------------------------------------------

class EventLabelEnum:
    FIXATION = "fixation"
    SACCADE = "saccade"
    UNDEFINED = "undefined"


class Config:
    EPSILON = 1e-10


cnfg = Config()


class Constants:
    X = "x"
    Y = "y"


cnst = Constants()


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def _px_to_deg(arr: np.ndarray, pixel_size_cm: float, viewer_distance_cm: float) -> np.ndarray:
    """Convert pixel coordinates to degrees of visual angle (vectorised)."""
    scale = (pixel_size_cm / viewer_distance_cm) * 57.29577951308232  # 180/pi
    return arr * scale


# -----------------------------------------------------------------------------
# Base class (unchanged except for type hints)
# -----------------------------------------------------------------------------

class BaseDetector(ABC):
    def __init__(self, missing_value: float, min_event_duration: float, pad_blinks_ms: float, name: str | None = None):
        self._missing_value = missing_value
        self._min_event_duration = min_event_duration
        self._pad_blinks_ms = pad_blinks_ms
        self._metadata: Dict[str, float] = {}
        self._sr: float = np.nan  # sampling rate – must be filled in by user
        self._name = name or self.__class__.__name__

    @abstractmethod
    def _detect_impl(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        viewer_distance_cm: float,
        pixel_size_cm: float,
    ) -> np.ndarray:
        ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        *,
        viewer_distance_cm: float,
        pixel_size_cm: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        t, x, y = self._reshape_vectors(t, x, y)
        labels = np.full_like(t, EventLabelEnum.UNDEFINED, dtype=object)
        detected_labels = self._detect_impl(t, x, y, labels, viewer_distance_cm, pixel_size_cm)
        return detected_labels, copy.deepcopy(self._metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reshape_vectors(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = t.reshape(-1)
        x = x.reshape(-1)
        y = y.reshape(-1)
        if len(t) != len(x) or len(t) != len(y):
            raise ValueError("t, x and y must be the same length")
        return t, x, y


# -----------------------------------------------------------------------------
# EngbertDetector implementation
# -----------------------------------------------------------------------------

class EngbertDetector(BaseDetector):
    """Velocity‑based microsaccade/saccade detector (Engbert & Kliegl, 2003)."""

    _DEFAULT_LAMBDA_PARAM = 5
    _DEFAULT_DERIVATION_WINDOW_SIZE = 5

    __THRESHOLD_VELOCITY_STR = "threshold_velocity"
    __LAMBDA_PARAM_STR = "lambda_param"
    __DERIV_WINDOW_SIZE_STR = "deriv_window_size"

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(
        self,
        missing_value: float,
        min_event_duration: float,
        pad_blinks_ms: float,
        *,
        name: str | None = None,
        lambda_param: float = _DEFAULT_LAMBDA_PARAM,
        deriv_window_size: int = _DEFAULT_DERIVATION_WINDOW_SIZE,
    ) -> None:
        super().__init__(missing_value, min_event_duration, pad_blinks_ms, name)
        if lambda_param <= 0:
            raise ValueError("lambda_param must be > 0")
        if deriv_window_size <= 0:
            raise ValueError("deriv_window_size must be > 0")
        self._lambda_param = lambda_param
        self._deriv_window_size = deriv_window_size

    @classmethod
    def get_default_params(cls) -> Dict[str, float]:
        return {
            cls.__LAMBDA_PARAM_STR: cls._DEFAULT_LAMBDA_PARAM,
            cls.__DERIV_WINDOW_SIZE_STR: cls._DEFAULT_DERIVATION_WINDOW_SIZE,
        }

    # ------------------------------------------------------------------
    # Core detection logic
    # ------------------------------------------------------------------

    @override
    def _detect_impl(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        viewer_distance_cm: float,
        pixel_size_cm: float,
    ) -> np.ndarray:
        labels = np.asarray(copy.deepcopy(labels), dtype=object)

        # 1. Convert from px → deg so that thresholds are in deg/s.
        x_deg = _px_to_deg(x, pixel_size_cm, viewer_distance_cm)
        y_deg = _px_to_deg(y, pixel_size_cm, viewer_distance_cm)

        # 2. Compute velocities (deg/s) along each axis.
        x_velocity = self._axial_velocities(x_deg, self._sr, self._deriv_window_size)
        y_velocity = self._axial_velocities(y_deg, self._sr, self._deriv_window_size)

        # 3. Robust SDs & thresholds.
        x_thresh = self._median_standard_deviation(x_velocity) * self._lambda_param
        y_thresh = self._median_standard_deviation(y_velocity) * self._lambda_param

        # 4. Elliptic criterion.
        ellipse = (x_velocity / x_thresh) ** 2 + (y_velocity / y_thresh) ** 2
        labels[ellipse < 1] = EventLabelEnum.FIXATION
        labels[ellipse >= 1] = EventLabelEnum.SACCADE

        # 5. Store metadata (deg/s).
        self._metadata.update(
            {
                f"{cnst.X}_{self.__THRESHOLD_VELOCITY_STR}_deg_s": x_thresh,
                f"{cnst.Y}_{self.__THRESHOLD_VELOCITY_STR}_deg_s": y_thresh,
            }
        )
        return labels

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def lambda_param(self) -> float:
        return self._lambda_param

    @property
    def deriv_window_size(self) -> int:
        return self._deriv_window_size

    # ------------------------------------------------------------------
    # Internals – velocity & robust SD
    # ------------------------------------------------------------------

    @staticmethod
    def _axial_velocities(arr: np.ndarray, sr: float, window_size: int) -> np.ndarray:
        """Central‑difference velocity estimate (deg/s) over `window_size` samples."""
        if not (sr > 0 and np.isfinite(sr)):
            raise ValueError("Sampling rate must be positive and finite")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        half_ws = window_size // 2 if window_size % 2 == 0 else window_size // 2 + 1
        vel = np.full_like(arr, np.nan, dtype=float)
        for idx in range(half_ws, len(arr) - half_ws):
            sum_before = np.sum(arr[idx - half_ws : idx])
            sum_after = np.sum(arr[idx + 1 : idx + half_ws + 1])
            diff = sum_after - sum_before
            vel[idx] = diff * sr / window_size
        return vel

    @staticmethod
    def _median_standard_deviation(arr: np.ndarray) -> float:
        """Median‑based SD (unchanged, keeps original minus‑median² formulation)."""
        squared_median = np.power(np.nanmedian(arr), 2)
        median_of_squares = np.nanmedian(np.power(arr, 2))
        sd = np.sqrt(median_of_squares - squared_median)
        return float(np.nanmax([sd, cnfg.EPSILON]))

    # ------------------------------------------------------------------
    # Vector length sanity
    # ------------------------------------------------------------------

    @override
    def _reshape_vectors(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t, x, y = super()._reshape_vectors(t, x, y)
        if len(x) < 2 * self._deriv_window_size:
            raise ValueError("Derivation window is longer than half the data length")
        return t, x, y


# -----------------------------------------------------------------------------
# Quick self‑test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic 1‑s 1000 Hz trace.
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 10 * t) * 100 + 500  # px
    y = np.cos(2 * np.pi * 10 * t) * 100 + 500  # px

    detector = EngbertDetector(missing_value=np.nan, min_event_duration=50, pad_blinks_ms=0)
    detector._sr = 1000  # Hz

    labels, meta = detector.detect(t, x, y, viewer_distance_cm=60, pixel_size_cm=0.024)

    print("Event labels (first 20):", labels[:20])
    print("Metadata:", meta)