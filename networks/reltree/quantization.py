"""Fixed scalar bins; optional explicit train-only calibration, no fitting by gradient."""

from dataclasses import asdict, dataclass

import numpy as np

from .geometry import wrap_yaw


ATTRIBUTES = ("x", "y", "z", "yaw", "log_sx", "log_sy", "log_sz")


@dataclass(frozen=True)
class QuantizationConfig:
    # Positive log ratios are allowed: OBJECT and parent axes can differ.
    ranges: tuple[tuple[float, float], ...] = ((-.5, .5), (-.5, .5), (-.5, .5),
                                             (-np.pi, np.pi), (-8., 2.), (-8., 2.), (-8., 2.))
    bins: tuple[int, ...] = (256, 256, 256, 255, 256, 256, 256)
    source: str = "fixed"
    calibration_samples: int = 0

    def __post_init__(self):
        ranges = np.asarray(self.ranges, dtype=float)
        bins = np.asarray(self.bins)
        if ranges.shape != (7, 2) or not np.isfinite(ranges).all() or np.any(ranges[:, 0] >= ranges[:, 1]):
            raise ValueError("ranges must contain seven finite (min, max) intervals")
        if bins.shape != (7,) or bins.dtype.kind not in "iu" or np.any(bins < 2):
            raise ValueError("bins must contain seven integers >= 2")
        if not np.array_equal(ranges[3], [-np.pi, np.pi]):
            raise ValueError("yaw range must be [-pi, pi)")
        if self.source not in ("fixed", "train"):
            raise ValueError("quantization source must be fixed or train")
        object.__setattr__(self, "ranges", tuple(tuple(float(x) for x in r) for r in ranges))
        object.__setattr__(self, "bins", tuple(int(x) for x in bins))

    def to_dict(self):
        return asdict(self)


class UniformQuantizer:
    def __init__(self, config: QuantizationConfig | None = None):
        self.config = config or QuantizationConfig()
        self.lower, self.upper = np.asarray(self.config.ranges).T
        self.bins = np.asarray(self.config.bins, dtype=np.int64)

    def encode(self, relative):
        values = np.array(relative, dtype=float, copy=True)
        if values.ndim < 1 or values.shape[-1] != 7 or not np.isfinite(values).all():
            raise ValueError("relative geometry must be finite with final dimension seven")
        values[..., 3] = wrap_yaw(values[..., 3])
        clipped = (values < self.lower) | (values > self.upper)
        clipped[..., 3] = False
        unit = np.clip((values - self.lower) / (self.upper - self.lower), 0, 1)
        codes = np.minimum(self.bins - 1, np.floor(unit * self.bins).astype(np.int64))
        return codes, clipped

    def decode(self, codes):
        codes = np.asarray(codes)
        if codes.ndim < 1 or codes.shape[-1] != 7 or codes.dtype.kind not in "iu":
            raise ValueError("codes must be integer arrays with final dimension seven")
        if np.any(codes < 0) or np.any(codes >= self.bins):
            raise ValueError("quantized code outside configured vocabulary")
        values = self.lower + (codes + .5) / self.bins * (self.upper - self.lower)
        values[..., 3] = wrap_yaw(values[..., 3])
        return values

    def calibrate(self, relative, *, split: str, quantiles=(0.0, 1.0), margin: float = .01):
        """Return a new quantizer. Caller must supply only training-set edges.

        The explicit split guard prevents accidental use of named val/test inputs;
        it cannot inspect the provenance of an arbitrary NumPy array.
        """
        if split != "train":
            raise ValueError("range calibration is permitted only on split='train'")
        values = np.asarray(relative, dtype=float)
        if values.ndim != 2 or values.shape[1] != 7 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("calibration requires a nonempty finite [edges, 7] array")
        if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
            raise ValueError("invalid calibration quantiles")
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError("calibration margin must be positive")
        lo, hi = np.quantile(values, quantiles, axis=0)
        padding = np.maximum((hi - lo) * margin, 1e-6)
        ranges = np.stack((lo - padding, hi + padding), axis=1)
        ranges[3] = [-np.pi, np.pi]
        return UniformQuantizer(QuantizationConfig(ranges, self.config.bins, "train", len(values)))
