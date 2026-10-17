"""MiDiffusion/ThreedFront floor-plan sampling and Scale.scale normalization.

Sampling adapted from MIT-SPARK/ThreedFront at
de929ef80e1678bf95bb0369afa6e0d854b291ad:
threed_front/datasets/floorplan_utils.py (itself adapted from LEGO-Net).
The optional RNG makes the preprocessing draw reproducible per scene.

BSD 2-Clause License
Copyright (c) 2024, MIT-SPARK

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from scipy.interpolate import interp1d


def clockwise_polygon(points):
    """Adapt metric XZ corners to upstream's clockwise, non-repeated contour."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Room polygon must contain finite XZ coordinates')
    # Remove repeated endpoints/consecutive corners without changing the frame.
    points = points[np.any(points != np.roll(points, 1, axis=0), axis=1)]
    if len(points) < 3:
        raise ValueError('Room polygon has fewer than three distinct corners')
    following = np.roll(points, -1, axis=0)
    area = np.sum(points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1])
    if abs(area) < 1e-8:
        raise ValueError('Room polygon has zero area')
    if area > 0:
        # Keep the starting corner while reversing traversal.
        points = np.concatenate([points[:1], points[:0:-1]], axis=0)
    return points


def fp_line_normal(fpoc):
    """Upstream unit normals, pointing inward for clockwise contours."""
    fp_line_n = np.zeros((fpoc.shape[0], 2))
    for i in range(fpoc.shape[0]):
        line_vec = fpoc[(i + 1) % fpoc.shape[0]] - fpoc[i]
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            raise ValueError('Room polygon has a zero-length edge')
        fp_line_n[i, 0] = line_vec[1] / line_len
        fp_line_n[i, 1] = -line_vec[0] / line_len
    return fp_line_n


def scene_sample_fpbp(fpoc, num_sampled_points=256, rng=None):
    """Upstream stratified random perimeter sampling in physical coordinates."""
    if num_sampled_points <= 0:
        raise ValueError('num_sampled_points must be positive')
    rng = np.random if rng is None else rng
    x = np.append(fpoc[:, 0], [fpoc[0, 0]])
    y = np.append(fpoc[:, 1], [fpoc[0, 1]])
    fp_line_n = fp_line_normal(fpoc)
    dist_bins = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2
                                 + np.ediff1d(y, to_begin=0)**2))
    dist_bins = dist_bins / dist_bins[-1]
    fx, fy = interp1d(dist_bins, x), interp1d(dist_bins, y)
    seg_len = float(1) / num_sampled_points
    seg_starts = np.linspace(0, 1, num_sampled_points + 1)[:-1]
    per_seg_displacement = rng.uniform(low=0.0, high=seg_len, size=num_sampled_points)
    sampled_distance = seg_starts + per_seg_displacement
    fpbp = np.stack([fx(sampled_distance), fy(sampled_distance)], axis=1)
    bin_idx = np.digitize(sampled_distance, dist_bins) - 1
    return np.concatenate([fpbp, fp_line_n[bin_idx]], axis=1)


def validate_bounds_fpbpn(bounds):
    bounds = np.asarray(bounds, dtype=np.float32)
    if (bounds.shape != (8,) or not np.isfinite(bounds).all()
            or not np.all(bounds[4:] > bounds[:4])
            or not np.array_equal(bounds[[2, 3, 6, 7]], [-1, -1, 1, 1])):
        raise ValueError('Invalid bounds_fpbpn: expected [xmin,zmin,-1,-1,xmax,zmax,1,1]')
    return bounds


def scale_fpbpn(points, bounds):
    """Exact upstream Scale.scale: clip, per-axis min/max, then map to [-1,1]."""
    minimum, maximum = np.split(validate_bounds_fpbpn(bounds), 2)
    X = points.astype(np.float32)
    X = np.clip(X, minimum, maximum)
    X = ((X - minimum) / (maximum - minimum))
    X = 2 * X - 1
    return X
