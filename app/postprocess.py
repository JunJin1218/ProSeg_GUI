from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage


DEFAULT_POSTPROCESS_CONFIG: dict[int, dict[str, Any]] = {
    1: {"keep_largest_k": 1, "min_component_size": 400, "closing_iters": 1, "fill_holes": True, "contour_sigma": 0.9},
    2: {"keep_largest_k": 1, "min_component_size": 400, "closing_iters": 1, "fill_holes": True, "contour_sigma": 0.9},
    3: {"keep_largest_k": 2, "min_component_size": 250, "opening_iters": 1, "closing_iters": 1, "contour_sigma": 1.1},
    4: {"keep_largest_k": 1, "min_component_size": 800, "closing_iters": 2, "fill_holes": True, "contour_sigma": 1.2},
    5: {"keep_largest_k": 1, "min_component_size": 300, "closing_iters": 1, "contour_sigma": 1.0},
    6: {"keep_largest_k": 3, "min_component_size": 50, "opening_iters": 1, "closing_iters": 1, "contour_sigma": 0.9},
}


def postprocess_mask(
    mask: np.ndarray,
    class_idx: int,
    config_map: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg_all = config_map or DEFAULT_POSTPROCESS_CONFIG
    cfg = dict(cfg_all.get(int(class_idx), {}))

    mask_bool = np.asarray(mask) > 0
    stats = {"enabled": bool(cfg), "voxels_before": int(mask_bool.sum())}
    if not cfg or stats["voxels_before"] == 0:
        stats["voxels_after"] = stats["voxels_before"]
        return mask_bool.astype(np.uint8), stats

    if mask_bool.ndim == 3:
        out = np.zeros_like(mask_bool, dtype=bool)
        for z in range(mask_bool.shape[0]):
            out[z] = _postprocess_single_mask(mask_bool[z], cfg)
        mask_bool = out
    else:
        mask_bool = _postprocess_single_mask(mask_bool, cfg)

    stats["voxels_after"] = int(mask_bool.sum())
    stats["config"] = cfg
    stats["mode"] = "2d-slicewise"
    return mask_bool.astype(np.uint8), stats


def _structure(ndim: int) -> np.ndarray:
    return ndimage.generate_binary_structure(ndim, 1)


def _postprocess_single_mask(mask_bool: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    if cfg.get("fill_holes"):
        mask_bool = ndimage.binary_fill_holes(mask_bool)

    opening_iters = int(cfg.get("opening_iters", 0) or 0)
    if opening_iters > 0:
        mask_bool = ndimage.binary_opening(
            mask_bool,
            structure=_structure(mask_bool.ndim),
            iterations=opening_iters,
        )

    closing_iters = int(cfg.get("closing_iters", 0) or 0)
    if closing_iters > 0:
        mask_bool = ndimage.binary_closing(
            mask_bool,
            structure=_structure(mask_bool.ndim),
            iterations=closing_iters,
        )

    min_component_size = int(cfg.get("min_component_size", 0) or 0)
    if min_component_size > 0:
        mask_bool = _remove_small_components(mask_bool, min_component_size)

    keep_largest_k = int(cfg.get("keep_largest_k", 0) or 0)
    if keep_largest_k > 0:
        mask_bool = _keep_largest_components(mask_bool, keep_largest_k)

    contour_sigma = float(cfg.get("contour_sigma", 0.0) or 0.0)
    if contour_sigma > 0:
        mask_bool = _smooth_contour(mask_bool, sigma=contour_sigma)

    if cfg.get("fill_holes"):
        mask_bool = ndimage.binary_fill_holes(mask_bool)

    return mask_bool


def _remove_small_components(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    labeled, num = ndimage.label(mask, structure=_structure(mask.ndim))
    if num == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    keep_labels = {i + 1 for i, size in enumerate(np.asarray(sizes)) if int(size) >= int(min_component_size)}
    if not keep_labels:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, list(keep_labels))


def _keep_largest_components(mask: np.ndarray, k: int) -> np.ndarray:
    labeled, num = ndimage.label(mask, structure=_structure(mask.ndim))
    if num == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    ranked = sorted(((int(size), i + 1) for i, size in enumerate(np.asarray(sizes))), reverse=True)
    keep_labels = [label for _size, label in ranked[:k]]
    return np.isin(labeled, keep_labels)


def _smooth_contour(mask: np.ndarray, sigma: float) -> np.ndarray:
    if not np.any(mask):
        return mask
    if mask.ndim == 2:
        return _smooth_contour_2d(mask, sigma)
    if mask.ndim == 3:
        out = np.zeros_like(mask, dtype=bool)
        for z in range(mask.shape[0]):
            if np.any(mask[z]):
                out[z] = _smooth_contour_2d(mask[z], sigma)
        return out
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    signed_distance = inside - outside
    signed_distance = ndimage.gaussian_filter(signed_distance.astype(np.float32), sigma=float(sigma))
    return signed_distance > 0.0


def _smooth_contour_2d(mask2d: np.ndarray, sigma: float) -> np.ndarray:
    inside = ndimage.distance_transform_edt(mask2d)
    outside = ndimage.distance_transform_edt(~mask2d)
    signed_distance = inside - outside
    signed_distance = ndimage.gaussian_filter(signed_distance.astype(np.float32), sigma=float(sigma))
    return signed_distance > 0.0
