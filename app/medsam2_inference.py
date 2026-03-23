from __future__ import annotations

import json
import os
import sys
import tempfile
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pydicom
from fastapi import UploadFile
from PIL import Image
from pydantic import BaseModel


class PromptBox(BaseModel):
    classId: int
    sliceIndex: int
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class NormStats:
    lo: float
    hi: float
    min_v: float
    max_v: float


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "MedSAM2_models"
DEFAULT_MEDSAM2_REPO = PROJECT_ROOT.parent / "MedSAM2"
_MODEL_CACHE: dict[tuple[str, str, str | None, str | None], Any] = {}


def resolve_sam2_config_path(sam2_cfg: Optional[str]) -> str:
    cfg = sam2_cfg or os.getenv("MEDSAM2_SAM2_CONFIG", "configs/sam2.1_hiera_t512.yaml")
    return cfg[len("sam2/") :] if cfg.startswith("sam2/") else cfg


def _build_predictor_from_training_config(
    repo_path: Path,
    checkpoint_path: str,
    resolved_device: str,
):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    train_cfg_path = repo_path / "sam2" / "configs" / "prostate_train.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_cfg_path}")

    cfg = OmegaConf.load(train_cfg_path)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.trainer.model, resolve=True))
    model_cfg["_target_"] = "sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ"
    model_cfg["binarize_mask_from_pts_for_mem_enc"] = True
    model_cfg["fill_hole_area"] = 8
    model_cfg["sam_mask_decoder_extra_args"] = {
        "dynamic_multimask_via_stability": True,
        "dynamic_multimask_stability_delta": 0.05,
        "dynamic_multimask_stability_thresh": 0.98,
    }

    model = instantiate(model_cfg, _recursive_=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys[:10]}")
    if missing_keys:
        print(f"Warning: missing checkpoint keys: {missing_keys[:10]}")
    model = model.to(resolved_device)
    model.eval()
    return model


def available_model_names() -> list[str]:
    if not DEFAULT_MODEL_DIR.exists():
        return []
    return sorted(
        path.name
        for path in DEFAULT_MODEL_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".pt", ".pth"}
    )


def resolve_model_path(model_name: Optional[str] = None) -> Path:
    if model_name:
        candidate = DEFAULT_MODEL_DIR / model_name
        if candidate.exists():
            return candidate
        candidate = Path(model_name)
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Model not found: {model_name}")

    env_model = os.getenv("MEDSAM2_CHECKPOINT")
    if env_model:
        candidate = Path(env_model)
        if candidate.exists():
            return candidate.resolve()

    models = available_model_names()
    if models:
        return (DEFAULT_MODEL_DIR / models[0]).resolve()

    fallback = PROJECT_ROOT / "checkpoint_10_MedSAM2_US_Heart.pt"
    if fallback.exists():
        return fallback.resolve()

    raise FileNotFoundError(
        "No MedSAM2 checkpoint found. Add a .pt file under MedSAM2_models/ or set MEDSAM2_CHECKPOINT."
    )


def parse_prompts(payload: Optional[str]) -> list[PromptBox]:
    if not payload:
        return []
    raw = json.loads(payload)
    if not isinstance(raw, list):
        raise ValueError("prompts must be a JSON array")
    return [PromptBox.model_validate(item) for item in raw]


async def load_sorted_volume_from_uploads(files: list[UploadFile]) -> tuple[np.ndarray, list[bytes]]:
    parsed_files: list[tuple[int, str, bytes]] = []
    for file in files:
        file_bytes = await file.read()
        ds = pydicom.dcmread(BytesIO(file_bytes), stop_before_pixels=True)
        instance_num = int(getattr(ds, "InstanceNumber", 0))
        parsed_files.append((instance_num, file.filename or "", file_bytes))

    parsed_files.sort(key=lambda item: (item[0], item[1]))
    slices = [load_dicom_slice(item[2]) for item in parsed_files]
    if not slices:
        raise ValueError("No readable DICOM slices were uploaded.")
    volume = np.stack(slices, axis=0)
    return volume, [item[2] for item in parsed_files]


def load_dicom_slice(file_bytes: bytes) -> np.ndarray:
    ds = pydicom.dcmread(BytesIO(file_bytes))
    pixel_array = ds.pixel_array.astype(np.float32)

    if "RescaleSlope" in ds and "RescaleIntercept" in ds:
        pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    return pixel_array


def compute_case_norm_stats(
    volume: np.ndarray,
) -> NormStats:
    vol = volume.astype(np.float32)
    lo, hi = np.percentile(vol, (10.0, 90.0))
    vol_clip = np.clip(vol, lo, hi)
    return NormStats(
        lo=float(lo),
        hi=float(hi),
        min_v=float(vol_clip.min()),
        max_v=float(vol_clip.max()),
    )


def to_rgb_uint8(image: np.ndarray, norm_stats: Optional[NormStats] = None) -> np.ndarray:
    if image.ndim == 3 and image.shape[-1] == 3:
        return image.astype(np.uint8)

    img = image.astype(np.float32)
    if norm_stats is None:
        lo, hi = np.percentile(img, (10.0, 90.0))
        clipped = np.clip(img, lo, hi)
        min_v = float(clipped.min())
        max_v = float(clipped.max())
    else:
        lo, hi = norm_stats.lo, norm_stats.hi
        min_v, max_v = norm_stats.min_v, norm_stats.max_v

    img = np.clip(img, lo, hi)
    if max_v > min_v:
        img = (img - min_v) / (max_v - min_v)
    else:
        img = img * 0.0

    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def save_volume_as_jpg_sequence(
    volume: np.ndarray,
    dir_path: str,
    norm_stats: Optional[NormStats] = None,
) -> None:
    for z in range(volume.shape[0]):
        rgb = to_rgb_uint8(volume[z], norm_stats=norm_stats)
        Image.fromarray(rgb).save(os.path.join(dir_path, f"{z:05d}.jpg"), quality=95)


def resize_grayscale_to_rgb_and_resize(array: np.ndarray, image_size: int) -> np.ndarray:
    depth, _, _ = array.shape
    resized_array = np.zeros((depth, 3, image_size, image_size), dtype=np.float32)
    for index in range(depth):
        img_pil = Image.fromarray(array[index].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size), resample=Image.BILINEAR)
        resized_array[index] = np.asarray(img_resized, dtype=np.float32).transpose(2, 0, 1)
    return resized_array


def preprocess_volume_to_tensor(
    volume: np.ndarray,
    image_size: int,
    device: str,
    norm_stats: NormStats,
):
    import torch

    normalized = np.stack(
        [to_rgb_uint8(slice_2d, norm_stats=norm_stats)[:, :, 0] for slice_2d in volume],
        axis=0,
    )
    height, width = normalized.shape[1:]
    if height != image_size or width != image_size:
        img_resized = resize_grayscale_to_rgb_and_resize(normalized, image_size)
    else:
        img_resized = np.repeat(normalized[:, None, :, :], 3, axis=1).astype(np.float32)

    img_resized = img_resized / 255.0
    img_t = torch.from_numpy(img_resized).to(device)
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=device)[:, None, None]
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=device)[:, None, None]
    img_t = (img_t - img_mean) / img_std
    return img_t


def load_medsam2_model(
    checkpoint_path: Path,
    predictor_type: str = "video",
    device: Optional[str] = None,
    sam2_cfg: Optional[str] = None,
):
    import torch

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (str(checkpoint_path), predictor_type, resolved_device, sam2_cfg)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_path = str(checkpoint_path)
    checkpoint = torch.load(model_path, map_location="cpu")

    if hasattr(checkpoint, "eval"):
        model = checkpoint.to(resolved_device)
        model.eval()
        _MODEL_CACHE[cache_key] = model
        return model

    state_dict = None
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

    repo_path = Path(os.getenv("MEDSAM2_REPO", str(DEFAULT_MEDSAM2_REPO))).resolve()
    if repo_path.exists() and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        from training.utils.train_utils import register_omegaconf_resolvers

        register_omegaconf_resolvers()
    except Exception:
        pass

    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model_cfg = resolve_sam2_config_path(sam2_cfg)

    predictor_type = str(predictor_type or "video").lower()
    if predictor_type == "video":
        video_errors: list[str] = []
        try:
            from sam2.build_sam import build_sam2_video_predictor_npz

            model = build_sam2_video_predictor_npz(
                model_cfg,
                ckpt_path=model_path,
                device=resolved_device,
            )
            _MODEL_CACHE[cache_key] = model
            return model
        except Exception as exc:
            video_errors.append(f"build_sam2_video_predictor_npz failed: {exc!r}")
        try:
            model = _build_predictor_from_training_config(
                repo_path=repo_path,
                checkpoint_path=model_path,
                resolved_device=resolved_device,
            )
            _MODEL_CACHE[cache_key] = model
            return model
        except Exception as exc:
            video_errors.append(f"prostate_train-derived predictor failed: {exc!r}")
        try:
            model = build_sam2_video_predictor(model_cfg, model_path, device=resolved_device)
            _MODEL_CACHE[cache_key] = model
            return model
        except Exception as exc:
            video_errors.append(f"build_sam2_video_predictor failed: {exc!r}")
            raise RuntimeError(" ; ".join(video_errors)) from exc

    try:
        sam2_model = build_sam2(model_cfg, model_path)
    except Exception:
        sam2_model = build_sam2(model_cfg, None)
        if state_dict is None:
            raise ValueError("Checkpoint did not contain a loadable state_dict.")
        sam2_model.load_state_dict(state_dict, strict=False)

    sam2_model.to(resolved_device)
    sam2_model.eval()
    model = SAM2ImagePredictor(sam2_model)
    _MODEL_CACHE[cache_key] = model
    return model


def sanitize_box(box: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    ax1, ay1, ax2, ay2 = box
    x1 = int(np.clip(min(ax1, ax2), 0, w - 1))
    x2 = int(np.clip(max(ax1, ax2), 0, w - 1))
    y1 = int(np.clip(min(ay1, ay2), 0, h - 1))
    y2 = int(np.clip(max(ay1, ay2), 0, h - 1))
    return x1, y1, x2, y2


def prompts_by_class(prompts: list[PromptBox], height: int, width: int) -> dict[int, dict[int, tuple[int, int, int, int]]]:
    grouped: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    for prompt in prompts:
        class_id = int(prompt.classId)
        slice_index = int(prompt.sliceIndex)
        box = sanitize_box((prompt.x1, prompt.y1, prompt.x2, prompt.y2), h=height, w=width)
        grouped.setdefault(class_id, {})[slice_index] = box
    return grouped


def _extract_mask2d_from_video_logits(video_res_masks: Any) -> np.ndarray:
    if hasattr(video_res_masks, "detach"):
        arr = video_res_masks.detach().cpu().numpy()
    else:
        arr = np.asarray(video_res_masks)
    if arr.ndim >= 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0]
    return (np.asarray(arr) > 0).astype(np.uint8)


def predict_mask_for_prompts(
    model: Any,
    volume: np.ndarray,
    prompt_boxes_by_frame: dict[int, tuple[int, int, int, int]],
    key_slice_idx: int,
    norm_stats: NormStats,
) -> np.ndarray:
    if not prompt_boxes_by_frame:
        return np.zeros(volume.shape, dtype=np.uint8)

    sorted_prompt_frames = sorted(int(k) for k in prompt_boxes_by_frame.keys())

    if hasattr(model, "init_state") and hasattr(model, "add_new_points_or_box") and hasattr(model, "propagate_in_video"):
        mask3d = np.zeros(volume.shape, dtype=np.uint8)
        try:
            image_size = int(getattr(model, "image_size", 512))
            model_device = str(getattr(model, "device", "cpu"))
            images_t = preprocess_volume_to_tensor(
                volume=volume,
                image_size=image_size,
                device=model_device,
                norm_stats=norm_stats,
            )
            inference_state = model.init_state(images_t, volume.shape[1], volume.shape[2])
        except TypeError:
            with tempfile.TemporaryDirectory(prefix="sam2_frames_") as frames_dir:
                save_volume_as_jpg_sequence(volume, frames_dir, norm_stats=norm_stats)
                inference_state = model.init_state(
                    video_path=frames_dir,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    async_loading_frames=False,
                )

        for frame_idx in sorted_prompt_frames:
            box_arr = np.array(prompt_boxes_by_frame[frame_idx], dtype=np.float32)
            _, _, seed_masks = model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=int(frame_idx),
                obj_id=1,
                box=box_arr,
            )
            mask3d[int(frame_idx)] = _extract_mask2d_from_video_logits(seed_masks)
        for frame_idx, _obj_ids, video_res_masks in model.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=int(min(sorted_prompt_frames)),
            reverse=False,
        ):
            mask3d[int(frame_idx)] = _extract_mask2d_from_video_logits(video_res_masks)
        for frame_idx, _obj_ids, video_res_masks in model.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=int(max(sorted_prompt_frames)),
            reverse=True,
        ):
            mask3d[int(frame_idx)] = _extract_mask2d_from_video_logits(video_res_masks)
        if hasattr(model, "reset_state"):
            model.reset_state(inference_state)
        return mask3d

    if hasattr(model, "set_image") and hasattr(model, "predict"):
        mask3d = np.zeros(volume.shape, dtype=np.uint8)
        for z in range(volume.shape[0]):
            nearest = min(sorted_prompt_frames, key=lambda frame: abs(frame - z))
            box_arr = np.array(prompt_boxes_by_frame[int(nearest)], dtype=np.float32)
            image = to_rgb_uint8(volume[z], norm_stats=norm_stats)
            model.set_image(image)
            masks, _, _ = model.predict(
                point_coords=None,
                point_labels=None,
                box=box_arr,
                multimask_output=False,
            )
            mask3d[z] = (masks[0] > 0).astype(np.uint8)
        return mask3d

    if hasattr(model, "predict"):
        ref_frame = int(key_slice_idx if key_slice_idx in prompt_boxes_by_frame else sorted_prompt_frames[0])
        box = prompt_boxes_by_frame[ref_frame]
        try:
            mask = model.predict(volume=volume, box=box, key_slice_idx=ref_frame)
        except TypeError:
            mask = model.predict(volume, box, ref_frame)
        return (np.asarray(mask) > 0).astype(np.uint8)

    if callable(model):
        ref_frame = int(key_slice_idx if key_slice_idx in prompt_boxes_by_frame else sorted_prompt_frames[0])
        mask = model(volume, prompt_boxes_by_frame[ref_frame], ref_frame)
        return (np.asarray(mask) > 0).astype(np.uint8)

    raise TypeError("Loaded model is not callable and has no compatible predict() method.")
