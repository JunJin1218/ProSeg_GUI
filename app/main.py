from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.medsam2_inference import (
    PromptBox,
    available_model_names,
    compute_case_norm_stats,
    load_medsam2_model,
    load_sorted_volume_from_uploads,
    parse_prompts,
    predict_mask_for_prompts,
    prompts_by_class,
    resolve_model_path,
)
from app.postprocess import postprocess_mask


app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PNAMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"access-control-allow-private-network", b"true"))
            await send(message)

        await self.app(scope, receive, send_wrapper)


app.add_middleware(PNAMiddleware)


@app.get("/")
def read_root():
    return "MedSAM2 backend is running. Visit /docs for API documentation."


@app.get("/models")
def list_models():
    return {"models": available_model_names()}


def _build_slice_payload(label_map: np.ndarray, slice_index: int) -> dict:
    mask_map = label_map[slice_index].astype(np.uint8)
    height, width = mask_map.shape
    class_errors: dict[int, bool] = {}
    total_pixels = height * width
    for class_val in np.unique(mask_map):
        if class_val == 0:
            continue
        class_pixels = int(np.sum(mask_map == class_val))
        class_errors[int(class_val)] = class_pixels > 0.5 * total_pixels

    return {
        "sliceIndex": slice_index,
        "width": width,
        "height": height,
        "data": mask_map.flatten().tolist(),
        "confidences": {},
        "errors": class_errors,
    }


@app.post("/segment")
async def segment_dicoms(
    files: list[UploadFile] = File(...),
    prompts: Optional[str] = Form(default=None),
    model_name: Optional[str] = Form(default=None),
    predictor_type: str = Form(default="video"),
    device: Optional[str] = Form(default=None),
    sam2_cfg: Optional[str] = Form(default=None),
):
    try:
        prompt_items: list[PromptBox] = parse_prompts(prompts)
        if not prompt_items:
            raise ValueError("At least one bbox prompt is required.")
        print(f"/segment prompt count: {len(prompt_items)}")
        print("/segment prompts:", [item.model_dump() for item in prompt_items])

        volume, _sorted_files = await load_sorted_volume_from_uploads(files)
        num_slices, height, width = volume.shape

        model_path = resolve_model_path(model_name)
        model = load_medsam2_model(
            checkpoint_path=Path(model_path),
            predictor_type=predictor_type,
            device=device,
            sam2_cfg=sam2_cfg,
        )

        norm_stats = compute_case_norm_stats(volume)
        grouped_prompts = prompts_by_class(prompt_items, height=height, width=width)

        label_map = np.zeros((num_slices, height, width), dtype=np.uint8)
        for class_id in sorted(grouped_prompts.keys()):
            class_prompts = {
                int(np.clip(slice_idx, 0, num_slices - 1)): box
                for slice_idx, box in grouped_prompts[class_id].items()
            }
            print(f"/segment class {class_id} prompts: {class_prompts}")
            key_slice_idx = sorted(class_prompts.keys())[0]
            class_mask = predict_mask_for_prompts(
                model=model,
                volume=volume,
                prompt_boxes_by_frame=class_prompts,
                key_slice_idx=key_slice_idx,
                norm_stats=norm_stats,
            )
            class_mask, _stats = postprocess_mask(class_mask, class_id)
            print(f"/segment class {class_id} voxels after postprocess: {int(class_mask.sum())}")
            label_map[class_mask > 0] = int(class_id)

        masks = [_build_slice_payload(label_map, slice_index) for slice_index in range(num_slices)]
        print("/segment total labeled voxels:", int((label_map > 0).sum()))

        return {
            "masks": masks,
            "model": model_path.name,
            "predictorType": predictor_type,
        }
    except Exception as exc:
        print("MedSAM2 /segment failed:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
