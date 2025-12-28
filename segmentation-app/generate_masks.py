import os
import glob
from pathlib import Path
import json
import logging

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from sam2.build_sam import build_sam2_video_predictor
from src.helpers import load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

set_seed(42)

# ---- Load config ----
cfg = load_config("config.yaml")

MASKS_DIR = Path(cfg["data"]["masks_dir"])
VIDEOS_DIR = Path(cfg["data"]["videos_dir"])
FRAMES_DIR = Path(cfg["data"]["frames_dir"])
STATES_PATH = Path(cfg["data"]["inference_states_dir"])
ANNOTATIONS_OUTPUT_DIR = Path(cfg["data"]["annotations_dir"])
CHECKPOINT = cfg["sam2_model_path"]["checkpoint"]
MODEL_CFG = cfg["sam2_model_path"]["config"]
VIDEOS_DF_PATH = Path(cfg["data"]["videos_df_path"])

VIDEOS_DF = pd.read_csv(VIDEOS_DF_PATH)

def list_frames(frame_dir: Path):
    """
    Return a sorted list of frame filepaths and a map: idx -> path.
    Only keep common image file extensions.
    """
    patterns = ["*.*"]
    candidates = sorted(glob.glob(str(frame_dir / patterns[0])))
    allowed = {".png", ".jpg", ".jpeg", ".bmp"}
    frame_paths = [p for p in candidates if Path(p).suffix.lower() in allowed]
    idx_to_path = {i: p for i, p in enumerate(frame_paths)}
    return frame_paths, idx_to_path

def make_mask_dict(height: int, width: int, idx_to_path: dict):
    """Create a dict of black (False) masks for all frame indices."""
    base = np.zeros((height, width), dtype=bool)
    return {i: base.copy() for i in idx_to_path.keys()}

def overlay_mask_on_frame(frame_bgr: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.45):
    """
    Overlay a single-channel boolean mask onto a BGR frame in-place and return the blended frame.
    The mask is drawn as a red overlay (red channel = 255).
    """
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    color = np.zeros_like(frame_bgr)
    color[..., 2] = 255  # red in BGR
    selector = cv2.merge([mask_u8, mask_u8, mask_u8]) > 0
    blended = frame_bgr.copy()
    blended[selector] = cv2.addWeighted(frame_bgr, 1 - alpha, color, alpha, 0)[selector]
    return blended

def write_video(frames_bgr: list, out_path: str, fps: int = 25):
    """Write a list of BGR frames to an mp4 using imageio (converting to RGB)."""
    if not frames_bgr:
        raise ValueError("No frames to write.")

    frames_rgb = [f[..., ::-1] for f in frames_bgr]  # BGR -> RGB

    iio.imwrite(
        out_path,
        frames_rgb,
        fps=fps,
        codec="libx264",
        quality=8,            # lower = smaller file, higher = better quality
        pixelformat="yuv420p"  # ensures player compatibility
    )


def process_annotation_file(json_annotation_file_path: Path):
    """Process a single annotations JSON file and produce overlay videos and mask files."""
    with open(json_annotation_file_path, "r") as f:
        data = json.load(f)

    # Reformat points into per-video -> per-frame structure
    reformatted = {}
    for vid_name, vid_info in data.items():
        if not vid_info.get("points_valid", False):
            logger.info(f"Points are invalid for: {vid_name}, skipping")
            continue
        reformatted[vid_name] = {}
        for point, label, frame_num in zip(
            vid_info.get("selected_points", []),
            vid_info.get("selected_labels", []),
            vid_info.get("selected_frames", []),
        ):
            if frame_num not in reformatted[vid_name]:
                reformatted[vid_name][frame_num] = {"points": [], "labels": []}
            reformatted[vid_name][frame_num]["points"].append(point)
            reformatted[vid_name][frame_num]["labels"].append(label)

    # Process each video in the annotation file
    for vid_name, frames in tqdm(reformatted.items(), desc="Videos", leave=False):
        # Get video path from VIDEOS_DF
        try:
            video_path = VIDEOS_DF.loc[VIDEOS_DF["video_name"] == vid_name, "video_path"].iloc[0]
        except Exception as e:
            logger.warning(f"Could not find video path {vid_name}, skipping.")
            continue


        frame_dir = (Path(FRAMES_DIR) / Path(video_path).relative_to(VIDEOS_DIR)).with_suffix("")
        mask_dir = (Path(MASKS_DIR) / Path(video_path).relative_to(VIDEOS_DIR)).with_suffix("")
        state_file_path = (Path(STATES_PATH) / Path(video_path).relative_to(VIDEOS_DIR)).with_suffix(".pt")

        os.makedirs(mask_dir, exist_ok=True)

        frame_paths, idx_to_path = list_frames(frame_dir)
        if not frame_paths:
            logger.warning(f"No frames found in  {frame_dir}, skipping.")
            continue
        first = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
        if first is None:
            logger.warning(f"Could not read first frame in {frame_dir}, skipping.")
            continue

        H, W = first.shape[:2]
        merged_masks = make_mask_dict(H, W, idx_to_path)

        out_video_path = os.path.join(mask_dir, "overlay.mp4")
        #if overlay exists, skip
        if os.path.exists(out_video_path):
            continue

        # Build the SAM predictor 
        sam_model = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT)

        # Load inference state if available (original script continued on failure)
        try:
            inference_state = torch.load(state_file_path, weights_only=True)
        except:
            logger.warning(f"Could not loaf inference state for {vid_name}, skipping.")
            continue

        # Process frames that have points
        for frame_num, content in frames.items():
            points = np.array(content["points"])
            labels = np.array(content["labels"])
            if len(points) == 0:
                continue
            
            # Propagate masks from the selected frame through the video
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                new_points = []
                new_labels = []
                for i in range(0, len(points)):
                    new_points.append(points[i])
                    new_labels.append(labels[i])
                    _, obj_ids, logits = sam_model.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_num,
                        obj_id=0,
                        points=np.array(new_points),
                        labels=np.array(new_labels),
                    )

                for f_idx, obj_ids, logits in sam_model.propagate_in_video(inference_state, start_frame_idx=frame_num):
                    pred = (logits[0] > 0).squeeze().cpu().numpy().astype(bool)
                    merged_masks[f_idx] |= pred

        # Write out masks and create overlaid frames
        for idx, frame_path in idx_to_path.items():
            mask_bool = merged_masks[idx]
            mask_u8 = (mask_bool.astype(np.uint8) * 255)
            out_name = Path(frame_path).stem + ".png"
            out_path = mask_dir / out_name
            cv2.imwrite(str(out_path), mask_u8)

        overlaid_frames = []
        for idx in range(len(idx_to_path)):
            frame = cv2.imread(idx_to_path[idx], cv2.IMREAD_COLOR)
            if frame is None:
                # keep lengths consistent; skip overlay for missing frame (preserves original behavior)
                logger.warning(f" Could not read {idx_to_path[idx]}, skipping overlay for this frame.")
                continue
            over = overlay_mask_on_frame(frame, merged_masks[idx], alpha=0.45)
            overlaid_frames.append(over)

        # Write the overlay video (same fps as original)
        try:
            write_video(overlaid_frames, out_video_path, fps=25)
            logger.info(f"Saved overlay video to: {out_video_path}")
        except Exception as e:
            logger.warn(f"Failed to write video: {out_video_path}: {e}")

if __name__ == "__main__":
    annotations_json_files = sorted(os.listdir(ANNOTATIONS_OUTPUT_DIR))
    for annotations_file in annotations_json_files:
        json_annotation_file_path = ANNOTATIONS_OUTPUT_DIR / annotations_file
        process_annotation_file(json_annotation_file_path)