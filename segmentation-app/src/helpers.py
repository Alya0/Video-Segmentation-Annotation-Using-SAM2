import glob
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import random
import yaml
from omegaconf import OmegaConf
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        logger.error("Config file %s not found.", path)
        raise FileNotFoundError(path)
    with p.open("r") as f:
        return OmegaConf.load("config.yaml")

def load_user_credentials(env_var: str = "USER_CREDENTIALS") -> Tuple[Dict[str, str], bool]:
    """Return (credentials_dict, used_real_credentials_bool)."""
    load_dotenv(find_dotenv(), override=True)
    raw = os.environ.get(env_var, "{}")
    try:
        creds = json.loads(raw)
        if not isinstance(creds, dict) or len(creds) == 0:
            logger.warning("No valid USER_CREDENTIALS found; falling back to single local annotator.")
            return ({"local_annotator": "local"}, False)
        return (creds, True)
    except json.JSONDecodeError:
        logger.warning("Could not parse USER_CREDENTIALS; falling back to single local annotator.")
        return ({"local_annotator": "local"}, False)

@dataclass
class AnnotatorState:
    idx: int
    user_key: str
    videos_df: pd.DataFrame
    selected_points: List = field(default_factory=list)
    selected_labels: List = field(default_factory=list)
    selected_frames: List = field(default_factory=list)
    selected_masks: Dict = field(default_factory=dict)
    cur_label_val: float = 1.0
    points_are_valid: bool = True
    current_video: Any = None
    frames: List = field(default_factory=list)
    sam_model: Any = None
    inference_state: Any = None

def make_annotator_states(user_keys: List[str], videos_df: pd.DataFrame, video_col: str = "video_path") -> List[AnnotatorState]:
    states = []
    for i, key in enumerate(user_keys):
        current_vid_df = videos_df[videos_df['annotator'] == i]
        states.append(AnnotatorState(idx=i, user_key=key, videos_df=current_vid_df))
    return states


# helpers for handlers
def draw_points(img_arr, frame_idx, points, labels,frames_idxs, radius=5):
    img = img_arr.copy()
    h, w = img.shape[:2]
    if len(points) == 0:
        return img
    for (x, y), lab, f in zip(points, labels, frames_idxs):
        if f != frame_idx:
            continue
        x = int(round(x)); y = int(round(y))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if lab == 0:
            color = (255, 0, 0)
        elif lab == 1:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.circle(img, (x, y), 4, color, thickness=-1, lineType=cv2.LINE_AA)
    return img

def overlay_mask(img, mask, alpha=0.8):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    color = np.array([0, 255, 0], dtype=np.uint8)
    overlay = img.copy()
    overlay[mask] = (alpha * overlay[mask] + (1 - alpha) * color).astype(np.uint8)
    return overlay
    
def load_frames(frame_dir, max_workers=os.cpu_count()):
    paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    def _read(p):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None: 
            return None
        return im[..., ::-1]               

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        imgs = list(ex.map(_read, paths))

    keep = [(p, im) for p, im in zip(paths, imgs) if im is not None]
    if not keep:
        return [], []

    paths, frames = zip(*keep)  # tuples
    return list(paths), list(frames)