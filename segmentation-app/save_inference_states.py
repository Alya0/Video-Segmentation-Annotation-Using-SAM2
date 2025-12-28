import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
import torch
import time
import yaml
from tqdm import tqdm

from src.helpers import load_config
from sam2.build_sam import build_sam2_video_predictor

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---- Load config ----

cfg = load_config("config.yaml")

VIDEOS_DIR = Path(cfg["data"]["videos_dir"])
FRAMES_DIR = Path(cfg["data"]["frames_dir"])
INFERENCE_STATE_OUT = Path(cfg["data"]["inference_states_dir"])
VIDEOS_DF_PATH = Path(cfg['data']['videos_df_path'])
CHECKPOINT = cfg["sam2_model_path"]["checkpoint"]
MODEL_CFG = cfg["sam2_model_path"]["config"]



def extract_frames(video_path: Path, frames_out_dir: Path) -> None:
    """Extract frames from a video using ffmpeg into frames_out_dir (creates dir)."""
    os.makedirs(frames_out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "auto", "-i", str(video_path),
        "-map", "0:v:0", "-vsync", "0",
        "-q:v", "2", "-threads", "0", "-n",
        str(frames_out_dir / "%05d.jpg")
    ]
    subprocess.run(cmd, check=False)


def get_videos_path_csv() -> pd.DataFrame:
    """Return a DataFrame with video paths (video_idx, video_path) and write a CSV next to VIDEOS_DIR."""
    # Change logic depending on how the videos are in the database
    videos = os.listdir(VIDEOS_DIR)
    data = [{'video_idx': i,'video_name':os.path.basename(video) ,'video_path':os.path.join(VIDEOS_DIR, video), 'annotated': False} for i, video in enumerate(videos)]
    df = pd.DataFrame(data)
    df.to_csv(VIDEOS_DF_PATH, index=False)
    logging.info("Saved video paths CSV to %s (rows=%d)", VIDEOS_DF_PATH, len(df))
    return df

def main() -> None:
    videos_df = get_videos_path_csv()
    
    for _ ,row in tqdm(videos_df.iterrows(), desc="Extracting Inference States", total=len(videos_df)):
        current_video_path = row.video_path
        output_state_file_path = (Path(INFERENCE_STATE_OUT) / Path(current_video_path).relative_to(VIDEOS_DIR)).with_suffix(".pt")

        if os.path.exists(output_state_file_path):
            logging.info("%s already has a state file", current_video_path)
            continue

        frames_out_dir= (Path(FRAMES_DIR) / Path(current_video_path).relative_to(VIDEOS_DIR)).with_suffix("")
        extract_frames(current_video_path, frames_out_dir)

        sam_model = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT)
        inference_state = sam_model.init_state(video_path=str(frames_out_dir))

        # ensure parent folder exists before saving
        output_state_file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(inference_state, output_state_file_path)


if __name__ == "__main__":
    main()