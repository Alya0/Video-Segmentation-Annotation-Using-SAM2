from pathlib import Path

from .helpers import load_config

# ---- Load config ----
cfg=load_config("config.yaml")

VIDEOS_DIR = Path(cfg["data"]["videos_dir"])
FRAMES_DIR = Path(cfg["data"]["frames_dir"])
MASKS_BY_ANOT_DIR = Path(cfg['data']['masks_by_annotators_dir'])
ANNOTATIONS_OUTPUT_DIR = Path(cfg['data']['annotations_dir'])
STATES_PATH = Path(cfg["data"]["inference_states_dir"])
VIDEOS_DF_PATH = Path(cfg['data']['videos_df_path'])
CHECKPOINT = cfg["sam2_model_path"]["checkpoint"]
MODEL_CFG = cfg["sam2_model_path"]["config"]

# ---- Annotator states ----
annotator_states = [] 

# ---- Users ----
USER_CREDENTIALS, USE_CREDENTIALS = None, None
USER_TO_IDX = None 

# ---- videos df ---- 
VIDEOS_DF = None