import pandas as pd
from src.helpers import *                
from src.ui import init_gradio_app
import src.app_state as app_state  
# ---- Seed ----
set_seed(42)

# ---- Load users ----
app_state.USER_CREDENTIALS, app_state.USE_CREDENTIALS = load_user_credentials()
app_state.USER_TO_IDX = {user: i for i, user in enumerate(app_state.USER_CREDENTIALS.keys())}
NUM_ANNOTATORS = len(app_state.USER_TO_IDX)

# ---- Load annotator states ----
app_state.VIDEOS_DF = pd.read_csv(app_state.VIDEOS_DF_PATH)
app_state.VIDEOS_DF['annotator']=[i % NUM_ANNOTATORS for i in range(len(app_state.VIDEOS_DF))]
app_state.VIDEOS_DF.to_csv(app_state.VIDEOS_DF_PATH, index=False)
app_state.annotator_states = make_annotator_states(list(app_state.USER_TO_IDX.keys()), app_state.VIDEOS_DF, video_col='video_path')

if __name__ == "__main__":
    demo, launch_kwargs = init_gradio_app(app_state.USE_CREDENTIALS)
    demo.launch(
        **launch_kwargs,
        server_name="0.0.0.0",
        server_port=8080,
        allowed_paths=[app_state.FRAMES_DIR, app_state.VIDEOS_DIR, app_state.MASKS_BY_ANOT_DIR],
        share=True,
    )

