import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import cv2

from sam2.build_sam import build_sam2_video_predictor

from .helpers import draw_points, load_frames, overlay_mask
from . import app_state

def get_mask(frame_idx, user_idx):
    st = app_state.annotator_states[user_idx]

    points = np.array(st.selected_points, dtype=np.float32)
    labels = np.array(st.selected_labels, dtype=np.int32)
    frames = np.array(st.selected_frames)
    mask = (frames == frame_idx)
    labels = labels[mask]
    points = points[mask]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, obj_ids, logits = st.sam_model.add_new_points_or_box(
            inference_state=st.inference_state,
            frame_idx=frame_idx,
            obj_id=0,
            points=points,
            labels=labels
        )

    mask = (logits[0] > 0).squeeze().cpu().numpy()
    st.selected_masks[frame_idx] = mask

    return mask

def get_user(user):
    if isinstance(user, int):
        return user
    return app_state.USER_TO_IDX[user]

def reset_annotator_state(user_idx):
    st = app_state.annotator_states[user_idx]
    st.selected_points.clear()
    st.selected_labels.clear()
    st.selected_frames.clear()
    st.selected_masks.clear()
    st.cur_label_val = 1.0
    st.points_are_valid = True
    st.sam_model = build_sam2_video_predictor(app_state.MODEL_CFG, app_state.CHECKPOINT)
    st.inference_state = None

def set_video(video, current_user, origin="(unknown)"):
    if video is None or current_user is None:
        return (
            gr.update(maximum=0, value=0),
            None,
            "Select patient and video",
            gr.update(variant="primary"),   
            gr.update(variant="secondary"),
            None,
            gr.update(value = False)
        )
    user_idx=get_user(current_user)

    # reset per-annotator state
    reset_annotator_state(user_idx)

    # set current video in annotator state
    st = app_state.annotator_states[user_idx]
    st.current_video = video
    
    video_path = app_state.VIDEOS_DF.loc[app_state.VIDEOS_DF["video_name"] == video, "video_path"].iloc[0]

    # load frames
    frame_dir = (Path(app_state.FRAMES_DIR) / Path(video_path).relative_to(app_state.VIDEOS_DIR)).with_suffix("")
    _, frames = load_frames(frame_dir)
    st.frames = frames

    state_file_path = (Path(app_state.STATES_PATH) / Path(video_path).relative_to(app_state.VIDEOS_DIR)).with_suffix(".pt")

    try:
        st.inference_state = torch.load(state_file_path,weights_only=True)
    except Exception as e:
        st.inference_state =  st.sam_model.init_state(video_path=frame_dir)

    return (gr.update(maximum=len(st.frames)-1, value=0),
            st.frames[0], 
            "Mode: Positive clicks",
            gr.update(variant="primary"), 
            gr.update(variant="secondary"),
            video_path,
            gr.update(value = False))

def reset_state(current_user):
    user_idx= get_user(current_user)

    reset_annotator_state(user_idx)

    st=app_state.annotator_states[user_idx]

    # reset model initialization
    video = st.current_video 
    video_path = app_state.VIDEOS_DF.loc[app_state.VIDEOS_DF["video_name"] == video, "video_path"].iloc[0]
    frame_dir = (Path(app_state.FRAMES_DIR) / Path(video_path).relative_to(app_state.VIDEOS_DIR)).with_suffix("")
    state_file_path = (Path(app_state.STATES_PATH) / Path(video_path).relative_to(app_state.VIDEOS_DIR)).with_suffix(".pt")
    try:
        st.inference_state = torch.load(state_file_path,weights_only=True)
    except Exception as e:
        st.inference_state =  st.sam_model.init_state(video_path=frame_dir)
    
    return (gr.update(maximum=len(st.frames) - 1, value=0),
            st.frames[0],
            "State reset. Mode: Positive clicks",
            gr.update(variant="primary"),
            gr.update(variant="secondary"),
            gr.update(),
            gr.update(value=False))      

def show_frame(idx, current_user):
    user_idx= get_user(current_user)
    st=app_state.annotator_states[user_idx]
    if len(st.frames) == 0:
        return None
    return draw_points(st.frames[idx], idx, st.selected_points, st.selected_labels, st.selected_frames)

def add_point(frame_idx, current_user, transparency_slider, evt: gr.SelectData):

    user_idx= get_user(current_user)
    st=app_state.annotator_states[user_idx]
    
    x, y = evt.index
    st.selected_frames.append(frame_idx)
    st.selected_points.append([x, y])
    st.selected_labels.append(st.cur_label_val)

    mask = get_mask(frame_idx, user_idx)

    alpha = (100-transparency_slider)/100
    overlay = overlay_mask(st.frames[frame_idx], mask,alpha)

    overlay_with_points = draw_points(overlay,frame_idx, st.selected_points, st.selected_labels,st.selected_frames)

    return overlay_with_points

def set_mode(label_value, current_user):
    # 0 for negatives, 1 for positives,
    user_idx = get_user(current_user)
    st = app_state.annotator_states[user_idx]
    st.cur_label_val = label_value
    if label_value == 0:
        status_text = "Mode: Negative clicks"
        pos_update = gr.update(variant="secondary")
        neg_update = gr.update(variant="primary")
    elif label_value == 1:
        status_text = "Mode: Positive clicks"
        pos_update = gr.update(variant="primary")
        neg_update = gr.update(variant="secondary")
    return status_text, pos_update, neg_update

def toggle_negative(current_user):
    return set_mode(0,current_user)

def toggle_positive(current_user):
    return set_mode(1,current_user)

def undo_last_point(frame_idx, current_user, transparency_slider):

    user_idx= get_user(current_user)
    st=app_state.annotator_states[user_idx]

    if len(st.selected_frames) == 0:
        # if already nothing was pressed
        return st.frames[frame_idx],frame_idx
    
    frame_idx = st.selected_frames[-1]

    if st.selected_points:
        st.selected_points.pop()
        st.selected_labels.pop()
        st.selected_frames.pop()
    if len(st.selected_points):
        mask = get_mask(frame_idx, user_idx)
        alpha = (100-transparency_slider)/100
        overlay = overlay_mask(st.frames[frame_idx], mask,alpha)
        overlay_with_points = draw_points(overlay,frame_idx, st.selected_points, st.selected_labels,st.selected_frames)
    else:
        overlay_with_points = st.frames[frame_idx]

    return overlay_with_points, frame_idx
    
def save_annotations(user_idx, st):

    annotation_file_path = app_state.ANNOTATIONS_OUTPUT_DIR / f"annotations_{user_idx}.json"
    os.makedirs(app_state.ANNOTATIONS_OUTPUT_DIR, exist_ok=True)

    data = {}
    if os.path.exists(annotation_file_path):
        with open(annotation_file_path, "r") as f:
            data = json.load(f)

    video = st.current_video
    data[video] = {
        "points_valid" : st.points_are_valid,
        "selected_points": st.selected_points,
        "selected_labels": st.selected_labels,
        "selected_frames": st.selected_frames,
    }

    with open(annotation_file_path, "w") as f:
        json.dump(data, f, indent=2)
    
    # save the masks
    video_path = app_state.VIDEOS_DF.loc[app_state.VIDEOS_DF["video_name"] == video, "video_path"].iloc[0]
    masks_out_dir = (Path(app_state.MASKS_BY_ANOT_DIR) / Path(video_path).relative_to(app_state.VIDEOS_DIR)).with_suffix("")
    os.makedirs(masks_out_dir, exist_ok=True)

    for frame_idx, mask in st.selected_masks.items():
        cv2.imwrite(str(masks_out_dir / f"{frame_idx+1:05d}.png"), mask.astype(np.uint8) * 255)

    return f"Saved {video} with {len(st.selected_points)} points"

def load_next_vid(current_user):

    user_idx= get_user(current_user)

    st = app_state.annotator_states[user_idx]

    # mark it as annotated
    video = st.current_video
    app_state.VIDEOS_DF.loc[app_state.VIDEOS_DF["video_name"] == video, "annotated"] = True
    app_state.VIDEOS_DF.to_csv(app_state.VIDEOS_DF_PATH, index=False)

    # save the annotations and reset state
    save_annotations(user_idx, st)
    
    videos=app_state.VIDEOS_DF[(app_state.VIDEOS_DF["annotator"] == user_idx) & (~app_state.VIDEOS_DF["annotated"])]['video_name'].tolist()

    if len(videos) != 0:
        next_video = videos[0]
        video_update = gr.update(choices=videos, value=next_video, interactive=True)
        number_of_vids_done = app_state.VIDEOS_DF[(app_state.VIDEOS_DF["annotator"] == user_idx) & (app_state.VIDEOS_DF["annotated"])].shape[0]
        return video_update, f'## Number of videos completed: {number_of_vids_done}'

    return gr.update(choices=[], value=None), f'## All videos are done'

def init_on_launch(current_user):
    if not current_user:
        return (
            gr.update(choices=[], value=None),        # video_dd
            gr.update(maximum=0, value=0),            # frame_slider
            None,                                     # img_comp
            "Select patient and video",               # status
            gr.update(variant="primary"),             # pos_btn
            gr.update(variant="secondary"),           # neg_btn
            None                                      # video_comp
        )

    user_idx= get_user(current_user)
    videos = app_state.VIDEOS_DF[(app_state.VIDEOS_DF["annotator"] == user_idx) & (~app_state.VIDEOS_DF["annotated"])]['video_name'].tolist()
    if not videos:
        return (
            gr.update(choices=[], value=None),        # video_dd
            gr.update(maximum=0, value=0),            # frame_slider
            None,                                     # img_comp
            "All videos have been annotated",         # status
            gr.update(variant="primary"),             # pos_btn
            gr.update(variant="secondary"),           # neg_btn
            None                                      # video_comp
        )

    video_update = gr.update(choices=videos, value=videos[0], interactive=True)

    return (video_update,
            gr.update(maximum=0, value=0),            # frame_slider
            None,                                     # img_comp
            "Select patient and video",               # status
            gr.update(variant="primary"),             # pos_btn
            gr.update(variant="secondary"),           # neg_btn
            None )

def verify_user(username, password):
    return app_state.USER_CREDENTIALS.get(username) == password

def do_login(username, password):
        if verify_user(username, password):
            user_idx = get_user(username)
            st=app_state.annotator_states[user_idx]
            number_of_vids_done = app_state.VIDEOS_DF[(app_state.VIDEOS_DF["annotator"] == user_idx) & (app_state.VIDEOS_DF["annotated"])].shape[0]
            return (
                username,                         
                gr.update(visible=False),         
                gr.update(visible=True),           
                f"## Video Segmentation With Sam2 – user:{username}",
                f'## Number of videos completed: {number_of_vids_done}',
                "✅ Signed in.",
            )
        else:
            return gr.update(), gr.update(visible=True), gr.update(visible=False), gr.update(),gr.update(), "❌ Invalid credentials."

def auto_login():
    default_user = "local_annotator"
    number_of_vids_done = app_state.VIDEOS_DF[app_state.VIDEOS_DF["annotated"]].shape[0]
    return (
        default_user,   
        f"## Video Segmentation With Sam2",       
        f'## Number of videos completed: {number_of_vids_done}',        
        ""                          
    )

def on_tick_toggle(value, current_user):
    user_idx = get_user(current_user)
    app_state.annotator_states[user_idx].points_are_valid = (not value) 
    status = "Flagged points as valid"
    if value:
        status = "Flagged points as Invalid"
    return status

def change_mask_transparency(transparency_slider, current_user, frame_idx):
    user_idx = get_user(current_user)
    st = app_state.annotator_states[user_idx]

    mask = st.selected_masks[frame_idx]

    alpha = (100 - transparency_slider) / 100
    overlay = overlay_mask(st.frames[frame_idx], mask, alpha)

    overlay_with_points = draw_points(overlay,frame_idx, st.selected_points, st.selected_labels, st.selected_frames)
    return overlay_with_points
