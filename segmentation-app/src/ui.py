from functools import partial
from pathlib import Path

import gradio as gr

from . import app_state
from .handlers import *

CSS_PATH = Path(__file__).parent / "static" / "styles.css"
white_theme = gr.themes.Default().set(
    body_background_fill="#ffffff",
    background_fill_primary="#ffffff",
    background_fill_secondary="#ffffff",
    block_background_fill="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_text_color="#111111",
    button_secondary_border_color="#e5e7eb",

    # Primary color overrides
    button_primary_background_fill=app_state.cfg['app']['primary_color'],
    button_primary_border_color=app_state.cfg['app']['primary_color'],

     # hover/active/focus states
    button_primary_background_fill_hover=app_state.cfg['app']['hover_color'],   
    button_primary_border_color_hover=app_state.cfg['app']['hover_color'],

    slider_color=app_state.cfg['app']['primary_color'],
)

def load_css_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""

def init_gradio_app(use_credentials: bool = True):  
    with gr.Blocks() as demo:
        current_user = gr.State("")         # per-session user holder
        with gr.Row(elem_classes="row-spacing"):
            with gr.Column(scale=3):
                title_md = gr.Markdown("")
            with gr.Column(scale=1):
                vids_count_title = gr.Markdown("") 

        # --- login group ---
        with gr.Group(visible=use_credentials) as login_group:
            with gr.Row():
                gr.Column(scale=1) 
                with gr.Column(scale=1, min_width=250,elem_classes="login-card"):
                    u = gr.Textbox(label="Username")
                    p = gr.Textbox(label="Password", type="password")
                    login_btn = gr.Button("Sign in", variant="primary")
                    login_status = gr.Markdown("")
                gr.Column(scale=1) 

        # --- app group ---
        with gr.Group(visible=(not use_credentials)) as app_group:
            with gr.Row(elem_classes="row-spacing"):
                with gr.Column(scale=4,elem_classes="col-spacing app-card"):
                    img_comp = gr.Image(label="Frame", interactive=True, type="numpy")
                    frame_slider = gr.Slider(label="Frame", minimum=0, maximum=0, step=1, value=0)
                with gr.Column(scale=2, min_width=260,elem_classes="col-spacing app-card"):
                    with gr.Row(elem_classes="row-spacing"):
                        video_dd = gr.Dropdown(label="Select Video", choices=[])
                    status = gr.Textbox(label="Status", value="Select video", interactive=False)
                    with gr.Row(elem_classes="row-spacing button-row"):
                        pos_btn = gr.Button("Positive", variant="primary", elem_classes="equal-btn")
                        neg_btn = gr.Button("Negative", variant="secondary", elem_classes="equal-btn")
                    transparency_slider = gr.Slider(label="Mask transparency", minimum=0, maximum=100, step=5, value=30)
                    invalid_tick = gr.Checkbox(label="Points invalid", value=False)
                    with gr.Row(elem_classes="row-spacing button-row"):
                        undo_btn = gr.Button("Undo",variant="secondary",elem_classes="equal-btn")
                        reset_btn = gr.Button("Reset", variant="secondary",elem_classes="equal-btn")
                        save_btn = gr.Button('Save and next',variant="secondary", elem_classes="equal-btn")
                    video_comp = gr.Video(label="Video Preview", autoplay=True, loop=True)
        
        

        video_dd.change(partial(set_video, origin="video_dd.change"), [video_dd, current_user],[frame_slider, img_comp, status, pos_btn, neg_btn, video_comp, invalid_tick])
        img_comp.select(add_point, [frame_slider,current_user, transparency_slider], img_comp)
        pos_btn.click(toggle_positive,inputs=[current_user], outputs=[status, pos_btn, neg_btn])
        neg_btn.click(toggle_negative,inputs=[current_user], outputs=[status, pos_btn, neg_btn])
    
        undo_btn.click(undo_last_point, inputs=[frame_slider, current_user, transparency_slider], outputs=[img_comp,frame_slider])

        reset_btn.click(reset_state, inputs=[current_user], outputs=[frame_slider, img_comp, status, pos_btn, neg_btn, video_comp,invalid_tick])
        invalid_tick.change(on_tick_toggle, inputs=[invalid_tick, current_user], outputs=status)
        save_btn.click(load_next_vid, inputs=[current_user], outputs=[video_dd,vids_count_title])
        frame_slider.release(show_frame, [frame_slider, current_user], img_comp)
        transparency_slider.release(change_mask_transparency, [transparency_slider, current_user, frame_slider], img_comp)

        if use_credentials:
            login_btn.click(
                do_login,
                inputs=[u, p],
                outputs=[current_user, login_group, app_group, title_md,vids_count_title, login_status]
            ).then(
                init_on_launch,
                inputs=[current_user],
                outputs=[video_dd, frame_slider, img_comp, status, pos_btn, neg_btn, video_comp]
            ).then(
                partial(set_video, origin="login_then"),                  
                inputs=[video_dd, current_user],
                outputs=[frame_slider, img_comp, status,  pos_btn, neg_btn, video_comp,invalid_tick]
            )
        else:
            demo.load(
                auto_login,
                inputs=None,
                outputs=[current_user, title_md, vids_count_title, login_status]
            ).then(
                init_on_launch,
                inputs=[current_user],
                outputs=[video_dd, frame_slider, img_comp, status, pos_btn, neg_btn, video_comp]
            ).then(
                partial(set_video, origin="auto_load"),
                inputs=[video_dd, current_user],
                outputs=[frame_slider, img_comp, status, pos_btn, neg_btn, video_comp, invalid_tick]
            ) 

    return demo, {"theme": white_theme, "css": load_css_text(CSS_PATH)}