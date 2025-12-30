# Video-Segmentation-Annotation-Using-SAM2

## Installation
Make sure you have **Python installed** (recommended: Python 3.8+).  
A **CUDA-enabled GPU** with CUDA properly installed is also required, as the SAM2 model relies on GPU support.

Install all required libraries and create the Conda environment for the project by running:
```bash
chmod +x setup_env.sh
bash setup_env.sh
conda activate segmentation_app
cd segmentation-app
```
## Running the annotation

1. **Configure `config.yaml`.**  
   Set the dataset video root path (i.e., the folder containing your videos), following the dummy data example.  
   Adjust output paths as needed for storing masks.  
   You can also customize the UI primary and secondary colors.

2. **(Optional) Enable user authentication.**  
   To enable multiple users with passwords:
   - Rename `.env.example` to `.env`
   - Uncomment and update the usernames and passwords  
   You may add as many users as needed.


3. **Save inference states.** Run the following script to generate and save inference states:
   ```bash
   python save_inference_states.py
   ```
   If your dataset structure differs (e.g., videos are not all in a single folder), modify the logic in get_videos_path_csv() (line 44) so that the resulting CSV contains the full paths to all videos.

4. **Launch the annotation application.** Start the Gradio-based annotation interface:
    ```bash
    python segmentation_app.py
    ```
   This will generate a shareable link (valid for one week). It is recommended to run this inside a tmux session.

5. **Generate segmentation masks.** After completing all annotations, generate the final masks by running:
    ```bash
    python generate_masks.py
    ```

Masks will be saved to the directory specified in ```config.yaml```, along with an ```overlay.mp4``` file.


## Using the Annotation Interface.



