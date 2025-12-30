# Video Segmentation Annotation with Point Prompts Using SAM2

## Project Overview
This project enables video annotation for segmentation using SAM2 in a simplified workflow, where annotators only need to annotate a **single frame** of each video using point prompts. It provides an interactive UI based annotation interface built with Gradio that allows users to efficiently create segmentation masks.

## Notes

- A CUDA enabled GPU is required to run SAM2.
- Processing time depends on video length and dataset size (progress bar with ETA is provided).
- It is recommended to run long scripts inside a tmux session to avoid interruptions.

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

## Annotation Workflow (In-short)

1. Configure the dataset and output paths in `config.yaml`.
2. Save inference states from SAM2.
3. Annotate a single frame of each video using point prompts in the UI.
4. Automatically propagate segmentation masks across all frames.

## Annotation Workflow (Expanded)

1. **Configure `config.yaml`.**  
   Set the dataset video root path (i.e., the folder containing your videos), following the dummy data example.  
   Adjust output paths as needed for storing masks.  
   You can also customize the UI primary and secondary colors.

2. **(Optional) Enable user authentication.**  
   To enable multiple users to do annotations, with different videos assigned to each user, follow these steps:
   - Rename `.env.example` to `.env`
   - Uncomment and update the usernames and passwords, you may add as many users as needed.


3. **Save inference states.** Run the following script to generate and save inference states from SAM2.

   A typical dataset directory should follow the structure of the provided dummy dataset folder. If your dataset structure differs, for example if videos are not all in a single folder, modify the logic in `get_videos_path_csv()` (line 44) so that the resulting CSV contains the full paths to all videos.
   ```bash
   python save_inference_states.py
   ```

4. **Launch the annotation application.** Start the Gradio-based annotation interface. This will give you a shareable link, or you can use the local server running on port 8080.
    ```bash
    python segmentation_app.py
    ```
  
5. **Generate segmentation masks.** After completing all annotations, generate the final masks by running:
    ```bash
    python generate_masks.py
    ```

Masks will be saved to the directory specified in ```config.yaml```, along with an ```overlay.mp4``` file.


## Using the Annotation Interface

A short demo video or GIF illustrating the annotation workflow will be added here.

The interface allows users to add positive and negative point prompts on a selected frame, refine the segmentation mask interactively, and proceed to the next video once a satisfactory result is obtained.

## Acknowledgements
This project is built on top of the SAM2 model developed by Meta AI.  
The interactive annotation interface is implemented using Gradio.



