# Image and Video Processing Framework with TensorRT and Object Tracking

A Python-based framework for building real-time image and video processing pipelines, featuring object detection using NVIDIA TensorRT and object tracking with SORT.

## Features

*   Modular design for easy extension.
*   TensorRT integration for high-performance inference (requires user-provided `.engine` model).
*   Object tracking with SORT or DeepSORT algorithms.
*   Advanced object tracking with DeepSORT, utilizing appearance features from ReID models.
*   ReID feature extraction module (for use with DeepSORT).
*   Pipeline for processing both single images and video streams.
*   Basic visualization of detections and tracks.
*   Example usage script to demonstrate functionality.

## Directory Structure

-   `src/`: Contains the core source code.
    -   `io/`: Image and video input/output utilities.
    -   `tensorrt_utils/`: TensorRT detector implementation.
    -   `reid/`: Contains the ReID feature extraction logic.
        -   `feature_extractor.py`: `ReIDFeatureExtractor` class.
    -   `tracking/`: SORT tracker implementation.
        -   `deepsort_tracker.py`: `DeepSORTTracker` class.
    -   `pipeline/`: Main processing pipeline.
    -   `utils/`: Visualization utilities.
        -   `metrics.py`: Cosine distance and other similarity metrics.
-   `examples/`: Example scripts and dummy data.
    -   `run_pipeline.py`: Demonstrates how to use the processing pipeline.
    -   `temp_data/`: Temporary directory for example inputs/outputs (gitignored).
-   `models/`: Directory to store TensorRT `.engine` model files (gitignored by default, requires user to add their models).
-   `data/`: Directory for sample input images/videos (gitignored by default, requires user to add their data).
-   `tests/`: Unit tests (to be developed further).

## Setup and Installation

### Prerequisites

*   Python 3.7+
*   OpenCV (`opencv-python`)
*   NumPy (`numpy`)
*   SciPy (`scipy`)
*   **NVIDIA TensorRT**: Essential for the detection module. Installation is platform-specific and typically involves downloading from the [NVIDIA Developer website](https://developer.nvidia.com/tensorrt). Ensure it's correctly installed and configured in your environment.
*   **ReID Model**: If using DeepSORT, a ReID model (converted to TensorRT `.engine` format) is also required.
*   **NVIDIA CUDA and cuDNN**: Required by TensorRT.
*   **(Optional) PyCUDA**: May be needed for more direct CUDA interop by TensorRT or custom layers.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    The `requirements.txt` file lists primary Python packages.
    ```bash
    pip install -r requirements.txt
    ```
    **Note on TensorRT and PyCUDA**: `tensorrt` and `pycuda` are often not installed directly via pip from PyPI for all systems or might require specific versions matching your CUDA/TensorRT system setup. Please refer to the official NVIDIA documentation for installing TensorRT and, if needed, PyCUDA. The versions in `requirements.txt` are placeholders or might point to CPU-only stubs if available on PyPI.

## How to Use

### 1. TensorRT Model Preparation

This framework requires a pre-built TensorRT engine file (`.engine`). You need to convert your object detection model (e.g., from ONNX, TensorFlow, or PyTorch formats) into a TensorRT engine. NVIDIA provides tools like `trtexec` or APIs for this conversion. Place your `.engine` file in the `models/` directory or provide a path to it. Similarly, if you plan to use DeepSORT, you will need a ReID model converted to a TensorRT `.engine` file. The `ReIDFeatureExtractor` in `src/reid/feature_extractor.py` will then need its `preprocess` method and feature extraction logic customized for your specific ReID model.

The `src/tensorrt_utils/infer.py` module contains the `TensorRTDetector` class. Its `preprocess` and `postprocess` methods will likely need to be **customized** based on your specific model's input requirements and output format.

### 2. Running the Example

The `examples/run_pipeline.py` script demonstrates how to use the framework with a mock detector and dummy data:

```bash
# To run with default SORT tracker
python examples/run_pipeline.py --type all

# To run with DeepSORT tracker
python examples/run_pipeline.py --type all --tracker_type DeepSORT
```
This will:
- Create dummy image and video files in `examples/temp_data/`.
- Process them using the `ProcessingPipeline`.
- Save the output (with visualizations) in `examples/temp_data/`.

### 3. Integrating into Your Application

-   Initialize `TensorRTDetector` with the path to your detection `.engine` file.
-   If using DeepSORT, initialize `ReIDFeatureExtractor` with your ReID `.engine` file.
-   Instantiate `ProcessingPipeline`, providing the `tracker_type` ('SORT' or 'DeepSORT'), relevant configurations, and the `reid_model` if applicable.
-   Use `pipeline.process_image()` or `pipeline.process_video()` methods.

```python
from src.pipeline.main_pipeline import ProcessingPipeline
from src.tensorrt_utils.infer import TensorRTDetector # Ensure configured
from src.reid.feature_extractor import ReIDFeatureExtractor # Ensure configured if using DeepSORT
from src.tracking.tracker import SORTTracker # If using SORT explicitly
from src.tracking.deepsort_tracker import DeepSORTTracker # If using DeepSORT explicitly

# 1. Initialize detector (customize for your model)
# detector = TensorRTDetector(engine_path="models/your_detection_model.engine")

# 2. Initialize ReID Model (if using DeepSORT)
# This will require you to complete placeholder methods in ReIDFeatureExtractor
# reid_feature_ex = ReIDFeatureExtractor(engine_path="models/your_reid_model.engine")

# 3. Tracker configurations (examples)
tracker_to_use = 'DeepSORT' # or 'SORT'

sort_params = {'max_age': 30, 'min_hits': 3, 'iou_threshold': 0.3}
deepsort_params = {
    'max_age': 70, 
    'min_hits_to_confirm': 3, 
    'iou_threshold': 0.7, # IoU threshold for preliminary matching
    'max_cosine_distance': 0.2, # Threshold for appearance matching
    'nn_budget': 100 # Max features in track gallery
}

# 4. Initialize pipeline
# pipeline = ProcessingPipeline(
#     detector=detector,
#     tracker_type=tracker_to_use,
#     sort_tracker_config=sort_params if tracker_to_use == 'SORT' else None,
#     deepsort_tracker_config=deepsort_params if tracker_to_use == 'DeepSORT' else None,
#     reid_model=reid_feature_ex if tracker_to_use == 'DeepSORT' else None
# )

# Now process an image or video
# pipeline.process_image("path/to/your/image.jpg", "path/to/output/image.jpg")
# pipeline.process_video("path/to/your/video.mp4", "path/to/output/video.mp4")
```

## To Do / Future Improvements

*   Complete model-specific implementation for `preprocess` and `postprocess` in `TensorRTDetector`.
*   Complete model-specific implementation for `preprocess` and feature extraction in `ReIDFeatureExtractor`.
*   Refine Kalman Filter implementation in `DeepSORTTracker` for more robust state estimation.
*   Implement Mahalanobis distance gating in `DeepSORTTracker` for improved matching.
*   Implement a more sophisticated Kalman filter within `SORTTracker` (if continuing its separate development).
*   Add comprehensive unit tests for all modules.
*   Support for asynchronous processing.
*   Configuration file for pipeline parameters.
```
