# Image and Video Processing Framework with TensorRT and Object Tracking

A Python-based framework for building real-time image and video processing pipelines, featuring object detection using NVIDIA TensorRT and object tracking with SORT.

## Features

*   Modular design for easy extension.
*   TensorRT integration for high-performance inference (requires user-provided `.engine` model).
*   SORT algorithm for object tracking.
*   Pipeline for processing both single images and video streams.
*   Basic visualization of detections and tracks.
*   Example usage script to demonstrate functionality.

## Directory Structure

-   `src/`: Contains the core source code.
    -   `io/`: Image and video input/output utilities.
    -   `tensorrt_utils/`: TensorRT detector implementation.
    -   `tracking/`: SORT tracker implementation.
    -   `pipeline/`: Main processing pipeline.
    -   `utils/`: Visualization utilities.
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

This framework requires a pre-built TensorRT engine file (`.engine`). You need to convert your object detection model (e.g., from ONNX, TensorFlow, or PyTorch formats) into a TensorRT engine. NVIDIA provides tools like `trtexec` or APIs for this conversion. Place your `.engine` file in the `models/` directory or provide a path to it.

The `src/tensorrt_utils/infer.py` module contains the `TensorRTDetector` class. Its `preprocess` and `postprocess` methods will likely need to be **customized** based on your specific model's input requirements and output format.

### 2. Running the Example

The `examples/run_pipeline.py` script demonstrates how to use the framework with a mock detector and dummy data:

```bash
python examples/run_pipeline.py --type all
```
This will:
- Create dummy image and video files in `examples/temp_data/`.
- Process them using the `ProcessingPipeline`.
- Save the output (with visualizations) in `examples/temp_data/`.

### 3. Integrating into Your Application

-   Initialize `TensorRTDetector` with the path to your `.engine` file.
-   Initialize `SORTTracker` with desired parameters.
-   Create a `ProcessingPipeline` instance with the detector and tracker.
-   Use `pipeline.process_image()` or `pipeline.process_video()` methods.

```python
from src.pipeline.main_pipeline import ProcessingPipeline
from src.tensorrt_utils.infer import TensorRTDetector # Ensure this is configured for your model
from src.tracking.tracker import SORTTracker

# 1. Initialize detector (customize for your model engine and preprocessing/postprocessing)
# This will require you to complete the placeholder methods in TensorRTDetector
# detector = TensorRTDetector(engine_path="models/your_model.engine")

# 2. Initialize tracker
tracker = SORTTracker(max_age=20, min_hits=3, iou_threshold=0.3)

# 3. Initialize pipeline (using a real detector)
# pipeline = ProcessingPipeline(detector=detector, tracker=tracker)

# Now process an image or video
# pipeline.process_image("path/to/your/image.jpg", "path/to/output/image.jpg")
# pipeline.process_video("path/to/your/video.mp4", "path/to/output/video.mp4")
```

## To Do / Future Improvements

*   Complete model-specific implementation for `preprocess` and `postprocess` in `TensorRTDetector`.
*   Implement a more sophisticated Kalman filter within `SORTTracker`.
*   Add comprehensive unit tests for all modules.
*   Support for asynchronous processing.
*   Configuration file for pipeline parameters.
```
