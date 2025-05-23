import sys
import os
import shutil # For directory cleanup

# Add project root to Python path
# This assumes the script is in 'examples' and the 'src' directory is one level up
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(PROJECT_ROOT)

from src.pipeline.main_pipeline import ProcessingPipeline
# TensorRTDetector will be mocked, so we don't strictly need to import it if it causes issues
# in environments without TensorRT. However, for type hinting or if the mock inherits,
# it might be useful. For now, we'll assume it's not a problem.
from src.tensorrt_utils.infer import TensorRTDetector
from src.tracking.tracker import SORTTracker
from src.tracking.deepsort_tracker import DeepSORTTracker # For type hinting if needed
from src.reid.feature_extractor import ReIDFeatureExtractor # For type hinting if needed
import numpy as np
import cv2 # For creating dummy data
import argparse

# --- Mock/Dummy Components ---

class MockReIDFeatureExtractor:
    def __init__(self, engine_path="dummy_reid.engine"):
        self.engine_path = engine_path
        self.feature_dim = 128 # Example feature dimension
        print(f"MockReIDFeatureExtractor initialized with engine: {self.engine_path}")

    def extract_features(self, cropped_object_images: list[np.ndarray]) -> np.ndarray:
        num_objects = len(cropped_object_images)
        if num_objects == 0:
            return np.array([])
        
        # Simulate feature extraction
        # In a real scenario, each image in cropped_object_images would be preprocessed
        # and then passed to the ReID model.
        print(f"MockReIDFeatureExtractor: Extracting dummy features for {num_objects} objects.")
        dummy_features = np.random.rand(num_objects, self.feature_dim).astype(np.float32)
        return dummy_features

class MockTensorRTDetector:
    """
    A mock class for TensorRTDetector that mimics its interface.
    """
    def __init__(self, engine_path: str):
        """
        Initializes the MockTensorRTDetector.
        Args:
            engine_path: Path to the (dummy) TensorRT engine file.
        """
        self.engine_path = engine_path
        print(f"MockTensorRTDetector initialized with engine: {self.engine_path}")

    def detect(self, image: np.ndarray) -> list:
        """
        Simulates object detection on the input image.

        Args:
            image: NumPy array representing the input image.

        Returns:
            A list of detection dictionaries.
        """
        height, width, _ = image.shape
        detections = []
        
        # Add 1-2 dummy detections
        # Detection 1: A fixed-ish box
        detections.append({
            'bbox': [int(width*0.1), int(height*0.1), int(width*0.3), int(height*0.3)], 
            'score': 0.9, 
            'class_id': 0, 
            'label': 'object_A'
        })
        
        # Detection 2: Sometimes appears, more towards bottom-right
        if np.random.rand() > 0.3: # ~70% chance of this appearing
            detections.append({
                'bbox': [int(width*0.6), int(height*0.6), int(width*0.8), int(height*0.8)], 
                'score': 0.85, 
                'class_id': 1, 
                'label': 'object_B'
            })
            
        # Detection 3: A smaller, more central detection that might change per frame (if used in video)
        # To make it slightly dynamic for video, let's make its position vary a bit based on image sum
        # This is a crude way to make it seem like it's responding to the image.
        offset_x = int((np.sum(image) % 100) / 100.0 * width * 0.1) 
        offset_y = int((np.sum(image) % 70) / 70.0 * height * 0.1)
        detections.append({
            'bbox': [
                max(0, int(width*0.4) + offset_x), 
                max(0, int(height*0.4) + offset_y), 
                min(width, int(width*0.55) + offset_x), 
                min(height, int(height*0.55) + offset_y)
            ],
            'score': 0.78,
            'class_id': 2,
            'label': 'object_C'
        })

        print(f"MockDetector: Detected {len(detections)} objects.")
        return detections

def create_dummy_image(file_path: str, width: int = 640, height: int = 480):
    """
    Creates a simple dummy image and saves it.
    """
    # Create an image with a color gradient
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            image[i, j] = [int(i / height * 255), int(j / width * 255), (int(i / height * 255) + int(j / width * 255)) % 255]
    
    # Add a static circle
    cv2.circle(image, (width // 4, height // 4), 30, (0, 255, 0), -1)
    
    cv2.imwrite(file_path, image)
    print(f"Dummy image saved to {file_path}")

def create_dummy_video(file_path: str, width: int = 640, height: int = 480, num_frames: int = 50, fps: int = 10):
    """
    Creates a short dummy video with a simple moving object.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # or 'MP4V'
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {file_path}")
        return

    for frame_idx in range(num_frames):
        # Create a base image (e.g., changing color slightly per frame)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        # Simple background color change
        bg_color = (frame_idx * 5 % 255, (frame_idx * 3) % 100 + 50, 200 - (frame_idx * 2 % 150))
        image[:] = bg_color

        # Add a moving rectangle
        rect_size = 50
        # Move diagonally and bounce
        pos_x = int((frame_idx * 10) % (width * 2 - rect_size * 2))
        if pos_x > width - rect_size:
            pos_x = (width * 2 - rect_size * 2) - pos_x
        
        pos_y = int((frame_idx * 7) % (height * 2 - rect_size * 2))
        if pos_y > height - rect_size:
            pos_y = (height * 2 - rect_size * 2) - pos_y
            
        cv2.rectangle(image, (pos_x, pos_y), (pos_x + rect_size, pos_y + rect_size), (255, 0, 0), -1)
        
        # Add some text (frame number)
        cv2.putText(image, f"Frame: {frame_idx + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(image)
    
    out.release()
    print(f"Dummy video saved to {file_path} ({num_frames} frames, {fps} FPS)")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run ProcessingPipeline with dummy data.")
    parser.add_argument('--type', type=str, choices=['image', 'video', 'all'], default='all',
                        help="Specify whether to process an 'image', a 'video', or 'all'. Default: 'all'.")
    parser.add_argument('--tracker_type', type=str, choices=['SORT', 'DeepSORT'], default='SORT',
                        help="Specify the tracker type to use. Default: 'SORT'.")
    args = parser.parse_args()

    print(f"\nSelected processing type: {args.type}")
    print(f"Selected tracker type: {args.tracker_type}")

    # Setup paths
    base_dir = os.path.dirname(__file__)
    temp_data_dir = os.path.join(base_dir, "temp_data")
    
    # Create temp_data directory if it doesn't exist
    if os.path.exists(temp_data_dir):
        shutil.rmtree(temp_data_dir) # Clean up from previous runs
    os.makedirs(temp_data_dir, exist_ok=True)
    print(f"Using temporary data directory: {temp_data_dir}")

    dummy_engine_path = os.path.join(temp_data_dir, "dummy.engine")
    with open(dummy_engine_path, 'w') as f: # Create an empty dummy engine file
        f.write("This is a dummy TensorRT engine file.")
    dummy_reid_engine_path = os.path.join(temp_data_dir, "dummy_reid.engine")
    with open(dummy_reid_engine_path, 'w') as f: # Create an empty dummy reid engine file
        f.write("This is a dummy ReID engine file.")


    dummy_image_input_path = os.path.join(temp_data_dir, "dummy_input.png")
    dummy_image_output_path = os.path.join(temp_data_dir, f"dummy_output_image_{args.tracker_type}.png")
    
    dummy_video_input_path = os.path.join(temp_data_dir, "dummy_input.avi")
    dummy_video_output_path = os.path.join(temp_data_dir, f"dummy_output_video_{args.tracker_type}.avi")

    # Instantiate components
    mock_detector = MockTensorRTDetector(engine_path=dummy_engine_path)
    mock_reid_model = MockReIDFeatureExtractor(engine_path=dummy_reid_engine_path)

    sort_config = {'max_age': 30, 'min_hits': 3, 'iou_threshold': 0.3}
    deepsort_config = {
        'max_age': 70, 
        'min_hits_to_confirm': 3, 
        'iou_threshold': 0.7, # Note: DeepSORT often uses smaller IoU for cascade, this is for primary matching
        'max_cosine_distance': 0.2, 
        'nn_budget': 100
    }
    
    reid_for_pipeline = mock_reid_model if args.tracker_type == 'DeepSORT' else None
    
    print(f"\nInitializing ProcessingPipeline with {args.tracker_type} tracker.")
    pipeline = ProcessingPipeline(
        detector=mock_detector,
        tracker_type=args.tracker_type,
        sort_tracker_config=sort_config,
        deepsort_tracker_config=deepsort_config,
        reid_model=reid_for_pipeline
    )

    # Processing
    if args.type in ['image', 'all']:
        print(f"\n--- Processing Dummy Image with {args.tracker_type} ---")
        create_dummy_image(dummy_image_input_path)
        print(f"Attempting to process image: {dummy_image_input_path} -> {dummy_image_output_path}")
        try:
            pipeline.process_image(dummy_image_input_path, dummy_image_output_path)
            print(f"Image processing complete. Output: {dummy_image_output_path}")
        except Exception as e:
            print(f"Error during image processing with {args.tracker_type}: {e}")
            import traceback
            traceback.print_exc()

    if args.type in ['video', 'all']:
        print(f"\n--- Processing Dummy Video with {args.tracker_type} ---")
        create_dummy_video(dummy_video_input_path, num_frames=60, fps=15) # A bit longer video
        print(f"Attempting to process video: {dummy_video_input_path} -> {dummy_video_output_path}")
        try:
            pipeline.process_video(dummy_video_input_path, dummy_video_output_path)
            print(f"Video processing complete. Output: {dummy_video_output_path}")
        except Exception as e:
            print(f"Error during video processing with {args.tracker_type}: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup (optional, could be commented out for inspection)
    # print("\n--- Cleaning up temporary data ---")
    # try:
    #     shutil.rmtree(temp_data_dir)
    #     print(f"Removed temporary directory: {temp_data_dir}")
    # except OSError as e:
    #     print(f"Error removing {temp_data_dir}: {e.strerror}")

    # More specific cleanup based on generated files
    print("\n--- Cleanup ---")
    files_to_remove = [
        dummy_engine_path, dummy_reid_engine_path,
        dummy_image_input_path, dummy_video_input_path,
        os.path.join(temp_data_dir, f"dummy_output_image_SORT.png"),
        os.path.join(temp_data_dir, f"dummy_output_video_SORT.avi"),
        os.path.join(temp_data_dir, f"dummy_output_image_DeepSORT.png"),
        os.path.join(temp_data_dir, f"dummy_output_video_DeepSORT.avi"),
    ]
    for f_path in files_to_remove:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
                # print(f"Removed {f_path}")
            except OSError as e:
                print(f"Error removing {f_path}: {e.strerror}")
    
    # Optionally remove the directory if empty, or if you want to ensure it's fully cleaned.
    # For now, just removing specific files. If you want to remove the whole dir:
    # if os.path.exists(temp_data_dir) and not os.listdir(temp_data_dir): # Only if empty
    #     shutil.rmtree(temp_data_dir)

    print(f"\nScript finished. Check {temp_data_dir} for any remaining output files if cleanup was partial or disabled.")

if __name__ == '__main__':
    main()
