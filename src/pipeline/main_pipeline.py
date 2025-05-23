from src.io.rw import read_image, write_image, read_video, write_video
from src.tensorrt_utils.infer import TensorRTDetector # Assuming this class exists
from src.tracking.tracker import SORTTracker # Assuming this class exists
from src.utils.visualization import draw_detections, draw_tracks
import numpy as np
import cv2 # For getting video properties like FPS, frame_size

class ProcessingPipeline:
    """
    A pipeline for processing images or videos using object detection and tracking.
    """

    def __init__(self, detector: TensorRTDetector, tracker: SORTTracker):
        """
        Initializes the ProcessingPipeline.

        Args:
            detector: An instance of TensorRTDetector for object detection.
            tracker: An instance of SORTTracker for object tracking.
        """
        self.detector = detector
        self.tracker = tracker

    def _convert_detections_to_sort_format(self, detections: list) -> np.ndarray:
        """
        Converts detections from the detector's format to SORTTracker's format.

        Args:
            detections: A list of dictionaries from TensorRTDetector, where each dict
                        is e.g., {'bbox': [x1,y1,x2,y2], 'score': float, ...}.

        Returns:
            A NumPy array of shape (N, 5) where each row is [x1, y1, x2, y2, score].
            Returns an empty array if no detections.
        """
        if not detections:
            return np.empty((0, 5))
        
        sort_detections = []
        for det in detections:
            bbox = det['bbox']
            score = det['score']
            sort_detections.append([bbox[0], bbox[1], bbox[2], bbox[3], score])
        
        return np.array(sort_detections)

    def process_image(self, image_path: str, output_path: str):
        """
        Processes a single image: detects objects and tracks them (conceptually for a single frame).

        Args:
            image_path: Path to the input image.
            output_path: Path to save the processed image.
        """
        image_data = read_image(image_path)
        original_image_for_viz = image_data.copy() # For visualization later

        # Get detections from the detector
        # Detections are expected to be a list of dicts, e.g.,
        # [{'bbox': [x1,y1,x2,y2], 'score': float, 'class_id': int}, ...]
        detections = self.detector.detect(image_data) # Renamed detections_list to detections

        # Convert detections to the format expected by SORTTracker (NumPy array [x1,y1,x2,y2,score])
        np_detections = self._convert_detections_to_sort_format(detections)

        # Update the tracker with these detections
        # tracks will be [x1, y1, x2, y2, track_id]
        tracks = self.tracker.update(np_detections)

        # Visualize
        output_image = image_data.copy() # Make a copy to draw on
        if len(detections) > 0: # Ensure there are detections to draw
            output_image = draw_detections(output_image, detections)
        if len(tracks) > 0: # Ensure there are tracks to draw
            output_image = draw_tracks(output_image, tracks)
        
        write_image(output_path, output_image)
        print(f"Processed image saved to {output_path}. Tracks: {tracks}")


    def process_video(self, video_path: str, output_path: str):
        """
        Processes a video: detects and tracks objects frame by frame.

        Args:
            video_path: Path to the input video.
            output_path: Path to save the processed video.
        
        Raises:
            IOError: If video properties (FPS, frame size) cannot be read.
        """
        # Open video with OpenCV to get properties
        cap_temp = cv2.VideoCapture(video_path)
        if not cap_temp.isOpened():
            raise IOError(f"Cannot open video file to get properties: {video_path}")
        
        fps = cap_temp.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)
        cap_temp.release()

        if fps <= 0 or frame_width == 0 or frame_height == 0:
            raise IOError(f"Could not read valid properties (FPS, frame size) from video: {video_path}")

        processed_frames = []
        
        frame_count = 0
        for frame in read_video(video_path):
            original_frame_for_viz = frame.copy() # For visualization later
            frame_count += 1
            print(f"Processing frame {frame_count}...")

            # Get detections
            detections = self.detector.detect(frame) # Renamed detections_list to detections
            
            # Convert detections to SORT format
            np_detections = self._convert_detections_to_sort_format(detections)
            
            # Update tracker
            tracks = self.tracker.update(np_detections)

            # Visualize
            output_frame = frame.copy() # Make a copy to draw on
            if len(detections) > 0: # Ensure there are detections to draw
                output_frame = draw_detections(output_frame, detections)
            if len(tracks) > 0: # Ensure there are tracks to draw
                output_frame = draw_tracks(output_frame, tracks)
            
            processed_frames.append(output_frame)
            if tracks.shape[0] > 0:
                print(f"  Frame {frame_count} - Tracks: {tracks[:,4]}") # Print track IDs

        if not processed_frames:
            print(f"Warning: No frames were processed from {video_path}.")
            # Create an empty video or handle as an error if preferred
            # For now, we'll let write_video handle an empty iterator if that's the case.
            # However, read_video should raise an error if the video is unreadable.

        # Write the processed frames to a video file
        print(f"Writing processed video to {output_path} with {len(processed_frames)} frames, FPS: {fps}, Size: {frame_size}")
        write_video(output_path, iter(processed_frames), fps, frame_size)
        print(f"Processed video saved to {output_path}")

if __name__ == '__main__':
    # This is a placeholder for basic testing or demonstration.
    # To run this, you would need:
    # 1. A valid TensorRT engine file for TensorRTDetector.
    # 2. Dummy implementations or mocks for TensorRTDetector and SORTTracker if not fully available.
    # 3. Sample image/video files.

    print("ProcessingPipeline class defined. Basic usage example (requires setup):")

    # Dummy/Mock TensorRTDetector
    class MockTensorRTDetector:
        def __init__(self, engine_path):
            print(f"MockTensorRTDetector initialized with {engine_path}")
            # Simulate expected input/output shapes for preprocessing/postprocessing
            self.input_shape = (3, 640, 480) # Example CHW

        def detect(self, image: np.ndarray) -> list:
            print(f"MockTensorRTDetector.detect called with image shape {image.shape}")
            # Simulate some detections
            # Format: [{'bbox': [x1,y1,x2,y2], 'score': float, 'class_id': int}, ...]
            height, width, _ = image.shape
            detections = []
            if np.random.rand() > 0.3: # Simulate finding objects sometimes
                num_dets = np.random.randint(1, 4)
                for i in range(num_dets):
                    x1 = np.random.randint(0, width // 2)
                    y1 = np.random.randint(0, height // 2)
                    x2 = x1 + np.random.randint(50, width // 2)
                    y2 = y1 + np.random.randint(50, height // 2)
                    score = np.random.rand()
                    class_id = np.random.randint(0, 5)
                    detections.append({
                        'bbox': [min(x1,width-1), min(y1,height-1), min(x2,width-1), min(y2,height-1)],
                        'score': score,
                        'class_id': class_id
                    })
            print(f"  Mock Detections: {detections}")
            return detections

    # Dummy/Mock SORTTracker (already has a functional one in tracker.py)
    # from src.tracking.tracker import SORTTracker 

    try:
        # Create dummy files for testing
        # Create a dummy engine file (empty)
        with open("dummy.engine", "w") as f:
            f.write("dummy engine content")
        
        # Create a dummy image
        dummy_image_data = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        write_image("dummy_input.png", dummy_image_data)

        # Create a dummy video
        dummy_frames = [np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8) for _ in range(10)] # 10 frames
        # Ensure dummy_video_output directory exists or is handled by write_video
        # For simplicity, let's assume current dir is fine.
        write_video("dummy_input.avi", iter(dummy_frames), 10, (320, 240))


        # Initialize components
        mock_detector = MockTensorRTDetector(engine_path="dummy.engine")
        # Using the actual SORTTracker
        sort_tracker = SORTTracker(max_age=5, min_hits=2, iou_threshold=0.3) 
        
        pipeline = ProcessingPipeline(detector=mock_detector, tracker=sort_tracker)

        # Test image processing
        print("\nTesting image processing...")
        pipeline.process_image(image_path="dummy_input.png", output_path="dummy_output.png")
        print("Image processing test finished.")

        # Test video processing
        print("\nTesting video processing...")
        pipeline.process_video(video_path="dummy_input.avi", output_path="dummy_output.avi")
        print("Video processing test finished.")

    except ImportError as e:
        print(f"ImportError: {e}. Make sure all custom modules are accessible.")
        print("This example might fail if src.* modules are not in PYTHONPATH or structure is different.")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}. A dummy file might be missing.")
    except IOError as e:
        print(f"IOError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during example: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy files
        import os
        if os.path.exists("dummy.engine"): os.remove("dummy.engine")
        if os.path.exists("dummy_input.png"): os.remove("dummy_input.png")
        if os.path.exists("dummy_output.png"): os.remove("dummy_output.png")
        if os.path.exists("dummy_input.avi"): os.remove("dummy_input.avi")
        if os.path.exists("dummy_output.avi"): os.remove("dummy_output.avi")
        print("\nCleaned up dummy files.")
