import unittest
import os
import shutil
import numpy as np
import cv2 # For video properties and frame comparison

# Add this line to ensure src modules can be imported
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.io.rw import read_image, write_image, read_video, write_video

class TestIO(unittest.TestCase):
    def setUp(self):
        # Construct path relative to this test file's location
        self.base_dir = os.path.dirname(__file__)
        self.test_dir = os.path.join(self.base_dir, "temp_test_data")
        os.makedirs(self.test_dir, exist_ok=True)

        # Dummy data
        self.dummy_image_height = 100
        self.dummy_image_width = 150
        self.dummy_image_data = np.random.randint(0, 256, 
                                                  (self.dummy_image_height, self.dummy_image_width, 3), 
                                                  dtype=np.uint8)
        self.dummy_image_path = os.path.join(self.test_dir, "test_image.png")

        self.dummy_video_path = os.path.join(self.test_dir, "test_video.avi")
        self.video_frame_width = 80
        self.video_frame_height = 60
        self.video_fps = 10
        self.video_num_frames = 5
        self.dummy_video_frames = [
            np.random.randint(0, 256, (self.video_frame_height, self.video_frame_width, 3), dtype=np.uint8)
            for _ in range(self.video_num_frames)
        ]


    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_read_write_image(self):
        # Write the image
        write_image(self.dummy_image_path, self.dummy_image_data)
        self.assertTrue(os.path.exists(self.dummy_image_path), "Image file was not created.")

        # Read the image back
        read_img_data = read_image(self.dummy_image_path)
        self.assertIsNotNone(read_img_data, "Read image data is None.")
        self.assertEqual(read_img_data.shape, self.dummy_image_data.shape, "Read image shape mismatch.")
        np.testing.assert_array_equal(read_img_data, self.dummy_image_data, "Read image content mismatch.")

        # Test write_image error handling (attempting to write to an invalid path)
        # Forcing cv2.imwrite to fail is hard without mocking.
        # Instead, test a common user error: providing a directory that doesn't exist for the file.
        invalid_image_path = os.path.join(self.test_dir, "non_existent_dir", "invalid_image.png")
        with self.assertRaises(IOError): # cv2.imwrite returns False, which our wrapper converts to IOError
            write_image(invalid_image_path, self.dummy_image_data)

    def test_read_image_not_found(self):
        non_existent_image_path = os.path.join(self.test_dir, "non_existent_image.png")
        with self.assertRaises(FileNotFoundError):
            read_image(non_existent_image_path)

    def test_read_write_video(self):
        # Write the video
        write_video(self.dummy_video_path, iter(self.dummy_video_frames), 
                    self.video_fps, (self.video_frame_width, self.video_frame_height))
        self.assertTrue(os.path.exists(self.dummy_video_path), "Video file was not created.")

        # Read the video back
        read_frames_list = []
        for frame in read_video(self.dummy_video_path):
            read_frames_list.append(frame)
        
        self.assertEqual(len(read_frames_list), self.video_num_frames, "Number of read frames mismatch.")
        
        for i in range(self.video_num_frames):
            self.assertEqual(read_frames_list[i].shape, self.dummy_video_frames[i].shape, 
                             f"Frame {i} shape mismatch.")
            # Comparing raw video frames after compression can be tricky due to lossy compression.
            # For a robust test, one might compare checksums or structural similarity if exact match fails.
            # However, for simple, uncompressed-like AVIs with OpenCV, it often works.
            # If this fails due to compression, a more lenient comparison (e.g., PSNR > threshold)
            # or checking frame properties (sum, mean) might be needed.
            np.testing.assert_array_equal(read_frames_list[i], self.dummy_video_frames[i],
                                           f"Frame {i} content mismatch. This might be due to video compression.")

        # Test write_video error handling (e.g. invalid frame_size, fps)
        # Attempting to write with invalid parameters that VideoWriter might catch.
        # 1. Invalid FPS (e.g., 0 or negative) - OpenCV might default or error.
        # Our wrapper doesn't explicitly check for fps <=0, VideoWriter might handle it.
        # Let's test an invalid path for the video writer.
        invalid_video_path = os.path.join(self.test_dir, "non_existent_dir", "invalid_video.avi")
        with self.assertRaises(IOError): # Our wrapper raises IOError if VideoWriter fails to open
             write_video(invalid_video_path, iter(self.dummy_video_frames), 
                        self.video_fps, (self.video_frame_width, self.video_frame_height))

        # 2. Mismatched frame_size (if frames have different sizes than frame_size tuple)
        # This is harder to test directly for write_video as it expects an iterator.
        # The check for frame content is more of a read_video test after write.
        # The current write_video assumes all frames yielded by iterator match frame_size.
        # A more direct test would involve mocking cv2.VideoWriter.write to simulate failure.
        # For now, the invalid path test covers a basic IOError case for write_video.

    def test_read_video_not_found(self):
        non_existent_video_path = os.path.join(self.test_dir, "non_existent_video.avi")
        # The read_video is a generator. The FileNotFoundError should be raised when it's consumed
        # or when VideoCapture fails to open.
        with self.assertRaises(FileNotFoundError):
            video_generator = read_video(non_existent_video_path)
            # Attempt to consume the generator
            next(video_generator, None) 
            # For some OpenCV versions, the error might only occur if cap.isOpened() is checked
            # and explicitly raised, which our wrapper does.

if __name__ == '__main__':
    unittest.main()
