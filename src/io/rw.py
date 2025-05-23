import cv2
import numpy as np

def read_image(image_path: str) -> np.ndarray:
    """Reads an image from the specified path.

    Args:
        image_path: Path to the image file.

    Returns:
        The image as a NumPy array.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unable to read: {image_path}")
    return image

def read_video(video_path: str):
    """Reads a video from the specified path and yields frames.

    Args:
        video_path: Path to the video file.

    Yields:
        NumPy array representing a frame of the video.

    Raises:
        FileNotFoundError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found or unable to open: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def write_image(image_path: str, image_data: np.ndarray) -> None:
    """Writes an image to the specified path.

    Args:
        image_path: Path to save the image.
        image_data: NumPy array representing the image.

    Raises:
        IOError: If the image cannot be written.
    """
    success = cv2.imwrite(image_path, image_data)
    if not success:
        raise IOError(f"Unable to write image to: {image_path}")

def write_video(video_path: str, frames_iterator, fps: float, frame_size: tuple[int, int]) -> None:
    """Writes frames to a video file.

    Args:
        video_path: Path to save the video.
        frames_iterator: An iterator that yields NumPy array frames.
        fps: Frames per second for the output video.
        frame_size: A tuple (width, height) for the video frames.

    Raises:
        IOError: If the video writer cannot be opened or fails to write.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    if not out.isOpened():
        raise IOError(f"Unable to open video writer for: {video_path}")

    for frame in frames_iterator:
        out.write(frame)
    
    out.release()
