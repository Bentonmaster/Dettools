import cv2
import numpy as np
import random # For generating random colors

def draw_detections(image: np.ndarray, detections: list, color_map: dict = None) -> np.ndarray:
    """
    Draws detection bounding boxes and labels on an image.

    Args:
        image: The image (NumPy array) on which to draw.
        detections: A list of detection dictionaries, e.g.,
                    [{'bbox': [x1, y1, x2, y2], 'score': float, 
                      'class_id': int, 'label': str (optional)}, ...].
                    Bounding box values are expected to be integers.
        color_map: An optional dictionary mapping class_id to a BGR color tuple (B, G, R).
                   If not provided, or if a class_id is not in the map, random colors
                   will be used per class.

    Returns:
        The image (NumPy array) with detections drawn.
    """
    img_copy = image.copy()
    
    if color_map is None:
        color_map = {}
    
    # Internal map to store randomly assigned colors for classes not in color_map
    _class_colors = {} 

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (255, 255, 255) # White for text for general visibility

    for det in detections:
        bbox = det['bbox']
        score = det.get('score', -1.0) # Use -1.0 if score is not present
        class_id = det.get('class_id', -1) # Use -1 if class_id is not present
        label_override = det.get('label', None)

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Determine color
        color = color_map.get(class_id)
        if color is None:
            if class_id not in _class_colors:
                # Generate a new random color for this class_id
                _class_colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color = _class_colors[class_id]
        
        # Draw the rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Create label string
        if label_override:
            label_str = f"{label_override} ({score:.2f})" if score != -1.0 else label_override
        else:
            label_str = f"Class: {class_id}"
            if score != -1.0:
                label_str += f" S: {score:.2f}"
        
        # Put label text
        # Get text size to position it nicely
        (text_width, text_height), baseline = cv2.getTextSize(label_str, font, font_scale, font_thickness)
        
        # Position text above the box, or inside if it's near the top edge
        text_y = y1 - 10
        if text_y - text_height < 0: # If text goes above image, put it inside
            text_y = y1 + text_height + 5
            
        # Add a filled rectangle as background for the text for better readability
        cv2.rectangle(img_copy, (x1, text_y - text_height - baseline//2), (x1 + text_width, text_y + baseline//2), color, -1) #cv2.FILLED)
        cv2.putText(img_copy, label_str, (x1, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img_copy


def draw_tracks(image: np.ndarray, tracks: np.ndarray, color_seed: int = 42) -> np.ndarray:
    """
    Draws tracking bounding boxes and track IDs on an image.

    Args:
        image: The image (NumPy array) on which to draw.
        tracks: A NumPy array of tracks, shape (M, 5), where each row is
                [x1, y1, x2, y2, track_id].
                Bounding box values and track_id are expected to be integers.
        color_seed: An integer seed for random color generation to ensure somewhat
                    consistent colors for track IDs across calls if needed, though
                    the primary consistency is per-ID within a single call.

    Returns:
        The image (NumPy array) with tracks drawn.
    """
    img_copy = image.copy()
    
    # Predefined list of colors to cycle through for better distinction
    # (B, G, R) format
    PREDEFINED_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (0, 128, 128), (128, 0, 128),
        (255, 128, 0), (255, 0, 128), (0, 255, 128),
        (128, 255, 0), (0, 128, 255), (128, 0, 255),
        (255, 128, 128), (128, 255, 128), (128, 128, 255),
    ]
    
    _track_colors = {} # To store colors for track IDs within this call

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (0,0,0) # Black, assuming colors will be light enough or text bg is used

    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)

        # Generate a unique, consistent color for each track_id
        if track_id not in _track_colors:
            # Option 1: Cycle through predefined colors
            color = PREDEFINED_COLORS[track_id % len(PREDEFINED_COLORS)]
            # Option 2: Use random.seed for consistent random colors (can lead to similar colors)
            # random.seed(track_id * color_seed) # Multiply by color_seed for more variation if seed is fixed
            # color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            _track_colors[track_id] = color
        else:
            color = _track_colors[track_id]
        
        # Draw the rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Create label string
        label_str = f"ID: {track_id}"
        
        # Put track ID text
        (text_width, text_height), baseline = cv2.getTextSize(label_str, font, font_scale, font_thickness)
        
        text_y = y1 - 7
        text_x = x1 + 5
        if text_y - text_height < 0: # If text goes above image, put it at bottom-left of box
            text_y = y2 - 7 # text_height
        if text_x + text_width > img_copy.shape[1]: # if text goes beyond right edge
            text_x = x2 - text_width - 5


        # Add a filled rectangle as background for the text
        cv2.rectangle(img_copy, (text_x - 2, text_y - text_height - baseline//2), 
                                (text_x + text_width + 2, text_y + baseline//2), color, -1) #cv2.FILLED)
        cv2.putText(img_copy, label_str, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img_copy

if __name__ == '__main__':
    # Create a dummy image (black)
    dummy_image = np.zeros((600, 800, 3), dtype=np.uint8)

    # Example Detections
    detections_example = [
        {'bbox': [50, 50, 150, 150], 'score': 0.95, 'class_id': 0, 'label': 'Car'},
        {'bbox': [200, 100, 350, 280], 'score': 0.88, 'class_id': 1}, # No 'label', should use class_id
        {'bbox': [400, 150, 500, 300], 'score': 0.70, 'class_id': 0}, # Same class_id as first
        {'bbox': [50, 350, 180, 450], 'score': 0.92, 'class_id': 2, 'label': 'Pedestrian'},
        {'bbox': [600, 50, 700, 150], 'score': 0.65, 'class_id': 3, 'label': 'Bus'},
        {'bbox': [650, 400, 750, 550], 'class_id': 1}, # No score
    ]

    # Custom color map (optional)
    custom_colors = {
        0: (0, 0, 255),   # Class 0 (Car) -> Red
        2: (0, 255, 0),   # Class 2 (Pedestrian) -> Green
        # Class 1 and 3 will get random colors
    }

    img_with_detections = draw_detections(dummy_image.copy(), detections_example, color_map=custom_colors)
    
    # Example Tracks
    # [x1, y1, x2, y2, track_id]
    tracks_example = np.array([
        [50, 50, 150, 150, 1],
        [200, 100, 350, 280, 2],
        [400, 150, 500, 300, 1], # Same track_id as first, should be same color
        [50, 350, 180, 450, 3],
        [550, 200, 650, 350, 4]
    ])
    
    # First draw detections, then tracks on that image
    img_with_tracks_only = draw_tracks(dummy_image.copy(), tracks_example)
    img_with_both = draw_tracks(img_with_detections.copy(), tracks_example) # Draw tracks on image already having detections

    # In a real scenario, you'd likely use cv2.imshow or save the image
    try:
        cv2.imwrite("debug_detections_only.png", img_with_detections)
        cv2.imwrite("debug_tracks_only.png", img_with_tracks_only)
        cv2.imwrite("debug_detections_and_tracks.png", img_with_both)
        print("Visualization test images saved as debug_detections_only.png, debug_tracks_only.png, and debug_detections_and_tracks.png")
        
        # Test with empty inputs
        empty_dets_img = draw_detections(dummy_image.copy(), [])
        cv2.imwrite("debug_empty_detections.png", empty_dets_img)
        empty_tracks_img = draw_tracks(dummy_image.copy(), np.empty((0,5)))
        cv2.imwrite("debug_empty_tracks.png", empty_tracks_img)
        print("Empty input test images saved.")

    except Exception as e:
        print(f"Error saving test images: {e}. This might happen in environments without display or disk write access.")

    print("Visualization functions defined and basic test executed.")
