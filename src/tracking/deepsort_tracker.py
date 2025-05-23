import numpy as np
from scipy.optimize import linear_sum_assignment

# Attempt to import project-specific modules
# These will work if the script running this is in a context where 'src' is in sys.path
try:
    from src.reid.feature_extractor import ReIDFeatureExtractor
    from src.utils.metrics import cosine_distance
except ImportError:
    # This is a fallback for development or if the structure isn't perfectly recognized by linters.
    # Assume they will be available in the execution environment.
    print("Warning: DeepSORTTracker failed to import ReID/metrics. Ensure src is in PYTHONPATH.")
    ReIDFeatureExtractor = None # Placeholder
    cosine_distance = None # Placeholder


# --- IoU Function (copied from src.tracking.tracker.py for encapsulation) ---
def iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Assumes bb_test and bb_gt are in [x1, y1, x2, y2] format.
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
             + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    # Add epsilon for numerical stability if union is zero (e.g. for identical zero-area boxes)
    return wh / (union + 1e-7)


# --- Simplified Kalman Filter for DeepSORT ---
class KalmanFilterDeepSORT:
    """
    A simplified Kalman filter for DeepSORT.
    State: [cx, cy, a, h, vcx, vcy, va, vh]
    (center_x, center_y, aspect_ratio, height, and their velocities)
    """
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.ndim_state = 8
        self.ndim_measurement = 4

        # State transition matrix (F) - constant velocity model
        self._F = np.eye(self.ndim_state)
        for i in range(4):
            self._F[i, i + 4] = self.dt

        # Measurement matrix (H) - maps state to measurement space [cx, cy, a, h]
        self._H = np.eye(self.ndim_measurement, self.ndim_state)

        # Initial covariance estimate (P) - placeholder
        self._P_init = np.eye(self.ndim_state) * 1.0
        self._P_init[4:, 4:] *= 10.0 # Higher uncertainty for velocities

        # Process noise covariance (Q) - placeholder
        self._Q = np.eye(self.ndim_state) * 0.1
        self._Q[4:, 4:] *= 0.1 # Smaller noise for velocities

        # Measurement noise covariance (R) - placeholder
        self._R = np.eye(self.ndim_measurement) * 0.1

        self.mean = np.zeros(self.ndim_state)
        self.covariance = self._P_init.copy()

    def initiate(self, measurement_xyah: np.ndarray):
        """Initializes the filter state from the first detection."""
        self.mean[:4] = measurement_xyah
        self.mean[4:] = 0.0 # Initialize velocities to zero
        self.covariance = self._P_init.copy()
        return self.mean, self.covariance

    def predict(self):
        """Predicts the next state."""
        self.mean = self._F @ self.mean
        self.covariance = self._F @ self.covariance @ self._F.T + self._Q
        return self.mean, self.covariance

    def update(self, measurement_xyah: np.ndarray):
        """Updates the state with the new measurement."""
        # Kalman gain (K)
        S = self._H @ self.covariance @ self._H.T + self._R
        K = self.covariance @ self._H.T @ np.linalg.inv(S)

        # Innovation (y)
        y = measurement_xyah - self._H @ self.mean

        # Update state
        self.mean = self.mean + K @ y
        # Update covariance
        self.covariance = (np.eye(self.ndim_state) - K @ self._H) @ self.covariance
        return self.mean, self.covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space (cx, cy, a, h)."""
        # Not strictly needed for this simplified version if we directly use H,
        # but useful for gating if Mahalanobis distance is calculated in measurement space.
        projected_mean = self._H @ mean
        projected_covariance = self._H @ covariance @ self._H.T
        return projected_mean, projected_covariance


# --- Track Class ---
class Track:
    """
    Represents a single tracked object.
    """
    def __init__(self, track_id: int, initial_bbox_xyah: np.ndarray, initial_score: float, 
                 initial_feature: np.ndarray = None, nn_budget: int = 100):
        self.track_id = track_id
        self.bbox_xyah = initial_bbox_xyah # Current [cx, cy, a, h]
        self.score = initial_score
        
        self.kalman_filter = KalmanFilterDeepSORT()
        self.mean, self.covariance = self.kalman_filter.initiate(initial_bbox_xyah)
        
        self.hits = 1
        self.age = 0 # Age of the track in frames
        self.time_since_update = 0 # Frames since last measurement update
        
        self.state = 'Tentative' # Possible states: 'Tentative', 'Confirmed', 'Deleted'
        
        self.features = []
        if initial_feature is not None:
            self.features.append(initial_feature)
        self.nn_budget = nn_budget

    def predict(self):
        self.mean, self.covariance = self.kalman_filter.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection_xyah: np.ndarray, score: float, feature: np.ndarray = None):
        self.mean, self.covariance = self.kalman_filter.update(detection_xyah)
        self.bbox_xyah = self.mean[:4] # Update bbox from KF state
        self.score = score
        
        self.hits += 1
        self.time_since_update = 0
        
        if feature is not None:
            self.features.append(feature)
            if self.nn_budget is not None and len(self.features) > self.nn_budget:
                self.features.pop(0) # Remove oldest feature

    def mark_missed(self):
        # Called when a track is not associated with any detection
        # self.time_since_update is already incremented by predict()
        pass # No specific action other than time_since_update handling

    def is_tentative(self): return self.state == 'Tentative'
    def is_confirmed(self): return self.state == 'Confirmed'
    def is_deleted(self): return self.state == 'Deleted'


# --- DeepSORT Tracker ---
class DeepSORTTracker:
    def __init__(self, reid_model: ReIDFeatureExtractor, 
                 max_age: int = 70, 
                 min_hits_to_confirm: int = 3, 
                 iou_threshold: float = 0.6, # Slightly lower for cascade
                 max_cosine_distance: float = 0.2, 
                 nn_budget: int = 100):
        
        if ReIDFeatureExtractor is None or cosine_distance is None:
            raise ImportError("ReIDFeatureExtractor or cosine_distance is not available. Check imports.")

        self.reid_model = reid_model
        self.max_age = max_age
        self.min_hits_to_confirm = min_hits_to_confirm
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        
        self.tracks = []
        self.next_track_id = 0

    def _xyah_to_xyxy(self, xyah: np.ndarray) -> np.ndarray:
        """Converts [center_x, center_y, aspect_ratio, height] to [x1, y1, x2, y2]."""
        if xyah.ndim == 1: xyah = xyah.reshape(1,-1)
        cx, cy, a, h = xyah[:,0], xyah[:,1], xyah[:,2], xyah[:,3]
        w = a * h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    def _detections_to_xyah_with_score(self, detections_xyxy: np.ndarray) -> np.ndarray:
        """Converts [x1,y1,x2,y2,score] to [cx,cy,a,h,score]."""
        if detections_xyxy.ndim == 1: detections_xyxy = detections_xyxy.reshape(1,-1)
        if detections_xyxy.shape[0] == 0: return np.empty((0,5))
            
        x1, y1, x2, y2 = detections_xyxy[:,0], detections_xyxy[:,1], detections_xyxy[:,2], detections_xyxy[:,3]
        scores = detections_xyxy[:,4]
        w = x2 - x1
        h = y2 - y1
        # Handle zero width or height to avoid division by zero for aspect ratio
        # A default aspect ratio of 1.0 is used for zero width/height boxes.
        aspect_ratio = np.where(h > 0, w / h, 1.0)
        cx = x1 + w / 2
        cy = y1 + h / 2
        return np.stack([cx, cy, aspect_ratio, h, scores], axis=1)

    def predict(self):
        for track in self.tracks:
            track.predict()

    def _match(self, detections_xyah_score: np.ndarray, original_frame: np.ndarray):
        """
        Core matching logic.
        Returns: matches, unmatched_track_indices, unmatched_detection_indices
        """
        matches = [] # List of (track_idx, det_idx)
        unmatched_track_indices = list(range(len(self.tracks)))
        unmatched_detection_indices = list(range(detections_xyah_score.shape[0]))

        if not self.tracks or detections_xyah_score.shape[0] == 0:
            return matches, unmatched_track_indices, unmatched_detection_indices

        # Split tracks by state
        confirmed_tracks_indices = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        tentative_tracks_indices = [i for i, t in enumerate(self.tracks) if t.is_tentative()]

        # --- Stage 1: IoU Matching for Confirmed Tracks ---
        if confirmed_tracks_indices and unmatched_detection_indices:
            track_bboxes_xyxy = self._xyah_to_xyxy(np.array([self.tracks[i].bbox_xyah for i in confirmed_tracks_indices]))
            det_bboxes_xyxy = self._xyah_to_xyxy(detections_xyah_score[unmatched_detection_indices, :4])
            
            cost_matrix_iou = np.zeros((len(confirmed_tracks_indices), len(unmatched_detection_indices)))
            for i, trk_idx in enumerate(confirmed_tracks_indices):
                for j, det_idx in enumerate(unmatched_detection_indices):
                    cost_matrix_iou[i, j] = 1.0 - iou(track_bboxes_xyxy[i], det_bboxes_xyxy[j])
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix_iou)
            
            current_matches_iou = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix_iou[r, c] < (1.0 - self.iou_threshold):
                    original_track_idx = confirmed_tracks_indices[r]
                    original_det_idx = unmatched_detection_indices[c]
                    matches.append((original_track_idx, original_det_idx))
                    current_matches_iou.append((original_track_idx, original_det_idx))
            
            # Update unmatched lists
            for trk_idx, det_idx in current_matches_iou:
                if trk_idx in unmatched_track_indices: unmatched_track_indices.remove(trk_idx)
                if det_idx in unmatched_detection_indices: unmatched_detection_indices.remove(det_idx)
        
        # --- Stage 2: IoU Matching for Tentative Tracks ---
        # (Simplified: Re-use remaining detections for tentative tracks)
        if tentative_tracks_indices and unmatched_detection_indices:
            track_bboxes_xyxy = self._xyah_to_xyxy(np.array([self.tracks[i].bbox_xyah for i in tentative_tracks_indices]))
            det_bboxes_xyxy = self._xyah_to_xyxy(detections_xyah_score[unmatched_detection_indices, :4])

            cost_matrix_iou_tentative = np.zeros((len(tentative_tracks_indices), len(unmatched_detection_indices)))
            for i, trk_idx in enumerate(tentative_tracks_indices):
                for j, det_idx in enumerate(unmatched_detection_indices):
                     cost_matrix_iou_tentative[i,j] = 1.0 - iou(track_bboxes_xyxy[i], det_bboxes_xyxy[j])
            
            row_ind_t, col_ind_t = linear_sum_assignment(cost_matrix_iou_tentative)
            current_matches_iou_t = []
            for r,c in zip(row_ind_t, col_ind_t):
                if cost_matrix_iou_tentative[r,c] < (1.0 - self.iou_threshold):
                    original_track_idx = tentative_tracks_indices[r]
                    original_det_idx = unmatched_detection_indices[c]
                    matches.append((original_track_idx, original_det_idx))
                    current_matches_iou_t.append((original_track_idx, original_det_idx))

            for trk_idx, det_idx in current_matches_iou_t:
                if trk_idx in unmatched_track_indices: unmatched_track_indices.remove(trk_idx)
                if det_idx in unmatched_detection_indices: unmatched_detection_indices.remove(det_idx)


        # --- Stage 3: Appearance Matching for Remaining Confirmed Tracks & Detections ---
        # Only consider confirmed tracks that are still unmatched
        remaining_confirmed_unmatched_track_indices = [
            i for i in unmatched_track_indices if self.tracks[i].is_confirmed()
        ]

        if remaining_confirmed_unmatched_track_indices and unmatched_detection_indices:
            # Extract features for remaining detections
            det_bboxes_for_reid_xyxy = self._xyah_to_xyxy(detections_xyah_score[unmatched_detection_indices, :4])
            
            # Crop images for ReID
            cropped_images = []
            valid_det_indices_for_reid = [] # Keep track of which detections are valid for cropping
            for i, det_idx in enumerate(unmatched_detection_indices):
                x1, y1, x2, y2 = det_bboxes_for_reid_xyxy[i].astype(int)
                # Ensure valid crop dimensions
                if x1 < x2 and y1 < y2 and x1 >=0 and y1 >=0 and x2 <= original_frame.shape[1] and y2 <= original_frame.shape[0]:
                    crop = original_frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        cropped_images.append(crop)
                        valid_det_indices_for_reid.append(det_idx)

            if cropped_images:
                detection_features = self.reid_model.extract_features(cropped_images)
                
                if detection_features.size > 0:
                    cost_matrix_cosine = np.zeros((len(remaining_confirmed_unmatched_track_indices), detection_features.shape[0]))
                    
                    for i, trk_idx in enumerate(remaining_confirmed_unmatched_track_indices):
                        track_gallery = self.tracks[trk_idx].features
                        if not track_gallery: # Should not happen if track is confirmed, but as a safe guard
                            cost_matrix_cosine[i, :] = 1.0 # Max distance if no gallery
                            continue
                        
                        # Calculate distance between all features in gallery and current detection feature
                        # Taking the minimum distance as the cost
                        track_gallery_np = np.asarray(track_gallery)
                        dist_to_gallery = cosine_distance(track_gallery_np, detection_features) # (gallery_size, num_det_features)
                        min_dist_per_det = np.min(dist_to_gallery, axis=0)
                        cost_matrix_cosine[i, :] = min_dist_per_det
                    
                    row_ind_app, col_ind_app = linear_sum_assignment(cost_matrix_cosine)
                    current_matches_app = []
                    for r, c in zip(row_ind_app, col_ind_app):
                        if cost_matrix_cosine[r, c] < self.max_cosine_distance:
                            original_track_idx = remaining_confirmed_unmatched_track_indices[r]
                            original_det_idx = valid_det_indices_for_reid[c] # Map back to original detection index
                            
                            # Ensure this pair wasn't already matched by IoU (shouldn't be if logic is correct)
                            if original_track_idx in unmatched_track_indices and original_det_idx in unmatched_detection_indices:
                                matches.append((original_track_idx, original_det_idx))
                                current_matches_app.append((original_track_idx, original_det_idx))
                    
                    for trk_idx, det_idx in current_matches_app:
                        if trk_idx in unmatched_track_indices: unmatched_track_indices.remove(trk_idx)
                        if det_idx in unmatched_detection_indices: unmatched_detection_indices.remove(det_idx)
        
        return matches, unmatched_track_indices, unmatched_detection_indices


    def update(self, detections_xyxy_score: np.ndarray, original_frame: np.ndarray):
        self.predict() # Predict for all current tracks

        detections_xyah_score = self._detections_to_xyah_with_score(detections_xyxy_score)

        matches, unmatched_tracks, unmatched_detections = self._match(detections_xyah_score, original_frame)

        # Update matched tracks
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det_xyah = detections_xyah_score[det_idx, :4]
            det_score = detections_xyah_score[det_idx, 4]
            
            # Extract feature for this detection
            # Need to convert det_xyah back to xyxy for cropping
            det_bbox_xyxy = self._xyah_to_xyxy(det_xyah.reshape(1,-1))[0].astype(int)
            x1,y1,x2,y2 = det_bbox_xyxy
            feature = None
            if x1 < x2 and y1 < y2 and x1 >=0 and y1 >=0 and x2 <= original_frame.shape[1] and y2 <= original_frame.shape[0]:
                crop = original_frame[y1:y2, x1:x2]
                if crop.size > 0:
                    feature_arr = self.reid_model.extract_features([crop])
                    if feature_arr.size > 0:
                        feature = feature_arr[0]
            
            track.update(det_xyah, det_score, feature)
            
            if track.is_tentative() and track.hits >= self.min_hits_to_confirm:
                track.state = 'Confirmed'

        # Handle unmatched detections: Create new tracks
        for det_idx in unmatched_detections:
            det_xyah = detections_xyah_score[det_idx, :4]
            det_score = detections_xyah_score[det_idx, 4]
            
            # Extract feature for this new detection
            det_bbox_xyxy = self._xyah_to_xyxy(det_xyah.reshape(1,-1))[0].astype(int)
            x1,y1,x2,y2 = det_bbox_xyxy
            initial_feature = None
            if x1 < x2 and y1 < y2 and x1 >=0 and y1 >=0 and x2 <= original_frame.shape[1] and y2 <= original_frame.shape[0]:
                crop = original_frame[y1:y2, x1:x2]
                if crop.size > 0:
                    feature_arr = self.reid_model.extract_features([crop])
                    if feature_arr.size > 0:
                        initial_feature = feature_arr[0]

            new_track = Track(self.next_track_id, det_xyah, det_score, initial_feature, self.nn_budget)
            self.tracks.append(new_track)
            self.next_track_id += 1
            
        # Handle unmatched tracks: Mark for deletion if too old
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.mark_missed() # time_since_update already incremented in predict
            if track.time_since_update > self.max_age:
                track.state = 'Deleted'

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Prepare output
        output_tracks = []
        for track in self.tracks:
            # Only output confirmed tracks (or tentative if they meet some criteria, e.g. min_hits > 0)
            if track.is_confirmed() or (track.is_tentative() and track.hits > 0) : # Simple criteria for tentative
                bbox_xyxy = self._xyah_to_xyxy(track.bbox_xyah.reshape(1,-1))[0]
                output_tracks.append(np.concatenate((bbox_xyxy, [track.track_id, track.score]))) # Add score
        
        if not output_tracks:
            return np.empty((0, 6)) # x1,y1,x2,y2,track_id,score
        return np.array(output_tracks)


if __name__ == '__main__':
    # This section is for conceptual testing.
    # Full test requires a mock ReID model and sample data.
    print("DeepSORTTracker class defined.")

    # Example of how one might mock ReID for testing:
    class MockReIDFeatureExtractor:
        def __init__(self, engine_path="dummy_reid.engine"):
            print(f"MockReID initialized with {engine_path}")
            self.feature_dim = 128 # Example

        def extract_features(self, cropped_object_images: list[np.ndarray]) -> np.ndarray:
            if not cropped_object_images:
                return np.array([])
            num_objects = len(cropped_object_images)
            # print(f"MockReID: Extracting features for {num_objects} images.")
            return np.random.rand(num_objects, self.feature_dim).astype(np.float32)

    if ReIDFeatureExtractor is None: ReIDFeatureExtractor = MockReIDFeatureExtractor # Use mock if import failed
    if cosine_distance is None: 
        def cosine_distance(f1,f2): # Basic mock
            sim = np.dot(f1, f2.T) / (np.linalg.norm(f1,axis=1,keepdims=True) * np.linalg.norm(f2,axis=1,keepdims=True).T + 1e-7)
            return 1 - sim

    # Initialize tracker
    mock_reid = ReIDFeatureExtractor("dummy_reid.engine") # Use actual if imported, else mock
    tracker = DeepSORTTracker(reid_model=mock_reid, min_hits_to_confirm=1, max_age=3)

    # Dummy frame (e.g., 640x480)
    dummy_frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

    # Frame 1: One detection
    detections_f1 = np.array([[50, 50, 100, 100, 0.9]])
    print("\n--- Frame 1 ---")
    output_f1 = tracker.update(detections_f1, dummy_frame)
    print("Tracks F1:\n", output_f1) # Should be one tentative track

    # Frame 2: Detection moves slightly, should match
    detections_f2 = np.array([[55, 55, 105, 105, 0.92]])
    print("\n--- Frame 2 ---")
    output_f2 = tracker.update(detections_f2, dummy_frame)
    print("Tracks F2:\n", output_f2) # Track should be confirmed (if min_hits=1) and ID should be same

    # Frame 3: New detection, old one disappears (for a moment)
    detections_f3 = np.array([[200, 200, 250, 250, 0.8]])
    print("\n--- Frame 3 ---")
    output_f3 = tracker.update(detections_f3, dummy_frame)
    print("Tracks F3:\n", output_f3) # Old track might still be there (age 1), new tentative track

    # Frame 4: Old track reappears, new one also there
    detections_f4 = np.array([[60,60,110,110,0.94], [205,205,255,255,0.82]])
    print("\n--- Frame 4 ---")
    output_f4 = tracker.update(detections_f4, dummy_frame)
    print("Tracks F4:\n", output_f4)

    # Frame 5,6,7: No detections for first track
    print("\n--- Frame 5 (no det for track 0) ---")
    output_f5 = tracker.update(np.array([[210,210,260,260,0.81]]), dummy_frame)
    print("Tracks F5:\n", output_f5)
    
    print("\n--- Frame 6 (no det for track 0) ---")
    output_f6 = tracker.update(np.array([[215,215,265,265,0.83]]), dummy_frame)
    print("Tracks F6:\n", output_f6)

    # Track 0 should be max_age = 3. time_since_update: F3 (1), F4(0), F5(1), F6(2), F7(3) -> deleted
    print("\n--- Frame 7 (track 0 should be deleted) ---")
    output_f7 = tracker.update(np.array([[220,220,270,270,0.85]]), dummy_frame)
    print("Tracks F7:\n", output_f7)
    
    found_track_0 = False
    if output_f7.shape[0] > 0:
        for trk_data in output_f7:
            if trk_data[4] == 0: # Check track ID
                found_track_0 = True
    assert not found_track_0, "Track 0 should have been deleted by Frame 7"
    print("\nBasic DeepSORT example run completed.")
