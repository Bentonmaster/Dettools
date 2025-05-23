import numpy as np
from scipy.optimize import linear_sum_assignment

# Attempt to import filterpy.kalman.KalmanFilter.
# For now, we assume filterpy is NOT available to keep it simpler,
# and we'll implement a very basic constant velocity model prediction placeholder.
# try:
#     from filterpy.kalman import KalmanFilter
#     FILTERPY_AVAILABLE = True
# except ImportError:
#     FILTERPY_AVAILABLE = False
#     print("Warning: filterpy library not found. Using simplified SORT prediction logic.")

def iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.

    Args:
        bb_test: A NumPy array representing the test bounding box `[x1, y1, x2, y2]`.
        bb_gt: A NumPy array representing the ground truth bounding box `[x1, y1, x2, y2]`.

    Returns:
        The IoU score as a float.
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh + 1e-7) # Add epsilon for numerical stability
    return o

class SORTTracker:
    """
    A Simple Online and Realtime Tracking (SORT) algorithm implementation.

    Attributes:
        max_age (int): Maximum number of frames to keep a track alive without new detections.
        min_hits (int): Minimum number of hits (associated detections) to consider a track confirmed.
        iou_threshold (float): Minimum IoU for matching detections with tracks.
        tracks (list): List to store active tracks. Each track is a dictionary.
        next_track_id (int): Counter for assigning new track IDs.
    """

    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initializes the SORTTracker.

        Args:
            max_age: Maximum number of frames to keep a track alive without new detections.
            min_hits: Minimum number of hits (associated detections) to consider a track confirmed.
            iou_threshold: Minimum IoU for matching detections with tracks.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_track_id = 0
        # self.kf = None # Placeholder for Kalman Filter if filterpy was available

    def _predict_states(self) -> list[np.ndarray]:
        """
        Predicts the next state for each active track.
        
        Currently, this is a simplified prediction where the bounding box
        is assumed to remain the same. A Kalman Filter would typically be used here
        for more accurate motion prediction.
        
        Returns:
            A list of predicted bounding boxes for active tracks.
        """
        predicted_bboxes = []
        for track in self.tracks:
            # Simplified prediction: current bbox is the predicted bbox.
            # If using Kalman Filter:
            # track['kf'].predict()
            # track['bbox'] = track['kf'].x[:4] # Update bbox from state vector
            predicted_bboxes.append(track['bbox'])
        return predicted_bboxes

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Updates the tracker with new detections.

        Args:
            detections: A NumPy array of detections, shape (N, 5) where each
                        row is `[x1, y1, x2, y2, score]`. Detections with fewer
                        than 5 columns will be assumed to have a score of 1.0.

        Returns:
            A NumPy array of active tracks, shape (M, 5), where each row is
            `[x1, y1, x2, y2, track_id]`. Only tracks that meet `min_hits`
            are returned.
        """
        # Step 1: Predict new locations of existing tracks
        if len(self.tracks) > 0:
            predicted_bboxes = self._predict_states()
        else:
            predicted_bboxes = []

        # Ensure detections have scores, default to 1.0 if not provided
        if detections.shape[0] > 0 and detections.shape[1] == 4:
            # Add a column of ones for scores if only bboxes are provided
            detections = np.hstack((detections, np.ones((detections.shape[0], 1))))
        elif detections.shape[0] > 0 and detections.shape[1] < 4:
             # Not enough data for a bounding box
            detections = np.empty((0, 5))


        # Step 2: Associate detections with existing tracks
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(detections) > 0 and len(predicted_bboxes) > 0:
            iou_matrix = np.zeros((len(detections), len(predicted_bboxes)))
            for d, det in enumerate(detections):
                for t, trk_bbox in enumerate(predicted_bboxes):
                    iou_matrix[d, t] = iou(det[:4], trk_bbox)

            cost_matrix = 1 - iou_matrix
            # Note: linear_sum_assignment finds the minimum cost assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            current_matched_indices = [] # Use a temporary list for this iteration's matches
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    current_matched_indices.append((r, c))
                    if r in unmatched_detections:
                        unmatched_detections.remove(r)
                    if c in unmatched_tracks:
                        unmatched_tracks.remove(c)
            matched_indices = current_matched_indices


        # Step 3: Update tracks
        # Update matched tracks
        for det_idx, trk_idx in matched_indices:
            track = self.tracks[trk_idx]
            track['bbox'] = detections[det_idx][:4]
            track['score'] = detections[det_idx][4]
            track['hits'] += 1
            track['age'] = 0
            # If using Kalman Filter: track['kf'].update(detections[det_idx][:4])

        # Handle unmatched tracks
        for trk_idx in sorted(unmatched_tracks, reverse=True): # Iterate backwards for safe removal
            track = self.tracks[trk_idx]
            track['age'] += 1
            if track['age'] > self.max_age:
                self.tracks.pop(trk_idx)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det_data = detections[det_idx]
            new_track = {
                'id': self.next_track_id,
                'bbox': det_data[:4],
                'score': det_data[4] if len(det_data) == 5 else 1.0,
                'hits': 1,
                'age': 0,
                # 'kf': None # Initialize Kalman Filter if used
            }
            # if FILTERPY_AVAILABLE:
            #     kf = KalmanFilter(dim_x=7, dim_z=4) # Example KF dimensions
            #     # Initialize kf (state, covariance, etc.)
            #     new_track['kf'] = kf
            self.tracks.append(new_track)
            self.next_track_id += 1

        # Step 4: Output active tracks that meet min_hits
        active_tracks_output = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits: # or track['age'] == 0 to show new tracks immediately
                output_data = np.concatenate((track['bbox'], [track['id']])).reshape(1, 5)
                active_tracks_output.append(output_data)
        
        if len(active_tracks_output) > 0:
            return np.vstack(active_tracks_output)
        else:
            return np.empty((0, 5))

if __name__ == '__main__':
    # Example Usage
    tracker = SORTTracker(max_age=3, min_hits=1, iou_threshold=0.3)

    # Frame 1: Two detections
    detections_frame1 = np.array([
        [50, 50, 100, 100, 0.9],
        [120, 120, 180, 180, 0.85]
    ])
    print("Frame 1 Detections:\n", detections_frame1)
    tracked_objects_frame1 = tracker.update(detections_frame1)
    print("Frame 1 Tracked Objects:\n", tracked_objects_frame1) # Should be empty as min_hits=1 (for newly created tracks)

    # Frame 2: Detections shift slightly
    detections_frame2 = np.array([
        [55, 55, 105, 105, 0.92], # Matches first object
        [200, 200, 250, 250, 0.8] # New object
    ])
    print("\nFrame 2 Detections:\n", detections_frame2)
    tracked_objects_frame2 = tracker.update(detections_frame2)
    print("Frame 2 Tracked Objects:\n", tracked_objects_frame2)

    # Frame 3: One detection disappears, one continues
    detections_frame3 = np.array([
        [60, 60, 110, 110, 0.95] # Matches first object again
    ])
    print("\nFrame 3 Detections:\n", detections_frame3)
    tracked_objects_frame3 = tracker.update(detections_frame3)
    print("Frame 3 Tracked Objects:\n", tracked_objects_frame3)

    # Frame 4: No detections
    detections_frame4 = np.array([])
    print("\nFrame 4 Detections:\n", detections_frame4)
    tracked_objects_frame4 = tracker.update(detections_frame4)
    print("Frame 4 Tracked Objects:\n", tracked_objects_frame4) # Track 0 should still be alive (age 1)

    # Frame 5: Track 0 reappears
    detections_frame5 = np.array([
        [60, 60, 110, 110, 0.95] 
    ])
    print("\nFrame 5 Detections:\n", detections_frame5)
    tracked_objects_frame5 = tracker.update(detections_frame5)
    print("Frame 5 Tracked Objects:\n", tracked_objects_frame5)

    # Test iou function
    bb1 = np.array([50, 50, 100, 100])
    bb2 = np.array([60, 60, 110, 110]) # Overlapping
    bb3 = np.array([200, 200, 250, 250]) # Not overlapping with bb1
    print(f"\nIoU bb1, bb2: {iou(bb1, bb2):.4f}")
    print(f"IoU bb1, bb3: {iou(bb1, bb3):.4f}")

    # Test with 4-column detections (no scores)
    detections_frame6 = np.array([
        [50, 50, 100, 100],
        [120, 120, 180, 180]
    ])
    print("\nFrame 6 Detections (no scores):\n", detections_frame6)
    tracker_no_score_test = SORTTracker(min_hits=1) # Reset tracker for this test
    tracked_objects_frame6 = tracker_no_score_test.update(detections_frame6)
    print("Frame 6 Tracked Objects:\n", tracked_objects_frame6)
    # Each track should have score 1.0
    for track in tracker_no_score_test.tracks:
        assert track['score'] == 1.0
    print("Passed no-score detection test.")

    detections_frame7 = np.array([
        [50, 50, 100, 100, 0.9],
        [52, 52, 102, 102, 0.8] # Very similar detection
    ])
    print("\nFrame 7 Detections (highly overlapping):\n", detections_frame7)
    tracker_overlap_test = SORTTracker(min_hits=1, iou_threshold=0.1)
    tracked_objects_frame7 = tracker_overlap_test.update(detections_frame7)
    print("Frame 7 Tracked Objects:\n", tracked_objects_frame7)
    # Should create two tracks initially.
    assert len(tracker_overlap_test.tracks) == 2, "Should create two tracks for highly overlapping if iou_threshold is low enough"

    detections_frame8 = np.array([
        [50, 50, 100, 100, 0.9] # Only one matches
    ])
    tracked_objects_frame8 = tracker_overlap_test.update(detections_frame8)
    print("Frame 8 Tracked Objects:\n", tracked_objects_frame8)
    # One track should be updated, one aged.
    assert len(tracker_overlap_test.tracks) == 2
    assert any(t['hits']==2 for t in tracker_overlap_test.tracks)
    assert any(t['age']==1 for t in tracker_overlap_test.tracks)
    print("Passed highly overlapping detection test sequence.")

    # Test max_age
    tracker_maxage_test = SORTTracker(max_age=1, min_hits=1)
    dets1 = np.array([[10,10,20,20,0.9]])
    tracker_maxage_test.update(dets1) # Track 0 created
    print("\nMax Age Test:")
    print("Tracks after 1st update:", tracker_maxage_test.tracks)
    tracker_maxage_test.update(np.array([])) # No detections, track 0 age = 1
    print("Tracks after 2nd update (no dets):", tracker_maxage_test.tracks)
    assert tracker_maxage_test.tracks[0]['age'] == 1
    tracker_maxage_test.update(np.array([])) # No detections, track 0 age = 2, should be removed
    print("Tracks after 3rd update (no dets):", tracker_maxage_test.tracks)
    assert len(tracker_maxage_test.tracks) == 0, "Track should be removed due to max_age"
    print("Passed max_age test.")

    # Test min_hits
    tracker_minhits_test = SORTTracker(min_hits=2, iou_threshold=0.1)
    dets1 = np.array([[10,10,20,20,0.9]])
    dets2 = np.array([[11,11,21,21,0.9]])
    dets3 = np.array([[100,100,120,120,0.9]]) # new detection
    
    print("\nMin Hits Test:")
    res1 = tracker_minhits_test.update(dets1)
    print("Result after 1st update (1 hit):", res1) # Track 0, hits=1, not returned
    assert res1.shape[0] == 0
    assert len(tracker_minhits_test.tracks) == 1
    assert tracker_minhits_test.tracks[0]['hits'] == 1

    res2 = tracker_minhits_test.update(dets2) # Track 0, hits=2, should be returned
    print("Result after 2nd update (2 hits):", res2)
    assert res2.shape[0] == 1
    assert res2[0,4] == 0 # track_id
    assert tracker_minhits_test.tracks[0]['hits'] == 2
    
    res3 = tracker_minhits_test.update(dets3) # Track 0 (age 1), Track 1 (hits 1)
    print("Result after 3rd update (new det):", res3)
    assert res3.shape[0] == 1 # Only track 0 should be returned
    assert res3[0,4] == 0
    assert len(tracker_minhits_test.tracks) == 2
    assert tracker_minhits_test.tracks[0]['age'] == 1
    assert tracker_minhits_test.tracks[1]['hits'] == 1
    print("Passed min_hits test.")

    print("\nAll SORT example tests completed.")
