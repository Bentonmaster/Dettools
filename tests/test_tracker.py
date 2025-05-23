import unittest
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tracking.tracker import SORTTracker, iou

class TestSORTTracker(unittest.TestCase):

    def test_tracker_initialization(self):
        tracker = SORTTracker() # Default parameters
        self.assertEqual(len(tracker.tracks), 0, "Default tracker should have no active tracks.")
        self.assertEqual(tracker.next_track_id, 0, "Default tracker next_track_id should be 0.")
        self.assertEqual(tracker.max_age, 1, "Default max_age should be 1.")
        self.assertEqual(tracker.min_hits, 3, "Default min_hits should be 3.")
        self.assertAlmostEqual(tracker.iou_threshold, 0.3, msg="Default iou_threshold should be 0.3.")

        custom_max_age = 5
        custom_min_hits = 2
        custom_iou_threshold = 0.5
        tracker_custom = SORTTracker(max_age=custom_max_age, 
                                     min_hits=custom_min_hits, 
                                     iou_threshold=custom_iou_threshold)
        self.assertEqual(tracker_custom.max_age, custom_max_age, "Custom max_age not set correctly.")
        self.assertEqual(tracker_custom.min_hits, custom_min_hits, "Custom min_hits not set correctly.")
        self.assertAlmostEqual(tracker_custom.iou_threshold, custom_iou_threshold,
                               msg="Custom iou_threshold not set correctly.")

    def test_iou_calculation(self):
        # Identical boxes
        bb1 = np.array([0, 0, 10, 10])
        bb2 = np.array([0, 0, 10, 10])
        self.assertAlmostEqual(iou(bb1, bb2), 1.0, msg="IoU for identical boxes should be 1.0.")

        # Partially overlapping boxes
        bb3 = np.array([5, 5, 15, 15])
        # Intersection: (5,5) to (10,10) -> area = 5*5 = 25
        # Union: area1 + area2 - intersection = (10*10) + (10*10) - 25 = 100 + 100 - 25 = 175
        # IoU = 25 / 175 = 1/7
        self.assertAlmostEqual(iou(bb1, bb3), 25.0 / 175.0, msg="IoU for partially overlapping boxes calculation error.")
        self.assertTrue(0 < iou(bb1, bb3) < 1, "IoU for overlapping boxes should be between 0 and 1.")

        # Non-overlapping boxes
        bb4 = np.array([100, 100, 110, 110])
        self.assertAlmostEqual(iou(bb1, bb4), 0.0, msg="IoU for non-overlapping boxes should be 0.0.")

        # One box contained within another
        bb5 = np.array([2, 2, 8, 8]) # Contained within bb1
        # Intersection: area of bb5 = 6*6 = 36
        # Union: area of bb1 = 10*10 = 100
        # IoU = 36 / 100 = 0.36
        self.assertAlmostEqual(iou(bb1, bb5), 36.0 / 100.0, msg="IoU for contained box calculation error.")

        # Touching boxes (edge touch)
        bb6 = np.array([10, 0, 20, 10]) # Touches bb1 at x=10 edge
        self.assertAlmostEqual(iou(bb1, bb6), 0.0, msg="IoU for touching boxes should be 0.0.")
        
        # Epsilon test for stability (ensure no division by zero if areas are huge and wh is also huge but slightly less)
        bb_large1 = np.array([0,0,1000,1000])
        bb_large2 = np.array([1,1,1001,1001]) # Overlap is (999*999)
        # Area1 = 1000*1000, Area2 = 1000*1000
        # Intersection = 999*999
        # Union = 2 * (1000*1000) - (999*999)
        expected_iou_large = (999.0*999.0) / (2.0*1000.0*1000.0 - 999.0*999.0)
        self.assertAlmostEqual(iou(bb_large1, bb_large2), expected_iou_large, places=5)


    def test_track_creation(self):
        tracker = SORTTracker(min_hits=1) # New tracks are output immediately
        detection1 = np.array([[50, 50, 100, 100, 0.9]])
        
        tracks_out = tracker.update(detection1)
        
        self.assertEqual(len(tracker.tracks), 1, "One track should be created in internal list.")
        self.assertEqual(tracker.tracks[0]['id'], 0, "First track ID should be 0.")
        np.testing.assert_array_almost_equal(tracker.tracks[0]['bbox'], detection1[0, :4], 
                                             err_msg="Track bbox not matching detection.")
        
        self.assertEqual(tracks_out.shape[0], 1, "One track should be returned.")
        self.assertEqual(tracks_out[0, 4], 0, "Returned track ID should be 0.")
        np.testing.assert_array_almost_equal(tracks_out[0, :4], detection1[0, :4],
                                             err_msg="Returned track bbox not matching detection.")

    def test_track_update_and_matching(self):
        tracker = SORTTracker(min_hits=1, iou_threshold=0.5)
        
        # Frame 1
        detection1 = np.array([[50, 50, 100, 100, 0.9]])
        tracks_frame1 = tracker.update(detection1)
        track_id_1 = tracks_frame1[0, 4]
        self.assertEqual(track_id_1, 0, "Initial track ID should be 0.")

        # Frame 2: detection2 significantly overlaps with detection1
        detection2 = np.array([[55, 55, 105, 105, 0.88]]) # High overlap
        iou_val = iou(detection1[0,:4], detection2[0,:4])
        self.assertTrue(iou_val >= tracker.iou_threshold, f"Test setup error: IoU {iou_val} is less than threshold {tracker.iou_threshold}")

        tracks_frame2 = tracker.update(detection2)
        self.assertEqual(tracks_frame2.shape[0], 1, "Should still be one track.")
        self.assertEqual(tracks_frame2[0, 4], track_id_1, "Track ID should remain the same after matching.")
        np.testing.assert_array_almost_equal(tracks_frame2[0, :4], detection2[0, :4],
                                             err_msg="Track bbox should be updated to detection2's bbox.")
        self.assertEqual(tracker.tracks[0]['hits'], 2, "Track hits should be incremented.")

    def test_track_creation_multiple_detections(self):
        tracker = SORTTracker(min_hits=1)
        detections = np.array([
            [10, 10, 50, 50, 0.95],  # Detection A
            [100, 100, 150, 150, 0.92] # Detection B (non-overlapping)
        ])
        self.assertTrue(iou(detections[0,:4], detections[1,:4]) == 0.0, "Test setup error: detections should be non-overlapping.")

        tracks_out = tracker.update(detections)
        
        self.assertEqual(len(tracker.tracks), 2, "Two tracks should be created in internal list.")
        self.assertEqual(tracks_out.shape[0], 2, "Two tracks should be returned.")
        
        # Check IDs are unique and assigned sequentially
        returned_ids = sorted(tracks_out[:, 4].tolist())
        self.assertListEqual(returned_ids, [0, 1], "Returned track IDs should be 0 and 1.")
        
        internal_ids = sorted([t['id'] for t in tracker.tracks])
        self.assertListEqual(internal_ids, [0,1], "Internal track IDs should be 0 and 1.")


    def test_track_removal_by_age(self):
        tracker = SORTTracker(max_age=1, min_hits=1) # Track survives 1 frame without update

        # Frame 1: Create a track
        detection1 = np.array([[10, 10, 20, 20, 0.9]])
        tracks_f1 = tracker.update(detection1)
        self.assertEqual(tracks_f1.shape[0], 1, "Frame 1: One track should be active and returned.")
        self.assertEqual(tracker.tracks[0]['age'], 0, "Frame 1: Track age should be 0.")

        # Frame 2: No detections. Track should age.
        empty_detections = np.empty((0, 5))
        tracks_f2 = tracker.update(empty_detections)
        self.assertEqual(tracks_f2.shape[0], 1, "Frame 2: Track should still be active (age 1) and returned.")
        self.assertEqual(len(tracker.tracks), 1, "Frame 2: One track should be in internal list.")
        self.assertEqual(tracker.tracks[0]['age'], 1, "Frame 2: Track age should be 1.")

        # Frame 3: No detections. Track should be removed (age > max_age).
        tracks_f3 = tracker.update(empty_detections)
        self.assertEqual(tracks_f3.shape[0], 0, "Frame 3: Track should be removed, no tracks returned.")
        self.assertEqual(len(tracker.tracks), 0, "Frame 3: Internal track list should be empty.")

    def test_min_hits_filter(self):
        tracker = SORTTracker(min_hits=3, max_age=5) # Track needs 3 hits to be output

        det1 = np.array([[10, 10, 20, 20, 0.9]])
        det2 = np.array([[11, 11, 21, 21, 0.9]]) # Overlaps det1
        det3 = np.array([[12, 12, 22, 22, 0.9]]) # Overlaps det1/det2

        # Frame 1
        tracks_out_f1 = tracker.update(det1)
        self.assertEqual(tracks_out_f1.shape[0], 0, "Frame 1: No tracks returned (hits=1 < min_hits=3).")
        self.assertEqual(len(tracker.tracks), 1, "Frame 1: One track internally.")
        self.assertEqual(tracker.tracks[0]['hits'], 1, "Frame 1: Track hits should be 1.")

        # Frame 2
        tracks_out_f2 = tracker.update(det2)
        self.assertEqual(tracks_out_f2.shape[0], 0, "Frame 2: No tracks returned (hits=2 < min_hits=3).")
        self.assertEqual(len(tracker.tracks), 1, "Frame 2: Still one track internally.")
        self.assertEqual(tracker.tracks[0]['hits'], 2, "Frame 2: Track hits should be 2.")

        # Frame 3
        tracks_out_f3 = tracker.update(det3)
        self.assertEqual(tracks_out_f3.shape[0], 1, "Frame 3: One track returned (hits=3 == min_hits=3).")
        self.assertEqual(len(tracker.tracks), 1, "Frame 3: Still one track internally.")
        self.assertEqual(tracker.tracks[0]['hits'], 3, "Frame 3: Track hits should be 3.")
        self.assertEqual(tracks_out_f3[0, 4], 0, "Frame 3: Returned track ID should be 0.")

        # Frame 4: New detection, non-overlapping with previous
        det4 = np.array([[100, 100, 120, 120, 0.9]])
        tracks_out_f4 = tracker.update(det4)
        # Previous track (ID 0) is still returned as it met min_hits and is not too old.
        # New track (ID 1) is not returned as hits=1 < min_hits=3
        self.assertEqual(tracks_out_f4.shape[0], 1, "Frame 4: Only track 0 should be returned.")
        self.assertEqual(tracks_out_f4[0,4], 0, "Frame 4: Ensure track 0 is the one returned")
        self.assertEqual(len(tracker.tracks), 2, "Frame 4: Two tracks internally.")
        self.assertEqual(tracker.tracks[0]['hits'], 3, "Frame 4: Track 0 hits still 3, age becomes 1.")
        self.assertEqual(tracker.tracks[0]['age'], 1, "Frame 4: Track 0 age should be 1.")
        self.assertEqual(tracker.tracks[1]['hits'], 1, "Frame 4: Track 1 hits should be 1.")

if __name__ == '__main__':
    unittest.main()
