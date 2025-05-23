import unittest
import numpy as np
import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(PROJECT_ROOT)

from src.tracking.deepsort_tracker import DeepSORTTracker
# ReIDFeatureExtractor is needed for type hint by DeepSORTTracker's __init__
# but we will pass our mock version.
from src.reid.feature_extractor import ReIDFeatureExtractor 


class MockReIDFeatureExtractorForDeepSORTTests:
    def __init__(self, feature_dim=128):
        self.feature_dim = feature_dim
        self.call_count = 0
        # Store features for specific "identities" based on a simple bbox property (e.g., sum of coords)
        self.identity_features = {} 
        # A simple way to generate distinct features for different objects
        self.next_feature_val = 1.0 

    def _get_bbox_identity_key(self, image_crop_for_mock: np.ndarray):
        # Using sum of pixel values of the crop as a mock identifier.
        # This is very simplistic and assumes crops for the "same" object will be similar enough.
        # For test_appearance_matching, we'll rely more on specific feature assignment.
        if image_crop_for_mock.size == 0: return 0
        return int(np.sum(image_crop_for_mock) % 10000) # Modulo to keep keys manageable

    def extract_features(self, cropped_images: list[np.ndarray]) -> np.ndarray:
        num_objects = len(cropped_images)
        if num_objects == 0:
            return np.array([])
        
        output_features = []
        for i, crop in enumerate(cropped_images):
            # For test_appearance_matching, we might pre-assign features
            # based on a tag or specific crop properties.
            # Here, we generate features that are somewhat unique per call/crop.
            
            # Check if this crop has a pre-assigned "identity" feature for specific tests
            identity_key = self._get_bbox_identity_key(crop) # Use a property of the crop
            
            # Special handling for appearance matching test:
            # These keys are specific to the bboxes used in test_appearance_matching_unconfirmed_iou
            # d1_key (approx [10,10,50,50]), d3_key (approx [12,12,52,52])
            # These sums are illustrative and would need to be calculated from the actual dummy crops if used.
            # For simplicity in the test, we'll use a more direct feature assignment in the test itself
            # by manipulating self.identity_features.

            if identity_key in self.identity_features:
                feature = self.identity_features[identity_key]
            else:
                # Generate a new somewhat unique feature
                feature = np.full(self.feature_dim, self.next_feature_val, dtype=np.float32)
                # Store it if we want to recognize this "identity" again via this simple key
                # self.identity_features[identity_key] = feature 
                self.next_feature_val += 1.0 
                if self.next_feature_val > 200.0: self.next_feature_val = 1.0 # Cycle back
            output_features.append(feature)
        
        self.call_count += 1
        return np.array(output_features)


class TestDeepSORTTracker(unittest.TestCase):
    def setUp(self):
        self.feature_dim = 64 # Smaller dim for easier debugging if needed
        self.mock_reid_model = MockReIDFeatureExtractorForDeepSORTTests(feature_dim=self.feature_dim)
        
        self.tracker = DeepSORTTracker(
            reid_model=self.mock_reid_model,
            min_hits_to_confirm=1,
            max_age=5, # Frames a track can exist without updates
            max_cosine_distance=0.6, # Relaxed for predictable mock features (1-cos_sim)
                                     # If features are [A,A,...] and [B,B,...], cos_sim is 1 if A=B, else depends.
                                     # If features are normalized, [1,0] vs [0,1] -> sim 0, dist 1.
                                     # [1,1] vs [1,1] norm -> sim 1, dist 0.
                                     # [1,1] vs [1,2] norm -> sim !=1, dist >0
            nn_budget=10
        )
        # Dummy frame for tracker updates
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Make dummy frame non-zero to get different identity keys if MockReID uses crop sum
        self.dummy_frame[100:200, 100:200] = 128 


    def test_tracker_initialization(self):
        self.assertIsNotNone(self.tracker.reid_model, "ReID model should be set.")
        self.assertEqual(self.tracker.reid_model, self.mock_reid_model, "ReID model instance mismatch.")
        self.assertEqual(self.tracker.max_age, 5)
        self.assertEqual(self.tracker.min_hits_to_confirm, 1)

    def test_track_creation_with_features(self):
        detections = np.array([[50, 50, 100, 100, 0.9]]) # x1,y1,x2,y2,score
        
        # Ensure dummy_frame has some content for cropping
        cv2.rectangle(self.dummy_frame, (50,50), (100,100), (255,0,0), -1)

        output_tracks = self.tracker.update(detections, self.dummy_frame)
        
        self.assertEqual(len(self.tracker.tracks), 1, "One track should be created.")
        internal_track = self.tracker.tracks[0]
        
        self.assertFalse(internal_track.is_confirmed(), "Track should be tentative initially (min_hits_to_confirm=1, but first hit makes it confirmed after update).")
        # The state becomes confirmed *after* the first update if min_hits_to_confirm is 1.
        # The check `track.is_tentative() and track.hits >= self.min_hits_to_confirm` in DeepSORT
        # makes it confirmed. So for min_hits=1, it's confirmed on the first frame.
        self.assertTrue(internal_track.is_confirmed(), "Track should be confirmed after first update with min_hits=1.")

        self.assertGreater(len(internal_track.features), 0, "Feature gallery should not be empty.")
        self.assertEqual(internal_track.features[0].shape, (self.feature_dim,), "Feature dimension mismatch.")
        
        self.assertEqual(output_tracks.shape[0], 1, "One track should be in output.")
        self.assertEqual(output_tracks[0, 4], internal_track.track_id, "Output track ID mismatch.")


    def test_iou_matching_confirmed_track(self):
        # Frame 1: Create and confirm a track
        det1 = np.array([[50, 50, 100, 100, 0.9]])
        cv2.rectangle(self.dummy_frame, (50,50), (100,100), (10,10,10), -1)
        _ = self.tracker.update(det1, self.dummy_frame)
        track1_id = self.tracker.tracks[0].track_id
        initial_feature_count = len(self.tracker.tracks[0].features)

        # Frame 2: Highly overlapping detection
        det2 = np.array([[55, 55, 105, 105, 0.88]]) # High IoU with det1
        cv2.rectangle(self.dummy_frame, (55,55), (105,105), (20,20,20), -1) # Ensure crop is different
        
        # Reset reid call count to see if it's called
        self.mock_reid_model.call_count = 0
        
        output_tracks_f2 = self.tracker.update(det2, self.dummy_frame)
        
        self.assertEqual(len(self.tracker.tracks), 1, "Should still be one track.")
        self.assertEqual(self.tracker.tracks[0].track_id, track1_id, "Track ID should remain the same (matched).")
        self.assertEqual(output_tracks_f2[0, 4], track1_id, "Output track ID should be the same.")
        self.assertEqual(len(self.tracker.tracks[0].features), initial_feature_count + 1, "Feature gallery should grow.")
        
        # For a very high IoU match with a confirmed track, appearance matching might be skipped or confirm IoU.
        # The DeepSORT _match logic prioritizes IoU for confirmed. We expect reid_model not to be the primary source of match.
        # However, features *are* extracted for matched detections to update the gallery.
        self.assertEqual(self.mock_reid_model.call_count, 1, 
                         "ReID should be called to update features for the matched detection.")


    def test_appearance_matching_unconfirmed_iou(self):
        # Frame 1: d1 -> track t1
        d1_bbox = np.array([10, 10, 50, 50]) # x1,y1,x2,y2
        d1_crop_content = 10 # Unique value for this crop
        cv2.rectangle(self.dummy_frame, tuple(d1_bbox[:2]), tuple(d1_bbox[2:]), (d1_crop_content,)*3, -1)
        d1_key = self.mock_reid_model._get_bbox_identity_key(self.dummy_frame[d1_bbox[1]:d1_bbox[3], d1_bbox[0]:d1_bbox[2]])
        
        # Assign a specific feature for d1's identity
        d1_feature = np.full(self.feature_dim, 5.0, dtype=np.float32)
        self.mock_reid_model.identity_features[d1_key] = d1_feature

        detections_f1 = np.array([np.concatenate((d1_bbox, [0.9]))])
        _ = self.tracker.update(detections_f1, self.dummy_frame)
        self.assertEqual(len(self.tracker.tracks), 1, "Track t1 should be created.")
        track_t1 = self.tracker.tracks[0]
        self.assertEqual(len(track_t1.features), 1)
        np.testing.assert_array_equal(track_t1.features[0], d1_feature, "t1 initial feature mismatch")

        # Frame 2: 
        # d2: low IoU with t1's predicted position, different appearance
        d2_bbox = np.array([100, 100, 140, 140])
        d2_crop_content = 20
        cv2.rectangle(self.dummy_frame, tuple(d2_bbox[:2]), tuple(d2_bbox[2:]), (d2_crop_content,)*3, -1)
        d2_key = self.mock_reid_model._get_bbox_identity_key(self.dummy_frame[d2_bbox[1]:d2_bbox[3], d2_bbox[0]:d2_bbox[2]])
        d2_feature = np.full(self.feature_dim, 10.0, dtype=np.float32) # Different feature
        self.mock_reid_model.identity_features[d2_key] = d2_feature
        
        # d3: low IoU with t1's predicted position, but SAME appearance as d1
        d3_bbox = np.array([12, 12, 52, 52]) # Slightly shifted from d1, assume low IoU with prediction
        # Ensure d3_crop_content is the same as d1_crop_content to retrieve the same feature
        cv2.rectangle(self.dummy_frame, tuple(d3_bbox[:2]), tuple(d3_bbox[2:]), (d1_crop_content,)*3, -1)
        d3_key = self.mock_reid_model._get_bbox_identity_key(self.dummy_frame[d3_bbox[1]:d3_bbox[3], d3_bbox[0]:d3_bbox[2]])
        # Crucially, d3_key should match d1_key for the mock ReID to return the same feature.
        # This requires that the _get_bbox_identity_key is consistent or we force it.
        # Let's assume our _get_bbox_identity_key (sum of pixels) is consistent enough or use a simpler key for test.
        # Forcing d3_key to be d1_key for the test:
        self.mock_reid_model.identity_features[d3_key] = d1_feature # d3 gets d1's feature

        # Assume KF prediction makes d1 drift slightly away, so IoU with d2 and d3 are both poor.
        # For simplicity, we rely on max_cosine_distance being permissive enough for d1_feature vs d1_feature (dist 0)
        # and restrictive enough for d1_feature vs d2_feature (dist should be > max_cosine_distance).
        # Cosine distance between [5,5,...] and [10,10,...] will be 0 if normalized from all-same-value vectors.
        # Need to make features more distinct if using actual cosine distance.
        # Let d1_feature = [1,0,0...], d2_feature = [0,1,0...]. Cosine distance will be 1.0.
        d1_feature_distinct = np.zeros(self.feature_dim, dtype=np.float32); d1_feature_distinct[0] = 1.0
        d2_feature_distinct = np.zeros(self.feature_dim, dtype=np.float32); d2_feature_distinct[1] = 1.0
        self.mock_reid_model.identity_features[d1_key] = d1_feature_distinct
        self.mock_reid_model.identity_features[d2_key] = d2_feature_distinct
        self.mock_reid_model.identity_features[d3_key] = d1_feature_distinct # d3 gets d1's feature

        # Update t1's gallery with the new d1_feature_distinct
        track_t1.features[0] = d1_feature_distinct


        detections_f2 = np.array([
            np.concatenate((d2_bbox, [0.9])), 
            np.concatenate((d3_bbox, [0.9]))
        ])
        
        output_tracks_f2 = self.tracker.update(detections_f2, self.dummy_frame)

        self.assertEqual(len(self.tracker.tracks), 2, "Should be two tracks (t1 updated, new track for d2).")
        matched_to_t1 = False
        new_track_created_for_d2 = False

        for track in self.tracker.tracks:
            if track.track_id == track_t1.track_id:
                matched_to_t1 = True
                # Check if t1 was matched with d3
                np.testing.assert_array_almost_equal(track.bbox_xyah[:2], self.tracker._detections_to_xyah_with_score(d3_bbox.reshape(1,-1))[0,:2], decimal=1,
                                                     err_msg="Track t1 should be updated with d3's bbox.")
                self.assertEqual(len(track.features), 2, "t1 feature gallery should have grown.")
                np.testing.assert_array_equal(track.features[1], d1_feature_distinct, "t1 should have d3's (d1's) feature added.")
            elif np.allclose(track.bbox_xyah[:2], self.tracker._detections_to_xyah_with_score(d2_bbox.reshape(1,-1))[0,:2], atol=3):
                new_track_created_for_d2 = True
                np.testing.assert_array_equal(track.features[0], d2_feature_distinct, "New track should have d2's feature.")


        self.assertTrue(matched_to_t1, "Track t1 should have been matched and updated.")
        self.assertTrue(new_track_created_for_d2, "A new track should have been created for d2.")


    def test_track_confirmation_min_hits(self):
        self.tracker.min_hits_to_confirm = 2 # Override default for this test

        det1 = np.array([[10,10,50,50,0.9]])
        cv2.rectangle(self.dummy_frame, (10,10), (50,50), (30,30,30), -1)
        _ = self.tracker.update(det1, self.dummy_frame)
        self.assertEqual(len(self.tracker.tracks), 1)
        track1 = self.tracker.tracks[0]
        self.assertTrue(track1.is_tentative(), "Track should be Tentative after 1st hit.")
        self.assertEqual(track1.hits, 1)

        det2 = np.array([[12,12,52,52,0.9]]) # Overlapping
        cv2.rectangle(self.dummy_frame, (12,12), (52,52), (40,40,40), -1)
        _ = self.tracker.update(det2, self.dummy_frame)
        self.assertTrue(track1.is_confirmed(), "Track should be Confirmed after 2nd hit.")
        self.assertEqual(track1.hits, 2)


    def test_track_removal_max_age(self):
        self.tracker.max_age = 2 # Override for quicker test

        det1 = np.array([[10,10,50,50,0.9]])
        cv2.rectangle(self.dummy_frame, (10,10), (50,50), (50,50,50), -1)
        _ = self.tracker.update(det1, self.dummy_frame)
        self.assertEqual(len(self.tracker.tracks), 1)
        track1 = self.tracker.tracks[0]

        # Frame 2 (age 1, time_since_update 1)
        _ = self.tracker.update(np.empty((0,5)), self.dummy_frame)
        self.assertEqual(len(self.tracker.tracks), 1)
        self.assertEqual(track1.time_since_update, 1)

        # Frame 3 (age 2, time_since_update 2)
        _ = self.tracker.update(np.empty((0,5)), self.dummy_frame)
        self.assertEqual(len(self.tracker.tracks), 1)
        self.assertEqual(track1.time_since_update, 2)
        
        # Frame 4 (age 3, time_since_update 3 -> should be removed as > max_age=2)
        _ = self.tracker.update(np.empty((0,5)), self.dummy_frame)
        self.assertEqual(len(self.tracker.tracks), 0, "Track should be removed after max_age.")


    def test_nn_budget_feature_gallery(self):
        self.tracker.nn_budget = 2 # Max 2 features in gallery
        self.tracker.min_hits_to_confirm = 1 # Confirm quickly

        base_bbox = [10,10,50,50]
        detections = []
        for i in range(4): # 4 frames
            # Slightly move bbox to ensure they are treated as updates to same track
            # And ensure crop content changes to generate potentially "new" features if mockreid relies on it.
            dx = i*2
            current_bbox = [base_bbox[0]+dx, base_bbox[1]+dx, base_bbox[2]+dx, base_bbox[3]+dx, 0.9]
            detections.append(np.array([current_bbox]))
            cv2.rectangle(self.dummy_frame, (current_bbox[0],current_bbox[1]), (current_bbox[2],current_bbox[3]), (60+i*10,)*3, -1)

        # Frame 1
        _ = self.tracker.update(detections[0], self.dummy_frame)
        track1 = self.tracker.tracks[0]
        self.assertEqual(len(track1.features), 1)
        feature1 = track1.features[0].copy()

        # Frame 2
        _ = self.tracker.update(detections[1], self.dummy_frame)
        self.assertEqual(len(track1.features), 2)
        feature2 = track1.features[1].copy()
        np.testing.assert_array_equal(track1.features[0], feature1, "Feature 1 should still be there.")

        # Frame 3 - nn_budget exceeded, oldest (feature1) should be popped
        _ = self.tracker.update(detections[2], self.dummy_frame)
        self.assertEqual(len(track1.features), 2, "Gallery size should not exceed nn_budget.")
        feature3 = track1.features[1].copy() # Newest is at the end
        np.testing.assert_array_equal(track1.features[0], feature2, "Feature 2 should now be the oldest.")
        # Check that feature1 is gone (not equal to feature3, and feature2 is now first)
        self.assertFalse(np.array_equal(track1.features[0], feature1)) 

        # Frame 4
        _ = self.tracker.update(detections[3], self.dummy_frame)
        self.assertEqual(len(track1.features), 2)
        np.testing.assert_array_equal(track1.features[0], feature3, "Feature 3 should now be the oldest.")


if __name__ == '__main__':
    # Need to import cv2 for the setUp and some tests if not already.
    # It's used to create dummy content in self.dummy_frame for cropping.
    import cv2
    unittest.main()
