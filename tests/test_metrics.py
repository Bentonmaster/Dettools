import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.metrics import cosine_distance

class TestMetrics(unittest.TestCase):
    def test_cosine_distance_basic(self):
        f1 = np.array([[1.0, 0.0, 0.0]])
        f2 = np.array([[1.0, 0.0, 0.0]]) # Identical
        self.assertAlmostEqual(cosine_distance(f1, f2)[0, 0], 0.0, places=6)

        f3 = np.array([[-1.0, 0.0, 0.0]]) # Opposite
        self.assertAlmostEqual(cosine_distance(f1, f3)[0, 0], 2.0, places=6)
        
        f4 = np.array([[0.0, 1.0, 0.0]]) # Orthogonal
        self.assertAlmostEqual(cosine_distance(f1, f4)[0, 0], 1.0, places=6)

    def test_cosine_distance_multiple_features(self):
        f1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        f2 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        
        expected_distances = np.array([
            [0.0, 1.0, 2.0], # Distances from f1[0] to f2[0], f2[1], f2[2]
            [1.0, 0.0, 1.0]  # Distances from f1[1] to f2[0], f2[1], f2[2]
        ])
        distances = cosine_distance(f1, f2)
        np.testing.assert_array_almost_equal(distances, expected_distances, decimal=6)

    def test_cosine_distance_different_magnitudes(self):
        f1 = np.array([[2.0, 0.0, 0.0]]) # Magnitude 2
        f2 = np.array([[5.0, 0.0, 0.0]]) # Magnitude 5, same direction
        # Cosine distance depends only on angle, not magnitude
        self.assertAlmostEqual(cosine_distance(f1, f2)[0, 0], 0.0, places=6)

        f3 = np.array([[-3.0, 0.0, 0.0]]) # Opposite direction, magnitude 3
        self.assertAlmostEqual(cosine_distance(f1, f3)[0, 0], 2.0, places=6)


    def test_cosine_distance_zero_vectors(self):
        f1_zero = np.array([[0.0, 0.0, 0.0]])
        f2_valid = np.array([[1.0, 2.0, 3.0]])
        
        # Distance from zero vector to a valid vector
        dist_zero_to_valid = cosine_distance(f1_zero, f2_valid)
        self.assertAlmostEqual(dist_zero_to_valid[0, 0], 1.0, places=6, 
                               msg="Distance from zero vector to valid vector should be 1.0")

        # Distance from a valid vector to a zero vector
        dist_valid_to_zero = cosine_distance(f2_valid, f1_zero)
        self.assertAlmostEqual(dist_valid_to_zero[0, 0], 1.0, places=6,
                               msg="Distance from valid vector to zero vector should be 1.0")

        f3_zero = np.array([[0.0, 0.0, 0.0]])
        # Distance between two zero vectors
        dist_zero_to_zero = cosine_distance(f1_zero, f3_zero)
        self.assertAlmostEqual(dist_zero_to_zero[0, 0], 1.0, places=6,
                               msg="Distance between two zero vectors should be 1.0")

        # Test with multiple zero vectors and valid vectors
        f_multi1 = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]])
        f_multi2 = np.array([[0.0, 0.0], [3.0, 3.0], [0.0, 0.0]])
        
        expected_distances = np.array([
            [1.0, 0.0, 1.0], # [1,1] vs [0,0], [1,1] vs [3,3], [1,1] vs [0,0]
            [1.0, 1.0, 1.0], # [0,0] vs [0,0], [0,0] vs [3,3], [0,0] vs [0,0]
            [1.0, 0.0, 1.0]  # [2,2] vs [0,0], [2,2] vs [3,3], [2,2] vs [0,0]
        ])
        # Note: [1,1] vs [3,3] -> dist 0.0
        #       [0,0] vs [3,3] -> dist 1.0
        #       [0,0] vs [0,0] -> dist 1.0
        distances = cosine_distance(f_multi1, f_multi2)
        np.testing.assert_array_almost_equal(distances, expected_distances, decimal=6)


    def test_cosine_distance_empty_input(self):
        f_empty_N0_D3 = np.empty((0, 3)) # N=0, D=3
        f_empty_N0_D0 = np.empty((0, 0)) # N=0, D=0 (special case if allowed, though typically D > 0)

        f_valid_N1_D3 = np.array([[1.0, 0.0, 0.0]])
        f_valid_N2_D3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        # Test with first argument empty
        dist1 = cosine_distance(f_empty_N0_D3, f_valid_N1_D3)
        self.assertEqual(dist1.shape, (0, 1))

        dist1_b = cosine_distance(f_empty_N0_D3, f_valid_N2_D3)
        self.assertEqual(dist1_b.shape, (0, 2))

        # Test with second argument empty
        dist2 = cosine_distance(f_valid_N1_D3, f_empty_N0_D3)
        self.assertEqual(dist2.shape, (1, 0))

        dist2_b = cosine_distance(f_valid_N2_D3, f_empty_N0_D3)
        self.assertEqual(dist2_b.shape, (2, 0))

        # Test with both arguments empty
        dist3 = cosine_distance(f_empty_N0_D3, f_empty_N0_D3)
        self.assertEqual(dist3.shape, (0, 0))
        
        # Test with D=0 if N>0 (should likely error or be handled by norm, but problem implies NxM output)
        # My implementation's np.linalg.norm would raise on D=0 if N > 0.
        # The problem statement focuses on N=0.
        # If features1.shape[0] == 0, it returns empty (N, M) = (0, M).
        # If N > 0 but D = 0, linalg.norm will raise.
        # This is fine; D must be > 0 for "features".
        # The prompt's test case implies (0,3) and (1,3) -> (0,1) output, which is handled.

    def test_cosine_distance_dimension_mismatch(self):
        f1 = np.array([[1.0, 0.0]])      # D=2
        f2 = np.array([[1.0, 0.0, 0.0]]) # D=3
        with self.assertRaisesRegex(ValueError, "Feature dimensions must match: 2 != 3"):
            cosine_distance(f1, f2)

    def test_single_feature_vectors(self):
        # Test when inputs are 1D arrays (single feature vector)
        f1_1d = np.array([1.0, 0.0, 0.0])
        f2_1d_ident = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_distance(f1_1d, f2_1d_ident)[0,0], 0.0, places=6)

        f2_1d_orth = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(cosine_distance(f1_1d, f2_1d_orth)[0,0], 1.0, places=6)

        f1_2d_single = np.array([[1.0, 0.0, 0.0]]) # Already 2D
        self.assertAlmostEqual(cosine_distance(f1_1d, f1_2d_single)[0,0], 0.0, places=6)
        self.assertAlmostEqual(cosine_distance(f1_2d_single, f1_1d)[0,0], 0.0, places=6)

    def test_numerical_stability_clipping(self):
        # Create vectors that are extremely close to identical or opposite
        # such that dot product might slightly exceed 1.0 or -1.0
        f_base = np.random.rand(1, 128).astype(np.float32) # Use float32 for more pronounced errors
        f_base = f_base / np.linalg.norm(f_base)

        # Almost identical vector
        f_almost_ident = f_base.copy()
        f_almost_ident[0,0] += 1e-8 # Tiny perturbation
        # No need to renormalize, perturbation is too small to change norm significantly for this test.
        
        # Almost opposite vector
        f_almost_opp = -f_base.copy()
        f_almost_opp[0,0] += 1e-8

        # Without clipping, 1.0 - (1.0 + epsilon) could be negative.
        # Or 1.0 - (-1.0 - epsilon) could be > 2.0.
        dist_ident = cosine_distance(f_base, f_almost_ident)[0,0]
        self.assertTrue(0.0 <= dist_ident <= 2.0, f"Identical distance out of bounds: {dist_ident}")
        self.assertAlmostEqual(dist_ident, 0.0, places=5) # Should be very close to 0

        dist_opp = cosine_distance(f_base, f_almost_opp)[0,0]
        self.assertTrue(0.0 <= dist_opp <= 2.0, f"Opposite distance out of bounds: {dist_opp}")
        self.assertAlmostEqual(dist_opp, 2.0, places=5) # Should be very close to 2


if __name__ == '__main__':
    unittest.main()
