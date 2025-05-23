import numpy as np

def cosine_distance(features1: np.ndarray, features2: np.ndarray) -> np.ndarray:
    """
    Calculates the cosine distance matrix between two sets of feature embeddings.

    Args:
        features1: A NumPy array of shape (N, D) where N is the number of
                   features and D is the feature dimension.
        features2: A NumPy array of shape (M, D) where M is the number of
                   features and D is the feature dimension.

    Returns:
        A NumPy array of shape (N, M) where element (i, j) is the
        cosine distance between features1[i] and features2[j].
        Distance is 1 - cosine_similarity.
    """
    # Handle empty inputs first
    if features1.shape[0] == 0 or features2.shape[0] == 0:
        return np.empty((features1.shape[0], features2.shape[0]))

    if features1.ndim == 1:
        features1 = features1.reshape(1, -1)
    if features2.ndim == 1:
        features2 = features2.reshape(1, -1)

    if features1.shape[1] != features2.shape[1]:
        raise ValueError(f"Feature dimensions must match: {features1.shape[1]} != {features2.shape[1]}")

    # Normalize each feature vector to unit length
    norm_f1 = np.linalg.norm(features1, axis=1, keepdims=True)
    norm_f2 = np.linalg.norm(features2, axis=1, keepdims=True)
    
    # Create masks for zero-norm vectors
    # These masks identify rows in features1 or features2 that were originally zero vectors
    mask_f1_zero_norm_rows = (norm_f1 == 0).flatten()
    mask_f2_zero_norm_rows = (norm_f2 == 0).flatten()

    # Replace zero norms with 1.0 to avoid division by zero during normalization.
    # The dot product involving these original zero vectors will be zero anyway.
    norm_f1[norm_f1 == 0] = 1.0
    norm_f2[norm_f2 == 0] = 1.0
    
    normalized_f1 = features1 / norm_f1
    normalized_f2 = features2 / norm_f2

    # Cosine similarity matrix
    similarity_matrix = np.dot(normalized_f1, normalized_f2.T)
    
    # For any row in features1 that was a zero vector, its similarity to all vectors in features2 is 0.
    similarity_matrix[mask_f1_zero_norm_rows, :] = 0.0
    # For any row in features2 that was a zero vector, its similarity to all vectors in features1 is 0.
    similarity_matrix[:, mask_f2_zero_norm_rows] = 0.0

    # Ensure similarity values are clipped between -1 and 1 due to potential floating point inaccuracies
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
    
    # Cosine distance is 1 - cosine_similarity
    distance_matrix = 1.0 - similarity_matrix
    return distance_matrix
