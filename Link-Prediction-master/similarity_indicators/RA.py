import numpy as np
import time

def RA(MatrixAdjacency_Train):
    similarity_StartTime = time.process_time()

    RA_Train = np.sum(MatrixAdjacency_Train, axis=0)
    RA_Train = RA_Train.reshape((RA_Train.shape[0], 1))

    # Check for zero values in the denominator and handle them
    zero_indices = np.where(RA_Train == 0)
    non_zero_indices = np.where(RA_Train != 0)

    RA_Train[zero_indices] = 1  # Assign a non-zero value to avoid division by zero
    MatrixAdjacency_Train_Log = MatrixAdjacency_Train / RA_Train

    MatrixAdjacency_Train_Log[zero_indices] = 0  # Set the values as 0 where the denominator was zero

    Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train_Log)

    similarity_EndTime = time.process_time()
    print("    SimilarityTime: %f s" % (similarity_EndTime - similarity_StartTime))
    return Matrix_similarity
