import numpy as np
import time

def PA(MatrixAdjacency_Train):
    similarity_StartTime = time.process_time()

    deg_row = np.sum(MatrixAdjacency_Train, axis=0)
    deg_row = deg_row.reshape((deg_row.shape[0], 1))
    deg_row_T = deg_row.T

    Matrix_similarity = np.dot(deg_row, deg_row_T)

    similarity_EndTime = time.process_time()
    print("    SimilarityTime: %f s" % (similarity_EndTime - similarity_StartTime))
    return Matrix_similarity