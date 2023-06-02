import numpy as np
import time

def LP(MatrixAdjacency_Train):
    similarity_StartTime = time.process_time()

    Parameter = 0.01
    Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train) + np.dot(np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train), MatrixAdjacency_Train) * Parameter

    similarity_EndTime = time.process_time()
    print("    SimilarityTime: %f s" % (similarity_EndTime - similarity_StartTime))

    return Matrix_similarity
