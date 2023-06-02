import numpy as np
import time

def Cn(MatrixAdjacency_Train):
    similarity_StartTime = time.perf_counter()
    Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train)
    
    similarity_EndTime = time.perf_counter()
    print(f"    SimilarityTime: {similarity_EndTime - similarity_StartTime} s")
    return Matrix_similarity