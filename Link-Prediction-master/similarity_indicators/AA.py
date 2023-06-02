import numpy as np
import time

def AA(MatrixAdjacency_Train):
    similarity_StartTime = time.process_time()

    logTrain = np.log1p(np.sum(MatrixAdjacency_Train, axis=0))
    logTrain = np.nan_to_num(logTrain)
    logTrain = logTrain.reshape((logTrain.shape[0], 1))
    MatrixAdjacency_Train_Log = np.divide(MatrixAdjacency_Train, logTrain, out=np.zeros_like(MatrixAdjacency_Train), where=logTrain!=0)
    MatrixAdjacency_Train_Log = np.nan_to_num(MatrixAdjacency_Train_Log)

    Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train_Log)

    similarity_EndTime = time.process_time()
    print("SimilarityTime: %f s" % (similarity_EndTime - similarity_StartTime))
    return Matrix_similarity
