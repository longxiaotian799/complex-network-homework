# import numpy as np
# import time

# def ACT(MatrixAdjacency_Train):
#     similarity_StartTime = time.perf_counter()

#     Matrix_D = np.diag(np.sum(MatrixAdjacency_Train, axis=0))
#     Matrix_Laplacian = Matrix_D - MatrixAdjacency_Train
#     INV_Matrix_Laplacian = np.linalg.pinv(Matrix_Laplacian)

#     Array_Diag = np.diag(INV_Matrix_Laplacian)
#     Matrix_ONE = np.ones([MatrixAdjacency_Train.shape[0], MatrixAdjacency_Train.shape[0]])
#     Matrix_Diag = Array_Diag * Matrix_ONE

#     Matrix_similarity = Matrix_Diag + Matrix_Diag.T - (2 * INV_Matrix_Laplacian)

#     # Check for zero values in the denominator and handle them
#     zero_indices = np.where(Matrix_similarity == 0)
#     non_zero_indices = np.where(Matrix_similarity != 0)

#     Matrix_similarity[zero_indices] = 1  # Assign a non-zero value to avoid division by zero
#     Matrix_similarity = Matrix_ONE / Matrix_similarity

#     Matrix_similarity[zero_indices] = 0  # Revert back to zero values

#     similarity_EndTime = time.perf_counter()
#     print(f"    SimilarityTime: {similarity_EndTime - similarity_StartTime} s")
#     return Matrix_similarity
# import numpy as np
# import time
# def ACT(MatrixAdjacency_Train):
#     similarity_StartTime = time.perf_counter()
#     # print(f"MatrixAdjacency_Train:\n{MatrixAdjacency_Train}")
#     # Generate degree matrix
#     Matrix_D = np.diag(np.sum(MatrixAdjacency_Train, axis=1))
    
#     # Calculate Laplacian matrix
#     Matrix_Laplacian = Matrix_D - MatrixAdjacency_Train
    
#     # Calculate pseudo inverse of Laplacian matrix
#     INV_Matrix_Laplacian = np.linalg.pinv(Matrix_Laplacian)
    
#     # Generate matrix with diagonal elements of pseudo inverse of Laplacian matrix
#     Matrix_Diag = np.tile(np.diag(INV_Matrix_Laplacian), (MatrixAdjacency_Train.shape[1], 1))
#     # print(f"Matrix_Diag:\n{Matrix_Diag}")
#     # Calculate denominator and handle potential zero values
#     denominator = Matrix_Diag + Matrix_Diag.T - 2 * INV_Matrix_Laplacian
#     denominator[denominator == 0] = np.finfo(float).eps

#     # Then use this denominator for division
#     Matrix_similarity = 1.0 / denominator

#     # Handle potential NaN and infinite values in Matrix_similarity
#     Matrix_similarity[np.isnan(Matrix_similarity)] = 0
#     Matrix_similarity[np.isinf(Matrix_similarity)] = 0

#     similarity_EndTime = time.perf_counter()
#     print(f"    SimilarityTime: {similarity_EndTime - similarity_StartTime} s")
    
#     return Matrix_similarity
import numpy as np
import time

def ACT(MatrixAdjacency_Train):
    similarity_StartTime = time.process_time()

    # 计算拉普拉斯矩阵L和其伪逆L_plus
    DegreeMatrix = np.diagflat(np.sum(MatrixAdjacency_Train, axis=1))
    LaplaceMatrix = DegreeMatrix - MatrixAdjacency_Train
    L_plus = np.linalg.pinv(LaplaceMatrix)

    # 计算ACT相似性指标
    n = MatrixAdjacency_Train.shape[0]
    Matrix_similarity = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            lxx = L_plus[x, x]
            lyy = L_plus[y, y]
            lxy = L_plus[x, y]

            # 检查lxx, lyy, lxy中是否有无效的值（例如NaN或者inf）
            if np.isnan(lxx) or np.isinf(lxx) or np.isnan(lyy) or np.isinf(lyy) or np.isnan(lxy) or np.isinf(lxy):
                Matrix_similarity[x, y] = 0
            else:
                Matrix_similarity[x, y] = 1 / (lxx + lyy - 2 * lxy + 1e-20)  # 添加一个非常小的数值以避免除以零
    np.savetxt('Sxy.txt',Matrix_similarity[:10,:10], fmt='%f.1', delimiter=' ', newline='\n')
    similarity_EndTime = time.process_time()
    print("    SimilarityTime: %f s" % (similarity_EndTime - similarity_StartTime))

    return Matrix_similarity
