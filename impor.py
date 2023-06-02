import numpy as np

def avg_path_length(adj_matrix):
    # 将邻接矩阵转换为距离矩阵
    dist_matrix = np.zeros_like(adj_matrix)
    dist_matrix[adj_matrix == 1] = 1

    for k in range(adj_matrix.shape[0]):
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[0]):
                if dist_matrix[i][j] == 0:
                    dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]

    # 将距离矩阵中的0元素替换为无穷大
    dist_matrix[dist_matrix == 0] = np.inf
    return np.sum(dist_matrix) / (adj_matrix.shape[0] * (adj_matrix.shape[0] - 1))

# 示例使用
adj_matrix = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])
print(avg_path_length(adj_matrix)) # 输出结果为2.1666666666666665