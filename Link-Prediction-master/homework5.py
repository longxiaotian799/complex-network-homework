import numpy as np
import time
import os
import Initialize
import Evaluation_Indicators.AUC
import similarity_indicators.CommonNeighbor
import similarity_indicators.AA
import similarity_indicators.RA
import similarity_indicators.LP
import similarity_indicators.Katz
import similarity_indicators.ACT

def precision(train_matrix, test_matrix, similarity_matrix, L, max_node_num):
    """计算链接预测的精确度。
    参数:
    train_matrix: 训练数据的邻接矩阵
    test_matrix: 测试数据的邻接矩阵
    similarity_matrix: 相似度得分矩阵
    L: 计算精确度的参数
    max_node_num: 网络中的总节点数量
    返回:
    precision_score: 链接预测的精确度"""

    # 基于训练数据调整相似度矩阵，并只保留上三角部分
    adjusted_similarity_matrix = np.triu(similarity_matrix - similarity_matrix * train_matrix)

    # 仅保留测试矩阵的上三角部分，以避免重复计算
    test_matrix = np.triu(test_matrix)
    # print(f"test_matrix:{test_matrix}")
    # 计算测试集和不存在的边的并集
    non_edge_mask = np.ones(max_node_num) - train_matrix - np.eye(max_node_num)
    upper_triangular_non_edge_mask = np.triu(non_edge_mask)

    # 计算测试集和不存在的边的相似度得分
    similarity_scores = adjusted_similarity_matrix * upper_triangular_non_edge_mask

    # 将相似度得分按降序排序并获取索引
    sort_index = np.argsort(similarity_scores, axis=None)[::-1]

    # 获取前L条边的索引
    top_L_indices = np.unravel_index(sort_index[:L], similarity_scores.shape)
    # print(f"top_L_indices:{top_L_indices}")
    # 计算在测试集中的前L条边的数量
    m = np.sum(test_matrix[top_L_indices] != 0)
    # print(f"test_matrix[top_L_indices]:{test_matrix[top_L_indices]}")
    # print(f"m:{m}")
    # 计算精确度
    precision_score = float(m) / L

    return precision_score


for _ in range(1):
    startTime = time.time()
    NetFile = 'Data/Email.txt'
    NetName = 'Email'

    print("\nLink Prediction start:\n")
    TrainFile_Path = 'Data\\'+NetName+'\\Train.txt'
    if os.path.exists(TrainFile_Path):
        Train_File = 'Data\\'+NetName+'\\Train.txt'
        Test_File = 'Data\\'+NetName+'\\Test.txt'
        MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum = Initialize.Init2(Test_File, Train_File)
    else:
        MatrixAdjacency_Net, MaxNodeNum = Initialize.Init(NetFile)
        MatrixAdjacency_Train, MatrixAdjacency_Test = Initialize.Divide(NetFile, MatrixAdjacency_Net, MaxNodeNum, NetName)
    # print(f"MatrixAdjacency_Test:\n{MatrixAdjacency_Test}")
    # print(f"MatrixAdjacency_Test_sum:{np.sum(MatrixAdjacency_Test)}")
    similarity_StartTime = time.time()

    for Method in range(6):
        if Method == 0:
            print('----------Cn----------')
            Matrix_similarity = similarity_indicators.CommonNeighbor.Cn(MatrixAdjacency_Train)
        elif Method == 1:
            print('----------RA----------')
            Matrix_similarity = similarity_indicators.RA.RA(MatrixAdjacency_Train)
        elif Method == 2:
            print('----------Katz----------')
            Matrix_similarity = similarity_indicators.Katz.Katz(MatrixAdjacency_Train)
        elif Method == 3:
            print('----------ACT----------')
            Matrix_similarity = similarity_indicators.ACT.ACT(MatrixAdjacency_Train)
        elif Method == 4:
            print('----------AA----------')
            Matrix_similarity = similarity_indicators.AA.AA(MatrixAdjacency_Train)
        elif Method == 5:
            print('----------LP----------')
            Matrix_similarity = similarity_indicators.LP.LP(MatrixAdjacency_Train)
        else:
            print("Method Error!")
        Evaluation_Indicators.AUC.Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, MaxNodeNum)
        print(f"precision(L=100):{precision(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, 100, MaxNodeNum)}")
        print(f"Precision(L=200):{precision(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, 200, MaxNodeNum)}")

    similarity_EndTime = time.time()
    print('----------汇总----------')
    print("All SimilarityTime: {} s".format(similarity_EndTime - similarity_StartTime))

    endTime = time.time()
    print(f"\nRunTime: {endTime - startTime} s")
