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

def Precision(Train,Test,Sim,L,Max_Nodenum):
    """计算Precision指标
        --input--
    Train: 训练集邻接矩阵
    Test：测试集邻接矩阵
    Sim: 相似性指标矩阵
    L 参数
    Max_Nodenum 网络总结点数
    --return--
    p 精确度"""
    Sim=np.triu(Sim-Sim*Train) # 训练集中不存在的边的相似度
    Test = np.triu(Test)  # 无向图,只需取矩阵的上三角，避免重复计算边
    # print(f'sim\n{Sim}')
    UandNE=np.ones(Max_Nodenum)-Train-np.eye(Max_Nodenum)   #原始集+不存在边集合中的边
    UNE=np.triu(UandNE)
    UNEPre=Sim*UNE#测试集+不存在边集合中的边的相似度矩阵
    # print(f'UNEpre:\n{UNEPre}')
    rows=UNEPre.shape[0]
    cols=UNEPre.shape[1]
    sort=UNEPre.flatten()#转化为1维数组
    sort_index=np.argsort(sort)[::-1]#实现降序排列
    #从大到小排序，取前L条边
    max_L=sort_index[0:L:1]
    #找到前l个在原矩阵中的对应下标
    index=[(int(i/cols),i%cols) for i in max_L]
    # print(f'index:{index}')
    m=0#记录有几条边在测试集中
    for k in range(len(index)):
        if Test[index[k][0],index[k][1]]!=0:
            m+=1
    # print(m)
    return float(m/L)

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

similarity_StartTime = time.time()
print(f'{MatrixAdjacency_Train}')
print('----------相似性指标----------')
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

    print("AUC: ", Evaluation_Indicators.AUC.Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, MaxNodeNum))
    print("Precision@100: ", Precision(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, 100, MaxNodeNum))
    print("Precision@200: ", Precision(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, 200, MaxNodeNum))

similarity_EndTime = time.time()
print('----------汇总----------')
print("All SimilarityTime: {} s".format(similarity_EndTime - similarity_StartTime))

# Calculate AUC
Evaluation_Indicators.AUC.Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, MaxNodeNum)

endTime = time.time()
print(f"\nRunTime: {endTime - startTime} s")
