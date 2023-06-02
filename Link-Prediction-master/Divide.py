def Divide(NetFile,netname,train_rate):
    """按比例划分训练集与测试集并且保证训练集连通
    返回训练集、测试集邻接矩阵并写入文件中"""
    #1.读入文件
    data = np.loadtxt(NetFile)
    enum = len(data)  # 网络总边数
    enum_test = int(float(1 - train_rate) * enum)  # 测试集边数
    V = set()  # 顶点集合
    for item in data:
        V.add(int(item[0]))
        V.add(int(item[1]))
    vnum = len(V)
    # 2.创建邻接矩阵
    A = np.zeros([vnum, vnum], dtype=int)
    for item in data:
        v1 = int(item[0]) - 1
        v2 = int(item[1]) - 1
        A[v1, v2] = A[v2, v1] = 1
    #3.计算训练集以及测试集邻接矩阵
    Matrix_Train = A    #初始训练集
    # print(f'初始训练集\n{Matrix_Train}')
    Matrix_Test = np.zeros((vnum, vnum))#初始测试集
    # 3.1在原有边中随机选enum_test条边，存在测试集合中，并从原集合中删除
    while len(np.nonzero(Matrix_Test)[0]) < enum_test:
        index = np.random.randint(low=0, high=len(data), size=1)
        v1 = int(data[index, 0]) - 1
        v2 = int(data[index, 1]) - 1
        Matrix_Train[v1, v2] = Matrix_Train[v2, v1] = 0
        # 3.2判断所选边的两个端点是否可达，若不可达需要重新选边
        temp = Matrix_Train[v1]  # 存储v1的邻居
        flag = 0  # 标记这条边是否可以被删除
        count = 0  # 记录v1到v2之间的步数
        v1_v2 = np.dot(temp, Matrix_Train) + temp  # v1 2步可到达的点
        if v1_v2[v2] > 0:
            flag = 1
        else:
            count = 1
        temp1 = np.int64(v1_v2 > 0)
        # 3.3直到v1可达的点到达稳定状态，如果仍然不能到达v2，则v1-v2不可达
        while len((temp1 - temp).nonzero()[0]) != 0:
            temp = temp1
            v1_v2 = np.dot(temp, Matrix_Train) + temp# v1 n步可到达的点
            count += 1
            if v1_v2[v2] > 0:
                flag = 1
                break
            if count >= vnum:
                flag = 0
        if flag == 1:
            data = np.delete(data, index, axis=0)
            Matrix_Test[v1, v2] = 1
        else:
            data = np.delete(data, index, axis=0)
            Matrix_Train[v1, v2] = Matrix_Train[v2, v1] = 1
    Matrix_Test = Matrix_Test + Matrix_Test.T
    print(f'网络大小：顶点数：{vnum}，边数：{enum},训练集比例：{train_rate}')
    # print(f'训练集邻接矩阵：\n{Matrix_Train}\n测试集邻接矩阵\n{Matrix_Test}')
    #4.写入文件
    WriteFile("Train", Matrix_Train, netname)
    WriteFile("Test", Matrix_Test, netname)
    return Matrix_Train, Matrix_Test
 
def WriteFile(filename,Matrix,netname):
    M=np.triu(Matrix)
    index=np.argwhere(M!=0)
    with open (f'data/{netname}/{filename}.txt','w') as f:
        np.savetxt(f,index,fmt="%d")
