'''
Created on 2016年10月25日

@author: ZWT
'''
import csv
import numpy as np
import random

method = input("请输入1选择ProbS推荐，输入2选择HeatS推荐:")
Re_length = int(input("请输入所需推荐列表的长度:"))
timeNum = int(input("清输入AUC循环次数:"))

TrainSet_CsvFile = u"data/votes.csv"
ScoialSet_CsvFIle = u'data/social.csv'
ProbeSet_CsvFile = u"data/probeSet.csv"

TrainSet_userList = []#训练集user列表
TrainSet_itemList = []#训练集item列表
ProbeSet_userList = []#测试集user列表
ProbeSet_itemList = []#测试集item列表


######训练集TrainSet_CsvFile文件转邻接矩阵
with open(TrainSet_CsvFile,'r') as f:
    TrainSet = csv.reader(f)
    TrainSetData = []
    for line in TrainSet:
        TrainSet_userList.append(int(line[1])-1)
        TrainSet_itemList.append(int(line[2])-1)
        TrainSetData.append([int(line[1]),int(line[2])])
TrainSet_userList = set(TrainSet_userList)
TrainSet_itemList = set(TrainSet_itemList)
Matrix_Adjacency = np.zeros([len(TrainSet_userList),len(TrainSet_itemList)])
for n in range(len(TrainSetData)):
    i = int(TrainSetData[n][0]) - 1
    j = int(TrainSetData[n][1]) - 1
    Matrix_Adjacency[i,j] = 1
print("邻接矩阵转换："+str(Matrix_Adjacency.shape))

######社交网络ScoialSet_CsvFile文件求社交网络矩阵
with open(ScoialSet_CsvFIle,'r') as f:
    ScoialSet = csv.reader(f)
    ScoialSetData = []
    ScoialSet_rowList = []
    ScoialSet_colList = []
    for line in ScoialSet:
        ScoialSet_rowList.append(int(line[0]))
        ScoialSet_colList.append(int(line[1]))
        ScoialSetData.append([int(line[0]),int(line[1])])
ScoialSet_rowList = set(ScoialSet_rowList)
ScoialSet_colList = set(ScoialSet_colList)
Matrix_Scoial = np.zeros([len(ScoialSet_rowList),len(ScoialSet_colList)])
for n in range(len(ScoialSetData)):
    i = int(ScoialSetData[n][0]) - 1
    j = int(ScoialSetData[n][1]) - 1
    Matrix_Scoial[i,j] = 1
print("社交矩阵转换："+str(Matrix_Scoial.shape))

######测试集ProbeSet_CsvFile文件求用户喜欢列表
with open(ProbeSet_CsvFile,'r') as f:
    ProbeSet = csv.reader(f)
    ProbeSetData = []
    for line in ProbeSet:
        ProbeSet_userList.append(int(line[1])-1)
        ProbeSet_itemList.append(int(line[2])-1)
        ProbeSetData.append([int(line[1]),int(line[2])])
ProbeSet_userList = set(ProbeSet_userList)
ProbeSet_itemList = set(ProbeSet_itemList)
Re_item = {} #用户喜欢的商品列表字典
for i in range(len(ProbeSetData)):
    if int(ProbeSetData[i][0]) not in Re_item.keys():
        Re_item[int(ProbeSetData[i][0])] = []
        Re_item[int(ProbeSetData[i][0])].append(int(ProbeSetData[i][1])-1)
    else:
        Re_item[int(ProbeSetData[i][0])].append(int(ProbeSetData[i][1])-1)
print("用户喜欢的商品列表转换："+str(len(Re_item.keys())))

######训练集TrainSet_CsvFile文件求用户度，商品度
row = Matrix_Adjacency.shape[0]
col = Matrix_Adjacency.shape[1]
TrainSet_u_degree = np.zeros([row,1])
TrainSet_v_degree = np.zeros([col,1])
for i in range(row):
    TrainSet_u_degree[i] = np.sum(Matrix_Adjacency[i,:])
for i in range(col):
    TrainSet_v_degree[i] = np.sum(Matrix_Adjacency[:,i])

######算法
Matrix_F = np.zeros([row,col])
for t in range(10):
    if method == "1":
        Matrix_F = np.dot(np.dot(np.dot(Matrix_Adjacency, np.linalg.inv(Matrix_Adjacency.T + np.eye(row))), Matrix_Adjacency), np.linalg.inv(np.eye(col) + np.dot(Matrix_Adjacency.T, Matrix_Adjacency)))
    elif method == "2":
        Matrix_F = np.dot(Matrix_Adjacency, np.linalg.inv(np.eye(col) + np.dot(Matrix_Adjacency.T, Matrix_Adjacency)))

#######推荐列表
ReList = {}
for i in range(row):
    ReList[i+1] = []
    x = Matrix_F[i,:]
    x[i] = -1
    for j in range(Re_length):
        max_v = np.where(x == np.max(x))
        if len(max_v[0]) > 1:
            a = random.randint(0,len(max_v[0])-1)
            ReList[i+1].append(max_v[0][a])
            x[max_v[0][a]] = -1
        else:
            ReList[i+1].append(max_v[0][0])
            x[max_v[0][0]] = -1
print("推荐列表转换："+str(len(ReList.keys())))

#########测试集AUC
AUC = 0
for user in Re_item.keys():
    n = 0
    m = 0
    Re_item_v = Re_item[user]
    ReList_v = ReList[user]
    for t in range(timeNum):
        i = random.randint(0,len(Re_item_v)-1)
        j = random.randint(0,len(ReList_v)-1)
        if Matrix_F[user-1,Re_item_v[i]] > Matrix_F[user-1,ReList_v[j]]:
            n += 1
        elif Matrix_F[user-1,Re_item_v[i]] == Matrix_F[user-1,ReList_v[j]]:
            m += 1
    AUC += (n + 0.5*m)/timeNum
AUC /= len(Re_item.keys())
print("AUC值为："+str(AUC))