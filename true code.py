import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['KaiTi']     #设置字体为楷体
plt.rcParams['axes.unicode_minus']=False        #显示负号


def main():
    """"""
    path=[r"C:\Users\zhangwentao\Desktop\大三下学期\Complex Network\dolphins.txt"]
    name=['dolphins']
    result(path[0],name[0])

def result(path,name):
    """读入文件中的网络，返回平均路径长度、平均聚类系数以及统计度分布"""
    #1.读取文件
    G,A= Load_Graph(path)
    #2.计算平均路径
    avepath= Ave_path(A)
    #3.计算平均聚类系数
    c=clustering_coef(A)
    # 4.统计度分布
    K = np.sum(A, axis=0)  # 每个结点的度
    # print(K)
    deg, num = np.unique(K, return_counts=True)
    print("=============================")
    print(f'\t\t网络{name}')
    print(f'平均路径：{avepath}\n平均聚类系数：{c}\n度分布序列：{num}')
    plt.title("网络度分布")
    plt.bar(deg, num, width=0.80, color='b')
    plt.savefig(f"{name}网络度分布", dpi=600)
    plt.close()
    print("\npython程序包验证")
    model(G)



def model(G):
    """python包
    求图的平均路径长度，平均聚类系数，度分布"""
    print("Average path length:" + str(nx.average_shortest_path_length(G)))
    print("Average clustering coefficient:" + str(nx.average_clustering(G)))
    print("Degree distribution:" + str(nx.degree_histogram(G)))

def Floyed(adjMatrix):
    """Floyed算法获得最短路径矩阵
    --input--
    adjMatrix 连通片邻接矩阵
    --return--
    D 最短距离矩阵   D[i][j]第i个结点到第j个结点的最短距离
    """
    m = adjMatrix.shape[0]  # 结点数
    # 1.初始化距离矩阵 dij=1 or inf
    D = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            D[i][j] = adjMatrix[i][j]
            # 将不相连的结点距离设为无穷大
            if D[i][j] == 0 and i != j:
                inf = float('inf')
                D[i][j] = inf
    # 2.计算最短距离矩阵
    for k in range(m):
        for i in range(m):
            for j in range(m):
                if (D[i][k] + D[k][j] < D[i][j]):
                    D[i][j] = D[i][k] + D[k][j]
    return D


def Ave_path(adjMatrix): 
    """根据图的邻接矩阵求平均最短路径长度"""
    D = Floyed(adjMatrix)
    N = D.shape[0]  # 结点数
    L = np.sum(D)  # 对D求和
    ave_path = L / (N * (N - 1))
    return ave_path

def clustering_coef(adjMatrix):
    '''求聚类系数
    --input---
    adiMatrix 邻接矩阵
    ---return---
    C 数组 返回各个结点的聚类系数'''
    n = adjMatrix.shape[0]
    E = np.zeros((n, 1), dtype=int)  # 存储结点i邻居中的实际连边数
    C = np.zeros((n, 1), dtype=float)  # 存储结点i的聚类系数
    for i in range(n):
        neighbors = np.nonzero(adjMatrix[i])
        neighbors = neighbors[0]  # 返回结点i的邻居结点编号
        E[i] = 0  # 记录结点i邻居中的实际连边数
        for j in neighbors:
            for k in neighbors:
                if adjMatrix[j][k] == 1:
                    E[i] += 1
        k = len(neighbors)
        if k > 1:
            C[i] = E[i] / (k * (k - 1))
    c=float(sum(C) / len(C))
    return c

def Load_Graph(path):
    """读取文件，转化为点、边集,图"""
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    nodes = {}
    edges = []
    for line in lines:
        n = line.split()
        if not n:
            break
        nodes[n[0]] = 1
        nodes[n[1]] = 1
        w = 1
        if len(n) == 3:
            w = int(n[2])
        edges.append((n[0], n[1], w))
    nodes_, edges_ = in_order(nodes, edges)
    G = nx.Graph()
    G.add_nodes_from(nodes_)
    G.add_weighted_edges_from(edges_)
    A= nx.to_scipy_sparse_array(G).todense()
    return G,A


def in_order(nodes, edges):
    # rebuild graph with successive identifiers
    nodes = list(nodes.keys())
    nodes.sort()
    i = 0
    nodes_ = []
    d = {}
    for n in nodes:
        nodes_.append(i)
        d[n] = i
        i += 1
    edges_ = []
    for e in edges:
        edges_.append((d[e[0]], d[e[1]], e[2]))
    return (nodes_, edges_)

if __name__ == '__main__':
    main()