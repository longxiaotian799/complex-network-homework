import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
def generate_ba_network(n, m, seed=None):
    random.seed(seed)  # 初始化随机数生成器的种子
    nodes = list(range(n))  # 创建节点列表，包含从 0 到 n-1 的整数
    edges = [(i, j) for i in nodes for j in nodes if i != j]  # 创建边列表，包含所有不重复的边
    adj_dict = {i: set() for i in nodes}  # 创建邻接字典，包含所有节点的邻居
    for i in nodes:
        adj_dict[i].update([j for j in nodes if i != j])  # 将所有节点的邻居初始化为所有其他节点
    for i in range(m, n):
        targets = set()
        while len(targets) < m:
            chosen = random.choice(nodes)  # 随机选择一个节点作为目标节点
            targets.add(chosen)  # 将目标节点添加到目标集合中
        nodes.append(i)  # 将新节点添加到节点列表中
        for target in targets:
            edges.append((i, target))  # 将新节点与目标节点之间添加一条边
            adj_dict[i].add(target)  # 将目标节点添加到新节点的邻居中
            adj_dict[target].add(i)  # 将新节点添加到目标节点的邻居中
    return adj_dict  # 返回邻接字典


n = 100
m = 1
seed = 42
adj_dict = generate_ba_network(n, m, seed)
# print(adj_dict)

def clustering_coefficient(G):
    '''在上述代码中, 我们首先将聚类系数 coefficient 初始化为 0 , 然后遍历图中的每个节点。
对于每个节点, 我们获取其所有邻居节点, 并计 算该节点的度数 $k$ 。
如果该节点的度数大于 1 , 则计算该节点的聚类系数 num_edges, 并将其加入总聚类系数中。
在计算聚类系数时, 我们使用 combinations 函数生成该节点所有邻居节点中的两两组合, 然后计算相邻节点之间是否存在边。
最后, 我们将总聚类系数除以节点数, 得到平均聚类系数, 并返回该值。
这样做的时间复杂度为 $O\left(n^2\right)$, 其中 $n$ 是节点数。由于 combinations 函数的空间复杂度为 $O\left(k^2\right)$, 其中 $k$ 是节点的平均度数, 因此总的空间复杂度为 $O\left(n k^2\right)$ 。'''
    # 计算图对象 G 的平均聚类系数,输入是图对象，输出是平均聚类系数
    coefficient = 0  # 初始化聚类系数为0
    for node in G.nodes:  # 遍历图中每个节点
        neighbors = list(G.neighbors(node))  # 获取该节点的所有邻居节点
        k = len(neighbors)  # 该节点的度数
        if k > 1:  # 只有度数大于1的节点才有聚类系数
            # 计算该节点的聚类系数
            num_edges = sum(1 for v, w in combinations(neighbors, 2) if G.has_edge(v, w))
            coefficient += num_edges / (k * (k - 1))
    print('\nAverage Clustering Coefficient:')
    return 2 * coefficient / G.number_of_nodes()  # 返回平均聚类系数

G = nx.Graph(adj_dict)
print(clustering_coefficient(G))
