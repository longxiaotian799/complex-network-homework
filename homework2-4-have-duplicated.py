from collections import Counter
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import copy
'''
配置模型（configuration model）是一种随机生成具有给定度分布的复杂网络的方法。生成配置网络的基本步骤如下： 

1. 确定网络的顶点（节点）数量 N。
2. 为每个顶点分配一个度数。度数可以通过输入的度分布（如幂律分布、泊松分布等）生成，或者你可以自定义每个节点的度数。
3. 为每个顶点创建度数对应的“桩子”（stubs），桩子是连接顶点的一半边。
4. 随机选择一对桩子并连接它们，形成一条边。请注意，多重边（同一对节点间存在多条边）和自环（节点连接到其自身）是允许的，但在某些应用中可能需要避免。
5. 重复步骤4，直到所有桩子都被连接。
'''


def degree_distribution(G):
    # 获取所有节点的度并创建 Counter 对象，记录每个度出现的次数
    counts = Counter(deg for _, deg in G.degree)
    
    # 初始化一个空列表来存储输出
    output = []
    
    # 遍历 Counter 对象
    for degree, count in counts.items():
        # 将每个度数添加到输出列表中，次数等于其在 Counter 中的计数值
        output.extend([degree] * count)
    print(output)
    return output
def create_configuration_model(N, degree_sequence):
    # 创建一个空的无向图
    G = nx.MultiGraph()

    # 添加 N 个节点
    G.add_nodes_from(range(N))

    # 生成每个节点的桩子
    stubs = [node for node, degree in enumerate(degree_sequence) for _ in range(degree)]
    print(stubs)
    # 随机连接桩子，生成边
    i = 0
    while len(stubs) > 1:
        stub1 = random.choice(stubs)
        stub = copy.deepcopy(stubs)
        stub.remove(stub1)
        stub2 = random.choice(stub)
        print(f"{i}:{stub1} {stub2}")
        G.add_edge(stub1, stub2)
        print(f"add {stub1} {stub2}")
        i += 1
        print(i)
        stubs.remove(stub1)
        stubs.remove(stub2)
        print(f"remove {stub1} {stub2}")
        print(stubs)
    return G

# 定义文件路径
file_path = 'karate1.txt'

# 从文件中读取边列表，创建一个图，其中节点的类型为整数，边的权重为整数
G = nx.read_edgelist(
    file_path,
    nodetype=int,
    data=(('weight', int),)
)
nx.draw(G, with_labels=True)
plt.show()
# 获取原始图的节点数和度分布
N = G.number_of_nodes()
degree_sequence = degree_distribution(G)
print(degree_sequence)
# 创建一个具有email网络节点个数和给定度序列的配置模型
G_config = create_configuration_model(N, degree_sequence)
nx.draw(G_config, with_labels=True)
plt.show()
# 计算配置模型的度直方图并删除值为 0 的元素
histogram = nx.degree_histogram(G_config)
histogram = [x for x in histogram if x != 0]

# 计算配置模型的度分布
counts = Counter(deg for _, deg in G_config.degree)
degree_counts = list(counts.values())

# 比较度直方图和度分布是否相同
print(f"{histogram}")
print(f"{degree_counts}")
comparison_result = sorted(histogram, reverse=True) == sorted(degree_counts, reverse=True)
print(f"度分布进行比较：{comparison_result}")

# N = G.number_of_nodes()
# degree_sequence = degree_distribution(G)
# G = create_configuration_model(N, degree_sequence)
# histogram = nx.degree_histogram(G)
# for x in histogram[:]:
#     if x == 0:
#         histogram.remove(x)
# counts = Counter(deg for _, deg in G.degree)
# dic = dict(counts)
# d = []
# for u, v in dic.items():
#     d.append(v)
# print(f"度分布进行比较：{sorted(histogram, reverse=True) == sorted(d, reverse=True)}")