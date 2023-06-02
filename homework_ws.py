from itertools import combinations
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from collections import deque
import math

# 参数设置
N = 2000
k = 6
p = 0.1
# for _ in range(1, 10):
#     # 生成 WS 小世界网络
#     G = nx.watts_strogatz_graph(N, k, p)

#     # 计算最大连通子图的规模
#     largest_cc = max(nx.connected_components(G), key=len)
#     print(f"最大连通子图的规模: {len(largest_cc)}")

#     # 计算最大连通子图的平均路径长度
#     largest_cc_subgraph = G.subgraph(largest_cc)
#     average_shortest_path_length = nx.average_shortest_path_length(largest_cc_subgraph)
#     print(f"最大连通子图的平均路径长度: {average_shortest_path_length}")
#     print(f"By formula, Average path length is:{math.log(N*k*p/(k*k*p))}" )
    

# 生成 WS 小世界网络
G = nx.watts_strogatz_graph(N, k, p)
# print(nx.average_shortest_path_length(G))
# def degree_distribution(G):
#     # 获取所有节点的度并创建Counter对象，记录每个度出现的次数
#     counts = Counter(deg for _, deg in G.degree)
    
#     # 找到最小和最大的度
#     min_degree, max_degree = min(counts), max(counts)
    
#     # 遍历所有可能的度，输出对应的出现次数
#     for d in range(min_degree, max_degree + 1):
#         count = counts[d] # 如果d不存在于counts中，返回0
        
#     # 绘制度分布直方图
#     plt.bar(counts.keys(), counts.values())
#     plt.xlabel('Degree')
#     plt.ylabel('Count')
#     plt.title('Degree Distribution')
#     plt.xticks(range(min_degree, max_degree + 1))
#     plt.show()
    
# degree_distribution(G)
# 计算最大连通子图的规模
# largest_cc = max(nx.connected_components(G), key=len)
# print(f"最大连通子图的规模: {len(largest_cc)}")

# 计算最大连通子图的平均路径长度
# largest_cc_subgraph = G.subgraph(largest_cc)
# average_shortest_path_length = nx.average_shortest_path_length(largest_cc_subgraph)

print(f"最大连通子图的平均路径长度: {nx.average_shortest_path_length(G)}")
print(f"By formula, Average path length is:{math.log(N*k*p)/(k*k*p)}" )