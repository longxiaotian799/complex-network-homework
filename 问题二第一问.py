import networkx as nx
import numpy as np

# 生成N=100，连边概率p=0.05的随机网络
G = nx.erdos_renyi_graph(100, 0.05)

# 计算最大连通子图的规模
largest_cc = max(nx.connected_components(G), key=len)
print(f"最大连通子图的规模为: {len(largest_cc)}")

# 求最大连通子图的平均路径长度和平均聚类系数
avg_path_length = nx.average_shortest_path_length(G.subgraph(largest_cc))
avg_clustering_coef = nx.average_clustering(G.subgraph(largest_cc))
print(f"最大连通子图的平均路径长度为: {avg_path_length}")
print(f"最大连通子图的平均聚类系数为: {avg_clustering_coef}")

# 统计度分布
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
degree_count = np.zeros(max(degree_sequence) + 1)
for degree in degree_sequence:
    degree_count[degree] += 1
degree_distribution = degree_count / sum(degree_count)

# 输出统计结果
print("度分布: ")
for i, p in enumerate(degree_distribution):
    print(f"度为 {i} 的节点比例为: {p:.4f}")