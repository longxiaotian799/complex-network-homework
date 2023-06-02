import networkx as nx
import numpy as np

file_path = "karate1.txt"
# 从文件中读取边列表，创建一个图，其中节点的类型为整数，边的权重为整数
G = nx.read_edgelist(
    file_path,
    nodetype=int,
    data=(('weight', int),)
)

# 获取节点数量
num_nodes = G.number_of_nodes()

# 创建空的邻接矩阵（使用 numpy）
shortest_path_count_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

# 计算所有节点对之间的最短路径
shortest_paths = dict(nx.all_pairs_shortest_path(G))

# 统计所有节点对之间的最短路径数量并填充邻接矩阵
for source_node, paths_from_source in shortest_paths.items():
    for target_node, path in paths_from_source.items():
        # 将节点标签转换为矩阵索引（假设节点标签从 1 开始）
        source_idx = source_node - 1
        target_idx = target_node - 1

        # 统计最短路径数量
        shortest_path_count = 1 if source_node != target_node else 0

        # 在矩阵中存储最短路径数量
        shortest_path_count_matrix[source_idx, target_idx] = shortest_path_count

# 输出邻接矩阵
np.savetxt('shortest_path_count_matrix_gpt.txt', shortest_path_count_matrix, fmt='%d', delimiter=' ')
print(shortest_path_count_matrix)