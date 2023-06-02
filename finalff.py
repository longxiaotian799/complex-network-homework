from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

# 记录程序开始时间
start_time = time.time()

def clustering_coefficient(G):
    # 计算图对象 G 的平均聚类系数,输入是图对象，输出是平均聚类系数
    connect_number = 0
    coefficient = 0
    for node in list(G.nodes):
        neighbors = list(G.neighbors(node))
        for neighbors1 in neighbors: 
            for neighbors2 in neighbors:
                if neighbors1 == neighbors2:
                    continue
                connect_number = connect_number + int(G.has_edge(neighbors1, neighbors2))
            if len(list(G.neighbors(node))) != 1:
                coefficient = coefficient + connect_number / (len(list(G.neighbors(node))) * (len(list(G.neighbors(node))) - 1))
            connect_number = 0
    print('Average Clustering Coefficient:')
    return coefficient / G.number_of_nodes()
    # (nodes for nodes in list(G.neighbors(nodes)) G.adj, list(G.neighbors(1)))
    
    # print(G.degree)
# def clustering_coefficient(G):
#     # 计算图对象 G 的平均聚类系数,输入是图对象，输出是平均聚类系数
#     coefficient = 0  # 初始化聚类系数为0
#     for node in G.nodes:  # 遍历图中每个节点
#         neighbors = list(G.neighbors(node))  # 获取该节点的所有邻居节点
#         k = len(neighbors)  # 该节点的度数
#         if k > 1:  # 只有度数大于1的节点才有聚类系数
#             # 计算该节点的聚类系数
#             num_edges = sum(1 for v, w in combinations(neighbors, 2) if G.has_edge(v, w))
#             coefficient += num_edges / (k * (k - 1))
#     print('Average Clustering Coefficient:')
#     return 2 * coefficient / G.number_of_nodes()  # 返回平均聚类系数


# # 输出图的各种信息
# G.nodes # 返回一个节点列表。
# G.edges # 返回一组边列表。
# G.degree # 返回一个字典，其中键是节点，值是该节点的度数。 
# G.number_of_nodes() # 返回节点数。
# G.number_of_edges() # 返回边数。
# G.adj # 这是一个字典类型的变量，它的键为源节点，值为目标节点和权重。这个字典表示了一个加权有向图的邻接表。每个节点的所有出边在邻接表中都被存储为一个键值对，其中键为目标节点，值是一个包含一条边的属性字典，其中包括该边的权重。
# adj_matrix = nx.to_numpy_array(G) # 返回图的邻接矩阵。
# #输入上面的变量
# print("G.nodes:", G.nodes)
# print("G.edges:", G.edges)
# print("G.degree:", G.degree)
# print("G.nodes:", G.nodes)
# print("G.number_of_nodes:", G.number_of_nodes())
# print("G.number_of_edges:", G.number_of_edges())
# print("G.adj:", G.adj)
# print("nx.to_numpy_array(G):", nx.to_numpy_array(G))

'''输出结果
G.nodes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 18, 20, 22, 32, 31, 10, 28, 29, 33, 17, 34, 15, 16, 19, 21, 23, 24, 26, 30, 25, 27]
G.edges: [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 11), (1, 12), (1, 13), (1, 14), (1, 18), (1, 20), (1, 22), (1, 32), (2, 3), (2, 4), (2, 8), (2, 14), (2, 18), (2, 20), (2, 22), (2, 31), (3, 4), (3, 8), (3, 9), (3, 10), (3, 14), (3, 28), (3, 29), (3, 33), (4, 8), (4, 13), (4, 14), (5, 7), (5, 11), (6, 7), (6, 11), (6, 17), (7, 17), (9, 31), (9, 33), (9, 34), (14, 34), (20, 34), (32, 25), (32, 26), (32, 29), (32, 33), (32, 34), (31, 33), (31, 34), (10, 34), (28, 24), (28, 25), (28, 34), (29, 34), (33, 15), (33, 16), (33, 19), (33, 21), (33, 23), (33, 24), (33, 30), (33, 34), (34, 15), (34, 16), (34, 19), (34, 21), (34, 23), (34, 24), (34, 27), (34, 30), (24, 26), (24, 30), (26, 25), (30, 27)]
G.number_of_nodes: 34
G.number_of_edges: 78
G.adj: {1: {2: {'weight': 1}, 3: {'weight': 1}, 4: {'weight': 1}, 5: {'weight': 1}, 6: {'weight': 1}, 7: {'weight': 1}, 8: {'weight': 1}, 9: {'weight': 1}, 11: {'weight': 1}, 12: {'weight': 1}, 13: {'weight': 1}, 14: {'weight': 1}, 18: {'weight': 1}, 20: {'weight': 1}, 22: {'weight': 1}, 32: {'weight': 1}}, 2: {1: {'weight': 1}, 3: {'weight': 1}, 4: {'weight': 1}, 8: {'weight': 1}, 14: {'weight': 1}, 18: {'weight': 1}, 20: {'weight': 1}, 22: {'weight': 1}, 31: {'weight': 1}}, 3: {1: {'weight': 1}, 2: {'weight': 1}, 4: {'weight': 1}, 8: {'weight': 1}, 9: {'weight': 1}, 10: {'weight': 1}, 14: {'weight': 1}, 28: {'weight': 1}, 29: {'weight': 1}, 33: {'weight': 1}}, 4: {1: {'weight': 1}, 2: {'weight': 1}, 3: {'weight': 1}, 8: {'weight': 1}, 13: {'weight': 1}, 14: {'weight': 1}}, 5: {1: {'weight': 1}, 7: {'weight': 1}, 11: {'weight': 1}}, 6: {1: {'weight': 1}, 7: {'weight': 1}, 11: {'weight': 1}, 17: {'weight': 1}}, 7: {1: {'weight': 1}, 5: {'weight': 1}, 6: {'weight': 1}, 17: {'weight': 1}}, 8: {1: {'weight': 1}, 2: {'weight': 1}, 3: {'weight': 1}, 4: {'weight': 1}}, 9: {1: {'weight': 1}, 3: {'weight': 1}, 31: {'weight': 1}, 33: {'weight': 1}, 34: {'weight': 1}}, 11: {1: {'weight': 1}, 5: {'weight': 1}, 6: {'weight': 1}}, 12: {1: {'weight': 1}}, 13: {1: {'weight': 1}, 4: {'weight': 1}}, 14: {1: {'weight': 1}, 2: {'weight': 1}, 3: {'weight': 1}, 4: {'weight': 1}, 34: {'weight': 1}}, 18: {1: {'weight': 1}, 2: {'weight': 1}}, 20: {1: {'weight': 1}, 2: {'weight': 1}, 34: {'weight': 1}}, 22: {1: {'weight': 1}, 2: {'weight': 1}}, 32: {1: {'weight': 1}, 25: {'weight': 1}, 26: {'weight': 1}, 29: {'weight': 1}, 33: {'weight': 1}, 34: {'weight': 1}}, 31: {2: {'weight': 1}, 9: {'weight': 1}, 33: {'weight': 1}, 34: {'weight': 1}}, 10: {3: {'weight': 1}, 34: {'weight': 1}}, 28: {3: {'weight': 1}, 24: {'weight': 1}, 25: {'weight': 1}, 34: {'weight': 1}}, 29: {3: {'weight': 1}, 32: {'weight': 1}, 34: {'weight': 1}}, 33: {3: {'weight': 1}, 9: {'weight': 1}, 15: {'weight': 1}, 16: {'weight': 1}, 19: {'weight': 1}, 21: {'weight': 1}, 23: {'weight': 1}, 24: {'weight': 1}, 30: {'weight': 1}, 31: {'weight': 1}, 32: {'weight': 1}, 34: {'weight': 1}}, 17: {6: {'weight': 1}, 7: {'weight': 1}}, 34: {9: {'weight': 1}, 10: {'weight': 1}, 14: {'weight': 1}, 15: {'weight': 1}, 16: {'weight': 1}, 19: {'weight': 1}, 20: {'weight': 1}, 21: {'weight': 1}, 23: {'weight': 1}, 24: {'weight': 1}, 27: {'weight': 1}, 28: {'weight': 1}, 29: {'weight': 1}, 30: {'weight': 1}, 31: {'weight': 1}, 32: {'weight': 1}, 33: {'weight': 1}}, 15: {33: {'weight': 1}, 34: {'weight': 1}}, 16: {33: {'weight': 1}, 34: {'weight': 1}}, 19: {33: {'weight': 1}, 34: {'weight': 1}}, 21: {33: {'weight': 1}, 34: {'weight': 1}}, 23: {33: {'weight': 1}, 34: {'weight': 1}}, 24: {26: {'weight': 1}, 28: {'weight': 1}, 30: {'weight': 1}, 33: {'weight': 1}, 34: {'weight': 1}}, 26: {24: {'weight': 1}, 25: {'weight': 1}, 32: {'weight': 1}}, 30: {24: {'weight': 1}, 27: {'weight': 1}, 33: {'weight': 1}, 34: {'weight': 1}}, 25: {26: {'weight': 1}, 28: {'weight': 1}, 32: {'weight': 1}}, 27: {30: {'weight': 1}, 34: {'weight': 1}}}
G.degree: [(1, 16), (2, 9), (3, 10), (4, 6), (5, 3), (6, 4), (7, 4), (8, 4), (9, 5), (11, 3), (12, 1), (13, 2), (14, 5), (18, 2), (20, 3), (22, 2), (32, 6), (31, 4), (10, 2), (28, 4), (29, 3), (33, 12), (17, 2), (34, 17), (15, 2), (16, 2), (19, 2), (21, 2), (23, 2), (24, 5), (26, 3), (30, 4), (25, 3), (27, 2)]
adj_matrix: 
[[0. 1. 1. ... 0. 0. 0.]
 [1. 0. 1. ... 0. 0. 0.]
 [1. 1. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 1. 0. 0.]]
'''
file_path = r'C:\Users\zhangwentao\Desktop\大三下学期\Complex-Network\karate.txt'
# 从文件中读取边列表，创建一个图，其中节点的类型为整数，边的权重为整数
G = nx.read_edgelist(file_path, nodetype=int, data=(('weight',int),))

print(clustering_coefficient(G))
# 记录程序结束时间
end_time = time.time()

# 计算程序执行时间（单位：秒）
elapsed_time = end_time - start_time
print(f"程序执行时间：{elapsed_time:.8f} 秒")
# print('Degree Distribution:')
# print(degree_distribution(G))

# print('Average Path Length:')
# print(average_path_length(G))

# source, target = 1, 3
# print(f'Breadth First Search Distance between {source} and {target}:')
# print(breadth_first_search(G, source, target))

# print('Is Connected?:')
# print(is_connected(G))

# print('Largest Connected Component:')
# print(largest_connected_component(G))

# def compute_yu_average(G):
#     # 定义一个字典，用于存储每个节点的 Yu 平均度数
#     yu_average = {}
#     # 迭代处理每个节点
#     for node in G.nodes():
#         # 获取当前节点的邻居节点列表
#         neighbors = list(G.neighbors(node))
#         # 如果当前节点有邻居，则计算其邻居节点的度数总和并除以邻居节点数，得到平均度数
#         if len(neighbors) > 0:
#             yu_average[node] = sum(d for n, d in G.degree(neighbors)) / len(neighbors)
#         # 如果当前节点没有邻居，则将其平均度数设置为 0
#         else:
#             yu_average[node] = 0
#     # 返回每个节点的 Yu 平均度数
#     return yu_average

# # 计算网络的余平均度
# yu_average = compute_yu_average(G)

# # 打印结果
# for node, yu_aver in yu_average.items():
#     print(f"Node {node}: Yu Average = {yu_aver}")

# print('Assortativity:')
# print(assortativity(G))

# # 绘制图形
# pos = nx.circular_layout(G)

# # 绘制节点
# nx.draw_networkx_nodes(G, pos, node_size=500, node_color="yellow")

# # 绘制边
# nx.draw_networkx_edges(G, pos, edge_color='blue')

# # 绘制节点标签
# nx.draw_networkx_labels(G, pos, font_size=16, font_family='sans-serif')

# # 关闭坐标轴
# plt.axis('off')

# # 显示图形
# plt.show()

# # 计算富人俱乐部系数
# # 找到度数为 0 的节点并移除它们
# degree_zero_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
# G.remove_nodes_from(degree_zero_nodes)

# rc = nx.rich_club_coefficient(G)
# rc_random = nx.rich_club_coefficient(G, normalized=True, Q=100)

# # 输出结果
# for k, v in rc.items():
#     print(f"Degree {k}: Rich Club Coefficient = {v}, Randomized RC = {rc_random[k]}")
    
# # 获取所有节点的度数
# degrees = dict(G.degree())
# degrees = {node: degree for node, degree in degrees.items() if degree != 0}
# rc = {node: val for node, val in rc.items() if node in degrees}

# # 绘制关系图
# plt.plot(degrees.values(), rc.values(), 'r.-', label='RC')
# plt.plot(degrees.values(), rc_random.values(), 'b.-', label='Randomized RC')
# plt.legend(loc='best')
# plt.xlabel('Node Degree')
# plt.ylabel('Rich Club Coefficient')
# plt.show()




# def save_adjacency_matrix(G, file_path):
#     # 将图对象 G 转换为邻接矩阵，并保存到文件中
#     adj_matrix = nx.to_numpy_array(G)
#     np.savetxt(file_path, adj_matrix, fmt='%d')


# def save_edge_list(G, file_path):
#     # 将图对象 G 转换为边列表，并保存到文件中
#     edge_list = list(G.edges())
#     with open(file_path, 'w') as f:
#         for edge in edge_list:
#             f.write(f'{edge[0]} {edge[1]}\n')


# def degree_distribution(G):
#     # 计算图对象 G 的度分布序列并返回
#     degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
#     return degree_sequence


# def average_path_length(G):
#     # 如果图对象 G 是连通的，则计算其平均最短路径长度并返回
#     if nx.is_connected(G):
#         return nx.average_shortest_path_length(G)
#     # 如果图对象 G 不是连通的，则返回 None
#     else:
#         return None


# def breadth_first_search(G, source, target):
#     return nx.shortest_path_length(G, source=source, target=target, weight='weight', method='dijkstra')


# def is_connected(G):
#     # 判断图对象 G 是否是连通的
#     return nx.is_connected(G)


# def largest_connected_component(G):
#     # 如果图对象 G 不是连通的，则返回最大连通子图的节点集合
#     if not is_connected(G):
#         return max(nx.connected_components(G), key=len)
#     # 如果图对象 G 是连通的，则返回 None
#     else:
#         return None


# def assortativity(G):
#     # 计算图对象 G 的度同配性系数并返回
#     return nx.degree_assortativity_coefficient(G)


# def draw_network(G):
#     # 使用 Spring 布局算法生成节点的位置信息，并绘制图形
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
#     plt.show()
#     # 将图形保存到文件中
#     plt.savefig('C:\\Users\\zhangwentao\\Desktop\\大三下学期\\Complex-Network\\karate.jpg')
    

# file_path = r'C:\Users\zhangwentao\Desktop\大三下学期\Complex-Network\karate.txt'
# # 从文件中读取边列表，创建一个图，其中节点的类型为整数，边的权重为整数
# G = nx.read_edgelist(file_path, nodetype=int, data=(('weight',int),))
# clustering_coefficient(G)
# # Save adjacency matrix
# adj_matrix_path = 'C:\\Users\\zhangwentao\\Desktop\\大三下学期\\Complex-Network\\adj_matrix.txt'
# save_adjacency_matrix(G, adj_matrix_path)
# print(f'Saved adjacency matrix to {adj_matrix_path}')

# # Save edge list
# edge_list_path = 'C:\\Users\\zhangwentao\\Desktop\\大三下学期\\Complex-Network\\edge_list.txt'
# save_edge_list(G, edge_list_path)
# print(f'Saved edge list to {edge_list_path}')

# plt.rcParams['font.sans-serif'] = ['KaiTi']  #设置字体为楷体
# plt.rcParams['axes.unicode_minus']=False     #显示负号