from itertools import combinations
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from collections import deque



def average_path_length(G):
    n = G.number_of_nodes()
    matrix = [np.zeros((n, n))]
    shortest_path = np.zeros((n, n))
    adj_matrix = np.array(nx.to_numpy_matrix(G))
    matrix = adj_matrix
    np.fill_diagonal(shortest_path, np.inf)
    for i in range(1, n):
        if shortest_path.all() != 0:
            break
        for p in range(n):
            for q in range(n):
                if q != p:
                    if matrix[p][q] != 0 and shortest_path[p][q] == 0:
                        shortest_path[p][q] = i
        matrix = matrix @ adj_matrix
    # 将对角线上的值设置为0，因为节点到自身的距离为0
    np.fill_diagonal(shortest_path, 0)
    
    # 计算平均最短路径长度
    average = np.sum(shortest_path) / (n * (n - 1))
    print(f'Average Path Length is {average}')
    return average

# def power_adj_matrix(adj_matrix, k):
#     # 获取邻接矩阵的行数和列数
#     n = adj_matrix.shape[0]

#     # 初始化三维数组
#     matrix = np.zeros((k + 1, n, n))
#     matrix[0] = adj_matrix

#     # 计算邻接矩阵的幂
#     for i in range(1, k + 1):
#         # 计算下一个矩阵
#         matrix[i] = matrix[i - 1] @ matrix[i - 1]

#     return matrix[k]

# def average_path_length(G):
#     # 获取图中节点的数量
#     n = G.number_of_nodes()

#     # 从图中获取邻接矩阵
#     adj_matrix = np.array(nx.to_numpy_matrix(G))

#     # 计算邻接矩阵的幂，并更新最短路径矩阵
#     k = int(np.ceil(np.log2(n)))
#     power_matrix = power_adj_matrix(adj_matrix, k)
#     shortest_path = np.zeros((n, n))
#     for i in range(k, -1, -1):
#         mask = (power_adj_matrix(adj_matrix, i) != 0)
#         shortest_path[mask] = i

#     # 将对角线上的值设置为0，因为节点到自身的距离为0
#     np.fill_diagonal(shortest_path, 0)

#     # 计算平均最短路径长度
#     average = np.sum(shortest_path) / (n * (n - 1))

#     # 打印平均最短路径长度
#     print(f'Average Path Length is {average}')

#     # 返回平均最短路径长度
#     return average




# 定义文件路径
file_path = 'karate.txt'

# 从文件中读取边列表，创建一个图，其中节点的类型为整数，边的权重为整数
G = nx.read_edgelist(
    file_path,
    nodetype=int,
    data=(('weight', int),)
)
# 将图转换为 NumPy 矩阵
adj_matrix = nx.to_numpy_matrix(G)

# # 使用格式化字符串打印邻接矩阵
# print(f"邻接矩阵为:\n{np.array(adj_matrix)}\n")

# # 生成三元组，表示节点之间的边和权重，如 (1, 2, 1) 表示节点 1 和节点 2 之间有一条权重为 1 的边
# edges = [(x[0], x[1], 1) for x in G.edges]

# # 使用格式化字符串打印三元组
# print(f"三元组为: {edges}")

# # 输出平均聚类系数
# print(clustering_coefficient(G), "\n")

# # 输出度分布序列
# degree_distribution(G)

# 输出平均最短路径长度
average_path_length(G)

# plt.clf()
print(f"python包验证{nx.average_shortest_path_length(G)}")

# breadth_first_search(G,1,4)

# is_connected(G)
# print(get_largest_connected_component(G))

# source, target = 1, 31
# print(f'Breadth First Search Distance between {source} and {target}:')
# print(breadth_first_search(G, source, target))

# print('Is Connected?:')
# print(is_connected(G))

# print('Largest Connected Component:')
# print(get_largest_connected_component(G))

# # 计算网络的余平均度
# yu_average = compute_yu_average(G)

# # 打印结果
# for node, yu_aver in yu_average.items():
#     print(f"Node {node}: Yu Average = {yu_aver}")

# # 计算网络的标准化同配系数
# print("Normalized Assortativity Coefficient:", normalized_assortativity(G))
# nx.degree_assortativity_coefficient(G)

# # 绘制图形
# pos = nx.spring_layout(G, seed=14)  # 使用弹簧布局

# # 绘制节点
# nx.draw_networkx_nodes(G, pos, node_size=500, node_color="orange", alpha=0.7, edgecolors="black")

# # 绘制边
# nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)

# # 绘制节点标签
# nx.draw_networkx_labels(G, pos, font_size=16, font_family='sans-serif', font_weight='bold')

# # 关闭坐标轴
# plt.axis('off')

# # 设置背景颜色
# plt.gca().set_facecolor("white")

# # 显示图形
# plt.show()

# compare_rich_club_coefficients(G, p=0.5)













# # 用广度优先搜索算法计算两个节点之间的最短路径
# # for start in G.nodes:
# #     for end in G.nodes:
# #             bfs_result = breadth_first_search(G, start, end)
# #             bfs_result = bfs_result[0]
# #             nx_result = nx.shortest_path_length(G, start, end)
# #             print(f"From {start} to {end}: BFS = {bfs_result}, NetworkX = {nx_result}")
# #             assert bfs_result == nx_result
# # for start in G.nodes:
# #     for end in G.nodes:
# #             bfs_result = breadth_first_search(G, start, end)
# #             if bfs_result != np.inf:
# #                bfs_result = bfs_result[0]

# # # 输出图的各种信息
# # G.nodes # 返回一个节点列表。
# # G.edges # 返回一组边列表。
# # G.degree # 返回一个字典，其中键是节点，值是该节点的度数。 
# # G.number_of_nodes() # 返回节点数。
# # G.number_of_edges() # 返回边数。
# # G.adj # 这是一个字典类型的变量，它的键为源节点，值为目标节点和权重。这个字典表示了一个加权有向图的邻接表。每个节点的所有出边在邻接表中都被存储为一个键值对，其中键为目标节点，值是一个包含一条边的属性字典，其中包括该边的权重。
# # adj_matrix = nx.to_numpy_array(G) # 返回图的邻接矩阵。
# # #输入上面的变量
# # print("G.nodes:", G.nodes)
# # print("G.edges:", G.edges)
# # print("G.degree:", G.degree)
# # print("G.nodes:", G.nodes)
# # print("G.number_of_nodes:", G.number_of_nodes())
# # print("G.number_of_edges:", G.number_of_edges())
# # print("G.adj:", G.adj)
# # print("nx.to_numpy_array(G):", nx.to_numpy_array(G))

# '''karate输出结果
# G.nodes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 18, 20, 22, 32, 31, 10, 28, 29, 33, 17, 34, 15, 16, 19, 21, 23, 24, 26, 30, 25, 27]
# G.edges: [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 11), (1, 12), (1, 13), (1, 14), (1, 18), (1, 20), (1, 22), (1, 32), (2, 3), (2, 4), (2, 8), (2, 14), (2, 18), (2, 20), (2, 22), (2, 31), (3, 4), (3, 8), (3, 9), (3, 10), (3, 14), (3, 28), (3, 29), (3, 33), (4, 8), (4, 13), (4, 14), (5, 7), (5, 11), (6, 7), (6, 11), (6, 17), (7, 17), (9, 31), (9, 33), (9, 34), (14, 34), (20, 34), (32, 25), (32, 26), (32, 29), (32, 33), (32, 34), (31, 33), (31, 34), (10, 34), (28, 24), (28, 25), (28, 34), (29, 34), (33, 15), (33, 16), (33, 19), (33, 21), (33, 23), (33, 24), (33, 30), (33, 34), (34, 15), (34, 16), (34, 19), (34, 21), (34, 23), (34, 24), (34, 27), (34, 30), (24, 26), (24, 30), (26, 25), (30, 27)]
# G.number_of_nodes: 34
# G.number_of_edges: 78
# G.adj: {1: {2: {'weight': 1}, 3: {'weight': 1}, 4: {'weight': 1}, 5: {'weight': 1}, 6: {'weight': 1}, 7: {'weight': 1}, 8: {'weight': 1}, 9: {'weight': 1}, 11: {'weight': 1}, 12: {'weight': 1}, 13: {'weight': 1}, 14: {'weight': 1}, 18: {'weight': 1}, 20: {'weight': 1}, 22: {'weight': 1}, 32: {'weight': 1}}, 2: {1: {'weight': 1}, 3: {'weight': 1}, 4: {'weight': 1}, 8: {'weight': 1}, 14: {'weight': 1}, 18: {'weight': 1}, 20: {'weight': 1}, 22: {'weight': 1}, 31: {'weight': 1}}, 3: {1: {'weight': 1}, 2: {'weight': 1}, 4: {'weight': 1}, 8: {'weight': 1}, 9: {'weight': 1}, 10: {'weight': 1}, 14: {'weight': 1}, 28: {'weight': 1}, 29: {'weight': 1}, 33: {'weight': 1}}, 4: {1: {'weight': 1}, 2: {'weight': 1}, 3: {'weight': 1}, 8: {'weight': 1}, 13: {'weight': 1}, 14: {'weight': 1}}, 5: {1: {'weight': 1}, 7: {'weight': 1}, 11: {'weight': 1}}, 6: {1: {'weight': 1}, 7: {'weight': 1}, 11: {'weight': 1}, 17: {'weight': 1}}, 7: {1: {'weight': 1}, 5: {'weight': 1}, 6: {'weight': 1}, 17: {'weight': 1}}, 8: {1: {'weight': 1}, 2: {'weight': 1}, 3: {'weight': 1}, 4: {'weight': 1}}, 9: {1: {'weight': 1}, 3: {'weight': 1}, 31: {'weight': 1}, 33: {'weight': 1}, 34: {'weight': 1}}, 11: {1: {'weight': 1}, 5: {'weight': 1}, 6: {'weight': 1}}, 12: {1: {'weight': 1}}, 13: {1: {'weight': 1}, 4: {'weight': 1}}, 14: {1: {'weight': 1}, 2: {'weight': 1}, 3: {'weight': 1}, 4: {'weight': 1}, 34: {'weight': 1}}, 18: {1: {'weight': 1}, 2: {'weight': 1}}, 20: {1: {'weight': 1}, 2: {'weight': 1}, 34: {'weight': 1}}, 22: {1: {'weight': 1}, 2: {'weight': 1}}, 32: {1: {'weight': 1}, 25: {'weight': 1}, 26: {'weight': 1}, 29: {'weight': 1}, 33: {'weight': 1}, 34: {'weight': 1}}, 31: {2: {'weight': 1}, 9: {'weight': 1}, 33: {'weight': 1}, 34: {'weight': 1}}, 10: {3: {'weight': 1}, 34: {'weight': 1}}, 28: {3: {'weight': 1}, 24: {'weight': 1}, 25: {'weight': 1}, 34: {'weight': 1}}, 29: {3: {'weight': 1}, 32: {'weight': 1}, 34: {'weight': 1}}, 33: {3: {'weight': 1}, 9: {'weight': 1}, 15: {'weight': 1}, 16: {'weight': 1}, 19: {'weight': 1}, 21: {'weight': 1}, 23: {'weight': 1}, 24: {'weight': 1}, 30: {'weight': 1}, 31: {'weight': 1}, 32: {'weight': 1}, 34: {'weight': 1}}, 17: {6: {'weight': 1}, 7: {'weight': 1}}, 34: {9: {'weight': 1}, 10: {'weight': 1}, 14: {'weight': 1}, 15: {'weight': 1}, 16: {'weight': 1}, 19: {'weight': 1}, 20: {'weight': 1}, 21: {'weight': 1}, 23: {'weight': 1}, 24: {'weight': 1}, 27: {'weight': 1}, 28: {'weight': 1}, 29: {'weight': 1}, 30: {'weight': 1}, 31: {'weight': 1}, 32: {'weight': 1}, 33: {'weight': 1}}, 15: {33: {'weight': 1}, 34: {'weight': 1}}, 16: {33: {'weight': 1}, 34: {'weight': 1}}, 19: {33: {'weight': 1}, 34: {'weight': 1}}, 21: {33: {'weight': 1}, 34: {'weight': 1}}, 23: {33: {'weight': 1}, 34: {'weight': 1}}, 24: {26: {'weight': 1}, 28: {'weight': 1}, 30: {'weight': 1}, 33: {'weight': 1}, 34: {'weight': 1}}, 26: {24: {'weight': 1}, 25: {'weight': 1}, 32: {'weight': 1}}, 30: {24: {'weight': 1}, 27: {'weight': 1}, 33: {'weight': 1}, 34: {'weight': 1}}, 25: {26: {'weight': 1}, 28: {'weight': 1}, 32: {'weight': 1}}, 27: {30: {'weight': 1}, 34: {'weight': 1}}}
# G.degree: [(1, 16), (2, 9), (3, 10), (4, 6), (5, 3), (6, 4), (7, 4), (8, 4), (9, 5), (11, 3), (12, 1), (13, 2), (14, 5), (18, 2), (20, 3), (22, 2), (32, 6), (31, 4), (10, 2), (28, 4), (29, 3), (33, 12), (17, 2), (34, 17), (15, 2), (16, 2), (19, 2), (21, 2), (23, 2), (24, 5), (26, 3), (30, 4), (25, 3), (27, 2)]
# adj_matrix: 
# [[0. 1. 1. ... 0. 0. 0.]
#  [1. 0. 1. ... 0. 0. 0.]
#  [1. 1. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 1.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 1. 0. 0.]]
# '''
# # def random_network(p):
# #     # 创建一个空的网络
# #     G = nx.Graph()

# #     # 添加节点
# #     G.add_nodes_from(range(1, 1001))

# #     # 添加边
# #     for i in range(1, 1001):
# #         for j in range(i + 1, 1001):
# #             if random.random() < p:
# #                 G.add_edge(i, j)
# #                 G.add_edge(j, i)
# #     return G


def breadth_first_search(G, source, target=None):
    '''
    输入：

    G：一个图，表示节点及其相互之间的连接关系。在这里，它应该是一个 NetworkX 图对象。
    source：源节点，是 BFS 算法的起点。
    target（可选）：目标节点，如果提供了这个参数，函数将返回源节点到目标节点的最短距离。
    输出：

    如果没有指定目标节点，函数将返回一个字典，键为图中的每个节点，值为一个元组，包含两个元素：该节点到源节点的最短距离和该节点是否已被访问过。
    如果指定了目标节点，函数将返回一个整数，表示源节点到目标节点的最短距离。
    '''
    # 初始化队列，将源节点加入队列
    queue = deque([source])
    # 为图中的每个节点创建一个字典，键为节点，值为一个包含两个元素的元组：距离和是否访问过的布尔值
    node_info = {node: (0, False) for node in G.nodes}
    # 将源节点的距离设置为 0，并将其访问状态设置为 True
    node_info[source] = (0, True)

    # 当队列非空时，继续执行循环
    while queue:
        # 从队列左侧移除并返回一个节点，将其作为当前节点
        current_node = queue.popleft()

        # 如果目标节点不为空，且当前节点等于目标节点，退出循环
        if target is not None and current_node == target:
            break

        # 获取当前节点的相邻节点列表
        neighbors = list(G.neighbors(current_node))
        # 遍历相邻节点
        for neighbor in neighbors:
            # 获取相邻节点的距离和访问状态
            distance, visited = node_info[neighbor]
            # 如果相邻节点未访问过
            if not visited:
                # 更新相邻节点的距离和访问状态
                node_info[neighbor] = (node_info[current_node][0] + 1, True)
                # 将相邻节点添加到队列的右侧
                queue.append(neighbor)

    # 如果没有指定目标节点，返回包含所有节点信息的字典
    if target is None:
        return node_info
    # 如果指定了目标节点，返回目标节点的距离
    else:
        return node_info[target][0]


def bfs(graph, start_node, visited):
    queue = [start_node]
    visited.add(start_node)

    while queue:
        current_node = queue.pop(0)
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)


def is_connected(G):
    start = next(iter(G.nodes))  # 选择任意一个节点作为起点
    target = None  # 设置目标节点为 None，因为我们只关心访问的节点数量
    node_distances = breadth_first_search(G, start, target)
    
    visited_nodes = [node for node, (distance, visited) in node_distances.items() if visited]
    
    if len(visited_nodes) == len(G.nodes):
        print("Graph is connected")
        return True
    else:
        print("Graph is not connected")
        return False


def get_largest_connected_component(graph):
    visited = set()
    largest_cc = set()
    for node in graph:
        if node not in visited:
            current_cc = set()
            bfs(graph, node, current_cc)
            if len(current_cc) > len(largest_cc):
                largest_cc = current_cc
    return largest_cc


def compute_yu_average(G):
    # 定义一个字典，用于存储每个节点的 Yu 平均度数
    yu_average = {}
    # 迭代处理每个节点
    for node in G.nodes():
        # 获取当前节点的邻居节点列表
        neighbors = list(G.neighbors(node))
        # 如果当前节点有邻居，则计算其邻居节点的度数总和并除以邻居节点数，得到平均度数
        if len(neighbors) > 0:
            yu_average[node] = sum(d for n, d in G.degree(neighbors)) / len(neighbors)
        # 如果当前节点没有邻居，则将其平均度数设置为 0
        else:
            yu_average[node] = 0
    # 返回每个节点的 Yu 平均度数
    return yu_average


def normalized_assortativity(G):
    # 计算网络中每个节点的度数
    d = dict(G.degree())

    # 提取网络中的边
    edges = list(G.edges())

    # 计算边的数量
    K = len(edges)

    # 提取与边关联的节点度数
    di = [d[edge[0]] for edge in edges]
    dj = [d[edge[1]] for edge in edges]

    # 计算同配系数
    di_times_dj_sum = sum(di_i * dj_i for di_i, dj_i in zip(di, dj))
    di_plus_dj_sum = sum(di_i + dj_i for di_i, dj_i in zip(di, dj))
    di_squared_plus_dj_squared_sum = sum(di_i**2 + dj_i**2 for di_i, dj_i in zip(di, dj))
    r = (di_times_dj_sum / K - (di_plus_dj_sum / (2 * K))**2) / (di_squared_plus_dj_squared_sum / (2 * K) - (di_plus_dj_sum / (2 * K))**2)
    return r


def random_network(G1, p):
    # 创建一个空的网络
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(1, G1.number_of_nodes()+1))

    # 添加边
    for i in range(1, G1.number_of_nodes()):
        for j in range(i + 1, G1.number_of_nodes()):
            if random.random() < p:
                G.add_edge(i, j)
                G.add_edge(j, i)
    return G


def rich_club_coefficient(G):
    # 获得网络中每个节点的度数，转化为字典
    d = dict(G.degree())
    rich_club = {}
    for i in range(0, max(d.values())):
        # 提取网络中度数大于等于 i 的节点
        nodes = [node for node in d if d[node] > i]
        sub_maj_matrix = nx.to_numpy_matrix(G, nodelist=nodes)
        # 统计sub_maj_matrix中1的个数
        sum = np.sum(sub_maj_matrix)
        if len(nodes) == 1:
            # print(f'度大于{i}的富人俱乐部系数为1\n')
            rich_club.update({i: 1})
        else:
            # print(f'度大于{i}的富人俱乐部系数为{sum / (len(nodes) * (len(nodes) - 1))}\n')
            rich_club.update({i: sum / (len(nodes) * (len(nodes) - 1))})
    return rich_club


def compare_rich_club_coefficients(G, p):
    random_G = random_network(G, p)
    
    rich_club_G = rich_club_coefficient(G)
    rich_club_random_G = rich_club_coefficient(random_G)
    
    plt.plot(list(rich_club_G.keys()), list(rich_club_G.values()), 'b', label='rich-club coefficient')
    plt.plot(list(rich_club_random_G.keys()), list(rich_club_random_G.values()), 'r', label='random rich-club coefficient')
    
    plt.xlabel('Degree')
    plt.ylabel('Rich Club')
    plt.legend()
    plt.show()
def clustering_coefficient(G):
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
    '''在上述代码中, 我们首先将聚类系数 coefficient 初始化为 0 , 然后遍历图中的每个节点。
    对于每个节点, 我们获取其所有邻居节点, 并计 算该节点的度数 $k$ 。
    如果该节点的度数大于 1 , 则计算该节点的聚类系数 num_edges, 并将其加入总聚类系数中。
    在计算聚类系数时, 我们使用 combinations 函数生成该节点所有邻居节点中的两两组合, 然后计算相邻节点之间是否存在边。
    最后, 我们将总聚类系数除以节点数, 得到平均聚类系数, 并返回该值。
    这样做的时间复杂度为 $O\left(n^2\right)$, 其中 $n$ 是节点数。由于 combinations 函数的空间复杂度为 $O\left(k^2\right)$, 其中 $k$ 是节点的平均度数, 因此总的空间复杂度为 $O\left(n k^2\right)$ 。'''


def degree_distribution(G):
    # 获取所有节点的度并创建Counter对象，记录每个度出现的次数
    counts = Counter(deg for _, deg in G.degree)
    
    # 找到最小和最大的度
    min_degree, max_degree = min(counts), max(counts)
    
    # 遍历所有可能的度，输出对应的出现次数
    for d in range(min_degree, max_degree + 1):
        count = counts[d] # 如果d不存在于counts中，返回0
        
    # 绘制度分布直方图
    plt.bar(counts.keys(), counts.values())
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution')
    plt.xticks(range(min_degree, max_degree + 1))
    plt.show()
