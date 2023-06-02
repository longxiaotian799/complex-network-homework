from itertools import combinations
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from collections import deque


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
    print(f"\nAverage Clustering Coefficient:{2 * coefficient / G.number_of_nodes()}")
    return 2 * coefficient / G.number_of_nodes()  # 返回平均聚类系数

def degree_distribution(G):
    # 获取所有节点的度并创建Counter对象，记录每个度出现的次数
    counts = Counter(deg for _, deg in G.degree)

    # 找到最小和最大的度
    min_degree, max_degree = min(counts), max(counts)
    
    max_degree = max(counts.keys())
    degree_counts = [0] * (max_degree + 1)

    for degree, count in counts.items():
        degree_counts[degree] = count
    print('\nDegree Distribution:')
    print(degree_counts)

        
    # 绘制度分布直方图
    plt.bar(counts.keys(), counts.values())
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution')
    # plt.xticks(range(min_degree, max_degree + 1))
    plt.show()


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
    
def average_path_length(G):
    n = G.number_of_nodes()
    matrix = [np.zeros((n, n))]
    shortest_path = np.zeros((n, n))
    adj_matrix = np.array(nx.to_numpy_array(G))
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


def generate_ws_small_world_network(N, k, p):
    # 创建一个空的NetworkX图对象G和一个大小为(N, N)的零邻接矩阵adj
    G = nx.Graph()
    adj = np.zeros((N, N))

    # 将邻接矩阵adj初始化为一个环形的规则图，每个节点连接到其相邻的k // 2个节点
    for i in range(N):
        for j in range(1, k // 2 + 1):
            adj[i, (i + j) % N] = 1
            adj[i, (i - j) % N] = 1

    # 将邻接矩阵表示为NetworkX图对象G。我们使用np.triu()函数将邻接矩阵的下三角设置为零
    adj1 = np.triu(adj)
    G = nx.from_numpy_array(adj1)

    # 计算G的平均度数，并将G的边列表存储在edges变量中
    avg_degree = sum(dict(G.degree()).values()) / len(G)
    edges = list(G.edges())

    # 按概率p对每条边进行重连
    for _, e in enumerate(edges):
        i, j = e
        if random.random() > 1 - p:
            non_one_indices = np.where((adj[i, :] != 1) & (np.arange(N) != i))[0]
            if len(non_one_indices) > 0:
                k = random.choice(non_one_indices)
                adj[i, k] = 1
                adj[k, i] = 1
                adj[i, j] = 0
                adj[j, i] = 0

    # 将更新后的邻接矩阵表示为NetworkX图对象G。我们再次使用np.triu()函数将邻接矩阵的下三角设置为零
    adj2 = np.triu(adj)
    G = nx.from_numpy_array(adj2)

    # 返回新生成的小世界网络G
    return G
N = 1000
k = 6
p = 0.02

G = generate_ws_small_world_network(N, k, p)
nx.draw(G, node_size=6)
plt.show()
# # 判断是否是连通图
# if not is_connected(G):
#     G_set = get_largest_connected_component(G)
#     G_list = list(G_set)
#     print(G_list)
#     for i in list(G.nodes()):
#         if i not in G_list:
#             G.remove_node(i)

# # 输出平均最短路径长度
# average_path_length(G)
# print(f"python NetworkX results:{nx.average_shortest_path_length(G)}")

# # 输出平均聚类系数
# clustering_coefficient(G)
# print(f"python NetworkX results:{nx.average_clustering(G)}")

# # 输出度分布序列
# degree_distribution(G)
# print(f"python NetworkX results:\n{nx.degree_histogram(G)}")