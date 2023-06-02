import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

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
# 幂迭代法计算特征向量中心性
def eigenvector_centrality(adj_matrix, max_iter=100, tol=1e-6):
    if not np.allclose(adj_matrix, adj_matrix.T):
        print("Adjacency matrix is not symmetric.")
    n = adj_matrix.shape[0]
    x = np.random.rand(n, 1)
    x /= np.linalg.norm(x)
    for _ in range(max_iter):
        x_next = adj_matrix @ x
        x_next /= np.linalg.norm(x_next)
        if np.linalg.norm(x_next - x) < tol:
            break
        x = x_next
    return x
    
def Closeness_Centrality(G):
    # 初始化一个零矩阵，用于存储图中所有节点间的最短路径长度
    shoretest_path_matrix = np.zeros((len(G.nodes()), len(G.nodes())))
    
    # 创建一个字典，将图中的节点映射到它们在矩阵中的索引
    dic = dict(zip(G.nodes(), range(len(G.nodes()))))
    
    # 遍历图中的所有节点，计算它们之间的最短路径长度
    for start in G.nodes:
        for end in G.nodes:
            if start != end:
                # 如果当前节点对的最短路径尚未计算
                if shoretest_path_matrix[dic.get(start), dic.get(end)] == 0:
                    # 使用广度优先搜索算法计算最短路径长度，并将其存储在矩阵中
                    shoretest_path_matrix[dic.get(start), dic.get(end)] = breadth_first_search(G, start, end)
                    continue
    
    # 打印最短路径矩阵
    print(shoretest_path_matrix)
    
    # 计算并返回接近中心性
    return (len(G.nodes()) - 1) / shoretest_path_matrix.sum(axis = 1)

    

def degree_centralized_distribution(G):
    degree = [d[1] for d in G.degree()]  # 获取图中每个节点的度（连接的边数）
    
    # 绘制每个节点的度中心化分布的柱状图
    plt.bar(G.nodes(), np.divide(degree, (len(G.nodes()) - 1)))
    plt.xlabel('Degree')  # 设置x轴标签为“Degree”
    plt.ylabel('Count')  # 设置y轴标签为“Count”
    plt.title('Degree Centralised Distribution')  # 设置图表标题为“Degree Centralised Distribution”
    plt.show()  # 显示图表
    
    # 返回每个节点的度中心化分布值
    return np.divide(degree, (len(G.nodes()) - 1))

def bfs_shortest_paths(G, source):
    visited = {source: 0}  # 初始化已访问节点字典，将起始节点的访问深度设为0
    queue = [source]  # 初始化队列，将起始节点加入队列
    paths = {node: [] for node in G.nodes}  # 初始化路径字典，用于存储从起始节点到每个节点的最短路径
    paths[source] = [[source]]  # 将起始节点的路径设为包含自身的列表

    # 当队列非空时，继续执行循环
    while queue:
        current = queue.pop(0)  # 从队列中取出第一个节点，并将其从队列中移除
        neighbors = G[current]  # 获取当前节点的邻居节点

        # 遍历邻居节点
        for neighbor in neighbors:
            # 如果邻居节点没有被访问过
            if neighbor not in visited:
                visited[neighbor] = visited[current] + 1  # 将邻居节点的访问深度设为当前节点访问深度加1
                queue.append(neighbor)  # 将邻居节点加入队列

            # 如果邻居节点的访问深度等于当前节点访问深度加1
            if visited[neighbor] == visited[current] + 1:
                # 更新从起始节点到邻居节点的最短路径
                paths[neighbor] += [path + [neighbor] for path in paths[current]]

    return paths  # 返回从起始节点到每个节点的最短路径字典


def betweenness_centrality(G):
    shoretest_path_matrix = shortest_path_number(G)
    dic = dict(zip(G.nodes(), range(len(G.nodes()))))
    Numerator = [np.zeros((len(G.nodes()), len(G.nodes()))) for _ in range(len(G.nodes()))]
    for node in G.nodes:
        for start in [n for n in G.nodes if n != node]:
            for end in [n for n in G.nodes if n != node]:
                if start != end:
                    # Calculate the number of shortest paths from start to end that pass through node
                    shortest_paths = bfs_shortest_paths(G, start)
                    count = sum(node in path for path in shortest_paths[end])
                    Numerator[dic.get(node)][dic.get(start)][dic.get(end)] += count
    
    # 将 shoretest_path_matrix 转换为 NumPy 数组
    shoretest_path_matrix_np = np.array(shoretest_path_matrix)
    # print(Numerator)
    Numerator_np = np.array(Numerator)
    # 然后计算 result
    re = []
    mask = shoretest_path_matrix_np != 0
    # print(mask)
    for i in Numerator_np:
        # 创建一个掩码，标识 B 中非零元素的位置
        # print(i)
        # 初始化一个与 A 相同形状的全零矩阵
        result = np.zeros_like(i)

        # 只对 B 中非零元素对应的位置进行除法操作
        result[mask] = i[mask] / shoretest_path_matrix_np[mask]

        re.append(np.sum(result))
    re = np.array(re)
    
    print("betweenness_centrality NetworkX:", nx.betweenness_centrality(G,normalized=True))
    # nx.betweenness_centrality(G,normalized=True) - betweenness_centrality(G)
    # 输出结果
    result = dict(zip(G.nodes(), re/((len(G.nodes()) - 1) * (len(G.nodes()) - 2))))
    return result
    
def shortest_path_number(G):
    n = G.number_of_nodes()  # 获取图 G 的节点数
    matrix = [np.zeros((n, n))]  # 初始化一个 n*n 的零矩阵
    shortest_path_number = np.zeros((n, n))  # 初始化一个 n*n 的零矩阵
    np.fill_diagonal(shortest_path_number, np.inf)  # 将 shortest_path_number 的对角线上的值设置为正无穷
    adj_matrix = np.array(nx.to_numpy_array(G))  # 将图 G 转换成邻接矩阵
    matrix = adj_matrix  # 将邻接矩阵赋值给 matrix
    mask = shortest_path_number != 0  # 创建一个掩码，用于标记 shortest_path_number 中已经计算出最短路径的节点对
    np.fill_diagonal(shortest_path_number, np.inf)  # 将 shortest_path_number 的对角线上的值设置为正无穷
    for i in range(1, n):  # 进行 n-1 次迭代
        for p in range(n):  # 遍历所有节点对
            for q in range(n):
                if q != p:  # 排除节点到自身的情况
                    if matrix[p][q] != 0 and shortest_path_number[p][q] == 0:  # 如果节点 p 和节点 q 之间有边，并且它们之间的最短路径还没有被计算出来
                        shortest_path_number[p][q] = matrix[p][q]  # 将它们之间的距离作为它们之间的最短路径
        matrix = matrix @ adj_matrix  # 计算 matrix 的下一次幂
    # 将对角线上的值设置为0，因为节点到自身的距离为0
    np.fill_diagonal(shortest_path_number, 0)
    return shortest_path_number
    
def k_shell_decomposition(graph):
    k_shell = {}  # 创建一个空字典，用于存储每个节点的k-壳值
    k = 1  # 初始化k的值为1，用于表示当前计算的k-壳层级

    # 当图中仍然有节点时，继续执行循环
    while graph.nodes():
        nodes_to_remove = []  # 创建一个空列表，用于存储本轮迭代中需要移除的节点

        # 遍历图中的所有节点
        for node in graph.nodes():
            # 如果当前节点的度（连接的边数）小于等于k
            if graph.degree(node) <= k:
                nodes_to_remove.append(node)  # 将当前节点添加到需要移除的节点列表中
                k_shell[node] = k  # 将当前节点的k-壳值设置为k

        # 如果本轮迭代没有需要移除的节点
        if not nodes_to_remove:
            k += 1  # 将k的值加1，计算下一个k-壳层级
        else:
            graph.remove_nodes_from(nodes_to_remove)  # 从图中移除本轮迭代中所有需要移除的节点

    return k_shell  # 返回字典k_shell，其中包含每个节点的k-壳值


file_path = "karate.txt"
# 从文件中读取边列表，创建一个图，其中节点的类型为整数，边的权重为整数
G = nx.read_edgelist(
    file_path,
    nodetype=int,
    data=(('weight', int),)
)
# print(betweenness_centrality(G))
# nx.draw(G, with_labels=True)
# plt.show()
# 将图转换为 NumPy 矩阵
# adj_matrix = nx.to_numpy_matrix(G)
# print(f"own code degeree contrality: {degree_centralized_distribution(G)}")
# print(f"NetworkX: {nx.degree_centrality(G)}")
# print(f"own code closeness contrality: {Closeness_Centrality(G)}")
# print(f"NetworkX: {nx.closeness_centrality(G)}")
# path = [] 

# 计算特征向量中心性
# centrality = nx.eigenvector_centrality(G)

# 判断是否是连通图
# if not is_connected(G):
#     G_set = get_largest_connected_component(G)
#     G_list = list(G_set)
#     # print(f"Largest connected component is {G_list}")
#     for i in list(G.nodes()):
#         if i not in G_list:
#             G.remove_node(i)
# print(f"networkX result based on norm 1:{nx.eigenvector_centrality(G)}")
# print(f"eigenvector_centrality based on norm 2:{eigenvector_centrality(adj_matrix)}")
betweenness_centrality(G)
# save_path = "shoretest_path_matrix_sum.txt"
# np.savetxt(save_path, shoretest_path_matrix.sum(axis = 1) / (len(G.nodes()) - 1), fmt = '%f', delimiter = ' ')
# # 输出结果
# print(centrality)
# nodes = [n for n in G.nodes() if n != 27]
# for n in nodes:
#     path.append(breadth_first_search(G, 27, n))
# print(path)
#     path.append(breadth_first_search(G, node))
# dict = zip(G.nodes(), path)
# print(list(dict))
    
# Centrifugal_centrality = []
# for node in G.nodes():
#     nodes = [n for n in G.nodes() if n != node]
#     for n in nodes:
#         path.append(breadth_first_search(G, node, n))
#     Centrifugal_centrality.append(max(path))
#     path = []
# List = zip(G.nodes(), Centrifugal_centrality)
# print(list(List))




    


