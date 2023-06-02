import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def load_network_data(file_path):
    # 从文件中读取边列表，创建一个图，其中节点的类型为整数，边的权重为浮点数
    G = nx.read_edgelist(file_path, nodetype=int, data=(('weight',int)))
    # 返回创建的图对象
    return G

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


def clustering_coefficient(G):
    # 计算图对象 G 的平均聚类系数,输入是图对象，输出是平均聚类系数
    adjacency_matrix = G.adjacency_matrix()#返回邻接矩阵表示法的矩阵
    print(adjacency_matrix)

def degree_distribution(G):
    # 计算图对象 G 的度分布序列并返回
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    return degree_sequence


def average_path_length(G):
    # 如果图对象 G 是连通的，则计算其平均最短路径长度并返回
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    # 如果图对象 G 不是连通的，则返回 None
    else:
        return None


def breadth_first_search(G, source, target):
    # 使用广度优先搜索算法计算从源节点 source 到目标节点 target 的最短路径长度
    '''计算图中的最短路径长度。
参数
----------
G : NetworkX 图

source : 节点, 可选
    路径的起始节点。
    如果未指定，则使用所有节点作为起始节点计算最短路径长度。

target : 节点, 可选
    路径的终止节点。
    如果未指定，则使用所有节点作为目标节点计算最短路径长度。

weight : None, 字符串或函数, 可选 (默认值 = None)
    如果为 None，则每条边的权重/距离/成本为 1。
    如果是字符串，则将此边属性用作边权重。
    不存在的任何边属性都默认为 1。
    如果这是一个函数，则边的权重是函数返回的值。函数必须接受正好三个位置参数：边的两个端点和该边的边属性字典。函数必须返回一个数字。

method : 字符串, 可选 (默认值 = 'dijkstra')
    用于计算路径长度的算法。
    支持的选项: 'dijkstra', 'bellman-ford'。
    其他输入会产生 ValueError。
    如果 `weight` 为 None，则使用无权图方法，此建议会被忽略。
    '''
    return nx.shortest_path_length(G, source=source, target=target, weight='weight', method='dijkstra')


def is_connected(G):
    # 判断图对象 G 是否是连通的
    return nx.is_connected(G)


def largest_connected_component(G):
    # 如果图对象 G 不是连通的，则返回最大连通子图的节点集合
    if not is_connected(G):
        return max(nx.connected_components(G), key=len)
    # 如果图对象 G 是连通的，则返回 None
    else:
        return None


def assortativity(G):
    # 计算图对象 G 的度同配性系数并返回
    return nx.degree_assortativity_coefficient(G)


def draw_network(G):
    # 使用 Spring 布局算法生成节点的位置信息，并绘制图形
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
    plt.show()
    # 将图形保存到文件中
    plt.savefig('C:\\Users\\zhangwentao\\Desktop\\大三下学期\\Complex-Network\\karate.jpg')
    

file_path = r'C:\Users\zhangwentao\Desktop\大三下学期\Complex-Network\karate.txt'
G = load_network_data(file_path)

# # Save adjacency matrix
# adj_matrix_path = 'C:\\Users\\zhangwentao\\Desktop\\大三下学期\\Complex-Network\\adj_matrix.txt'
# save_adjacency_matrix(G, adj_matrix_path)
# print(f'Saved adjacency matrix to {adj_matrix_path}')

# # Save edge list
# edge_list_path = 'C:\\Users\\zhangwentao\\Desktop\\大三下学期\\Complex-Network\\edge_list.txt'
# save_edge_list(G, edge_list_path)
# print(f'Saved edge list to {edge_list_path}')

print('Clustering Coefficient:')
print(clustering_coefficient(G))

print('Degree Distribution:')
print(degree_distribution(G))

print('Average Path Length:')
print(average_path_length(G))

source, target = 1, 3
print(f'Breadth First Search Distance between {source} and {target}:')
print(breadth_first_search(G, source, target))

print('Is Connected?:')
print(is_connected(G))

print('Largest Connected Component:')
print(largest_connected_component(G))

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

# 计算网络的余平均度
yu_average = compute_yu_average(G)

# 打印结果
for node, yu_aver in yu_average.items():
    print(f"Node {node}: Yu Average = {yu_aver}")

print('Assortativity:')
print(assortativity(G))

# 绘制图形
pos = nx.circular_layout(G)

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="yellow")

# 绘制边
nx.draw_networkx_edges(G, pos, edge_color='blue')

# 绘制节点标签
nx.draw_networkx_labels(G, pos, font_size=16, font_family='sans-serif')

# 关闭坐标轴
plt.axis('off')

# 显示图形
plt.show()

# 计算富人俱乐部系数
# 找到度数为 0 的节点并移除它们
degree_zero_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
G.remove_nodes_from(degree_zero_nodes)

rc = nx.rich_club_coefficient(G)
rc_random = nx.rich_club_coefficient(G, normalized=True, Q=100)

# 输出结果
for k, v in rc.items():
    print(f"Degree {k}: Rich Club Coefficient = {v}, Randomized RC = {rc_random[k]}")
    
# 获取所有节点的度数
degrees = dict(G.degree())
degrees = {node: degree for node, degree in degrees.items() if degree != 0}
rc = {node: val for node, val in rc.items() if node in degrees}

# 绘制关系图
plt.plot(degrees.values(), rc.values(), 'r.-', label='RC')
plt.plot(degrees.values(), rc_random.values(), 'b.-', label='Randomized RC')
plt.legend(loc='best')
plt.xlabel('Node Degree')
plt.ylabel('Rich Club Coefficient')
plt.show()