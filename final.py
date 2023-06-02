import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.assortativity.correlation import _numeric_ac
from networkx.algorithms.components.connected import _plain_bfs
from networkx.utils import arbitrary_element
import numpy as np
from collections import defaultdict
import heapq

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def load_network_data(file_path):
    G = nx.read_edgelist(file_path, nodetype=int, data=(('weight',float),))
    return G


def save_adjacency_matrix(G, file_path):
    adj_matrix = nx.to_numpy_array(G)
    np.savetxt(file_path, adj_matrix, fmt='%d')


def save_edge_list(G, file_path):
    edge_list = list(G.edges())
    with open(file_path, 'w') as f:
        for edge in edge_list:
            f.write(f'{edge[0]} {edge[1]}\n')


def clustering_coefficient(G):
    return nx.average_clustering(G)


def degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    
    return degree_sequence


def Floyd(adjMatrix):
    """Floyd算法获得最短路径矩阵
    --input--
    adjMatrix 连通片邻接矩阵
    --return--
    D 最短距离矩阵   D[i][j]第i个结点到第j个结点的最短距离
    """
    m = adjMatrix.shape[0]  # 结点数
    # 1.初始化距离矩阵 dij=1 or inf
    D = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            D[i][j] = adjMatrix[i][j]
            # 将不相连的结点距离设为无穷大
            if D[i][j] == 0 and i != j:
                inf = float('inf')
                D[i][j] = inf
    # 2.计算最短距离矩阵
    for k in range(m):
        for i in range(m):
            for j in range(m):
                if (D[i][k] + D[k][j] < D[i][j]):
                    D[i][j] = D[i][k] + D[k][j]
    return D


def Ave_path(adjMatrix):
    """根据图的邻接矩阵求平均最短路径长度"""
    D = Floyd(adjMatrix)
    N = D.shape[0]  # 结点数
    L = np.sum(D)  # 对D求和
    ave_path = L / (N * (N - 1))
    return ave_path



def adjacency_matrix_to_edge_list(adj_matrix):
    edge_list = []
    for i in range(adj_matrix.shape[0]):
        for j in range(i+1, adj_matrix.shape[1]):
            if adj_matrix[i, j] > 0:
                edge_list.append((i, j, adj_matrix[i, j]))
    return edge_list

def dijkstra_shortest_path_length(adj_matrix, source, target):
    edge_list = adjacency_matrix_to_edge_list(adj_matrix)
    num_nodes = adj_matrix.shape[0]
    
    dist = np.full(num_nodes, np.inf)
    dist[source] = 0
    visited = np.full(num_nodes, False)
    pq = [(0, source)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        if current_node == target:
            return current_dist

        if visited[current_node]:
            continue
        
        visited[current_node] = True

        for edge in edge_list:
            if edge[0] == current_node:
                neighbor = edge[1]
            elif edge[1] == current_node:
                neighbor = edge[0]
            else:
                continue

            if not visited[neighbor]:
                new_dist = current_dist + edge[2]
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
    return None

def is_connected(G):
    """Returns True if the graph is connected, False otherwise.

    Parameters
    ----------
    G : NetworkX Graph
       An undirected graph.

    Returns
    -------
    connected : bool
      True if the graph is connected, false otherwise.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.
    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined ", "for the null graph."
        )
    return sum(1 for node in _plain_bfs(G, arbitrary_element(G))) == len(G)


def largest_connected_component(G):
    if not is_connected(G):
        return max(nx.connected_components(G), key=len)
    else:
        return None


def compute_yu_average(G):
    yu_average = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            yu_average[node] = sum(d for n, d in G.degree(neighbors)) / len(neighbors)
        else:
            yu_average[node] = 0
    return yu_average


def degree_assortativity_coefficient(G, x="out", y="in", weight=None, nodes=None):
    """Compute degree assortativity of graph.

    Assortativity measures the similarity of connections
    in the graph with respect to the node degree.

    Parameters
    ----------
    G : NetworkX graph

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    nodes: list or iterable (optional)
        Compute degree assortativity only for nodes in container.
        The default is all nodes.

    Returns
    -------
    r : float
       Assortativity of graph by degree.
    """
    if nodes is None:
        nodes = G.nodes

    degrees = None

    if G.is_directed():
        indeg = (
            {d for _, d in G.in_degree(nodes, weight=weight)}
            if "in" in (x, y)
            else set()
        )
        outdeg = (
            {d for _, d in G.out_degree(nodes, weight=weight)}
            if "out" in (x, y)
            else set()
        )
        degrees = set.union(indeg, outdeg)
    else:
        degrees = {d for _, d in G.degree(nodes, weight=weight)}

    mapping = {d: i for i, d, in enumerate(degrees)}
    M = nx.degree_mixing_matrix(G, x=x, y=y, nodes=nodes, weight=weight, mapping=mapping)

    return _numeric_ac(M, mapping=mapping)


def draw_network(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
    plt.show()
    plt.savefig('C:\\Users\\zhangwentao\\Desktop\\大三下学期\\Complex-Network\\email-small.jpg')
    

def rich_club_coefficient(G, normalized=True, Q=100, seed=None):
    """Compute the rich club coefficient of each degree k.
    
    The rich club coefficient of degree k is defined as the ratio between
    the total number of edges between nodes of degree k or higher, and the
    maximum possible number of such edges.
    
    Parameters
    ----------
    G : NetworkX graph
        A graph object.
    normalized : bool, optional (default=True)
        If True, return the normalized rich club coefficient.
    Q : integer, optional (default=100)
        The number of randomized networks to compare against.
    seed : integer, optional
        Seed for the random number generator.
        
    Returns
    -------
    rc : dictionary
        A dictionary mapping degree values to rich club coefficients.
    """
    def _compute_rc(G):
        # Find nodes with degree 0 and remove them
        degree_zero_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
        G.remove_nodes_from(degree_zero_nodes)
        
        # Compute the total number of edges between nodes of degree >= k for each k
        total_rc_edges = defaultdict(int)
        for u in G:
            du = G.degree(u)
            for v in G[u]:
                dv = G.degree(v)
                if du > dv and dv >= 1:
                    total_rc_edges[dv] += 1
                elif dv > du and du >= 1:
                    total_rc_edges[du] += 1
        
        # Compute the maximum possible number of edges between nodes of degree >= k for each k
        max_rc_edges = {}
        for k in total_rc_edges:
            nodes = [node for node, degree in dict(G.degree()).items() if degree >= k and degree >= 1]
            max_rc_edges[k] = len(nodes) * (len(nodes) - 1) // 2
        
        # Compute the rich club coefficient for each k
        rc = {k: total_rc_edges[k] / max_rc_edges[k] if max_rc_edges[k] > 0 else 0 for k in total_rc_edges}
        return rc

    
    # Find nodes with degree 0 and remove them
    degree_zero_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(degree_zero_nodes)
    
    # Compute the rich club coefficient for the original network
    rc = _compute_rc(G)
    
    # Compute the rich club coefficient for Q randomized networks
    rcran = defaultdict(float)
    for i in range(Q):
        R = nx.double_edge_swap(G, Q * G.number_of_edges(), max_tries=Q * G.number_of_edges() * 10, seed=seed)
        rcr = _compute_rc(R)
        for k in rc:
            rcran[k] += rcr[k]
    
    # Compute the average rich club coefficient over the randomized networks
    rcran = {k: v / Q for k, v in rcran.items()}
    
    # Return the normalized rich club coefficient if requested
    if normalized:
        rc = {k: rc[k] / rcran[k] for k in rc}
    
    return rc



file_path = r'C:\Users\zhangwentao\Desktop\大三下学期\Complex-Network\karate.txt'
G = load_network_data(file_path)

# Save adjacency matrix
adj_matrix_path = 'C:\\Users\\zhangwentao\\Desktop\\大三下学期\\Complex-Network\\adj_matrix.txt'
save_adjacency_matrix(G, adj_matrix_path)
print(f'Saved adjacency matrix to {adj_matrix_path}')

# Save edge list
edge_list_path = 'C:\\Users\\zhangwentao\\Desktop\\大三下学期\\Complex-Network\\edge_list.txt'
save_edge_list(G, edge_list_path)
print(f'Saved edge list to {edge_list_path}')

print('Clustering Coefficient:')
print(clustering_coefficient(G))

print('Degree Distribution:')
print(degree_distribution(G))

print('Average Path Length:')
print(Ave_path(nx.to_numpy_array(G)))

source, target = 1, 3

# source 和 target 是节点索引
shortest_path_length = dijkstra_shortest_path_length(nx.to_numpy_array(G), source, target)
print(f'Dijkstra Shortest Path Length between {source} and {target}: {shortest_path_length}')
print('Is Connected?:')
print(is_connected(G))

print('Largest Connected Component:')
print(largest_connected_component(G))

def compute_yu_average(G):
    yu_average = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            yu_average[node] = sum(d for n, d in G.degree(neighbors)) / len(neighbors)
        else:
            yu_average[node] = 0
    return yu_average

# 计算网络的余平均度
yu_average = compute_yu_average(G)


# 打印结果
for node, yu_aver in yu_average.items():
    print(f"Node {node}: Yu Average = {yu_aver}")

print('Assortativity:')
print(degree_assortativity_coefficient(G))

# Plot the relationship between Rich Club Coefficient and Node Degree
pos = nx.circular_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=500, node_color="yellow")
nx.draw_networkx_edges(G, pos, edge_color='blue')
nx.draw_networkx_labels(G, pos, font_size=16, font_family='sans-serif')

plt.axis('off')
plt.show()

# Calculate Rich Club Coefficient
# Find nodes with degree 0 and remove them
degree_zero_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
G.remove_nodes_from(degree_zero_nodes)

# 计算富人俱乐部系数和随机富人俱乐部系数

rc = nx.rich_club_coefficient(G)
rc_random = nx.rich_club_coefficient(G, normalized=True, Q=100)

# 输出结果
for k, v in rc.items():
    print(f"Degree {k}: Rich Club Coefficient = {v}, Randomized RC = {rc_random[k]}")
    
# 获取所有节点的度数
degrees = dict(G.degree())
print(degrees)
degrees = {node: degree for node, degree in degrees.items()}
# rc = {node: val for node, val in rc.items() if node in degrees}

degrees_rc = {node: degrees[node] for node in degrees if node in rc}
# rc_degrees = {node: rc[node] for node in rc if node in degrees}

# rc_random = nx.rich_club_coefficient(G, normalized=True, Q=100)
if rc_random:  # 如果字典不为空
    rc_random.pop(next(iter(rc_random)))

if rc:  # 如果字典不为空
    rc.pop(next(iter(rc_random)))

# 为了清晰起见，将数据组织到列表中
degree_keys = list(rc.keys())
rc_values = [rc[k] for k in degree_keys]
# 如果键不存在于 rc_random 中，则返回默认值 1
rc_random_values = [rc_random.get(k, 1) for k in degree_keys]

# 使用 plt.plot() 绘制数据并将数据点连接起来
plt.plot(degree_keys, rc_values, 'r.-', label='RC')
plt.plot(degree_keys, rc_random_values, 'b.-', label='Randomized RC')

# 添加图例，轴标签和显示图形
plt.legend(loc='best')
plt.xlabel('Node Degree')
plt.ylabel('Rich Club Coefficient')
plt.show()