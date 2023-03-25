import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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


def average_path_length(G):
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        return None


def breadth_first_search(G, source, target):
    return nx.shortest_path_length(G, source=source, target=target, weight='weight', method='dijkstra')


def is_connected(G):
    return nx.is_connected(G)


def largest_connected_component(G):
    if not is_connected(G):
        return max(nx.connected_components(G), key=len)
    else:
        return None


def assortativity(G):
    return nx.degree_assortativity_coefficient(G)


def draw_network(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
    plt.show()


def rich_club_coefficient(G):
    return nx.rich_club_coefficient(G)


def plot_rich_club_vs_degree(G):
    rc = rich_club_coefficient(G)
    degrees = list(rc.keys())
    coefficients = list(rc.values())
    plt.plot(degrees, coefficients, 'bo-')
    plt.xlabel('Degree')
    plt.ylabel('Rich Club Coefficient')
    plt.show()


file_path = r'C:\Users\zhangwentao\Desktop\大三下学期\Complex Network\karate.txt'
G = load_network_data(file_path)

# Save adjacency matrix
adj_matrix_path = 'C:\\Users\\zhangwentao\\Desktop\\大三下学期\\Complex Network\\adj_matrix.txt'
save_adjacency_matrix(G, adj_matrix_path)
print(f'Saved adjacency matrix to {adj_matrix_path}')

# Save edge list
edge_list_path = 'C:\\Users\\zhangwentao\\Desktop\\大三下学期\\Complex Network\\edge_list.txt'
save_edge_list(G, edge_list_path)
print(f'Saved edge list to {edge_list_path}')

print('Clustering Coefficient:')
print(clustering_coefficient(G))

print('Degree Distribution:')
print(degree_distribution(G))

print('Average Path Length:')
print(average_path_length(G))

source, target = 1, 10
print(f'Breadth First Search Distance between {source} and {target}:')
print(breadth_first_search(G, source, target))

print('Is Connected?:')
print(is_connected(G))

print('Largest Connected Component:')
print(largest_connected_component(G))

print('Assortativity:')
print(assortativity(G))

draw_network(G)

# Calculate Rich Club Coefficient
# Find nodes with degree 0 and remove them
degree_zero_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
G.remove_nodes_from(degree_zero_nodes)
rc = nx.rich_club_coefficient(G)

# Extract node degrees and Rich Club Coefficients
degrees = dict(G.degree())
k_values = sorted(set(degrees.values()), reverse=True)
rc_values = [rc.get(k, 0) for k in k_values]

# Plot the relationship between Rich Club Coefficient and Node Degree
plt.plot(k_values, rc_values, 'o-', color='r')
plt.xscale('log')  # Set x-axis to log scale
plt.xlabel('Node Degree (log)')
plt.ylabel('Rich Club Coefficient')
plt.title('Rich Club vs Node Degree')
plt.show()