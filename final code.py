import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, id):
        self.id = id
        self.neighbors = set()
        
    def add_neighbor(self, neighbor_id):
        self.neighbors.add(neighbor_id)
        
class Graph:
    def __init__(self):
        self.nodes = {}
        self.num_nodes = 0
        
    def add_edge(self, u, v):
        if u not in self.nodes:
            self.nodes[u] = Node(u)
            self.num_nodes += 1
        if v not in self.nodes:
            self.nodes[v] = Node(v)
            self.num_nodes += 1
        self.nodes[u].add_neighbor(v)
        self.nodes[v].add_neighbor(u)
        
    def degree_sequence(self):
        degrees = [len(node.neighbors) for node in self.nodes.values()]
        degrees.sort(reverse=True)
        return degrees
    
    def clustering_coefficient(self):
        sum_cc = 0.0
        for node in self.nodes.values():
            num_neighbors = len(node.neighbors)
            if num_neighbors > 1:
                possible_edges = num_neighbors * (num_neighbors - 1) / 2
                cc = 0.0
                for neighbor in node.neighbors:
                    for other_neighbor in self.nodes[neighbor].neighbors:
                        if other_neighbor != node.id and other_neighbor in node.neighbors:
                            cc += 1
                cc /= possible_edges
                sum_cc += cc
        return sum_cc / self.num_nodes
    
    def average_path_length(self):
        distances = []
        for node1 in self.nodes.values():
            for node2 in self.nodes.values():
                if node1.id < node2.id:
                    distances.append(self.breadth_first_search(node1.id, node2.id))
        return sum(distances) / len(distances)
    
    def breadth_first_search(self, source, target):
        visited = {node_id: False for node_id in self.nodes.keys()}
        queue = [(source, 0)]
        visited[source] = True
        while len(queue) > 0:
            curr_node, curr_dist = queue.pop(0)
            if curr_node == target:
                return curr_dist
            for neighbor in self.nodes[curr_node].neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, curr_dist+1))
        return None
    
    def is_connected(self):
        visited = {node_id: False for node_id in self.nodes.keys()}
        dfs_order = []
        self.depth_first_search(next(iter(self.nodes)), visited, dfs_order)
        return all(visited.values())
    
    def largest_connected_component(self):
        visited = {node_id: False for node_id in self.nodes.keys()}
        dfs_order = []
        self.depth_first_search(next(iter(self.nodes)), visited, dfs_order)
        largest_component = set()
        for node_id in reversed(dfs_order):
            if not visited[node_id]:
                subgraph = set()
                self.depth_first_search(node_id, visited, subgraph.add)
                if len(subgraph) > len(largest_component):
                    largest_component = subgraph
        return largest_component
    
    def depth_first_search(self, node_id, visited, action_func):
        visited[node_id] = True
        action_func(node_id)
        for neighbor in self.nodes[node_id].neighbors:
            if not visited[neighbor]:
                self.depth_first_search(neighbor, visited, action_func)
                
    def assortativity(self):
        nodes = list(self.nodes.values())
        degrees = np.array([len(node.neighbors) for node in nodes])
        s = degrees.sum()
        s_squared = (degrees ** 2).sum()
        degrees_normalized = degrees / s
        edges = [edge for node in nodes for edge in [(node.id, neighbor_id) for neighbor_id in node.neighbors]]
        edges_normalized = np.array([(degrees_normalized[edge[0]], degrees_normalized[edge[1]]) for edge in edges])
        num_edges = len(edges)
        edge_sum = (edges_normalized[:,0] * edges_normalized[:,1]).sum()
        edge_squared_sum = (edges_normalized[:,0] ** 2).sum() + (edges_normalized[:,1] ** 2).sum()
        numerator = edge_sum * s - s_squared
        denominator = edge_squared_sum * s - s_squared
        return numerator / denominator
    
    def draw_network(self):
        pos = self.spring_layout()
        fig, ax = plt.subplots(figsize=(10,10))
        ax.axis('off')
        for node in self.nodes.values():
            x, y = pos[node.id]
            ax.scatter(x, y, s=200, c=node_colors[node.id])
        for neighbor in node.neighbors:
            x2, y2 = pos[neighbor]
            ax.plot([x, x2], [y, y2], c='black', alpha=0.5)
            plt.show()

def spring_layout(self):
    k = np.sqrt(1 / self.num_nodes)
    pos = {node_id: np.random.rand(2) for node_id in self.nodes.keys()}
    for _ in range(200): # number of iterations
        disp = np.zeros((self.num_nodes, 2))
        for i, node_id in enumerate(self.nodes.keys()):
            for neighbor_id in self.nodes[node_id].neighbors:
                delta_x = pos[node_id][0] - pos[neighbor_id][0]
                delta_y = pos[node_id][1] - pos[neighbor_id][1]
                dist = np.sqrt(delta_x ** 2 + delta_y ** 2)
                disp[i, 0] += delta_x * k ** 2 / dist
                disp[i, 1] += delta_y * k ** 2 / dist
        pos += disp
        pos = np.clip(pos, 0, 1)
    return pos

def rich_club_coefficient(self):
    degrees = np.array([len(node.neighbors) for node in self.nodes.values()])
    max_degree = degrees.max()
    rc = {}
    for k in range(1, max_degree+1):
        k_nodes = np.where(degrees >= k)[0]
        if len(k_nodes) > 1:
            subgraph = self.subgraph(k_nodes)
            edges_inside = len(subgraph.edges())
            nodes_outside = set(self.nodes.keys()) - set(k_nodes)
            num_possible_edges = len(nodes_outside) * k
            edges_outside = sum([1 for u in k_nodes for v in self.nodes[u].neighbors if v in nodes_outside and degrees[v-1] >= k])
            rc[k] = 2 * edges_inside / num_possible_edges if num_possible_edges > 0 else 0
    return rc

def subgraph(self, node_ids):
    subgraph = Graph()
    for node_id in node_ids:
        subgraph.add_edge(node_id, node_id)
        for neighbor in self.nodes[node_id].neighbors:
            if neighbor in node_ids and node_id < neighbor:
                subgraph.add_edge(node_id, neighbor)
    return subgraph

def plot_rich_club_vs_degree(self):
    rc = self.rich_club_coefficient()
    degrees = sorted(rc.keys())
    coefficients = [rc[k] for k in degrees]

    plt.plot(degrees, coefficients, 'bo-')
    plt.xscale('log') # 设置x轴为对数坐标系
    plt.xlabel('Degree (log)')
    plt.ylabel('Rich Club Coefficient')
    plt.title('Rich Club vs Node Degree')
    plt.show()

graph = Graph()
file_path = 'C:\\Users\\zhangwentao\\Desktop\\大三下学期\\Complex Network\\email.txt'
with open(file_path, 'r') as f:
    for line in f:
        u, v = map(int, line.strip().split())
        graph.add_edge(u, v)


graph.draw_network()

graph.plot_rich_club_vs_degree()