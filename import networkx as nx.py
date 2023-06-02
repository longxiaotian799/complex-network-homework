import networkx as nx

def compute_yu_average(G):
    yu_average = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            yu_average[node] = sum(d for n, d in G.degree(neighbors)) / len(neighbors)
        else:
            yu_average[node] = 0
    return yu_average

def load_network_data(file_path):
    G = nx.read_edgelist(file_path, nodetype=int, data=(('weight',float),))
    return G

file_path = r'C:\Users\zhangwentao\Desktop\大三下学期\Complex-Network\karate.txt'
G = load_network_data(file_path)
# 计算网络的余平均度
yu_average = compute_yu_average(G)

# 打印结果
for node, yu_aver in yu_average.items():
    print(f"Node {node}: Yu Average = {yu_aver}")