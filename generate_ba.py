import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

n = 1000
m = 3
seed = 42
ba_network = nx.barabasi_albert_graph(n, m, seed)
adj = list(nx.to_numpy_matrix(ba_network))
print(sum((sum(adj))) / len(adj))
# np.savetxt('ba_network.txt', nx.to_numpy_matrix(ba_network), fmt='%d')