import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
adj = np.zeros((9, 9))
G = nx.Graph(adj)
G.add_edges_from([(0, 1), (0, 7), (0, 8), (1, 2), (1, 8), (2, 7), (2, 3), (3, 4), (3, 5), (3, 6), (4, 5), (5, 6), (7, 8)])
nx.draw(G, with_labels=True)
plt.show()
# 划分图 G 的节点为若干个社团，得到社团划分列表
communities = [[0, 1, 7, 8, 2], [3, 4, 5, 6]]

# 计算模块度 Q 值
modularity = nx.algorithms.community.modularity(G, communities)
print(modularity)