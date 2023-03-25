import networkx as nx
import matplotlib.pyplot as plt
import time

start_time = time.time()  # 记录开始时间

# 在这里写入要测试的代码

G = nx.Graph()
G.add_node("A")
G.add_node("B")
G.add_node("C")
G.add_edge("A", "B")
G.add_edge("B", "C")
G.add_edge("C", "A")

pos = nx.circular_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=500, node_color="yellow")
nx.draw_networkx_edges(G, pos, edge_color='blue')
nx.draw_networkx_labels(G, pos, font_size=16, font_family='sans-serif')

plt.axis('off')
plt.show()

end_time = time.time()  # 记录结束时间

elapsed_time = end_time - start_time  # 计算耗时

print('代码运行时长为：', elapsed_time, '秒')
