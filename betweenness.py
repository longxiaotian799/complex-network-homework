import networkx as nx
file_path = "karate1.txt"
# 从文件中读取边列表，创建一个图，其中节点的类型为整数，边的权重为整数
G = nx.read_edgelist(
    file_path,
    nodetype=int,
    data=(('weight', int),)
)
print(type(G))
# 计算网络中的节点的介数中心性，并进行排序输出
def topNBetweeness(G):
	score = nx.betweenness_centrality(G,normalized=True)
	# score = sorted(score.items(), key=lambda item:item[1], reverse = True)
	print("betweenness_centrality: ", score)
	# output = []
	# for node in score:
	# 	output.append(node[0])
 
	# print(output)
	# fout = open("betweennessSorted.data", 'w')
	# for target in output:
	# 	fout.write(str(target)+" ")
 
topNBetweeness(G)

'''betweenness_centrality:  [(1, 0.4376352813852815), (24, 0.30407497594997596), (22, 0.14524711399711404), 
(3, 0.14365680615680618), (17, 0.13827561327561327), (9, 0.05592682780182782), (2, 0.05393668831168831), (13, 0.045863395863395856), (15, 0.03247504810004811),
(6, 0.02998737373737374), (7, 0.029987373737373736), (20, 0.022333453583453577), (30, 0.017613636363636363), (18, 0.014411976911976905), (4, 0.011909271284271283), (31, 0.0038404882154882154), (32, 0.0029220779220779218), 
(33, 0.0022095959595959595), (21, 0.0017947330447330447), (19, 0.0008477633477633478), (5, 0.0006313131313131313), (10, 0.0006313131313131313), (8, 0.0), 
(11, 0.0), (12, 0.0), (14, 0.0), (16, 0.0), (23, 0.0), (25, 0.0), (26, 0.0), (27, 0.0), (28, 0.0), (29, 0.0), (34, 0.0)]'''