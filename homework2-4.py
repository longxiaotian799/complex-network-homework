from collections import Counter
import networkx as nx
import random
import matplotlib.pyplot as plt
def degree_distribution(G):
    # 获取所有节点的度并创建 Counter 对象，记录每个度出现的次数
    counts = Counter(deg for _, deg in G.degree)
    
    # 初始化一个空列表来存储输出
    output = []
    
    # 遍历 Counter 对象
    for degree, count in counts.items():
        # 将每个度数添加到输出列表中，次数等于其在 Counter 中的计数值
        output.extend([degree] * count)
    
    return output
def create_configuration_model(N, degree_sequence):
    # 创建一个空的无向图
    G = nx.Graph()

    # 添加 N 个节点
    G.add_nodes_from(range(N))

    # 生成每个节点的桩子
    stubs = [node for node, degree in enumerate(degree_sequence) for _ in range(degree)]
    i = 0
    # 随机连接桩子，生成边
    while len(stubs) > 1:
        stub1 = random.choice(stubs)
        stub2 = random.choice(list(filter(lambda x: x != stub1, stubs)))  # 避免自环
        i += 1
        if not G.has_edge(stub1, stub2):  # 避免多重边
            G.add_edge(stub1, stub2)
        else:
            continue
        stubs.remove(stub1)
        stubs.remove(stub2)
    return G
def degree_distribution_plot(G):
    # 获取所有节点的度并创建Counter对象，记录每个度出现的次数
    counts = Counter(deg for _, deg in G.degree)

    # 找到最小和最大的度
    _, max_degree = min(counts), max(counts)
    
    max_degree = max(counts.keys())
    degree_counts = [0] * (max_degree + 1)

    for degree, count in counts.items():
        degree_counts[degree] = count
    print('\nDegree Distribution:')
    print(degree_counts)

        
    # 绘制度分布直方图
    plt.bar(counts.keys(), counts.values())
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution of Configuration Network')
    plt.show()

# 定义文件路径
file_path = 'email.txt'

# 从文件中读取边列表，创建一个图，其中节点的类型为整数，边的权重为整数
G = nx.read_edgelist(
    file_path,
    nodetype=int,
    data=(('weight', int),)
)

# 获取原始图的节点数和度分布
N = G.number_of_nodes()
degree_sequence = degree_distribution(G)
print("degree_sequence: ", degree_sequence)
# 创建一个具有email网络节点个数和给定度序列的配置模型
G_config = create_configuration_model(N, degree_sequence)

# 计算配置模型的度直方图并删除值为 0 的元素
histogram = nx.degree_histogram(G)
histogram = [x for x in histogram if x != 0]

# 计算配置模型的度分布
counts = Counter(deg for _, deg in G_config.degree)
degree_counts = list(counts.values())

# 比较度直方图和度分布是否相同
comparison_result = (sorted(histogram) == sorted(degree_counts))
print(f"Compare degree distribution: {comparison_result}")
degree_distribution_plot(G_config)
