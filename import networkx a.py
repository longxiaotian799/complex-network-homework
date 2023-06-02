import networkx as nx
sequence = nx.random_powerlaw_tree_sequence(10, tries=5)
print(sequence)
G = nx.configuration_model([])
# nx.configuration_model()