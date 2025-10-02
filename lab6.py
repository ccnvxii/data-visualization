import networkx as nx
import matplotlib.pyplot as plt
import random

# Load the graph
G = nx.read_gml("internet_routers-22july06.gml")

# Basic statistics
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

degrees = dict(G.degree())
print("Average degree:", sum(degrees.values()) / len(degrees))

# Largest connected component
largest_cc = max(nx.connected_components(G), key=len)
H_largest = G.subgraph(largest_cc)

print("Nodes in the largest component:", H_largest.number_of_nodes())
print("Edges in the largest component:", H_largest.number_of_edges())

# Visualization of the largest component (random layout)
plt.figure(figsize=(12, 8))
plt.suptitle("Figure 1: Largest component of the graph", fontsize=16, fontweight="bold")

pos = nx.random_layout(H_largest)
nx.draw(H_largest, pos,
        with_labels=False,
        node_size=10,
        node_color="lightgreen",
        edge_color="gray")

plt.title("Visualization using random layout", fontsize=12)
plt.xlabel("X coordinate", fontsize=11)
plt.ylabel("Y coordinate", fontsize=11)
plt.show()

# Visualization of a sample of 500 nodes for more detail
sample_nodes = random.sample(list(H_largest.nodes()), 500)
H_sample = H_largest.subgraph(sample_nodes)

plt.figure(figsize=(12, 8))
plt.suptitle("Figure 2: Sample subgraph (500 nodes)", fontsize=16, fontweight="bold")

pos = nx.spring_layout(H_sample, seed=42)
nx.draw(H_sample, pos,
        with_labels=True,
        font_size=6,
        node_size=30,
        node_color="orange",
        edge_color="gray")

plt.title("Visualization using spring layout", fontsize=12)
plt.xlabel("X coordinate", fontsize=11)
plt.ylabel("Y coordinate", fontsize=11)
plt.show()
