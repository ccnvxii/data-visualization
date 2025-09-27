# Create a binary tree according to the student number and perform its visualization with D. Knuth's algorithm.
# Integers in the range [-250, 300]

import matplotlib.pyplot as plt


# Binary Tree Implementation
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None


class Tree:
    def __init__(self):
        self.root = None

    def add_node(self, key, node=None):
        if self.root is None:
            self.root = Node(key)
            return
        if node is None:
            node = self.root
        if key <= node.key:
            if node.left is None:
                node.left = Node(key)
                node.left.parent = node
            else:
                self.add_node(key, node.left)
        else:
            if node.right is None:
                node.right = Node(key)
                node.right.parent = node
            else:
                self.add_node(key, node.right)


# Knuth-style layout
def knuth_layout(node, x=0, y=0, positions=None, level_gap=1.5, sibling_gap=1):
    if positions is None:
        positions = {}
    if node.left:
        knuth_layout(node.left, x - sibling_gap, y - level_gap, positions, level_gap, sibling_gap / 1.5)
    positions[node.key] = (x, y)
    if node.right:
        knuth_layout(node.right, x + sibling_gap, y - level_gap, positions, level_gap, sibling_gap / 1.5)
    return positions


# Tree Visualization
def draw_tree(tree):
    if tree.root is None:
        print("Tree is empty")
        return
    positions = knuth_layout(tree.root)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_axis_off()

    # Draw edges
    def draw_edges(node):
        if node.left:
            ax.plot([positions[node.key][0], positions[node.left.key][0]],
                    [positions[node.key][1], positions[node.left.key][1]], 'k-')
            draw_edges(node.left)
        if node.right:
            ax.plot([positions[node.key][0], positions[node.right.key][0]],
                    [positions[node.key][1], positions[node.right.key][1]], 'k-')
            draw_edges(node.right)

    draw_edges(tree.root)

    # Draw nodes
    for key, (x, y) in positions.items():
        ax.plot(x, y, 'o', markersize=20, color='skyblue')
        ax.text(x, y, str(key), fontsize=12, ha='center', va='center')

    plt.show()


# --------------------------------
t = Tree()
# Integers in the range [-250, 300]
numbers = [10, 13, 14, 8, 9, 7, 11, -50, 0, 25, 100, 200]
for num in numbers:
    t.add_node(num)

draw_tree(t)
