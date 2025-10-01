import os
import sys
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def is_hidden_dir(d):
    # Check if a directory is hidden
    if sys.platform.startswith("win"):
        try:
            p = subprocess.check_output(["attrib", d.encode('cp1251', errors='replace')])
            p = p.decode('cp1251', errors='replace')
            return 'H' in p[:12]  # Hidden attribute
        except:
            return False
    else:
        return os.path.basename(d).startswith('.')


def get_tree(tree, G=nx.DiGraph(), itr=0, max_itr=2000, max_depth=4, current_depth=0):
    # Recursively build a graph representation of the directory tree
    if not tree or current_depth > max_depth:
        return G
    point = tree.pop(0)
    itr += 1
    try:
        if not os.path.exists(point):
            return G
        if not os.access(point, os.R_OK):
            return G

        node_type = 'file' if os.path.isfile(point) else 'dir'
        G.add_node(point, type=node_type)

        if node_type == 'dir':
            items = os.listdir(point)
            sub_dirs = [
                os.path.join(point, x)
                for x in items
                if os.path.isdir(os.path.join(point, x)) and not is_hidden_dir(os.path.join(point, x))
            ]
            files = [
                os.path.join(point, x)
                for x in items
                if os.path.isfile(os.path.join(point, x))
            ]

            # Add edges between directory and its children
            for d in sub_dirs + files:
                target_type = 'file' if os.path.isfile(d) else 'dir'
                if d not in G.nodes:
                    G.add_node(d, type=target_type)
                G.add_edge(point, d)

            # Continue exploring subdirectories
            if sub_dirs:
                tree.extend(sub_dirs)

        # Continue recursion until max depth or iteration limit
        if tree and itr <= max_itr:
            return get_tree(tree, G, itr, max_itr, max_depth, current_depth + 1)
        else:
            return G
    except Exception:
        return G


def assign_colors(G):
    # Assign colors based on node type and contents
    colors = []
    for node in G.nodes:
        if 'type' not in G.nodes[node]:
            colors.append((0.5, 0.5, 0.5))  # Gray for unknown/error
            continue

        if G.nodes[node]['type'] == 'dir':
            try:
                count = len([x for x in os.listdir(node) if not is_hidden_dir(os.path.join(node, x))])
            except Exception:
                count = 0
            # More contents = brighter green
            green_intensity = min(1, count / 20)
            colors.append((0.0, green_intensity, 0.0))  # Shades of green for directories
        else:
            colors.append((0.0, 0.0, 1.0))  # Blue for files
    return colors


def main(root_dir: str):
    if not os.path.exists(root_dir):
        print(f"Root directory does not exist: {root_dir}")
        return
    if not os.access(root_dir, os.R_OK):
        print(f"No read access to root directory: {root_dir}")
        return

    G = get_tree(tree=[root_dir], max_depth=4)

    node_count = len(G.nodes)
    if node_count > 500:
        print(f"Graph is too large ({node_count} nodes). Using spring layout only.")
        layouts = [('spring', nx.draw_spring)]
    else:
        layouts = [
            ('kamada_kawai', nx.draw_kamada_kawai),
            ('circular', nx.draw_circular),
            ('spectral', nx.draw_spectral),
            ('spring', nx.draw_spring)
        ]

    node_colors = assign_colors(G)

    options = {
        'node_color': node_colors,
        'node_size': 30,
        'width': 0.5,
        'with_labels': False,
        'alpha': 0.7,
        'font_size': 3,
        'font_family': 'Arial'
    }

    plt.figure(figsize=(12, 12))

    for i, (layout_name, draw_func) in enumerate(layouts, 1):
        plt.subplot(2, 2, i)
        plt.title(f'{layout_name.capitalize()} Layout')
        try:
            draw_func(G, **options)

            # --- Add legend ---
            file_patch = mpatches.Patch(color=(0.0, 0.0, 1.0), label="Files")
            dir_patch = mpatches.Patch(color=(0.0, 0.7, 0.0), label="Directories (more = greener)")
            err_patch = mpatches.Patch(color=(0.5, 0.5, 0.5), label="Error/unknown")

            plt.legend(
                handles=[file_patch, dir_patch, err_patch],
                loc="upper right",
                fontsize=8,
                frameon=True
            )

        except Exception as e:
            print(f"Error rendering {layout_name} layout: {e}")
            plt.text(0.5, 0.5, f"Failed: {layout_name}\n{e}", ha='center', va='center')

    plt.tight_layout()
    plt.savefig('graphs.png', dpi=600)
    plt.show()


if __name__ == "__main__":
    root_dir = r"E:\khpi"  # directory with all KhPI files (labs, reports, documents)
    main(root_dir)