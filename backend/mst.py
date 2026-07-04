import pandas as pd
import numpy as np
import os

class KruskalMST:
    """
    Kruskal's Maximum Spanning Tree for weighted networks.
    Automatic separator and header detection.
    Column access is always positional (iloc[0], iloc[1], iloc[2]).
    """

    def __init__(self, test_mode=False, test_size=10000):
        self.test_mode = test_mode
        self.test_size = test_size

    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            x_root = self.find(x)
            y_root = self.find(y)
            if x_root != y_root:
                self.parent[y_root] = x_root

    def kruskal_maxst(self, edges, num_nodes):
        if num_nodes == 0:
            return []
        edges_sorted = sorted(edges, key=lambda x: -x[2])
        uf = self.UnionFind(num_nodes)
        maxst_edges = []
        for u, v, w in edges_sorted:
            if uf.find(u) != uf.find(v):
                uf.union(u, v)
                maxst_edges.append((u, v, w))
                if len(maxst_edges) == num_nodes - 1:
                    break
        return maxst_edges

    def load_data(self, input_file):
        with open(input_file, 'r') as f:
            first_line = f.readline().strip()

        separators = {'\t': first_line.count('\t'), ',': first_line.count(','),
                      ';': first_line.count(';'), ' ': first_line.count(' ')}
        sep = max(separators, key=lambda k: separators[k])
        if separators[sep] == 0:
            sep = '\t'

        nrows = self.test_size if self.test_mode else None

        df = pd.read_csv(input_file, sep=sep, header=0, nrows=nrows)
        print(f"  Columns detected: {list(df.columns)}")
        print(f"  Loaded {len(df)} rows")

        if df.shape[1] > 2:
            df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
            df = df.dropna(subset=[df.columns[2]])

        edges = []
        nodes = set()

        for _, row in df.iterrows():
            u = str(row.iloc[0]).strip()
            v = str(row.iloc[1]).strip()

            if not u or not v or u == 'nan' or v == 'nan' or u == v:
                continue

            w = 1.0
            if df.shape[1] > 2 and pd.notna(row.iloc[2]):
                w = float(row.iloc[2])

            if w <= 0:
                continue

            nodes.add(u)
            nodes.add(v)
            edges.append((u, v, w))

        nodes = sorted(nodes)
        node_to_id = {node: idx for idx, node in enumerate(nodes)}
        edges_idx = [(node_to_id[u], node_to_id[v], w) for u, v, w in edges]

        print(f"  Found {len(nodes)} nodes, {len(edges_idx)} edges")
        if edges_idx:
            weights = [w for _, _, w in edges_idx]
            print(f"  Weight range: {min(weights):.4f} - {max(weights):.4f}")

        return edges_idx, nodes

    def run_and_save(self, input_file, output_file):
        print(f"\n=== Kruskal Maximum Spanning Tree ===")
        print(f"Input: {input_file}")
        print(f"Test mode: {self.test_mode}")

        edges, nodes = self.load_data(input_file)

        if not edges or len(nodes) == 0:
            print("  No valid edges found")
            with open(output_file, 'w') as f:
                f.write("source\ttarget\tweight\n")
            return output_file

        print(f"Running Kruskal on {len(nodes)} nodes, {len(edges)} edges...")
        maxst_edges = self.kruskal_maxst(edges, len(nodes))

        id_to_node = {idx: node for idx, node in enumerate(nodes)}
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("source\ttarget\tweight\n")
            for u, v, w in maxst_edges:
                f.write(f"{id_to_node[u]}\t{id_to_node[v]}\t{w}\n")

        print(f"  MST edges: {len(maxst_edges)} (expected {len(nodes)-1})")
        print(f"  Saved to: {output_file}")
        return output_file

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "weighted_gene_edges_one_mode.tsv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "gene_maxst_edges.tsv"

    kruskal = KruskalMST(test_mode=False)
    kruskal.run_and_save(input_file, output_file)