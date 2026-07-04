import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix, csr_matrix
from collections import defaultdict
from tqdm import tqdm

class TrueMultiscaleBackbone:
    def __init__(self, max_iter=200, tol=1e-6, smoothing_power=2,
                 significance_level=0.05, sparsity_adjustment=True,
                 early_stop_percentile=5, min_edges=None):
        self.max_iter = max_iter
        self.tol = tol
        self.smoothing_power = smoothing_power
        self.significance_level = significance_level
        self.sparsity_adjustment = sparsity_adjustment
        self.early_stop_percentile = early_stop_percentile
        self.min_edges = min_edges

    def _detect_separator(self, file_path):
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        separators = {'\t': first_line.count('\t'), ',': first_line.count(','),
                      ';': first_line.count(';'), ' ': first_line.count(' ')}
        sep = max(separators, key=lambda k: separators[k])
        return sep if separators[sep] > 0 else '\t'

    def _make_doubly_stochastic(self, matrix):
        matrix = matrix.tocsr()
        for _ in range(self.max_iter):
            row_sums = matrix.sum(axis=1).A.ravel()
            row_sums[row_sums == 0] = 1
            matrix = matrix.multiply(1 / row_sums[:, None])

            col_sums = matrix.sum(axis=0).A.ravel()
            col_sums[col_sums == 0] = 1
            matrix = matrix.multiply(1 / col_sums)

            if np.max(np.abs(1 - matrix.sum(axis=1).A.ravel())) < self.tol and \
               np.max(np.abs(1 - matrix.sum(axis=0).A.ravel())) < self.tol:
                break
        return matrix.power(self.smoothing_power)

    def _tarjan_strongly_connected(self, adj_list):
        index = 0
        indices = {}
        lowlinks = {}
        on_stack = set()
        stack = []
        components = []

        def strongconnect(node):
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)

            for neighbor in adj_list.get(node, []):
                if neighbor not in indices:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif neighbor in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[neighbor])

            if lowlinks[node] == indices[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    component.append(w)
                    if w == node:
                        break
                components.append(component)

        for node in adj_list:
            if node not in indices:
                strongconnect(node)

        return components

    def _build_backbone(self, matrix, node_names):
        coo = matrix.tocoo()
        edges = sorted(zip(coo.row, coo.col, coo.data), key=lambda x: -x[2])

        weights = [w for _, _, w in edges]
        percentile_threshold = np.percentile(weights, self.early_stop_percentile)
        significance_threshold = self.significance_level if self.significance_level else 0

        G = nx.DiGraph()
        G.add_nodes_from(node_names)
        adj_list = defaultdict(list)
        backbone_edges = []

        for i, j, w in tqdm(edges, desc="Building backbone"):
            if w < significance_threshold:
                continue

            G.add_edge(node_names[i], node_names[j], weight=w)
            adj_list[i].append(j)
            backbone_edges.append((i, j, w))

            if self.min_edges and len(backbone_edges) < self.min_edges:
                continue

            if w < percentile_threshold:
                print(f"Stopping: edge weight {w:.4f} < {percentile_threshold:.4f} percentile")
                break

            if len(backbone_edges) % 1000 == 0:
                components = self._tarjan_strongly_connected(adj_list)
                if len(components) == 1:
                    print("Stopping: achieved strong connectivity")
                    break

        return G

    def extract_backbone(self, edge_list_path, output_path, mode='strict',
                         nrows=None, logger=None):
        if mode == 'strict':
            self.early_stop_percentile = 20
            self.significance_level = 0.05
            self.min_edges = None
        elif mode == 'balanced':
            self.early_stop_percentile = 1
            self.significance_level = 0.001
            self.min_edges = 100
        elif mode == 'full':
            self.early_stop_percentile = 0
            self.significance_level = 0
            self.min_edges = None

        print("Loading network...")
        try:
            sep = self._detect_separator(edge_list_path)

            df = pd.read_csv(edge_list_path, sep=sep, header=0, nrows=nrows)
            print(f"  Columns detected: {list(df.columns)}")

            if df.empty:
                print(f"  Warning: The file is empty.")
                return nx.DiGraph()

            df = df.rename(columns={
                df.columns[0]: 'source',
                df.columns[1]: 'target',
                df.columns[2]: 'weight'
            })

            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df = df.dropna(subset=['weight'])
            df = df[df['source'] != df['target']]

            if df.empty:
                print("  Warning: No valid edges found.")
                return nx.DiGraph()

        except FileNotFoundError:
            print(f"Error: File not found: '{edge_list_path}'")
            return nx.DiGraph()
        except Exception as e:
            print(f"Error while reading: {e}")
            return nx.DiGraph()

        nodes = sorted(set(df['source']) | set(df['target']))
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        if not nodes:
            print("  Warning: No nodes found.")
            return nx.DiGraph()

        sparse_matrix = coo_matrix(
            (df['weight'],
             ([node_to_idx[s] for s in df['source']],
              [node_to_idx[t] for t in df['target']])),
            shape=(len(nodes), len(nodes))
        ).tocsr()

        print(f"Network: {len(nodes)} nodes, {sparse_matrix.nnz} edges")

        if sparse_matrix.nnz == 0:
            print("Warning: The sparse matrix is empty.")
            return nx.DiGraph()

        print("Normalizing matrix...")
        normalized = self._make_doubly_stochastic(sparse_matrix)

        print("Extracting backbone...")
        backbone = self._build_backbone(normalized, nodes)

        print(f"\nResults:")
        print(f"  Nodes: {backbone.number_of_nodes()}")
        print(f"  Edges: {backbone.number_of_edges()}")
        print(f"  Strongly connected: {nx.is_strongly_connected(backbone)}")

        rows = []
        for u, v, data in backbone.edges(data=True):
            rows.append({'source': u, 'target': v, 'weight': data.get('weight', 1.0)})

        pd.DataFrame(rows, columns=['source', 'target', 'weight']).to_csv(
            output_path, index=False, sep='\t')
        print(f"  Saved to {output_path}")

        return backbone

if __name__ == "__main__":
    extractor = TrueMultiscaleBackbone()
    extractor.extract_backbone(
        "weighted_gene_edges_one_mode.tsv",
        "ds_backbone.csv",
        mode='strict',
        nrows=200000
    )