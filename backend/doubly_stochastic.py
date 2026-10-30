import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix, csr_matrix, triu
from collections import defaultdict
from tqdm import tqdm

class TrueMultiscaleBackbone:
    def __init__(self, max_iter=1000, tol=1e-12, epsilon=1e-4,
                 significance_level=0.05, sparsity_adjustment=True,
                 early_stop_percentile=5, min_edges=None, directed=False):
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
        self.significance_level = significance_level
        self.sparsity_adjustment = sparsity_adjustment
        self.early_stop_percentile = early_stop_percentile
        self.min_edges = min_edges
        self.directed = directed

    def _detect_separator(self, file_path):
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        separators = {'\t': first_line.count('\t'), ',': first_line.count(','),
                      ';': first_line.count(';'), ' ': first_line.count(' ')}
        sep = max(separators, key=lambda k: separators[k])
        return sep if separators[sep] > 0 else '\t'

    def _make_doubly_stochastic(self, matrix):
        """Sinkhorn-Knopp on W + epsilon (every cell, as in the reference code of
        Coscia and netbone): a sparse W usually has no doubly stochastic scaling
        (no total support), a positive matrix always has one. The dense matrix is
        never built: the scaled matrix is diag(r) (W + eps) diag(c), so
        row sums = r * (W c + eps * sum(c)) and column sums likewise.
        Stops when std(row sums) < tol, as the reference does.
        Returns the scaled values on the nonzero cells of W."""
        W = matrix.tocsr()
        eps = self.epsilon
        r = np.ones(W.shape[0]); c = np.ones(W.shape[1])
        for it in range(1, self.max_iter + 1):
            r = 1 / (W @ c + eps * c.sum())
            c = 1 / (W.T @ r + eps * r.sum())
            if np.std(r * (W @ c + eps * c.sum())) < self.tol:
                break
        else:
            print(f"Warning: Sinkhorn did not converge in {self.max_iter} iterations")
        print(f"Sinkhorn converged in {it} iterations (epsilon={eps})")
        coo = W.tocoo()
        return csr_matrix((r[coo.row] * (coo.data + eps) * c[coo.col], (coo.row, coo.col)), shape=W.shape)

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

    def _build_backbone_slater(self, edges, node_names, target_components):
        """Slater (2009), adapted to a disconnected input: add edges in decreasing
        order until the backbone has as many connected components as the input
        (weak components for a directed network)."""
        parent = list(range(len(node_names)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        G = nx.DiGraph() if self.directed else nx.Graph()
        G.add_nodes_from(node_names)
        components = len(node_names)
        for i, j, w in edges:
            G.add_edge(node_names[i], node_names[j], weight=w)
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                components -= 1
                if components == target_components:
                    print(f"Stopping: backbone has {components} components, as the input")
                    break
        return G

    def _build_backbone(self, matrix, node_names, target_components=None):
        """Edges are taken in decreasing order of the doubly stochastic value.

        mode 'slater': only Slater's rule (see _build_backbone_slater).
        Other modes apply, per edge, in this order:
          1. an edge below significance_level is skipped (the edges are sorted,
             so no later edge is kept either);
          2. the edge is added;
          3. stop if its value is below the early_stop_percentile-th
             percentile of all values (not applied before min_edges edges);
          4. every 1,000 edges, stop if the nodes reached so far form a single
             (strongly) connected component.
        """
        coo = matrix.tocoo()
        edges = sorted(zip(coo.row, coo.col, coo.data), key=lambda x: -x[2])
        if target_components is not None:
            return self._build_backbone_slater(edges, node_names, target_components)

        weights = [w for _, _, w in edges]
        percentile_threshold = np.percentile(weights, self.early_stop_percentile)
        significance_threshold = self.significance_level if self.significance_level else 0

        G = nx.DiGraph() if self.directed else nx.Graph()
        G.add_nodes_from(node_names)
        adj_list = defaultdict(list)
        backbone_edges = []

        for i, j, w in tqdm(edges, desc="Building backbone"):
            if w < significance_threshold:
                continue

            G.add_edge(node_names[i], node_names[j], weight=w)
            adj_list[i].append(j)
            if not self.directed:
                adj_list[j].append(i)
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

    def extract_backbone(self, edge_list_path, output_path, mode='slater',
                         nrows=None, logger=None):
        # mode 'slater' (default): Slater (2009) only. mode 'strict': a PRED-BiPN
        # heuristic, the significance_level given to __init__ (the interface's
        # alpha) plus a 20th-percentile stop; see _build_backbone for the order.
        if mode not in ('slater', 'strict'):
            raise ValueError(f"mode must be 'slater' or 'strict', not {mode!r}")
        if mode == 'strict':
            self.early_stop_percentile = 20
            self.min_edges = None
        print(f"Parameters: mode={mode}, directed={self.directed}"
              + (f", alpha={self.significance_level}" if mode == 'strict' else ''))

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

        if not self.directed:
            # undirected: A-B and B-A are the same edge, so normalise the
            # symmetric matrix and keep each pair once (upper triangle)
            sparse_matrix = (sparse_matrix + sparse_matrix.T).tocsr()

        print("Normalizing matrix...")
        normalized = self._make_doubly_stochastic(sparse_matrix)
        if not self.directed:
            normalized = triu(normalized.maximum(normalized.T), k=1).tocsr()

        target = None
        if mode == 'slater':
            inp = nx.DiGraph() if self.directed else nx.Graph()
            inp.add_nodes_from(nodes)
            inp.add_edges_from(zip(df['source'], df['target']))
            target = nx.number_weakly_connected_components(inp) if self.directed \
                else nx.number_connected_components(inp)
            print(f"Input network: {target} connected components")

        print("Extracting backbone...")
        backbone = self._build_backbone(normalized, nodes, target)

        print(f"\nResults:")
        print(f"  Nodes: {backbone.number_of_nodes()}")
        print(f"  Edges: {backbone.number_of_edges()}")
        print(f"  Connected: {nx.is_strongly_connected(backbone) if self.directed else nx.is_connected(backbone)}")

        rows = []
        for u, v, data in backbone.edges(data=True):
            rows.append({'source': u, 'target': v, 'weight': data.get('weight', 1.0)})

        pd.DataFrame(rows, columns=['source', 'target', 'weight']).to_csv(
            output_path, index=False, sep='\t')
        print(f"  Saved to {output_path}")

        return backbone

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Doubly stochastic backbone (Slater 2009)')
    parser.add_argument('input', help='Weighted edge list with a header row')
    parser.add_argument('output', help='Output TSV')
    parser.add_argument('--mode', choices=['slater', 'strict'], default='slater', help='Stopping rule (default slater)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level, strict rule only (default 0.05)')
    parser.add_argument('--directed', action='store_true', help='Treat the network as directed')
    args = parser.parse_args()
    TrueMultiscaleBackbone(significance_level=args.alpha, directed=args.directed).extract_backbone(
        args.input, args.output, mode=args.mode)
