"""Doubly stochastic backbone, Slater (2009).

1. The weight matrix W (symmetric for an undirected network) is scaled to a
   doubly stochastic matrix with the Sinkhorn-Knopp algorithm. As in the
   reference code (Coscia's backboning.py, netbone), epsilon = 1e-4 is added to
   every cell first: a sparse W usually has no doubly stochastic scaling (it
   lacks total support, Sinkhorn & Knopp 1967), a positive matrix always has
   one. Iteration stops when std(row sums) < 1e-12, as in the reference code.
2. Edges are added in decreasing order of their scaled value until the
   backbone is connected (strongly connected for a directed network). For an
   input that is not connected this condition is applied to each of its
   components: stop when the backbone has the same connected (strongly
   connected) components as the input.

There are no further parameters; epsilon and the tolerance are fixed to the
reference values.
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix, csr_matrix, triu

EPSILON = 1e-4
TOL = 1e-12
MAX_ITER = 1000


class TrueMultiscaleBackbone:
    def __init__(self, directed=False):
        self.directed = directed

    def _detect_separator(self, file_path):
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        separators = {'\t': first_line.count('\t'), ',': first_line.count(','),
                      ';': first_line.count(';'), ' ': first_line.count(' ')}
        sep = max(separators, key=lambda k: separators[k])
        return sep if separators[sep] > 0 else '\t'

    @staticmethod
    def _make_doubly_stochastic(matrix, return_iterations=False):
        """Sinkhorn-Knopp on W + EPSILON (every cell). The dense matrix is never
        built: the scaled matrix is diag(r) (W + eps) diag(c), so
        row sums = r * (W c + eps * sum(c)) and column sums likewise.
        Returns the scaled values on the nonzero cells of W."""
        W = matrix.tocsr()
        r = np.ones(W.shape[0]); c = np.ones(W.shape[1])
        for it in range(1, MAX_ITER + 1):
            r = 1 / (W @ c + EPSILON * c.sum())
            c = 1 / (W.T @ r + EPSILON * r.sum())
            if np.std(r * (W @ c + EPSILON * c.sum())) < TOL:
                break
        else:
            it = None   # did not converge
        coo = W.tocoo()
        scaled = csr_matrix((r[coo.row] * (coo.data + EPSILON) * c[coo.col], (coo.row, coo.col)), shape=W.shape)
        return (scaled, it) if return_iterations else scaled

    def _n_components(self, n, edges):
        G = nx.DiGraph() if self.directed else nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from((i, j) for i, j, _ in edges)
        return nx.number_strongly_connected_components(G) if self.directed else nx.number_connected_components(G)

    def _slater_prefix(self, edges, n, target):
        """Number of edges, taken in the given (decreasing) order, after which the
        backbone first has `target` (strongly) connected components. Adding an
        edge never splits a component, so the count only falls as edges are added."""
        if not self.directed:   # union-find, one pass
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            comps = n
            for k, (i, j, _) in enumerate(edges, 1):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
                    comps -= 1
                    if comps == target:
                        return k
            return len(edges)
        lo, hi = 0, len(edges)   # strongly connected: binary search on the prefix length
        while lo < hi:
            mid = (lo + hi) // 2
            if self._n_components(n, edges[:mid]) == target:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def extract_backbone(self, edge_list_path, output_path, logger=None):
        say = logger.info if logger else print           # the app passes its logger, so these
        warn = logger.warning if logger else print       # lines reach the run's log
        say(f"Parameters: directed={self.directed}, epsilon={EPSILON}, tolerance={TOL}")
        print("Loading network...")
        try:
            sep = self._detect_separator(edge_list_path)
            df = pd.read_csv(edge_list_path, sep=sep, header=0)
            print(f"  Columns detected: {list(df.columns)}")
            if df.empty or df.shape[1] < 3:
                print("  Warning: the file is empty or has no weight column.")
                return nx.DiGraph()
            df = df.rename(columns={df.columns[0]: 'source', df.columns[1]: 'target', df.columns[2]: 'weight'})
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df = df.dropna(subset=['weight'])
            df = df[(df['source'] != df['target']) & (df['weight'] > 0)]
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
        idx = {v: i for i, v in enumerate(nodes)}
        W = coo_matrix((df['weight'], ([idx[s] for s in df['source']], [idx[t] for t in df['target']])),
                       shape=(len(nodes), len(nodes))).tocsr()
        print(f"Network: {len(nodes)} nodes, {W.nnz} edges")
        if not self.directed:
            # undirected: A-B and B-A are the same edge, so scale the symmetric
            # matrix and keep each pair once (upper triangle)
            W = (W + W.T).tocsr()

        scaled, iterations = self._make_doubly_stochastic(W, return_iterations=True)
        if iterations is None:
            warn(f"Sinkhorn did not converge in {MAX_ITER} iterations")
        else:
            say(f"Sinkhorn converged in {iterations} iterations (epsilon={EPSILON})")
        if not self.directed:
            scaled = triu(scaled.maximum(scaled.T), k=1).tocsr()
        coo = scaled.tocoo()
        # decreasing value; equal values by node order, so the result does not depend on row order
        edges = sorted(zip(coo.row, coo.col, coo.data), key=lambda e: (-e[2], e[0], e[1]))

        all_edges = [(i, j, 0) for i, j, _ in edges]
        target = self._n_components(len(nodes), all_edges)
        kind = 'strongly connected' if self.directed else 'connected'
        say(f"Input network: {target} {kind} components")
        if self.directed and target == len(nodes):
            warn("The directed input has no cycles, so every node is its own strongly connected "
                  "component and Slater's rule keeps no edge. An edge list that lists each pair once "
                  "(such as a projection) should be run as Undirected.")
        k = self._slater_prefix(edges, len(nodes), target)
        say(f"Stopping after {k} of {len(edges)} edges: the backbone has the same {target} {kind} components as the input")

        # weight = the input weight of the pair (for undirected input, w(A,B) + w(B,A)
        # if the file lists both directions, as in the scaled matrix)
        Wd = W.todok()
        rows = [(nodes[i], nodes[j], Wd[i, j], val) for i, j, val in edges[:k]]
        out = pd.DataFrame(rows, columns=['source', 'target', 'weight', 'ds_value'])
        out.to_csv(output_path, index=False, sep='\t')
        say(f"Results: {out[['source', 'target']].stack().nunique()} nodes, {len(out)} edges")

        backbone = nx.DiGraph() if self.directed else nx.Graph()
        backbone.add_nodes_from(nodes)
        backbone.add_weighted_edges_from(out[['source', 'target', 'weight']].itertuples(index=False))
        return backbone


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Doubly stochastic backbone (Slater 2009)')
    parser.add_argument('input', help='Weighted edge list with a header row')
    parser.add_argument('output', help='Output TSV: source, target, weight, ds_value')
    parser.add_argument('--directed', action='store_true', help='Treat the network as directed')
    args = parser.parse_args()
    TrueMultiscaleBackbone(directed=args.directed).extract_backbone(args.input, args.output)
