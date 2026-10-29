import networkx as nx
import time
import pandas as pd
from community import community_louvain
import numpy as np
from scipy import sparse
from tqdm import tqdm

class SparseModularityBackbone:
    """Modularity-vitality backbone (Magelinski et al.): repeatedly remove the
    node whose removal changes the modularity Q of the Louvain partition the
    least, until `percentage` of the nodes is left.

    Q = sum_c [ in_c / m - (tot_c / 2m)^2 ], m = total edge weight. Removing
    node i (community c, strength k_i, weight K[i,d] to community d) gives
        m' = m - k_i,  in_c' = in_c - K[i,c],
        tot_d' = tot_d - K[i,d]  (d != c),  tot_c' = tot_c - K[i,c] - k_i,
    so the vitality of every node is one vectorised pass over K per step.
    """

    def __init__(self, G, seed=42):
        self.G = G
        self.seed = seed
        # sorted so that ties (and the result) do not depend on the file's row order
        self.nodes = sorted(G.nodes(), key=str)
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.n_nodes = len(self.nodes)

        rows, cols, data = [], [], []
        for u, v, d in G.edges(data=True):
            if u == v:
                continue
            i, j, w = self.node_index[u], self.node_index[v], d.get('weight', 1.0)
            rows += [i, j]; cols += [j, i]; data += [w, w]
        self.adj = sparse.csr_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))

    @staticmethod
    def vitality(K, k, tot, comm, S_in, m):
        """Q(G) - Q(G without node i) for every node i."""
        own = K[np.arange(len(k)), comm]
        S_sq = (tot ** 2).sum()
        Q = S_in / m - S_sq / (4 * m * m)
        m2 = m - k
        a = tot[comm] - own
        S_sq2 = S_sq - 2 * (K @ tot) + (K * K).sum(axis=1) - 2 * a * k + k * k
        with np.errstate(divide='ignore', invalid='ignore'):
            Q2 = (S_in - own) / m2 - S_sq2 / (4 * m2 * m2)
        return np.where(m2 > 0, Q - Q2, 0.0)

    def extract_backbone(self, percentage=0.3):
        print("Computing initial partition using Louvain method...")
        partition = community_louvain.best_partition(self.G, weight='weight', random_state=self.seed)
        comm_ids = {c: k for k, c in enumerate(sorted(set(partition.values())))}
        comm = np.array([comm_ids[partition[n]] for n in self.nodes])
        n, C = self.n_nodes, len(comm_ids)

        # Dense node x community matrix, O(n*C) memory; switch K to
        # sparse if networks reach ~1e5 nodes with hundreds of communities
        member = sparse.csr_matrix((np.ones(n), (np.arange(n), comm)), shape=(n, C))
        K = np.asarray((self.adj @ member).todense())
        k = K.sum(axis=1)
        tot = np.bincount(comm, weights=k, minlength=C)
        own = K[np.arange(n), comm]
        S_in = own.sum() / 2
        m = k.sum() / 2
        alive = np.ones(n, dtype=bool)

        target_size = max(1, int(n * percentage))
        print(f"Extracting backbone with target size {target_size} nodes...")
        pbar = tqdm(total=n - target_size)
        while alive.sum() > target_size and m > 0:
            own = K[np.arange(n), comm]
            vit = self.vitality(K, k, tot, comm, S_in, m)
            vit[~alive] = np.inf
            r = int(np.argmin(np.abs(vit)))

            # remove node r and its edges
            c = comm[r]
            tot -= K[r]
            tot[c] -= k[r]
            S_in -= own[r]
            m -= k[r]
            row = self.adj.getrow(r)
            K[row.indices, c] -= row.data
            k[row.indices] -= row.data
            K[r] = 0
            k[r] = 0
            alive[r] = False
            pbar.update(1)
        pbar.close()

        return self.G.subgraph([self.nodes[i] for i in np.flatnonzero(alive)]).copy()

def load_graph_sparse(file_path, max_lines=None):
    """
    Load graph from file with automatic separator and header detection.
    Always uses positional access: iloc[0]=source, iloc[1]=target, iloc[2]=weight.
    Header row (if present) is used as column names, never as data.
    """
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    separators = {'\t': first_line.count('\t'), ',': first_line.count(','),
                  ';': first_line.count(';'), ' ': first_line.count(' ')}
    sep = max(separators, key=lambda k: separators[k])
    if separators[sep] == 0:
        sep = '\t'

    df = pd.read_csv(file_path, sep=sep, header=0,
                     nrows=max_lines if max_lines else None)
    print(f"  Columns detected: {list(df.columns)}")
    print(f"  Loaded {len(df)} rows")

    if df.shape[1] > 2:
        df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')

    G = nx.Graph()
    for _, row in df.iterrows():
        u = str(row.iloc[0]).strip()
        v = str(row.iloc[1]).strip()

        w = 1.0
        if df.shape[1] > 2 and pd.notna(row.iloc[2]):
            w = float(row.iloc[2])

        if w <= 0:
            continue

        G.add_edge(u, v, weight=w)

    return G

def save_backbone_sparse(backbone, output_file):
    """Save backbone as CSV with columns source, target, weight."""
    rows = [(u, v, d.get('weight', 1.0)) for u, v, d in backbone.edges(data=True)]
    pd.DataFrame(rows, columns=['source', 'target', 'weight']).to_csv(
        output_file, index=False, sep='\t')
    print(f"Backbone saved to {output_file}")

class ModularityFilter:
    """Wrapper class for Flask server."""

    def __init__(self, percentage=0.3, test_mode=False, max_lines=None, seed=42):
        self.seed = seed
        self.percentage = percentage
        self.test_mode = test_mode
        self.max_lines = max_lines

    def run(self, input_file, output_file):
        try:
            print(f"\n=== Modularity Filter ===")
            print(f"Input: {input_file}, Percentage: {self.percentage}")

            max_lines = self.max_lines if not self.test_mode else 10000
            G = load_graph_sparse(input_file, max_lines)
            print(f"Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

            backbone = SparseModularityBackbone(G, seed=self.seed).extract_backbone(self.percentage)
            print(f"Backbone: {len(backbone.nodes())} nodes, {len(backbone.edges())} edges")

            save_backbone_sparse(backbone, output_file)

            return {
                'success': True,
                'original_nodes': len(G.nodes()),
                'original_edges': len(G.edges()),
                'backbone_nodes': len(backbone.nodes()),
                'backbone_edges': len(backbone.edges())
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'success': False, 'message': str(e)}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Modularity Vitality Backbone Extraction')
    parser.add_argument('input', help='Input edge list file')
    parser.add_argument('output', help='Output backbone file')
    parser.add_argument('--percentage', type=float, default=0.3)
    parser.add_argument('--max_lines', type=int)
    args = parser.parse_args()

    G = load_graph_sparse(args.input, args.max_lines)
    print(f"Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

    backbone = SparseModularityBackbone(G).extract_backbone(args.percentage)
    print(f"Backbone: {len(backbone.nodes())} nodes, {len(backbone.edges())} edges")

    save_backbone_sparse(backbone, args.output)