import networkx as nx
import time
import pandas as pd
from community import community_louvain
from collections import defaultdict
import numpy as np
from scipy import sparse
from tqdm import tqdm


class SparseModularityBackbone:
    def __init__(self, G):
        self.original_G = G.copy()
        self.G = G.copy()
        self.nodes = list(self.G.nodes())
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.n_nodes = len(self.nodes)

        rows, cols, data = [], [], []
        for u, v, d in self.G.edges(data=True):
            weight = d.get('weight', 1.0)
            rows.append(self.node_index[u])
            cols.append(self.node_index[v])
            data.append(weight)
            if u != v:
                rows.append(self.node_index[v])
                cols.append(self.node_index[u])
                data.append(weight)

        self.adj_matrix = sparse.csr_matrix((data, (rows, cols)),
                                            shape=(self.n_nodes, self.n_nodes))
        self.partition: dict | None = None
        self.communities: dict | None = None
        self.total_weight: float | None = None
        self.community_metrics: dict | None = None

    def compute_initial_partition(self):
        print("Computing initial partition using Louvain method...")
        start = time.time()
        self.partition = community_louvain.best_partition(self.G, weight='weight')
        self._update_community_structures()
        self.total_weight = float(self.adj_matrix.sum()) / 2
        self._precompute_community_metrics()
        print(f"Initial partition completed in {time.time()-start:.2f} seconds")

    def _update_community_structures(self):
        self.communities = defaultdict(set)
        assert self.partition is not None
        for node, comm in self.partition.items():
            self.communities[comm].add(node)

    def _precompute_community_metrics(self):
        self.community_metrics = {}
        assert self.communities is not None
        for comm, node_ids in self.communities.items():
            node_indices = [self.node_index[n] for n in node_ids]
            submatrix = self.adj_matrix[node_indices, :][:, node_indices]
            sum_in = submatrix.sum() / 2
            sum_tot = self.adj_matrix[node_indices, :].sum()
            self.community_metrics[comm] = {
                'sum_in': sum_in,
                'sum_tot': sum_tot,
                'node_ids': node_ids
            }

    def compute_modularity(self):
        if self.total_weight == 0:
            return 0.0
        Q = 0.0
        assert self.community_metrics is not None
        assert self.total_weight is not None
        for metrics in self.community_metrics.values():
            sum_in = metrics['sum_in']
            sum_tot = metrics['sum_tot']
            Q += (sum_in / self.total_weight) - (sum_tot / (2 * self.total_weight)) ** 2
        return Q

    def compute_vitality(self, node_id):
        assert self.partition is not None
        assert self.communities is not None
        assert self.community_metrics is not None
        if node_id not in self.partition:
            return 0.0
        node_idx = self.node_index[node_id]
        comm = self.partition[node_id]
        original_modularity = self.compute_modularity()

        node_row = self.adj_matrix[node_idx, :]
        node_sum_tot = node_row.sum()
        comm_node_indices = [self.node_index[n] for n in self.communities[comm]]
        node_sum_in = node_row[:, comm_node_indices].sum()

        temp_metrics = {c: dict(metrics) for c, metrics in self.community_metrics.items()}
        temp_metrics[comm]['sum_in'] -= node_sum_in
        temp_metrics[comm]['sum_tot'] -= node_sum_tot

        m = self.total_weight - node_sum_tot
        if m <= 0:
            return 0.0

        new_modularity = 0.0
        for metrics in temp_metrics.values():
            new_modularity += (metrics['sum_in'] / m) - (metrics['sum_tot'] / (2 * m)) ** 2

        return original_modularity - new_modularity

    def remove_node_and_update(self, node_id):
        assert self.partition is not None
        assert self.communities is not None
        assert self.community_metrics is not None
        if node_id not in self.partition:
            return
        node_idx = self.node_index[node_id]
        comm = self.partition[node_id]

        node_row = self.adj_matrix[node_idx, :]
        node_sum_tot = node_row.sum()
        comm_node_indices = [self.node_index[n] for n in self.communities[comm]]
        node_sum_in = node_row[:, comm_node_indices].sum()

        self.community_metrics[comm]['sum_in'] -= node_sum_in
        self.community_metrics[comm]['sum_tot'] -= node_sum_tot
        self.total_weight -= node_sum_tot

        self.adj_matrix[node_idx, :] = 0
        self.adj_matrix[:, node_idx] = 0

        self.communities[comm].remove(node_id)
        del self.partition[node_id]

        if not self.communities[comm]:
            del self.communities[comm]
            del self.community_metrics[comm]

    def extract_backbone(self, percentage=0.3):
        self.compute_initial_partition()
        target_size = max(1, int(self.n_nodes * percentage))
        print(f"Extracting backbone with target size {target_size} nodes...")
        pbar = tqdm(total=self.n_nodes - target_size)

        while self.adj_matrix.nnz > 0 and len(self.partition) > target_size:  # type: ignore[arg-type]
            vitals = {n: self.compute_vitality(n) for n in list(self.partition.keys())}  # type: ignore[union-attr]
            if not vitals:
                break
            min_node = min(vitals.keys(), key=lambda x: abs(vitals[x]))
            self.remove_node_and_update(min_node)
            pbar.update(1)

        pbar.close()
        remaining_nodes = list(self.partition.keys())  # type: ignore[union-attr]
        backbone = self.G.subgraph(remaining_nodes).copy()

        remaining_nodes = list(self.partition.keys())  # type: ignore[union-attr]
        backbone = self.G.subgraph(remaining_nodes).copy()
        return backbone
            


def load_graph_sparse(file_path, max_lines=None):
    """
    Load graph from file with automatic separator and header detection.
    Always uses positional access: iloc[0]=source, iloc[1]=target, iloc[2]=weight.
    Header row (if present) is used as column names, never as data.
    """
    # Ανίχνευση διαχωριστικού
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    separators = {'\t': first_line.count('\t'), ',': first_line.count(','),
                  ';': first_line.count(';'), ' ': first_line.count(' ')}
    sep = max(separators, key=lambda k: separators[k])
    if separators[sep] == 0:
        sep = '\t'

    # Διάβασμα με header=0 — η πρώτη γραμμή γίνεται ονόματα, δεν μπαίνει ως δεδομένο
    df = pd.read_csv(file_path, sep=sep, header=0,
                     nrows=max_lines if max_lines else None)
    print(f"  Columns detected: {list(df.columns)}")
    print(f"  Loaded {len(df)} rows")

    # Μετατροπή βάρους positional
    if df.shape[1] > 2:
        df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
        #df = df.dropna(subset=[df.columns[2]])

    G = nx.Graph()
    for _, row in df.iterrows():
        u = str(row.iloc[0]).strip()
        v = str(row.iloc[1]).strip()

        #if not u or not v or u == 'nan' or v == 'nan' or u == v:
        #   continue

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

    def __init__(self, percentage=0.3, test_mode=False, max_lines=None):
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

            backbone = SparseModularityBackbone(G).extract_backbone(self.percentage)
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