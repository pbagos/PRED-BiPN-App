"""
Girvan-Newman Community Detection Algorithm for Gene Networks.
Memory-optimized implementation for large-scale network analysis.
"""

import networkx as nx
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count
import math
from typing import List, Set, Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import uuid
import traceback
import gc
import pandas as pd
from tqdm import tqdm


def brandes_algorithm_worker(sources, adjacency_dict, index_to_node):
    local_betweenness = defaultdict(float)
    for source in sources:
        source_node = index_to_node[source]
        S, P, sigma = [], defaultdict(list), defaultdict(float)
        sigma[source_node] = 1.0
        d = defaultdict(lambda: -1)
        d[source_node] = 0
        Q = deque([source_node])
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in adjacency_dict[v]:
                if d[w] < 0:
                    Q.append(w); d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]; P[w].append(v)
        delta = defaultdict(float)
        while S:
            w = S.pop()
            for v in P[w]:
                edge = tuple(sorted((v, w)))
                c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                local_betweenness[edge] += c
                delta[v] += c
    return local_betweenness


class GirvanNewmanOptimized:

    def __init__(self, graph: nx.Graph, num_processes: Optional[int] = None, logger=None, batch_size: int = 100):
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.num_processes = num_processes if num_processes else max(1, cpu_count() - 1)
        self.batch_size = batch_size
        self.communities_history: List[List[Set]] = []
        self.modularity_history: List[float] = []
        self.edge_removal_order: List[Tuple] = []
        self.runtime = 0.0
        self.node_list = list(self.graph.nodes())
        self.node_to_index = {node: i for i, node in enumerate(self.node_list)}
        self.index_to_node = {i: node for i, node in enumerate(self.node_list)}
        self.logger = logger
        self.adjacency_dict = self._create_adjacency_dict()
        self.current_edges = set()
        for u, v in self.graph.edges():
            self.current_edges.add(tuple(sorted((u, v))))

    def _log(self, message, level='info'):
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            print(message)

    def _create_adjacency_dict(self):
        adj_dict = defaultdict(set)
        for u, v in self.graph.edges():
            adj_dict[u].add(v)
            adj_dict[v].add(u)
        return adj_dict

    def _normalize_edge(self, edge: Tuple) -> Tuple:
        return tuple(sorted(edge))

    def _calculate_edge_betweenness_parallel(self) -> Dict[Tuple, float]:
        n = len(self.node_list)
        total_betweenness = defaultdict(float)
        all_nodes = list(range(n))

        chunk_size = max(1, n // (self.num_processes * 4))
        node_chunks = [all_nodes[i:i + chunk_size] for i in range(0, n, chunk_size)]

        adj_snapshot = defaultdict(set, {k: set(v) for k, v in self.adjacency_dict.items()})
        args = [(chunk, adj_snapshot, self.index_to_node) for chunk in node_chunks]

        try:
            with Pool(self.num_processes) as pool:
                for result in pool.starmap(brandes_algorithm_worker, args):
                    for edge, val in result.items():
                        total_betweenness[edge] += val
        except Exception:
            for arg in args:
                for edge, val in brandes_algorithm_worker(*arg).items():
                    total_betweenness[edge] += val

        for edge in total_betweenness:
            total_betweenness[edge] = round(total_betweenness[edge] / 2, 6)
        return total_betweenness

    def _calculate_modularity_fast(self, communities: List[Set]) -> float:
        m = self.original_graph.number_of_edges()
        if m == 0:
            return 0.0
        degrees = dict(self.original_graph.degree())
        q = 0.0
        for community in communities:
            cl = list(community)
            sum_in = sum(1 for idx, i in enumerate(cl) for j in cl[idx + 1:]
                         if self.original_graph.has_edge(i, j))
            sum_total = sum(degrees[node] for node in community)
            q += (sum_in / m) - (sum_total / (2 * m)) ** 2
        return round(q, 6)

    def _remove_highest_betweenness_edge(self) -> bool:
        edge_betweenness = self._calculate_edge_betweenness_parallel()
        if not edge_betweenness:
            return False
        max_edge = max(edge_betweenness.items(), key=lambda x: x[1])[0]
        max_edge_n = self._normalize_edge(max_edge)
        u, v = max_edge_n
        if not self.graph.has_edge(u, v):
            self._log(f"Edge {max_edge_n} not in graph", 'warning')
            return False
        self.graph.remove_edge(u, v)
        self.adjacency_dict[u].discard(v)
        self.adjacency_dict[v].discard(u)
        self.current_edges.discard(max_edge_n)
        self.edge_removal_order.append((max_edge_n, edge_betweenness[max_edge]))
        gc.collect()
        return True

    def detect_communities(self, early_stopping: bool = True,
                           max_iterations: Optional[int] = None,
                           min_communities: int = 2,
                           progress_callback=None) -> List[Set]:
        start_time = time.time()
        self._log(f"Girvan-Newman: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        if progress_callback:
            progress_callback(10)
        max_modularity = -math.inf
        best_communities: List[Set] = []
        best_graph = self.graph.copy()
        iteration = 0
        no_improvement_count = 0
        total_iterations = max_iterations if max_iterations else self.graph.number_of_edges()
        current_communities: List[Set] = []

        while iteration < total_iterations and self.graph.number_of_edges() > 0:
            iteration += 1
            try:
                current_communities = list(nx.connected_components(self.graph))
            except Exception as e:
                self._log(f"Component error: {e}"); break
            self.communities_history.append(current_communities)

            if len(current_communities) >= min_communities:
                try:
                    q = self._calculate_modularity_fast(current_communities)
                except Exception:
                    q = -math.inf
                self.modularity_history.append(q)
                self._log(f"Step {iteration}: {len(current_communities)} communities, Q={q}")
                if q > max_modularity:
                    max_modularity = q
                    best_communities = current_communities
                    best_graph = self.graph.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                if early_stopping and no_improvement_count >= 3:
                    self._log("Early stopping"); break
            else:
                self.modularity_history.append(-math.inf)

            if progress_callback:
                progress_callback(min(90, 10 + int(80 * iteration / total_iterations)))
            if not self._remove_highest_betweenness_edge():
                break
            gc.collect()

        self.runtime = time.time() - start_time
        self._log(f"Done in {self.runtime:.2f}s | {len(best_communities)} communities | Q={max_modularity}")
        if progress_callback:
            progress_callback(100)
            
        # Retain the network state with the highest modularity for edgelist output
        if best_communities:
            self.graph = best_graph
            
        return best_communities if best_communities else current_communities

    def visualize_network(self, communities, output_path, show_labels=True, show_edges=True):
        if not communities:
            return
        try:
            pos = (nx.spring_layout(self.original_graph, k=1, iterations=20)
                   if self.original_graph.number_of_nodes() > 500
                   else nx.spring_layout(self.original_graph, k=0.5, iterations=50))
            plt.figure(figsize=(16, 12))
            colors = plt.colormaps['tab20'](np.linspace(0, 1, min(len(communities), 20)))
            if show_edges:
                nx.draw_networkx_edges(self.original_graph, pos, alpha=0.3, edge_color='gray', width=0.5)
            for i, comm in enumerate(communities[:20]):
                nx.draw_networkx_nodes(self.original_graph, pos, nodelist=list(comm),
                                       node_color=[colors[i % len(colors)]] * len(comm),
                                       node_size=100, alpha=0.8, edgecolors='black', linewidths=0.3)
            if show_labels and self.original_graph.number_of_nodes() < 100:
                nx.draw_networkx_labels(self.original_graph, pos,
                                        {n: n for n in self.original_graph.nodes()},
                                        font_size=6, font_weight='bold')
            plt.title(f"Network Communities\n{len(communities)} communities", fontsize=14, fontweight='bold')
            plt.axis('off'); plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close()
        except Exception as e:
            self._log(f"Visualization error: {e}")

    def visualize_modularity(self, output_path):
        try:
            valid_q = [m for m in self.modularity_history if m > -math.inf]
            valid_i = [i for i, m in enumerate(self.modularity_history) if m > -math.inf]
            plt.figure(figsize=(10, 5))
            if valid_q:
                plt.plot(valid_i, valid_q, 'b-o', markersize=4)
                max_idx = valid_i[np.argmax(valid_q)]
                plt.plot(max_idx, max(valid_q), 'ro', markersize=8, label=f'Max: {max(valid_q):.4f}')
                plt.legend()
            plt.title("Modularity Evolution"); plt.xlabel("Step"); plt.ylabel("Modularity")
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close()
        except Exception as e:
            self._log(f"Modularity plot error: {e}")

    def save_communities_to_file(self, communities: List[Set], filename: str):
        """
        Saves community membership as CSV.
        Columns: source (node), size (community_size), community (community_id)
        Comment lines at the top summarise the run.
        """
        try:
            valid_q  = [m for m in self.modularity_history if m > -math.inf]
            max_mod  = max(valid_q) if valid_q else 0.0
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Girvan-Newman Community Detection Results\n")
                f.write(f"# Total communities: {len(communities)}\n")
                f.write(f"# Execution time: {self.runtime:.2f} seconds\n")
                f.write(f"# Maximum modularity: {max_mod:.6f}\n#\n")
                f.write("source\tsize\tcommunity\n")
                for comm_id, comm in enumerate(communities, start=1):
                    comm_size = len(comm)
                    for node in sorted(comm):
                        f.write(f"{node}\t{comm_size}\t{comm_id}\n")
        except Exception as e:
            self._log(f"Save error: {e}")
            
    def save_edgelist_to_file(self, filename: str):
        """
        Saves the filtered network (remaining edges) as TSV.
        Columns: Source, Target, Weight
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Source\tTarget\tWeight\n")
                for u, v in self.graph.edges():
                    w = self.original_graph[u][v].get('weight', 1.0) if self.original_graph.has_edge(u, v) else 1.0
                    f.write(f"{u}\t{v}\t{w}\n")
        except Exception as e:
            self._log(f"Save edgelist error: {e}")

    def get_communities_summary(self, communities: List[Set]) -> List[Dict]:
        return [{'id': i + 1, 'size': len(c), 'nodes': sorted(list(c))[:5]}
                for i, c in enumerate(communities[:20])]


# ------------------------------------------------------------------
# Graph loader — positional access, auto-detect separator, tqdm
# ------------------------------------------------------------------
_HEADER_KEYWORDS = {'source', 'target', 'node', 'gene', 'from', 'to', 'src', 'dst',
                    'protein', 'id', 'name', 'weight', 'score', 'edge'}


def _detect_separator_from_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readline()
    if '\t' in line:  return '\t'
    if ',' in line:   return ','
    if ';' in line:   return ';'
    return r'\s+'


def _has_header(filepath: str, sep: str) -> bool:
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
    if not line:
        return False
    parts = line.split(sep if sep != r'\s+' else None)
    if len(parts) < 2:
        return False
    return all(p.strip().lower() in _HEADER_KEYWORDS for p in parts[:2])


def _count_file_lines(filepath: str) -> int:
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def load_graph_from_file(filepath: str, max_lines: int = 500,
                         chunksize: int = 10000, logger=None) -> nx.Graph:
    """
    Loads a graph using positional column access (iloc) — no predefined column names.
        iloc[0] → source node
        iloc[1] → target node
        iloc[2] → edge weight  (optional; defaults to 1.0)
    Auto-detects separator (tab, comma, semicolon, whitespace).
    """
    G   = nx.Graph()
    sep = _detect_separator_from_file(filepath)
    skip = 1 if _has_header(filepath, sep) else 0
    total_lines  = _count_file_lines(filepath) - skip
    total_chunks = max(1, total_lines // chunksize)

    if logger:
        logger.info(f"Loading graph: sep={repr(sep)}, max_lines={max_lines}, skip_header={bool(skip)}")

    chunk_reader = pd.read_csv(
        filepath, sep=sep, header=None,
        skiprows=skip, chunksize=chunksize, dtype=str,
        engine='python', skipinitialspace=True
    )

    loaded = 0

    for chunk in tqdm(chunk_reader, total=total_chunks, desc="Processing network"):
        # Clean
        chunk = chunk.dropna()
        chunk = chunk[~chunk.iloc[:, 0].astype(str).str.contains('nan|None', case=False, na=False)]
        chunk = chunk[~chunk.iloc[:, 1].astype(str).str.contains('nan|None', case=False, na=False)]

        for _, row in chunk.iterrows():
            if max_lines and loaded >= max_lines:
                break
            u = str(row.iloc[0]).strip()
            v = str(row.iloc[1]).strip()
            if not u or not v:
                continue
            try:
                w = float(row.iloc[2]) if len(row) > 2 else 1.0
            except (ValueError, TypeError):
                w = 1.0
            G.add_edge(u, v, weight=w)
            loaded += 1

        if max_lines and loaded >= max_lines:
            break

    if logger:
        logger.info(f"Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ------------------------------------------------------------------
# Flask entry point
# ------------------------------------------------------------------
def run_girvan_newman_analysis(input_file, output_dir, max_lines=500, num_processes=None,
                               early_stopping=True, min_communities=2, max_iterations=10,
                               logger=None, progress_callback=None):
    start_time = time.time()
    task_id    = str(uuid.uuid4())[:8]

    if logger:
        logger.info(f"Starting Girvan-Newman (max_lines={max_lines})")
    if progress_callback:
        progress_callback(5)

    try:
        G = load_graph_from_file(input_file, max_lines=max_lines, logger=logger)
        if G.number_of_nodes() == 0:
            return {'success': False, 'error': 'No nodes loaded from file'}, None

        if progress_callback:
            progress_callback(20)

        gn = GirvanNewmanOptimized(G, num_processes=min(num_processes or cpu_count(), 4),
                                   logger=logger, batch_size=50)

        communities = gn.detect_communities(
            early_stopping=early_stopping,
            max_iterations=max_iterations,
            min_communities=min_communities,
            progress_callback=lambda p: progress_callback(20 + int(p * 0.6)) if progress_callback else None
        )

        if progress_callback:
            progress_callback(80)

        results = {}

        communities_file = os.path.join(output_dir, f"communities_{task_id}.tsv")
        gn.save_communities_to_file(communities, communities_file)
        results['communities_file'] = os.path.basename(communities_file)
        
        edgelist_file = os.path.join(output_dir, f"result_{task_id}.tsv")
        gn.save_edgelist_to_file(edgelist_file)
        results['result_file'] = os.path.basename(edgelist_file)

        if G.number_of_nodes() < 1000:
            network_img = os.path.join(output_dir, f"network_{task_id}.png")
            gn.visualize_network(communities, network_img, show_labels=False, show_edges=True)
            results['network_image'] = os.path.basename(network_img)

            modularity_img = os.path.join(output_dir, f"modularity_{task_id}.png")
            gn.visualize_modularity(modularity_img)
            results['modularity_image'] = os.path.basename(modularity_img)

        elapsed   = time.time() - start_time
        valid_q   = [m for m in gn.modularity_history if m > -math.inf]
        max_mod   = max(valid_q) if valid_q else 0

        stats = {
            'success':             True,
            'nodes':               G.number_of_nodes(),
            'edges':               G.number_of_edges(),
            'num_communities':     len(communities),
            'max_modularity':      round(max_mod, 6),
            'iterations':          len(gn.communities_history),
            'runtime':             round(elapsed, 2),
            'method':              'girvan_newman',
            'communities_summary': gn.get_communities_summary(communities)
        }

        if progress_callback:
            progress_callback(100)
        return stats, results

    except Exception as e:
        if logger:
            logger.error(f"Girvan-Newman failed: {e}")
            logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}, None