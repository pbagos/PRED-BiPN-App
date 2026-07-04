"""
High Salience Skeleton (HSS) extraction method for complex networks with automatic column detection.
Identifies the most "salient" (important) links in a network based on their participation in shortest paths.
"""

import pandas as pd
from collections import defaultdict, Counter
import heapq
import math
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
from tqdm import tqdm
import re

class FastLinkSalience:
    def __init__(self, weight_threshold=1e-6):
        """
        Initializes the FastLinkSalience class.
        :param weight_threshold: Weight threshold. Edges with weight smaller than this will be ignored.
        """
        self.weight_threshold = weight_threshold

    def build_network(self, edges):
        """
        Builds a network as an adjacency dictionary from a list of edges.
        Each edge is represented as (source, destination, weight).
        """
        network = defaultdict(dict)
        for u, v, w in edges:
            if w > self.weight_threshold:
                inv_weight = 1/w
                network[u][v] = inv_weight
                network[v][u] = inv_weight
        return network

    def dijkstra(self, network, start):
        """
        Dijkstra's algorithm to find the shortest paths from a starting node.
        """
        distances = {node: math.inf for node in network}
        distances[start] = 0
        predecessors = {node: [] for node in network}
        heap = [(0, start)]
        visited = set()

        while heap:
            current_dist, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            for neighbor, weight in network[current].items():
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = [current]
                    heapq.heappush(heap, (distance, neighbor))

        return predecessors

    def compute_for_node(self, network, node):
        """
        Calculates how many times each edge belongs to shortest paths starting from a node.
        """
        pred = self.dijkstra(network, node)
        edge_counts = Counter()

        for target in pred:
            if target == node:
                continue
            current = target
            while current in pred and pred[current]:
                predecessor = pred[current][0]
                edge = tuple(sorted((predecessor, current)))
                edge_counts[edge] += 1
                current = predecessor

        return edge_counts

    def compute_salience(self, edges, processes=None):
        """
        Parallel computation of edge salience.
        """
        network = self.build_network(edges)
        nodes = list(network.keys())
        N = len(nodes)

        if N == 0:
            return {}

        if processes is None:
            processes = cpu_count()

        network_to_pass = network
        compute_for_node_partial = partial(self.compute_for_node, network_to_pass)
        args_list = nodes

        total_counts = Counter()
        with Pool(processes=processes) as pool:
            salience_results_iter = pool.imap_unordered(compute_for_node_partial, args_list)

            for counter in tqdm(salience_results_iter, total=len(nodes), desc="Calculate Salience"):
                total_counts.update(counter)

        return {edge: count/N for edge, count in total_counts.items()}

class HighSalienceSkeleton:
    """
    High Salience Skeleton extractor with automatic column detection.
    """

    def __init__(self, threshold=0.9, weight_threshold=1e-6, n_jobs=None):
        """
        Initialize the High Salience Skeleton extractor.

        Parameters
        ----------
        threshold : float, optional (default=0.9)
            Salience threshold for keeping edges (0-1)
        weight_threshold : float, optional (default=1e-6)
            Minimum weight threshold for edges
        n_jobs : int, optional (default=None)
            Number of parallel jobs (uses all cores if None)
        """
        self.threshold = threshold
        self.weight_threshold = weight_threshold
        self.n_jobs = n_jobs or cpu_count()

        self.fast_link = FastLinkSalience(weight_threshold=weight_threshold)

        self.source_col = None
        self.target_col = None
        self.weight_col = None
        self.detected_sep = None
        self.progress_callback = None

    def set_progress_callback(self, callback):
        """
        Set a progress callback function.

        Parameters
        ----------
        callback : function
            Function that receives progress percentage (0-100)
        """
        self.progress_callback = callback

    def _update_progress(self, percent):
        """Update progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(percent)

    def _detect_separator(self, file_path):
        """Auto-detect the separator in the input file."""
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()

            separators = {
                '\t': first_line.count('\t'),
                ',': first_line.count(','),
                ';': first_line.count(';'),
                ' ': first_line.count(' ')
            }

            separators = {k: v for k, v in separators.items() if v > 0}

            if not separators:
                return '\t'
            return max(separators, key=lambda k: separators[k])

    def _detect_columns_from_data(self, file_path, sep):
        """
        Detect columns by reading the actual data values from the first row.
        This is smarter - it looks at the values, not just column names.
        """
        df_sample = pd.read_csv(file_path, sep=sep, nrows=5, header=None)

        source_candidates = []
        target_candidates = []
        weight_candidates = []

        for col_idx in range(len(df_sample.columns)):
            col_data = df_sample.iloc[:, col_idx]

            is_string = col_data.dtype == 'object'
            is_numeric = pd.api.types.is_numeric_dtype(col_data)

            if is_string:
                unique_vals = col_data.dropna().unique()

                sample_str = str(unique_vals[0]) if len(unique_vals) > 0 else ""

                if len(sample_str) < 50 and not sample_str.replace('.', '').replace('-', '').isdigit():
                    source_candidates.append(col_idx)
                    target_candidates.append(col_idx)

            elif is_numeric:
                if len(col_data) > 0:
                    min_val = col_data.min()
                    max_val = col_data.max()
                    if min_val >= 0 and max_val <= 1000:
                        weight_candidates.append(col_idx)

        if len(source_candidates) >= 2:
            source_col = source_candidates[0]
            target_col = source_candidates[1]
        elif len(source_candidates) >= 1:
            source_col = source_candidates[0]
            if len(weight_candidates) > 0:
                target_col = weight_candidates[0]
            else:
                target_col = 1 if len(df_sample.columns) > 1 else 0
        else:
            source_col = 0
            target_col = 1 if len(df_sample.columns) > 1 else 0

        if weight_candidates:
            weight_col = weight_candidates[0]
        elif len(df_sample.columns) > 2:
            weight_col = 2
        else:
            weight_col = None

        return source_col, target_col, weight_col

    def _detect_columns_by_name(self, file_path, sep):
        """
        Detect columns by checking header names if they exist.
        """
        try:
            df_sample = pd.read_csv(file_path, sep=sep, nrows=0)
            columns = df_sample.columns.tolist()

            source_patterns = ['source', 'from', 'node1', 'gene1', 'gene', 'protein', 'id', 'label', 'col1', 'column1', 'left', 'row', 'start']
            target_patterns = ['target', 'to', 'node2', 'gene2', 'disease', 'disease_id', 'col2', 'column2', 'right', 'end']
            weight_patterns = ['weight', 'shared', 'value', 'score', 'count', 'strength', 'shared_diseases']

            source_col = None
            target_col = None
            weight_col = None

            for i, col in enumerate(columns):
                col_lower = str(col).lower().strip()
                if col_lower in source_patterns:
                    source_col = i
                if col_lower in target_patterns:
                    target_col = i
                if col_lower in weight_patterns:
                    weight_col = i

            if source_col is not None and target_col is not None:
                return source_col, target_col, weight_col

            return None, None, None

        except Exception:
            return None, None, None

    def load_data(self, file_path, test_mode=False, max_rows=1000000):
        """
        Load edge data with automatic column detection.

        Parameters
        ----------
        file_path : str
            Path to input file
        test_mode : bool, optional (default=False)
            If True, load only max_rows
        max_rows : int, optional (default=1000000)
            Maximum rows to load in test mode

        Returns
        -------
        list
            List of tuples (source, target, weight)
        """
        self.detected_sep = self._detect_separator(file_path)

        source_idx, target_idx, weight_idx = self._detect_columns_by_name(file_path, self.detected_sep)

        if source_idx is None or target_idx is None:
            source_idx, target_idx, weight_idx = self._detect_columns_from_data(file_path, self.detected_sep)

        self.source_col = source_idx
        self.target_col = target_idx
        self.weight_col = weight_idx

        edges = []
        chunksize = 100000

        has_header = False
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                parts = first_line.split(self.detected_sep)
                if any(p.isalpha() and len(p) > 0 for p in parts):
                    has_header = True
        except Exception:
            has_header = False

        col_names = ['col_' + str(i) for i in range(max(source_idx, target_idx, weight_idx or 0) + 1)]

        total_rows = 0
        for chunk in pd.read_csv(
            file_path,
            sep=self.detected_sep,
            chunksize=chunksize,
            nrows=max_rows if test_mode else None,
            header=0 if has_header else None,
            names=col_names if has_header else None,
            dtype={source_idx: str, target_idx: str}
        ):
            source_data = chunk.iloc[:, source_idx].astype(str).str.strip()
            target_data = chunk.iloc[:, target_idx].astype(str).str.strip()

            if weight_idx is not None and weight_idx < len(chunk.columns):
                weight_data = pd.to_numeric(chunk.iloc[:, weight_idx], errors='coerce')
            else:
                weight_data = pd.Series([1.0] * len(chunk))

            temp_df = pd.DataFrame({
                'source': source_data,
                'target': target_data,
                'weight': weight_data
            })

            temp_df = temp_df.dropna(subset=['source', 'target', 'weight'])

            temp_df = temp_df[temp_df['weight'] > 0]

            temp_df = temp_df[temp_df['source'] != temp_df['target']]

            edges.extend(temp_df[['source', 'target', 'weight']].itertuples(index=False, name=None))

            total_rows += len(temp_df)
            if test_mode and total_rows >= max_rows:
                break

        return edges

    def get_column_info(self):
        """
        Get detected column information.

        Returns
        -------
        dict
            Information about detected columns
        """
        return {
            'separator': self.detected_sep,
            'source_column_index': self.source_col,
            'target_column_index': self.target_col,
            'weight_column_index': self.weight_col
        }

    def run(self, input_file, output_file, test_mode=False, logger=None):
        """
        Run High Salience Skeleton extraction.

        Parameters
        ----------
        input_file : str
            Path to input file
        output_file : str
            Path to output file (will be CSV with source, target, weight, salience)
        test_mode : bool, optional (default=False)
            If True, run in test mode with limited rows
        logger : logging.Logger, optional
            Logger for progress messages

        Returns
        -------
        pandas.DataFrame
            DataFrame with results
        """
        if logger is None:
            import logging
            logger = logging.getLogger(__name__)

        try:
            logger.info("Loading data with automatic column detection...")
            self._update_progress(10)
            edges = self.load_data(input_file, test_mode=test_mode)

            if not edges:
                logger.warning("No valid edges found in input file")
                empty_df = pd.DataFrame(columns=['source', 'target', 'weight', 'salience'])
                empty_df.to_csv(output_file, index=False, sep='\t')
                self._update_progress(100)
                return empty_df

            logger.info(f"Loaded {len(edges):,} edges")
            logger.info(f"Detected columns: {self.get_column_info()}")
            self._update_progress(30)

            logger.info("Computing edge salience...")
            self._update_progress(40)
            salience = self.fast_link.compute_salience(edges, processes=self.n_jobs)
            self._update_progress(80)

            logger.info(f"Filtering edges with salience >= {self.threshold}...")
            results = []
            for u, v, w in edges:
                edge_key = tuple(sorted((u, v)))
                s = salience.get(edge_key, 0)
                if s >= self.threshold:
                    results.append([u, v, w, s])

            df_results = pd.DataFrame(results, columns=['source', 'target', 'weight', 'salience'])

            logger.info(f"Saving results to {output_file}...")
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            df_results.to_csv(output_file, index=False, sep='\t')

            original = len(edges)
            hss = len(df_results)
            reduction = 0.0
            if original > 0:
                reduction = 100 * (original - hss) / original

            logger.info(f"Results:")
            logger.info(f"  Initial Edges: {original:,}")
            logger.info(f"  HSS Edges: {hss:,}")
            logger.info(f"  Reduction: {reduction:.1f}%")

            self._update_progress(100)

            return df_results

        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise

def create_hss_extractor(threshold=0.9, weight_threshold=1e-6, n_jobs=None):
    """
    Factory function to create a HighSalienceSkeleton instance.

    Parameters
    ----------
    threshold : float, optional (default=0.9)
        Salience threshold for keeping edges
    weight_threshold : float, optional (default=1e-6)
        Minimum weight threshold
    n_jobs : int, optional (default=None)
        Number of parallel jobs

    Returns
    -------
    HighSalienceSkeleton
    """
    return HighSalienceSkeleton(
        threshold=threshold,
        weight_threshold=weight_threshold,
        n_jobs=n_jobs
    )

def run_hss(input_file, output_file, threshold=0.9, test_mode=False, n_jobs=None):
    """
    Legacy function to run High Salience Skeleton extraction.

    Parameters
    ----------
    input_file : str
        Path to input file
    output_file : str
        Path to output file
    threshold : float, optional (default=0.9)
        Salience threshold
    test_mode : bool, optional (default=False)
        If True, run in test mode
    n_jobs : int, optional (default=None)
        Number of parallel jobs
    """
    extractor = create_hss_extractor(threshold=threshold, n_jobs=n_jobs)
    return extractor.run(input_file, output_file, test_mode=test_mode)

if __name__ == "__main__":
    input_path = "weighted_gene_edges_one_mode.tsv"
    output_path = "hss_filter_disease_edges_results.tsv"

    run_hss(input_path, output_path, test_mode=False)