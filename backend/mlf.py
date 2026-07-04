"""
Marginal Likelihood Filter (MLF) for weighted complex networks.
Reference: Dianati, N. (2016). Physical Review E, 93, 012304.

Column detection: always positional (iloc[0], iloc[1], iloc[2]).
Header is auto-detected and used as column names, never as data.
"""

from scipy.stats import poisson
import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix, save_npz
from multiprocessing import Pool, cpu_count
import warnings
import os
import time
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "0"

class MLFProcessor:
    def __init__(self, alpha=0.05, chunk_size=50000, test_mode=False,
                 test_size=10000, n_jobs=None):
        self.alpha = alpha
        self.chunk_size = chunk_size
        self.test_mode = test_mode
        self.test_size = test_size
        self.n_jobs = n_jobs if n_jobs else cpu_count()
        self.logger = None

    def _log(self, message: str, level: str = "info"):
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(message)

    def _detect_separator(self, file_path: str) -> str:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        separators = {'\t': first_line.count('\t'), ',': first_line.count(','),
                      ';': first_line.count(';'), ' ': first_line.count(' ')}
        sep = max(separators, key=lambda k: separators[k])
        return sep if separators[sep] > 0 else '\t'

    def _load_chunks(self, file_path: str, sep: str):
        """Load data in chunks. header=0 so first row becomes column names, not data."""
        nrows = self.test_size if self.test_mode else None
        return pd.read_csv(
            file_path,
            sep=sep,
            header=0,
            chunksize=self.chunk_size,
            nrows=nrows
        )

    @staticmethod
    def calculate_p_value(args):
        w_ij, s_i, s_j, T = args
        if T <= 0 or s_i <= 0 or s_j <= 0 or w_ij <= 0:
            return 1.0
        p_ij = (s_i * s_j) / (2 * T ** 2)
        lambda_val = T * p_ij
        try:
            return poisson.sf(w_ij - 1, lambda_val)
        except Exception:
            return np.nan

    def process_graph(self, input_file: str, output_file: str, logger=None) -> Tuple[int, pd.DataFrame]:
        """
        Process graph and return number of significant edges and full results dataframe.

        Returns:
            Tuple[int, pd.DataFrame]: (number_of_significant_edges, dataframe_with_all_results)
        """
        self.logger = logger
        self._log("=== MLF (Marginal Likelihood Filter) ===")
        self._log(f"Input: {input_file}")
        self._log(f"Alpha: {self.alpha}")

        start_time = time.time()

        sep = self._detect_separator(input_file)

        sample = pd.read_csv(input_file, sep=sep, header=0, nrows=1)
        self._log(f"Columns detected: {list(sample.columns)}")
        self._log("Using: iloc[0]=source, iloc[1]=target, iloc[2]=weight")

        self._log("\n[1/2] Computing node strengths...")
        node_index = {}
        current_idx = 0
        strength = []
        total_weight = 0
        edge_count = 0

        for chunk in self._load_chunks(input_file, sep):
            weight_vals = pd.to_numeric(chunk.iloc[:, 2], errors='coerce').fillna(1).astype(int)

            for i, row in chunk.iterrows():
                u = str(row.iloc[0]).strip()
                v = str(row.iloc[1]).strip()
                w = int(weight_vals.iloc[chunk.index.get_loc(i)])

                if w <= 0 or not u or not v or u == 'nan' or v == 'nan':
                    continue

                if u not in node_index:
                    node_index[u] = current_idx
                    strength.append(0)
                    current_idx += 1
                if v not in node_index:
                    node_index[v] = current_idx
                    strength.append(0)
                    current_idx += 1

                strength[node_index[u]] += w
                strength[node_index[v]] += w
                total_weight += w   # T = sum of edge weights W; sum of strengths = 2W. Do NOT halve.
                edge_count += 1

                if self.test_mode and edge_count >= self.test_size:
                    break
            if self.test_mode and edge_count >= self.test_size:
                break

        strength = np.array(strength, dtype=np.float64)
        self._log(f"  Nodes: {len(node_index)}, Edges: {edge_count}, Total weight: {total_weight:.2f}")

        self._log("\n[2/2] Filtering edges and computing statistics...")
        sparse_graph = dok_matrix((len(node_index), len(node_index)), dtype=np.float32)

        all_results = []
        significant_results = []

        processed_count = 0

        for chunk in self._load_chunks(input_file, sep):
            weight_vals = pd.to_numeric(chunk.iloc[:, 2], errors='coerce').fillna(1).astype(int)

            args, valid_rows = [], []
            for i, row in chunk.iterrows():
                u = str(row.iloc[0]).strip()
                v = str(row.iloc[1]).strip()
                w = int(weight_vals.iloc[chunk.index.get_loc(i)])

                if w <= 0 or u not in node_index or v not in node_index:
                    continue

                s_i = strength[node_index[u]]
                s_j = strength[node_index[v]]
                args.append((w, s_i, s_j, total_weight))
                valid_rows.append((u, v, w, s_i, s_j))
                processed_count += 1

                if self.test_mode and processed_count >= self.test_size:
                    break

            if args:
                with Pool(self.n_jobs) as pool:
                    p_values = pool.map(self.calculate_p_value, args)

                for (u, v, w, s_i, s_j), p_val in zip(valid_rows, p_values):
                    if total_weight > 0 and s_i > 0 and s_j > 0:
                        p_ij = (s_i * s_j) / (2 * total_weight ** 2)
                        lambda_val = total_weight * p_ij
                    else:
                        p_ij = 0.0
                        lambda_val = 0.0

                    edge_info = {
                        'source': u,
                        'target': v,
                        'weight': w,
                        'strength_source': s_i,
                        'strength_target': s_j,
                        'p_ij': p_ij,
                        'lambda_value': lambda_val,
                        'p_value': p_val,
                        'significant': p_val < self.alpha
                    }
                    all_results.append(edge_info)

                    if p_val < self.alpha:
                        i, j = node_index[u], node_index[v]
                        sparse_graph[i, j] = w
                        sparse_graph[j, i] = w

                        significant_results.append({
                            'source': u,
                            'target': v,
                            'weight': w,
                            'p_value': p_val
                        })

            if self.test_mode and processed_count >= self.test_size:
                break

        df_all_results = pd.DataFrame(all_results)
        df_significant = pd.DataFrame(significant_results)
        n_significant = len(df_significant)

        if n_significant > 0:
            df_significant[['source', 'target', 'weight', 'p_value']].to_csv(output_file, index=False, sep='\t')

            full_output_file = output_file.rsplit('.', 1)[0] + '_full.tsv'
            df_all_results.to_csv(full_output_file, index=False, sep='\t')
            npz_file = output_file.rsplit('.', 1)[0] + '_sparse.npz'
            save_npz(npz_file, sparse_graph.tocsr())

            self._log(f"\n  Saved {n_significant} significant edges with p-values to: {output_file}")
            self._log(f"    Columns: source, target, weight, p_value")
            self._log(f"  Saved full results ({len(df_all_results)} edges) with all fields to: {full_output_file}")
            self._log(f"    Columns: source, target, weight, strength_source, strength_target, p_ij, lambda_value, p_value, significant")
            self._log(f"  Sparse matrix saved to: {npz_file}")

            self._log(f"\n  Statistics:")
            self._log(f"    - Total edges analyzed: {len(df_all_results)}")
            self._log(f"    - Significant edges (p < {self.alpha}): {n_significant}")
            self._log(f"    - Retention rate: {n_significant/len(df_all_results)*100:.2f}%")
            self._log(f"    - Min p-value: {df_significant['p_value'].min():.2e}")
            self._log(f"    - Max p-value: {df_significant['p_value'].max():.2e}")

        else:
            self._log("\n  No significant edges found.")
            full_output_file = output_file.rsplit('.', 1)[0] + '_full.tsv'
            df_all_results.to_csv(full_output_file, index=False, sep='\t')

            pd.DataFrame(columns=['source', 'target', 'weight', 'p_value']).to_csv(output_file, index=False, sep='\t')

            self._log(f"  Saved full results ({len(df_all_results)} edges) with all fields to: {full_output_file}")

        elapsed = (time.time() - start_time) / 60
        self._log(f"\n=== Completed in {elapsed:.2f} minutes ===")

        return n_significant, df_all_results

def create_mlf_processor(alpha=0.05, chunk_size=50000, test_mode=False):
    return MLFProcessor(alpha=alpha, chunk_size=chunk_size, test_mode=test_mode)

def run_mlf(input_file, output_file, alpha=0.05, test_mode=False, logger=None):
    """
    Wrapper function for easy calling from the Flask app.

    Returns:
        Tuple[int, pd.DataFrame]: (number_of_significant_edges, full_results_dataframe)
    """
    processor = MLFProcessor(alpha=alpha, test_mode=test_mode)
    return processor.process_graph(input_file, output_file, logger=logger)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MLF Backbone Extraction')
    parser.add_argument('input_file', help='Path to input file')
    parser.add_argument('output_file', help='Path to output CSV file')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    processor = create_mlf_processor(alpha=args.alpha, test_mode=args.test)
    n_edges, full_results = processor.process_graph(args.input_file, args.output_file)
    print(f"\nResults: {n_edges} significant edges found")
    print(f"Full results shape: {full_results.shape}")