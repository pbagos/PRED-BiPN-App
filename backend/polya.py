"""
Polya Filter - Numerically Stable Implementation for Backbone Extraction
With auto-detection, POSITIONAL ACCESS and CSV output with source, target
(Includes fixes for FDR multiple testing and directed graph null models)
"""

import numpy as np
import scipy.special as sp
import scipy.stats
from typing import Tuple, Dict, Union, List, Optional
import os
import time
from statsmodels.stats.multitest import multipletests
import pandas as pd
from tqdm import tqdm
import gc
import logging
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count

def _log_polya_pmf_mp(w: int, k: int, s: int, a: float) -> float:
    if w < 0 or w > s or s <= 0 or k <= 1 or a <= 0:
        return -np.inf
    alpha = 1.0 / a
    beta_val = (k - 1.0) / a
    try:
        log_binom = sp.gammaln(s + 1) - sp.gammaln(w + 1) - sp.gammaln(s - w + 1)
        log_beta_ratio = sp.betaln(w + alpha, s - w + beta_val) - sp.betaln(alpha, beta_val)
        return log_binom + log_beta_ratio
    except ValueError:
        return -np.inf

def _polya_pvalue_mp(w: float, k: int, s: float, a: float, apr_lvl: float = 10) -> float:
    if k == 1:
        return 1.0
    w_int = int(round(w))
    s_int = int(round(s))
    if a == 0:
        return float(scipy.stats.binom.sf(w_int - 1, s_int, 1.0 / k))
    if w_int <= 0:
        return 1.0
    if w_int > s_int:
        return 0.0
    if apr_lvl < 0:
        if s > 0:
            p = 1.0 - sp.betainc(1.0 / a, (k - 1.0) / a, w / s)
            return float(min(max(p, 0.0), 1.0))
        return 1.0
    L = apr_lvl
    idx1 = (s - w) >= L * ((k - 1) / a + 1)
    idx2 = w >= L * max(1 / a, 1)
    idx3 = s >= L * max(k / a, 1)
    idx4 = k >= L * (a - 1 + 1e-20)
    if L > 0 and idx1 and idx2 and idx3 and idx4:
        p = 1.0 - sp.betainc(1.0 / a, (k - 1.0) / a, w / s)
        return float(min(max(p, 0.0), 1.0))
    log_probs = []
    for x in range(w_int, s_int + 1):
        lp = _log_polya_pmf_mp(x, k, s_int, a)
        if lp > -np.inf:
            log_probs.append(lp)
    if not log_probs:
        return 0.0
    log_max = max(log_probs)
    s_exp = sum(np.exp(lp - log_max) for lp in log_probs)
    return float(min(max(np.exp(log_max + np.log(s_exp)), 0.0), 1.0))

def _compute_edge_batch_mp(batch):
    """Compute p-values for a batch of edges."""
    results = []
    for i, j, w, node_i, node_j, a, apr_lvl, directed in batch:
        k1, s1 = node_i
        p1 = _polya_pvalue_mp(w, k1, s1, a, apr_lvl)

        if not directed and node_j is not None:
            k2, s2 = node_j
            p2 = _polya_pvalue_mp(w, k2, s2, a, apr_lvl)
        else:
            p2 = 1.0

        results.append((i, j, w, p1, p2))
    return results

class PolyaFilter:
    """
    Pólya Filter: Numerically stable implementation for backbone extraction.
    Uses positional access (iloc[0], iloc[1], iloc[2]) without predefined column names.
    """

    def __init__(self, significance_level: float = 0.05, method: str = 'fdr',
                 logger: Optional[logging.Logger] = None, n_workers: Optional[int] = None):
        """
        Initialize the Polya Filter.

        Args:
            significance_level (float): Significance level (α_u)
            method (str): Multiple-comparison method ('bonferroni' or 'fdr')
            logger (logging.Logger): Optional logger instance
        """
        self.alpha_u = significance_level
        self.method = method.lower()
        self.logger = logger or logging.getLogger(__name__)
        self.adj_dict = {}
        self.node_props = {}
        self.stats = {}
        self.n_workers = n_workers if n_workers is not None else 1

        self.detected_sep = None
        self.has_header = None
        self.num_columns = None

        if self.method not in ['bonferroni', 'fdr']:
            raise ValueError("The correction method must be 'bonferroni' or 'fdr'")

    def _log_message(self, message: str, level: str = 'info'):
        """Log message to both console and logger if available."""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
        else:
            print(message)

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
                self.detected_sep = '\t'
            else:
                self.detected_sep = max(separators, key=lambda k: separators[k])

            return self.detected_sep

    def _detect_header(self, file_path, sep):
        """Detect if file has header by analyzing first line."""
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                parts = first_line.split(sep)

                header_likely = False
                for part in parts:
                    try:
                        float(part)
                    except ValueError:
                        header_likely = True
                        break

                self.has_header = header_likely
                self.num_columns = len(parts)

                return self.has_header
        except:
            self.has_header = False
            return False

    def get_ML_estimate(self, directed: bool = False) -> float:
        """
        Computes the Maximum Likelihood estimate of parameter a,
        leading to ultra-sparse backbones adapted to the network's heterogeneity.
        """
        w_list = []
        k_list = []
        s_list = []
        for (u, v), w in self.adj_dict.items():
            if u in self.node_props:
                k_u, s_u = self.node_props[u]
                w_list.append(w)
                k_list.append(k_u)
                s_list.append(s_u)

            if not directed and v in self.node_props:
                k_v, s_v = self.node_props[v]
                w_list.append(w)
                k_list.append(k_v)
                s_list.append(s_v)

        w_arr = np.array(w_list)
        k_arr = np.array(k_list)
        s_arr = np.array(s_list)

        mask = k_arr > 1
        w_arr = w_arr[mask]
        k_arr = k_arr[mask]
        s_arr = s_arr[mask]

        if len(w_arr) == 0:
            self._log_message("  Warning: No links with degree > 1. Using a=1.0", 'warning')
            return 1.0

        def nll(a_val):
            a = a_val[0]
            if a <= 0: return 1e10
            A = 1.0 / a
            B = (k_arr - 1.0) / a

            log_p = (sp.gammaln(s_arr + 1) + sp.betaln(w_arr + A, s_arr - w_arr + B)
                     - sp.gammaln(w_arr + 1) - sp.gammaln(s_arr - w_arr + 1) - sp.betaln(A, B))
            val = -np.sum(log_p)
            if np.isinf(val) or np.isnan(val): return 1e10
            return val

        best_val, a_best = np.inf, 1.0
        for x0 in [0.1, 0.5, 1.0, 2.0, 3.0]:
            res = minimize(nll, x0=[x0], bounds=[(1e-5, 5.0)], method='L-BFGS-B')
            if res.fun < best_val:
                best_val = res.fun
                a_best = res.x[0]

        if a_best >= 4.9:
            self._log_message("  Warning: ML estimate hit upper bound. Network may be too heterogeneous. Falling back to a=1.0.", 'warning')
            a_best = 1.0

        self._log_message(f"  Estimation ended, a = {a_best:.6f}")
        return float(a_best)

    def _log_polya_pmf(self, w: int, k: int, s: int, a: float) -> float:
        """
        Computes the LOGARITHM of the probability function P(w|k,s,a).
        Numerically stable method.
        """
        if w < 0 or w > s or s <= 0 or k <= 1 or a <= 0:
            return -np.inf

        alpha = 1.0 / a
        beta = (k - 1.0) / a

        try:
            log_binom = sp.gammaln(s + 1) - sp.gammaln(w + 1) - sp.gammaln(s - w + 1)

            log_beta_ratio = sp.betaln(w + alpha, s - w + beta) - sp.betaln(alpha, beta)

            return log_binom + log_beta_ratio
        except ValueError:
            return -np.inf

    def polya_pvalue(self, w: float, k: int, s: float, a: float, apr_lvl: float = 10) -> float:
        """
        Computes the p-value π_P = P(X >= w).
        Supports asymptotic approximation (apr_lvl > 0).
        """
        if k == 1:
            return 1.0

        w_int = int(round(w))
        s_int = int(round(s))

        if a == 0:
            return float(scipy.stats.binom.sf(w_int - 1, s_int, 1.0 / k))

        if w_int <= 0:
            return 1.0
        if w_int > s_int:
            return 0.0

        if apr_lvl < 0:
            if s > 0:
                p = 1.0 - sp.betainc(1.0 / a, (k - 1.0) / a, w / s)
                return float(min(max(p, 0.0), 1.0))
            return 1.0

        L = apr_lvl
        idx1 = (s - w) >= L * ((k - 1) / a + 1)
        idx2 = w >= L * max(1 / a, 1)
        idx3 = s >= L * max(k / a, 1)
        idx4 = k >= L * (a - 1 + 1e-20)

        if L > 0 and idx1 and idx2 and idx3 and idx4:
            p = 1.0 - sp.betainc(1.0 / a, (k - 1.0) / a, w / s)
            return float(min(max(p, 0.0), 1.0))

        log_probs = []
        for x in range(w_int, s_int + 1):
            log_prob = self._log_polya_pmf(x, k, s_int, a)
            if log_prob > -np.inf:
                log_probs.append(log_prob)

        if not log_probs:
            return 0.0

        log_max = max(log_probs)
        s_exp = sum(np.exp(log_p - log_max) for log_p in log_probs)

        log_cumulative_prob = log_max + np.log(s_exp)
        p_value = np.exp(log_cumulative_prob)

        return float(min(max(p_value, 0.0), 1.0))

    def compute_link_significance(self, w: float, node1: Tuple[int, float],
                                  node2: Tuple[int, float], a: float, apr_lvl: float) -> Tuple[float, float]:
        """Compute p-values from the two node perspectives."""
        k1, s1 = node1
        k2, s2 = node2

        p1 = self.polya_pvalue(w, k1, s1, a, apr_lvl)
        p2 = self.polya_pvalue(w, k2, s2, a, apr_lvl)

        return p1, p2

    def load_network_data(self, file_path: str, directed: bool = False, chunksize: int = 100000):
        """
        Loads data from a file using POSITIONAL ACCESS (iloc[0], iloc[1], iloc[2]).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")

        self._detect_separator(file_path)
        self._log_message(f"  Detected separator: {repr(self.detected_sep)}")

        self._detect_header(file_path, self.detected_sep)
        self._log_message(f"  Detected header: {'Yes' if self.has_header else 'No'}")

        self._log_message("  Counting rows...")
        total_rows = 0
        with open(file_path, 'r') as f:
            if self.has_header:
                next(f)
            for _ in f:
                total_rows += 1
        self._log_message(f"  Total rows: {total_rows:,}")

        self._log_message("\n" + "="*60)
        self._log_message("FIRST PASS: Computing node properties")
        self._log_message("="*60)

        node_properties = {}
        edge_list = []

        chunk_reader = pd.read_csv(
            file_path,
            sep=self.detected_sep,
            header=0 if self.has_header else None,
            chunksize=chunksize,
            dtype=str
        )

        total_chunks = (total_rows // chunksize) + 1

        for chunk in tqdm(chunk_reader, total=total_chunks, desc="  Processing"):
            for _, row in chunk.iterrows():
                u = str(row.iloc[0]).strip()
                v = str(row.iloc[1]).strip()

                if len(row) > 2:
                    try:
                        w = float(row.iloc[2])
                    except (ValueError, TypeError):
                        w = 1.0
                else:
                    w = 1.0

                if not u or not v or u == 'nan' or v == 'nan' or w <= 0:
                    continue
                if u == v:
                    continue
                if not directed and u > v:
                    u, v = v, u

                edge_list.append((u, v, w))

                if u not in node_properties:
                    node_properties[u] = [0, 0.0]
                if not directed and v not in node_properties:
                    node_properties[v] = [0, 0.0]

                node_properties[u][0] += 1
                node_properties[u][1] += w

                if not directed and u != v:
                    node_properties[v][0] += 1
                    node_properties[v][1] += w

        self.node_props = {node: (int(props[0]), float(props[1]))
                          for node, props in node_properties.items() if props[0] > 0}

        self._log_message(f"\n  Found {len(self.node_props):,} unique nodes")

        self._log_message("\n" + "="*60)
        self._log_message("SECOND PASS: Building adjacency dictionary")
        self._log_message("="*60)

        self.adj_dict = {}
        processed_edges = set()

        for u, v, w in tqdm(edge_list, desc="  Building edges"):
            edge_key = (u, v)
            if edge_key in processed_edges:
                continue
            self.adj_dict[edge_key] = float(w)
            processed_edges.add(edge_key)

        self._log_message(f"\n  Unique edges: {len(self.adj_dict):,}")

        self.stats['input_edges'] = len(edge_list)
        self.stats['unique_edges'] = len(self.adj_dict)
        self.stats['nodes'] = len(self.node_props)

        return len(self.adj_dict)

    def extract_backbone(self, a: float = 1.0, directed: bool = False) -> List[Tuple]:
        """
        Extracts the backbone by applying the multiple-comparison correction.
        """
        if not self.adj_dict:
            raise ValueError("The network data has not been loaded.")

        num_links = len(self.adj_dict)
        num_tests = num_links if directed else 2 * num_links

        if a < 0:
            self._log_message("\n" + "="*60)
            self._log_message("ESTIMATING PARAMETER 'a' VIA MAXIMUM LIKELIHOOD")
            self._log_message("="*60)
            a = self.get_ML_estimate(directed=directed)

        has_non_int = any(w % 1 != 0 for w in self.adj_dict.values())
        apr_lvl = -1 if has_non_int else 10
        if has_non_int:
            self._log_message("  Non-integer weights detected. Forcing betainc approximation (Eq. 6 per paper).")

        p_values = []
        edges_data = []

        self._log_message("\n" + "="*60)
        self._log_message(f"CALCULATING P-VALUES (L = {num_tests:,})")
        self._log_message(f"  Parameter a = {a}")
        self._log_message(f"  Approximation level = {apr_lvl}")
        self._log_message(f"  Workers = {self.n_workers}")
        self._log_message("="*60)

        edge_items = []
        for (i, j), w in self.adj_dict.items():
            if directed:
                if i in self.node_props:
                    edge_items.append((i, j, w, self.node_props[i], None, a, apr_lvl, directed))
            else:
                if i in self.node_props and j in self.node_props:
                    edge_items.append((i, j, w, self.node_props[i], self.node_props[j], a, apr_lvl, directed))

        if self.n_workers > 1 and len(edge_items) > 500:
            batch_size = max(100, len(edge_items) // (self.n_workers * 4))
            batches = [edge_items[k:k + batch_size] for k in range(0, len(edge_items), batch_size)]
            self._log_message(f"  Parallel batches: {len(batches)} x ~{batch_size} edges")
            with Pool(processes=self.n_workers) as pool:
                for batch_result in tqdm(
                    pool.imap(_compute_edge_batch_mp, batches),
                    total=len(batches), desc="  Computing p-values"
                ):
                    for i, j, weight, p1, p2 in batch_result:
                        if directed:
                            p_values.append(p1)
                        else:
                            p_values.extend([p1, p2])
                        edges_data.append(((i, j), weight, p1, p2))
        else:
            for i, j, w, node_i, node_j, _a, _apr, _dir in tqdm(edge_items, desc="  Computing p-values"):
                k1, s1 = node_i
                p1 = _polya_pvalue_mp(w, k1, s1, _a, _apr)
                if not _dir:
                    k2, s2 = node_j
                    p2 = _polya_pvalue_mp(w, k2, s2, _a, _apr)
                    p_values.extend([p1, p2])
                else:
                    p2 = 1.0
                    p_values.append(p1)
                edges_data.append(((i, j), w, p1, p2))

        self.stats['total_p_values'] = len(p_values)

        self._log_message("\n" + "="*60)
        self._log_message(f"MULTIPLE COMPARISON CORRECTION - Method: {self.method.upper()}")
        self._log_message("="*60)

        rejected = np.array([], dtype=bool)
        if not p_values:
            return []

        if self.method == 'bonferroni':
            thresh = self.alpha_u / len(p_values)
            rejected = np.array(p_values) < thresh
            self._log_message(f"  Bonferroni threshold: {thresh:.6e}")
            self._log_message(f"  Significant edges: {sum(rejected):,}")
            self.stats['threshold'] = thresh

        elif self.method == 'fdr':
            rejected, p_corrected, _, _ = multipletests(p_values, alpha=self.alpha_u, method='fdr_bh')
            self._log_message(f"  FDR (Benjamini/Hochberg) applied.")
            self._log_message(f"  Significant edges: {sum(rejected):,}")
            self.stats['significant_fdr'] = int(sum(rejected))

        backbone_edges = []

        if directed:
            for idx, (edge, weight, p1, p2) in enumerate(edges_data):
                if rejected[idx]:
                    i, j = edge
                    r_ratio = self.compute_r_ratio(weight, self.node_props[i], (0, 0))
                    backbone_edges.append((i, j, weight, p1, p1, p2, r_ratio))
        else:
            rejected_pairs = rejected.reshape(-1, 2)
            for idx, (edge, weight, p1, p2) in enumerate(edges_data):
                if rejected_pairs[idx][0] or rejected_pairs[idx][1]:
                    i, j = edge
                    r_ratio = self.compute_r_ratio(weight, self.node_props[i], self.node_props[j])
                    backbone_edges.append((i, j, weight, min(p1, p2), p1, p2, r_ratio))

        self.stats['backbone_edges'] = len(backbone_edges)
        self.stats['retained_percentage'] = (len(backbone_edges) / num_links) * 100 if num_links > 0 else 0

        self._log_message(f"\n  Backbone edges: {len(backbone_edges):,} / {num_links:,} ({self.stats['retained_percentage']:.2f}%)")
        return backbone_edges

    def compute_r_ratio(self, w: float, node1_props: Tuple[int, float],
                        node2_props: Tuple[int, float]) -> float:
        """
        Computes the ratio r = max(r_i, r_j) (Eq. 4).
        """
        r_values = []

        k1, s1 = node1_props
        if s1 > 0:
            r_values.append((w / s1) * k1)

        k2, s2 = node2_props
        if s2 > 0:
            r_values.append((w / s2) * k2)

        return max(r_values) if r_values else 0

    def save_results(self, backbone_edges: List[Tuple], output_file: str):
        """
        Saves results to a CSV file with columns: source, target, weight, p_value.
        """
        try:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

            if not backbone_edges:
                self._log_message("  No results to save.")
                pd.DataFrame(columns=['source', 'target', 'weight', 'p_value']).to_csv(
                    output_file, index=False, sep='\t'
                )
                stats_file = output_file.replace('.tsv', '_stats.txt')
                self.save_summary_stats(stats_file)
                self.stats['output_file'] = output_file
                return

            self._log_message("\n" + "="*60)
            self._log_message("SAVING RESULTS")
            self._log_message("="*60)

            df_results = []
            for node1, node2, w, min_p, p1, p2, r in backbone_edges:
                df_results.append([node1, node2, w, min_p])

            df = pd.DataFrame(df_results, columns=['source', 'target', 'weight', 'p_value'])
            df.to_csv(output_file, index=False, sep='\t')

            self._log_message(f"  Main results: {output_file}")
            self._log_message(f"     Columns: source, target, weight, p_value")
            self._log_message(f"     Total edges: {len(backbone_edges):,}")

            detailed_file = output_file.replace('.tsv', '_detailed.tsv')
            df_detailed = pd.DataFrame(backbone_edges,
                                        columns=['source', 'target', 'weight',
                                                'p_value_min', 'p_value_node1',
                                                'p_value_node2', 'ratio_r'])
            df_detailed.to_csv(detailed_file, index=False, sep='\t')
            self._log_message(f"  Detailed results: {detailed_file}")

            stats_file = output_file.replace('.tsv', '_stats.txt')
            self.save_summary_stats(stats_file)

            self.stats['output_file'] = output_file
            self.stats['detailed_file'] = detailed_file

        except Exception as e:
            self._log_message(f"  Error saving results: {e}", 'error')

    def save_summary_stats(self, output_file: str):
        """Save summary statistics."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("POLYA FILTER - ANALYSIS STATISTICS\n")
                f.write("=" * 60 + "\n\n")

                f.write("EXECUTION PARAMETERS\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Correction method: {self.method.upper()}\n")
                f.write(f"  Significance level (α): {self.alpha_u}\n")
                f.write(f"  Separator: {repr(self.detected_sep)}\n")
                f.write(f"  Header: {'Yes' if self.has_header else 'No'}\n\n")

                f.write("NETWORK STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Input edges (with duplicates): {self.stats.get('input_edges', 0):,}\n")
                f.write(f"  Unique edges: {self.stats.get('unique_edges', 0):,}\n")
                f.write(f"  Nodes: {self.stats.get('nodes', 0):,}\n")
                f.write(f"  Total p-values calculated: {self.stats.get('total_p_values', 0):,}\n\n")

                f.write("BACKBONE RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Backbone edges: {self.stats.get('backbone_edges', 0):,}\n")
                f.write(f"  Retention percentage: {self.stats.get('retained_percentage', 0):.2f}%\n")

                if 'significant_fdr' in self.stats:
                    f.write(f"  Significant edges (FDR): {self.stats['significant_fdr']}\n")
                if 'threshold' in self.stats:
                    f.write(f"  Bonferroni threshold: {self.stats['threshold']:.6e}\n\n")

                f.write("EXECUTION TIME\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Total time: {self.stats.get('total_time', 0):.2f} seconds\n\n")

                f.write("OUTPUT FILES\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Main results: {self.stats.get('output_file', 'N/A')}\n")
                f.write(f"  Detailed results: {self.stats.get('detailed_file', 'N/A')}\n")
                f.write(f"  Statistics: {output_file}\n")

            self._log_message(f"  Statistics saved to: {output_file}")

        except Exception as e:
            self._log_message(f"  Error saving statistics: {e}", 'error')

    def get_column_info(self):
        """Returns information about detected columns."""
        return {
            'separator': self.detected_sep,
            'has_header': self.has_header,
            'num_columns': self.num_columns,
            'source_column': 'Column 0 (iloc[0]) - first column',
            'target_column': 'Column 1 (iloc[1]) - second column',
            'weight_column': 'Column 2 (iloc[2]) - third column (or default=1.0)',
            'nodes': self.stats.get('nodes', 0),
            'edges': self.stats.get('unique_edges', 0)
        }

    def run_full_analysis(self, input_file: str, output_file: str,
                          a: float = 1.0, directed: bool = False,
                          chunksize: int = 100000) -> Dict:
        """
        Runs the full Polya Filter analysis with POSITIONAL ACCESS.
        """
        start_time = time.time()

        self._log_message("\n" + "="*60)
        self._log_message("POLYA FILTER - BACKBONE EXTRACTION")
        self._log_message("="*60)
        self._log_message(f"  Input file: {input_file}")
        self._log_message(f"  Output file: {output_file}")
        self._log_message(f"  Method: {self.method.upper()}")
        self._log_message(f"  Significance level: {self.alpha_u}")
        self._log_message(f"  Parameter a: {a}")
        self._log_message(f"  Directed: {'Yes' if directed else 'No'}")
        self._log_message("="*60)

        try:
            self.load_network_data(input_file, directed=directed, chunksize=chunksize)
            backbone_edges = self.extract_backbone(a, directed)
            self.save_results(backbone_edges, output_file)

            total_time = time.time() - start_time
            self.stats['total_time'] = total_time

            self._log_message("\n" + "="*60)
            self._log_message("ANALYSIS COMPLETED SUCCESSFULLY")
            self._log_message("="*60)
            self._log_message(f"  Total time: {total_time:.2f} seconds")
            self._log_message(f"  Results saved to: {output_file}")
            self._log_message("="*60 + "\n")

            return {
                'success': True,
                'input_file': input_file,
                'output_file': output_file,
                'edges_input': self.stats.get('unique_edges', 0),
                'edges_backbone': self.stats.get('backbone_edges', 0),
                'retained_percentage': self.stats.get('retained_percentage', 0),
                'nodes': self.stats.get('nodes', 0),
                'method': self.method,
                'alpha': self.alpha_u,
                'a_parameter': a,
                'total_time': total_time,
                'column_info': self.get_column_info(),
                'stats': self.stats
            }

        except Exception as e:
            self._log_message(f"\nERROR IN ANALYSIS: {e}", 'error')
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

def create_polya_filter(significance_level: float = 0.05,
                        method: str = 'fdr',
                        logger: Optional[logging.Logger] = None,
                        n_workers: Optional[int] = None) -> PolyaFilter:
    return PolyaFilter(
        significance_level=significance_level,
        method=method,
        logger=logger,
        n_workers=n_workers,
    )

def run_polya_filter(input_file: str, output_file: str,
                     alpha: float = 0.05, method: str = 'fdr',
                     a: float = 1.0, directed: bool = False,
                     logger: Optional[logging.Logger] = None) -> Dict:
    filter_instance = create_polya_filter(
        significance_level=alpha,
        method=method,
        logger=logger
    )

    return filter_instance.run_full_analysis(
        input_file=input_file,
        output_file=output_file,
        a=a,
        directed=directed
    )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Polya Filter - Backbone Extraction from Networks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input_file', help='Input file (TSV/CSV)')
    parser.add_argument('output_file', help='Output file (CSV with columns source, target, weight, p_value)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level (default: 0.05)')
    parser.add_argument('--method', choices=['bonferroni', 'fdr'], default='fdr',
                       help='Correction method (default: fdr)')
    parser.add_argument('--a', type=float, default=1.0,
                       help='Polya parameter (a=1.0 = Disparity Filter, a<0 = ML Estimation)')
    parser.add_argument('--directed', action='store_true',
                       help='Directed network (default: undirected)')
    parser.add_argument('--chunksize', type=int, default=100000,
                       help='Chunk size for reading (default: 100000)')

    args = parser.parse_args()

    polya_filter = create_polya_filter(
        significance_level=args.alpha,
        method=args.method
    )

    results = polya_filter.run_full_analysis(
        input_file=args.input_file,
        output_file=args.output_file,
        a=args.a,
        directed=args.directed,
        chunksize=args.chunksize
    )

    if results['success']:
        print("\nSUMMARY")
        print("-" * 40)
        print(f"  Nodes: {results['nodes']:,}")
        print(f"  Input edges: {results['edges_input']:,}")
        print(f"  Backbone edges: {results['edges_backbone']:,}")
        print(f"  Retention: {results['retained_percentage']:.2f}%")
        print(f"  Time: {results['total_time']:.2f} sec")
        print(f"\n  Results: {results['output_file']}")
        print(f"  Detailed: {results['output_file'].replace('.tsv', '_detailed.tsv')}")
        print(f"  Statistics: {results['output_file'].replace('.tsv', '_stats.txt')}")