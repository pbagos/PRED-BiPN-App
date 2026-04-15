"""
Spearman Correlation Analyzer for Gene-Gene Interactions.
Optimized version — vectorized matrix computation instead of pair-by-pair loops.
Flask-compatible.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
import time
import uuid
import traceback

warnings.filterwarnings('ignore')


class BaseGeneCorrelationAnalyzer:

    def __init__(self, filepath, chunksize=10000, logger=None):
        self.filepath           = filepath
        self.chunksize          = chunksize
        self.gene_dict          = {}
        self.matrix             = None
        self.correlation_matrix = None
        self.logger             = logger
        self.progress_callback  = None

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def _log(self, message, level='info'):
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            print(message)

    def _update_progress(self, percent, message=None):
        if self.progress_callback:
            self.progress_callback(percent)
        if message:
            self._log(message)

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    def _detect_separator(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            line = f.readline()
        if '\t' in line:   sep = '\t'
        elif ',' in line:  sep = ','
        elif ';' in line:  sep = ';'
        else:              sep = r'\s+'
        self._log(f"Detected separator: {repr(sep)}")
        return sep

    def _count_lines(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    # ------------------------------------------------------------------
    # Cleaning — positional / vectorized
    # ------------------------------------------------------------------
    def _clean_data(self, chunk):
        """Vectorized clean — no iterrows."""
        chunk = chunk.dropna()
        mask = (
            ~chunk.iloc[:, 0].astype(str).str.contains('nan|None', case=False, na=False) &
            ~chunk.iloc[:, 1].astype(str).str.contains('nan|None', case=False, na=False)
        )
        return chunk[mask]

    # ------------------------------------------------------------------
    # Mappings — vectorized, no iterrows
    # ------------------------------------------------------------------
    def build_mappings(self):
        self._log("Building gene mappings...")
        self._update_progress(5, "Step 1/4: Building gene mappings...")

        sep          = self._detect_separator()
        total_lines  = self._count_lines()
        total_chunks = max(1, total_lines // self.chunksize)

        chunk_reader = pd.read_csv(
            self.filepath, sep=sep, header=None,
            chunksize=self.chunksize, dtype=str,
            engine='python', skipinitialspace=True,
            usecols=[0, 1]
        )

        gene_counter = 0
        chunk_count  = 0

        for chunk in tqdm(chunk_reader, total=total_chunks, desc="Processing network"):
            chunk = self._clean_data(chunk)

            # Vectorized: collect all unique genes from both columns at once
            genes = pd.concat([
                chunk.iloc[:, 0].str.strip(),
                chunk.iloc[:, 1].str.strip()
            ]).unique()

            for g in genes:
                if g and g not in self.gene_dict:
                    self.gene_dict[g] = gene_counter
                    gene_counter += 1

            chunk_count += 1
            self._update_progress(
                5 + min(15, int(15 * chunk_count / total_chunks)),
                f"Mapping chunk {chunk_count}/{total_chunks}..."
            )

        self._log(f"Total unique genes: {len(self.gene_dict)}")
        self._update_progress(20, f"Found {len(self.gene_dict)} unique genes")

    # ------------------------------------------------------------------
    # Sparse matrix — vectorized, no iterrows
    # ------------------------------------------------------------------
    def construct_sparse_matrix(self):
        self._log("Constructing gene-gene sparse matrix...")
        self._update_progress(25, "Step 2/4: Constructing sparse matrix...")

        sep          = self._detect_separator()
        total_lines  = self._count_lines()
        total_chunks = max(1, total_lines // self.chunksize)

        chunk_reader = pd.read_csv(
            self.filepath, sep=sep, header=None,
            chunksize=self.chunksize, dtype=str,
            engine='python', skipinitialspace=True
        )

        all_rows, all_cols, all_data = [], [], []
        chunk_count = 0

        gene_map = self.gene_dict   # local ref for speed

        for chunk in tqdm(chunk_reader, total=total_chunks, desc="Building matrix"):
            chunk = self._clean_data(chunk)

            # Vectorized column extraction
            col0 = chunk.iloc[:, 0].str.strip()
            col1 = chunk.iloc[:, 1].str.strip()

            # Weight column — vectorized coerce
            if chunk.shape[1] > 2:
                weights = pd.to_numeric(chunk.iloc[:, 2], errors='coerce').fillna(1.0).values
            else:
                weights = np.ones(len(chunk))

            # Map gene names → indices (vectorized via pandas map)
            r = col0.map(gene_map)
            c = col1.map(gene_map)

            # Drop rows where either gene is unknown
            valid = r.notna() & c.notna()
            r, c, w = r[valid].astype(int).values, c[valid].astype(int).values, weights[valid.values]

            all_rows.extend(r.tolist())
            all_cols.extend(c.tolist())
            all_data.extend(w.tolist())

            chunk_count += 1
            self._update_progress(
                25 + min(15, int(15 * chunk_count / total_chunks)),
                f"Matrix chunk {chunk_count}/{total_chunks}..."
            )

        if all_rows:
            n    = len(self.gene_dict)
            temp = csr_matrix((all_data, (all_rows, all_cols)), shape=(n, n))
            diff = (temp - temp.T).nnz
            self.matrix = temp if diff == 0 else (temp + temp.T) / 2
            assert self.matrix is not None
            self._log(f"Sparse matrix: {self.matrix.shape}, nnz={self.matrix.nnz}")
            self._update_progress(40, f"Matrix ready: {self.matrix.shape[0]} genes")  # type: ignore[index]
        else:
            raise ValueError("No valid data found to construct matrix")


# ------------------------------------------------------------------
# Batch worker — vectorized spearmanr on a sub-matrix slice
# ------------------------------------------------------------------
def _calculate_spearman_batch(args):
    """
    Compute Spearman correlations for a contiguous row slice [i_start, i_end).
    Uses scipy.stats.spearmanr on a dense sub-matrix — one call per batch row,
    not one call per pair.
    """
    from scipy.sparse import csr_matrix as _csr
    from scipy.stats import spearmanr as _spearmanr

    matrix_data, i_start, i_end, n, threshold = args
    nd, indices, indptr = matrix_data
    matrix = _csr((nd, indices, indptr), shape=(n, n))

    # Dense sub-matrix for the row slice
    sub = matrix[i_start:i_end].toarray()      # shape: (batch, n)

    rows_out, cols_out, vals_out = [], [], []

    for local_i, i in enumerate(range(i_start, i_end)):
        vec_i = sub[local_i]                   # already dense numpy array

        # Compute correlation of vec_i against ALL j > i in one spearmanr call
        # spearmanr(a, b) where b is a 2-D array computes column-wise
        j_range = np.arange(i + 1, n)
        if len(j_range) == 0:
            continue

        sub_j = matrix[j_range].toarray()      # shape: (n-i-1, n)

        # Stack vec_i as first row and compute
        combined = np.vstack([vec_i, sub_j])   # (1 + n-i-1, n) → transpose → (n, n-i)
        # spearmanr on columns: each column = one gene's profile
        rho, _ = _spearmanr(combined.T)        # rho shape: (n-i, n-i)

        # rho[0, k] = correlation between vec_i and sub_j[k-1]
        if np.isscalar(rho):
            # Only 2 variables — rho is a scalar
            if not np.isnan(rho) and abs(rho) >= threshold:  # type: ignore[arg-type]
                rows_out.append(i)
                cols_out.append(int(j_range[0]))
                vals_out.append(float(rho))  # type: ignore[arg-type]
        else:
            corrs = rho[0, 1:]                 # type: ignore[index]  # correlations of vec_i vs each j
            for k, corr in enumerate(corrs):
                if not np.isnan(corr) and abs(corr) >= threshold:
                    rows_out.append(i)
                    cols_out.append(int(j_range[k]))
                    vals_out.append(float(corr))

    return rows_out, cols_out, vals_out


# ------------------------------------------------------------------
# Main analyzer
# ------------------------------------------------------------------
class SpearmanCorrelationAnalyzer(BaseGeneCorrelationAnalyzer):

    def compute_correlations(self, threshold=0.0, row_batch=50):
        """
        Vectorized Spearman computation.
        Processes `row_batch` rows at a time per worker instead of one pair at a time.

        Args:
            threshold  : minimum |r| to store
            row_batch  : rows per parallel task (tune for memory vs parallelism)
        """
        self._log(f"Computing Spearman correlations (threshold={threshold}, batch={row_batch})...")
        self._update_progress(45, "Step 3/4: Computing Spearman correlations...")

        assert self.matrix is not None
        n           = self.matrix.shape[0]  # type: ignore[index]
        total_pairs = n * (n - 1) // 2
        self._log(f"Genes: {n}, pairs: {total_pairs}")

        matrix_csr  = self.matrix.tocsr()
        matrix_data = (matrix_csr.data, matrix_csr.indices, matrix_csr.indptr)

        # Build tasks: each task = one row-slice
        tasks = [
            (matrix_data, i, min(i + row_batch, n), n, threshold)
            for i in range(0, n, row_batch)
        ]
        total_tasks = len(tasks)
        self._log(f"Tasks: {total_tasks} row-batches of {row_batch}")

        # Collect COO triplets — much faster than lil_matrix random writes
        all_rows, all_cols, all_vals = [], [], []

        original_cb            = self.progress_callback
        self.progress_callback = None

        try:
            with Pool(cpu_count()) as pool:
                for task_idx, (r, c, v) in enumerate(
                    pool.imap_unordered(_calculate_spearman_batch, tasks)
                ):
                    all_rows.extend(r); all_cols.extend(c); all_vals.extend(v)
                    if (task_idx + 1) % 10 == 0 or (task_idx + 1) == total_tasks:
                        self._update_progress(
                            50 + int(40 * (task_idx + 1) / total_tasks),
                            f"Task {task_idx+1}/{total_tasks}, pairs so far: {len(all_vals)}"
                        )
        finally:
            self.progress_callback = original_cb

        # Build symmetric CSR in one shot — much faster than lil_matrix
        if all_rows:
            # Add symmetric pairs
            sym_rows = all_rows + all_cols
            sym_cols = all_cols + all_rows
            sym_vals = all_vals + all_vals
            self.correlation_matrix = csr_matrix(
                (sym_vals, (sym_rows, sym_cols)), shape=(n, n)
            )
        else:
            self.correlation_matrix = csr_matrix((n, n))

        significant = len(all_vals)
        self._log(f"Significant pairs (|r| >= {threshold}): {significant}")
        self._update_progress(90, f"Found {significant} significant correlations")
        return self.correlation_matrix

    def save_results(self, output_file, threshold=0.0):
        self._log(f"Saving results to {output_file}...")
        self._update_progress(95, "Step 4/4: Saving results...")

        keys = list(self.gene_dict.keys())
        coo  = self.correlation_matrix.tocoo()

        saved_pairs = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("source\ttarget\tweight\n")
            for i, j, v in zip(coo.row, coo.col, coo.data):
                if i < j and abs(v) >= threshold:   # upper triangle only
                    f.write(f"{keys[i]}\t{keys[j]}\t{v:.6f}\n")
                    saved_pairs += 1

        self._log(f"Saved {saved_pairs} pairs to {output_file}")
        self._update_progress(100, f"Done! Saved {saved_pairs} correlations")
        return saved_pairs


# ------------------------------------------------------------------
# Flask helper
# ------------------------------------------------------------------
def run_spearman_analysis(input_file, output_dir, threshold=0.0, chunksize=10000,
                          logger=None, progress_callback=None):
    start_time = time.time()

    analyzer = SpearmanCorrelationAnalyzer(input_file, chunksize=chunksize, logger=logger)
    if progress_callback:
        analyzer.set_progress_callback(progress_callback)

    task_id         = str(uuid.uuid4())[:8]
    output_filename = os.path.join(output_dir, f"spearman_backbone_{task_id}.tsv")

    try:
        analyzer.build_mappings()
        analyzer.construct_sparse_matrix()
        analyzer.compute_correlations(threshold=threshold)
        saved_pairs = analyzer.save_results(output_filename, threshold=threshold)

        elapsed = time.time() - start_time
        n       = analyzer.matrix.shape[0] if analyzer.matrix is not None else 0  # type: ignore[union-attr]
        max_p   = n * (n - 1) / 2 if n > 1 else 1

        stats = {
            'success':           True,
            'genes':             len(analyzer.gene_dict),
            'significant_pairs': saved_pairs,
            'threshold':         threshold,
            'execution_time':    round(elapsed, 2),
            'method':            'spearman',
            'nodes_in_graph':    n,
            'edges_in_graph':    analyzer.matrix.nnz if analyzer.matrix is not None else 0,
            'density':           round(saved_pairs / max_p, 6),
        }
        return stats, output_filename

    except Exception as e:
        if logger:
            logger.error(f"Spearman analysis failed: {e}")
            logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}, None