"""
Kendall Correlation Analyzer for Gene-Gene Interactions.
Computes Kendall rank correlation coefficients between genes.
Flask-compatible version.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from scipy.sparse import lil_matrix, csr_matrix, save_npz
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
import time
import uuid
import traceback
import functools

warnings.filterwarnings('ignore')


class BaseGeneCorrelationAnalyzer:
    """
    Base class for gene correlation analysis.
    Provides common functionality for building mappings and constructing sparse matrices.
    """

    def __init__(self, filepath, chunksize=10000, logger=None):
        self.filepath  = filepath
        self.chunksize = chunksize
        self.gene_dict = {}
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
    # Separator auto-detection
    # ------------------------------------------------------------------
    def _detect_separator(self):
        """
        Reads the first line of the file and returns the most likely separator.
        Supports tab, comma, semicolon, or any whitespace.
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        if '\t' in first_line:
            sep = '\t'
        elif ',' in first_line:
            sep = ','
        elif ';' in first_line:
            sep = ';'
        else:
            sep = r'\s+'          # fallback: any whitespace / space-separated
        self._log(f"Detected separator: {repr(sep)}")
        return sep

    def _count_lines(self):
        """Count total lines in file (all rows — no header assumed)."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    # ------------------------------------------------------------------
    # Cleaning  — positional access only
    # ------------------------------------------------------------------
    def _clean_data(self, chunk):
        """
        Drop NaN rows and rows where col-0 or col-1 contain 'nan'/'None'.
        Uses iloc so it works regardless of column names.
        """
        chunk = chunk.dropna()
        chunk = chunk[~chunk.iloc[:, 0].astype(str).str.contains('nan|None', case=False, na=False)]
        chunk = chunk[~chunk.iloc[:, 1].astype(str).str.contains('nan|None', case=False, na=False)]
        return chunk

    # ------------------------------------------------------------------
    # Mappings
    # ------------------------------------------------------------------
    def _process_chunk(self, chunk):
        """Build a local gene dict from one chunk using positional access."""
        chunk = self._clean_data(chunk)
        local = {}
        for _, row in chunk.iterrows():
            g1 = str(row.iloc[0]).strip()
            g2 = str(row.iloc[1]).strip()
            if not g1 or not g2:
                continue
            if g1 not in local:
                local[g1] = len(local)
            if g2 not in local:
                local[g2] = len(local)
        return local

    def build_mappings(self):
        """
        Reads file in chunks, builds gene → int index mappings.
        Uses positional access (iloc) — no predefined column names required.
        """
        self._log("Building gene mappings...")
        self._update_progress(5, "Step 1/4: Building gene mappings...")

        sep          = self._detect_separator()
        total_lines  = self._count_lines()
        total_chunks = max(1, total_lines // self.chunksize)

        chunk_reader = pd.read_csv(
            self.filepath, sep=sep, header=None,
            chunksize=self.chunksize, dtype=str,
            engine='python', skipinitialspace=True,
            usecols=[0, 1]          # only first two columns; ignores trailing separators
        )

        results     = []
        chunk_count = 0

        for chunk in tqdm(chunk_reader, total=total_chunks, desc="Processing network"):
            results.append(self._process_chunk(chunk))
            chunk_count += 1
            self._update_progress(
                5 + min(15, int(15 * chunk_count / total_chunks)),
                f"Mapping chunk {chunk_count}/{total_chunks}..."
            )

        gene_counter = 0
        for g_dict in results:
            for gene in g_dict:
                if gene not in self.gene_dict:
                    self.gene_dict[gene] = gene_counter
                    gene_counter += 1

        self._log(f"Total unique genes: {len(self.gene_dict)}")
        self._update_progress(20, f"Found {len(self.gene_dict)} unique genes")

    # ------------------------------------------------------------------
    # Sparse matrix
    # ------------------------------------------------------------------
    def _build_sparse_chunk(self, chunk):
        """
        Build COO triplets from one chunk.
        Uses iloc[0], iloc[1], iloc[2] — positional, no column names.
        """
        chunk = self._clean_data(chunk)
        rows, cols, data = [], [], []

        for _, row in chunk.iterrows():
            g1 = str(row.iloc[0]).strip()
            g2 = str(row.iloc[1]).strip()

            # Weight: third column if it exists, else 1.0
            try:
                weight = float(row.iloc[2]) if len(row) > 2 else 1.0
            except (ValueError, TypeError):
                weight = 1.0

            if not g1 or not g2:
                continue
            try:
                rows.append(self.gene_dict[g1])
                cols.append(self.gene_dict[g2])
                data.append(weight)
            except KeyError:
                continue

        return rows, cols, data

    def construct_sparse_matrix(self):
        """
        Constructs the gene × gene weighted sparse matrix.
        Uses positional access (iloc) — no predefined column names required.
        """
        self._log("Constructing gene-gene sparse matrix...")
        self._update_progress(25, "Step 2/4: Constructing sparse matrix...")

        sep          = self._detect_separator()
        total_lines  = self._count_lines()
        total_chunks = max(1, total_lines // self.chunksize)

        chunk_reader = pd.read_csv(
            self.filepath, sep=sep, header=None,
            chunksize=self.chunksize, dtype=str,
            engine='python', skipinitialspace=True
            # no usecols here — we may need col 2 for weight
        )

        all_rows, all_cols, all_data = [], [], []
        chunk_count = 0

        for chunk in tqdm(chunk_reader, total=total_chunks, desc="Building matrix"):
            r, c, d = self._build_sparse_chunk(chunk)
            all_rows.extend(r)
            all_cols.extend(c)
            all_data.extend(d)
            chunk_count += 1
            self._update_progress(
                25 + min(15, int(15 * chunk_count / total_chunks)),
                f"Matrix chunk {chunk_count}/{total_chunks}..."
            )

        if all_rows:
            n = len(self.gene_dict)
            temp = csr_matrix((all_data, (all_rows, all_cols)), shape=(n, n))
            diff = (temp - temp.T).nnz
            if diff == 0:
                self._log("Matrix is already symmetric")
                self.matrix = temp
            else:
                self._log(f"Making matrix symmetric (diff={diff})...")
                self.matrix = (temp + temp.T) / 2
            self._log(f"Sparse matrix: {self.matrix.shape}, nnz={self.matrix.nnz}")  # type: ignore[union-attr]
            assert self.matrix is not None
            self._update_progress(40, f"Matrix ready: {self.matrix.shape[0]} genes, {self.matrix.nnz} edges")  # type: ignore[union-attr,index]
        else:
            raise ValueError("No valid data found to construct matrix")


# ------------------------------------------------------------------
# Module-level static function (picklable for multiprocessing)
# ------------------------------------------------------------------
def _calculate_kendall_static(args):
    """Compute Kendall tau for a batch of gene pairs."""
    from scipy.sparse import csr_matrix as _csr
    matrix_data, gene_pairs, threshold = args
    n, data, indices, indptr = matrix_data
    matrix = _csr((data, indices, indptr), shape=(n, n))

    results = []
    for i, j in gene_pairs:
        vec_i = matrix.getrow(i).toarray()[0]
        vec_j = matrix.getrow(j).toarray()[0]
        try:
            _kt = kendalltau(vec_i, vec_j)
            corr_val = float(_kt[0])  # type: ignore[arg-type]
            if np.isnan(corr_val):
                continue
            if abs(corr_val) < threshold:
                continue
            results.append((i, j, corr_val))
        except Exception:
            continue
    return results


# ------------------------------------------------------------------
# Main analyzer
# ------------------------------------------------------------------
class KendallCorrelationAnalyzer(BaseGeneCorrelationAnalyzer):
    """Computes Kendall rank correlations between genes."""

    def compute_correlations(self, threshold=0.0):
        self._log(f"Computing Kendall correlations (threshold={threshold})...")
        self._update_progress(45, "Step 3/4: Computing Kendall correlations...")

        assert self.matrix is not None
        n     = self.matrix.shape[0]  # type: ignore[union-attr,index]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        total_pairs = len(pairs)
        self._log(f"Total gene pairs: {total_pairs}")

        if total_pairs < 10000:
            batch_size = max(100, total_pairs // (cpu_count() * 2))
        else:
            batch_size = min(1000, total_pairs // (cpu_count() * 10))
        batch_size = max(1, batch_size)

        batches       = [pairs[i:i + batch_size] for i in range(0, total_pairs, batch_size)]
        total_batches = len(batches)
        self._log(f"Batches: {total_batches} × ~{batch_size}")

        self.correlation_matrix = lil_matrix((n, n))
        processed_pairs = 0

        assert self.matrix is not None
        matrix_csr  = self.matrix.tocsr()  # type: ignore[union-attr]
        matrix_data = (n, matrix_csr.data, matrix_csr.indices, matrix_csr.indptr)
        args        = [(matrix_data, b, threshold) for b in batches]

        original_cb            = self.progress_callback
        self.progress_callback = None          # remove before fork

        try:
            with Pool(cpu_count()) as pool:
                for idx, batch_result in enumerate(pool.imap_unordered(_calculate_kendall_static, args)):
                    for i, j, val in batch_result:
                        self.correlation_matrix[i, j] = val
                        self.correlation_matrix[j, i] = val
                        processed_pairs += 1
                    if (idx + 1) % 100 == 0 or (idx + 1) == total_batches:
                        self._update_progress(
                            50 + int(40 * (idx + 1) / total_batches),
                            f"Batch {idx+1}/{total_batches}, pairs found: {processed_pairs}"
                        )
        finally:
            self.progress_callback = original_cb

        significant = self.correlation_matrix.nnz // 2
        self._log(f"Significant pairs (|r| ≥ {threshold}): {significant}")
        self._update_progress(90, f"Found {significant} significant correlations")
        return self.correlation_matrix

    def save_results(self, output_file, threshold=0.0):
        """
        Saves results as CSV with columns: source, target, weight
        """
        self._log(f"Saving results to {output_file}...")
        self._update_progress(95, "Step 4/4: Saving results...")

        keys        = list(self.gene_dict.keys())
        saved_pairs = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("source\ttarget\tweight\n")          # TSV header
            assert self.correlation_matrix is not None
            for i in range(self.correlation_matrix.shape[0]):  # type: ignore[union-attr,index]
                for j in range(i + 1, self.correlation_matrix.shape[1]):  # type: ignore[union-attr,index]
                    corr = self.correlation_matrix[i, j]
                    if corr != 0 and abs(corr) >= threshold:
                        f.write(f"{keys[i]}\t{keys[j]}\t{corr:.6f}\n")
                        saved_pairs += 1

        self._log(f"Saved {saved_pairs} pairs to {output_file}")
        self._update_progress(100, f"Done! Saved {saved_pairs} correlations")
        return saved_pairs


# ------------------------------------------------------------------
# Flask helper
# ------------------------------------------------------------------
def run_kendall_analysis(input_file, output_dir, threshold=0.0, chunksize=10000,
                         logger=None, progress_callback=None):
    """
    Run Kendall correlation analysis — entry point for Flask.

    Returns:
        tuple: (stats_dict, output_filename)
    """
    start_time = time.time()

    analyzer = KendallCorrelationAnalyzer(input_file, chunksize=chunksize, logger=logger)
    if progress_callback:
        analyzer.set_progress_callback(progress_callback)

    task_id         = str(uuid.uuid4())[:8]
    output_filename = os.path.join(output_dir, f"kendall_backbone_{task_id}.tsv")

    try:
        analyzer.build_mappings()
        analyzer.construct_sparse_matrix()
        analyzer.compute_correlations(threshold=threshold)
        saved_pairs = analyzer.save_results(output_filename, threshold=threshold)

        elapsed = time.time() - start_time
        _m = analyzer.matrix
        n       = _m.shape[0] if _m is not None else 0  # type: ignore[union-attr,index]
        max_p   = n * (n - 1) / 2 if n > 1 else 1

        stats = {
            'success':          True,
            'genes':            len(analyzer.gene_dict),
            'significant_pairs': saved_pairs,
            'threshold':        threshold,
            'execution_time':   round(elapsed, 2),
            'method':           'kendall',
            'nodes_in_graph':   n,
            'edges_in_graph':   analyzer.matrix.nnz if analyzer.matrix is not None else 0,
            'density':          round(saved_pairs / max_p, 6),
        }
        return stats, output_filename

    except Exception as e:
        if logger:
            logger.error(f"Kendall analysis failed: {e}")
            logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}, None