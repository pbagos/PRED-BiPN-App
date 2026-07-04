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
from pvalue_correction import correct
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
            sep = r'\s+'
        self._log(f"Detected separator: {repr(sep)}")
        return sep

    def _count_lines(self):
        """Count total lines in file (all rows, no header assumed)."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _has_header(self, sep):
        """Row 1 is a header iff both leading cells are exact header keywords
        (so real names like 'ALZHEIMER_DISEASE' survive), or col-2 flips
        non-numeric→numeric between rows 1 and 2 (weighted files)."""
        kw = {'source', 'target', 'from', 'to', 'node', 'node1', 'node2',
              'gene', 'genes', 'gene1', 'gene2', 'disease', 'diseases',
              'src', 'dst', 'id', 'name', 'weight', 'score', 'value',
              'col1', 'col2', 'edge', 'index'}
        splitter = None if sep == r'\s+' else sep
        with open(self.filepath, 'r', encoding='utf-8') as f:
            first = f.readline().strip()
            second = f.readline().strip()
        if not first:
            return False
        parts = first.split(splitter)
        if len(parts) < 2:
            return False
        if parts[0].strip().lower() in kw and parts[1].strip().lower() in kw:
            return True
        if second and len(parts) > 2:
            p2 = second.split(splitter)
            if len(p2) > 2:
                def _num(x):
                    try:
                        float(x)
                        return True
                    except ValueError:
                        return False
                if not _num(parts[2]) and _num(p2[2]):
                    return True
        return False

    def _clean_data(self, chunk):
        """
        Drop rows where iloc[0] or iloc[1] are NaN or literally 'nan'/'None'.
        Only inspects the first two columns, trailing separators or extra
        all-NaN columns must not cause every row to be dropped.
        """
        if chunk.shape[1] < 2:
            return chunk.iloc[0:0]
        chunk = chunk[chunk.iloc[:, 0].notna() & chunk.iloc[:, 1].notna()]
        chunk = chunk[~chunk.iloc[:, 0].astype(str).str.strip().str.lower().isin(['nan', 'none', ''])]
        chunk = chunk[~chunk.iloc[:, 1].astype(str).str.strip().str.lower().isin(['nan', 'none', ''])]
        return chunk

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
        Uses positional access (iloc), no predefined column names required.
        """
        self._log("Building gene mappings...")
        self._update_progress(5, "Step 1/4: Building gene mappings...")

        sep          = self._detect_separator()
        total_lines  = self._count_lines()
        total_chunks = max(1, total_lines // self.chunksize)

        chunk_reader = pd.read_csv(
            self.filepath, sep=sep, header=None,
            skiprows=1 if self._has_header(sep) else 0,
            chunksize=self.chunksize, dtype=str,
            engine='python', skipinitialspace=True,
            usecols=[0, 1]
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

    def _build_sparse_chunk(self, chunk):
        """
        Build COO triplets from one chunk.
        Uses iloc[0], iloc[1], iloc[2], positional, no column names.
        """
        chunk = self._clean_data(chunk)
        rows, cols, data = [], [], []

        for _, row in chunk.iterrows():
            g1 = str(row.iloc[0]).strip()
            g2 = str(row.iloc[1]).strip()

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
        Uses positional access (iloc), no predefined column names required.
        """
        self._log("Constructing gene-gene sparse matrix...")
        self._update_progress(25, "Step 2/4: Constructing sparse matrix...")

        sep          = self._detect_separator()
        total_lines  = self._count_lines()
        total_chunks = max(1, total_lines // self.chunksize)

        chunk_reader = pd.read_csv(
            self.filepath, sep=sep, header=None,
            skiprows=1 if self._has_header(sep) else 0,
            chunksize=self.chunksize, dtype=str,
            engine='python', skipinitialspace=True
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
            self._log(f"Sparse matrix: {self.matrix.shape}, nnz={self.matrix.nnz}")
            assert self.matrix is not None
            self._update_progress(40, f"Matrix ready: {self.matrix.shape[0]} genes, {self.matrix.nnz} edges")
        else:
            raise ValueError("No valid data found to construct matrix")

def _calculate_kendall_static(args):
    """Kendall tau + p-value between rows of a (possibly non-square) matrix for a
    batch of row-index pairs. Returns (n_valid, edges): n_valid counts pairs with a
    defined tau/p-value, edges holds (i, j, tau, p) only for p < alpha (the only
    pairs that can survive Benjamini-Hochberg given m total tests)."""
    from scipy.sparse import csr_matrix as _csr
    matrix_data, gene_pairs, alpha = args
    shape, data, indices, indptr = matrix_data
    matrix = _csr((data, indices, indptr), shape=shape)

    n_valid = 0
    edges = []
    for i, j in gene_pairs:
        vec_i = matrix.getrow(i).toarray()[0]
        vec_j = matrix.getrow(j).toarray()[0]
        try:
            tau, pval = kendalltau(vec_i, vec_j)
        except Exception:
            continue
        if np.isnan(tau) or np.isnan(pval):
            continue
        n_valid += 1
        if pval < alpha:
            edges.append((i, j, float(tau), float(pval)))
    return n_valid, edges

class KendallCorrelationAnalyzer(BaseGeneCorrelationAnalyzer):
    """Kendall rank correlation over BOTH one-mode projections of a bipartite
    input: col-0 nodes correlated over their col-1 profiles, and vice-versa."""

    def build_incidence(self):
        """Build a weighted type-1 × type-2 incidence matrix (col-0 = type-1,
        col-1 = type-2, optional col-2 = weight). Sets self.gene_dict (type-1),
        self.disease_dict (type-2) and self.incidence."""
        self._log("Building bipartite incidence matrix...")
        self._update_progress(10, "Step 1/3: Building incidence matrix...")

        sep  = self._detect_separator()
        skip = 1 if self._has_header(sep) else 0
        self.gene_dict = {}
        self.disease_dict = {}
        rows, cols, data = [], [], []

        reader = pd.read_csv(
            self.filepath, sep=sep, header=None, skiprows=skip,
            chunksize=self.chunksize, dtype=str,
            engine='python', skipinitialspace=True
        )
        for chunk in reader:
            chunk = self._clean_data(chunk)
            for _, row in chunk.iterrows():
                g = str(row.iloc[0]).strip()
                d = str(row.iloc[1]).strip()
                if not g or not d:
                    continue
                try:
                    w = float(row.iloc[2]) if len(row) > 2 else 1.0
                    if not np.isfinite(w):
                        w = 1.0
                except (ValueError, TypeError):
                    w = 1.0
                gi = self.gene_dict.setdefault(g, len(self.gene_dict))
                di = self.disease_dict.setdefault(d, len(self.disease_dict))
                rows.append(gi); cols.append(di); data.append(w)

        if not rows:
            raise ValueError("No valid data found to construct matrix")

        self.incidence = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.gene_dict), len(self.disease_dict))
        )
        self._log(f"Incidence: {self.incidence.shape[0]} type-1 x "
                  f"{self.incidence.shape[1]} type-2 nodes, nnz={self.incidence.nnz}")

    def _project(self, matrix, labels, alpha, correction):
        """Kendall tau between every pair of ROWS of `matrix`; keep edges that are
        significant at `alpha` under the chosen multiple-testing `correction`
        ('none' | 'fdr_bh' | 'bonferroni'). Returns
        [(labels[i], labels[j], tau, p), ...]."""
        matrix = matrix.tocsr()
        n = matrix.shape[0]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if not pairs:
            return []

        total = len(pairs)
        if total < 10000:
            batch_size = max(100, total // (cpu_count() * 2))
        else:
            batch_size = min(1000, total // (cpu_count() * 10))
        batch_size = max(1, batch_size)
        batches = [pairs[i:i + batch_size] for i in range(0, total, batch_size)]

        matrix_data = (matrix.shape, matrix.data, matrix.indices, matrix.indptr)
        args = [(matrix_data, b, alpha) for b in batches]

        m = 0
        cand = []
        cb = self.progress_callback
        self.progress_callback = None
        try:
            with Pool(cpu_count()) as pool:
                for n_valid, edges in pool.imap_unordered(_calculate_kendall_static, args):
                    m += n_valid
                    cand.extend(edges)
        finally:
            self.progress_callback = cb

        kept = correct(cand, m, alpha, correction)
        return [(labels[i], labels[j], tau, p) for (i, j, tau, p) in kept]

    @staticmethod
    def _write_edges(edges, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("source\ttarget\tkendall_tau\tp_value\n")
            for a, b, tau, p in edges:
                f.write(f"{a}\t{b}\t{tau:.6f}\t{p:.6g}\n")

    def run(self, type1_output, type2_output=None, alpha=0.05, correction='fdr_bh'):
        """Compute BOTH one-mode Kendall projections, keeping edges significant at
        `alpha` under the chosen multiple-testing `correction`. type2_output=None →
        only the type-1 projection is written."""
        start = time.time()
        self.build_incidence()

        gene_labels    = list(self.gene_dict.keys())
        disease_labels = list(self.disease_dict.keys())

        self._log(f"Type-1 projection (Kendall over type-2 profiles, alpha={alpha}, correction={correction})...")
        self._update_progress(40, "Step 2/3: Type-1 projection...")
        type1_edges = self._project(self.incidence, gene_labels, alpha, correction)
        self._write_edges(type1_edges, type1_output)
        self._log(f"Type-1: {len(gene_labels)} nodes, {len(type1_edges)} edges")

        type2_edges = []
        if type2_output:
            self._log(f"Type-2 projection (Kendall over type-1 profiles, alpha={alpha}, correction={correction})...")
            self._update_progress(70, "Step 3/3: Type-2 projection...")
            type2_edges = self._project(self.incidence.T.tocsr(), disease_labels, alpha, correction)
            self._write_edges(type2_edges, type2_output)
            self._log(f"Type-2: {len(disease_labels)} nodes, {len(type2_edges)} edges")

        self._update_progress(100, "Done")
        return {
            'success': True,
            'nodes_type1': len(gene_labels),
            'nodes_type2': len(disease_labels),
            'edges_type1': len(type1_edges),
            'edges_type2': len(type2_edges),
            'alpha': alpha,
            'correction': correction,
            'method': 'kendall',
            'columns': ['source', 'target', 'kendall_tau', 'p_value'],
            'dual_output': type2_output is not None,
            'execution_time': round(time.time() - start, 2),
        }

def run_kendall_analysis(input_file, output_dir, alpha=0.05, correction='fdr_bh',
                         chunksize=10000, logger=None, progress_callback=None):
    """
    Run Kendall correlation analysis, entry point for Flask.
    Produces BOTH one-mode projections of the bipartite input, keeping edges
    significant at `alpha` under the chosen `correction`
    ('none' | 'fdr_bh' | 'bonferroni').

    Returns:
        tuple: (stats_dict, type1_output, type2_output)
    """
    task_id      = str(uuid.uuid4())[:8]
    type1_output = os.path.join(output_dir, f"kendall_{task_id}_type1.tsv")
    type2_output = os.path.join(output_dir, f"kendall_{task_id}_type2.tsv")

    analyzer = KendallCorrelationAnalyzer(input_file, chunksize=chunksize, logger=logger)
    if progress_callback:
        analyzer.set_progress_callback(progress_callback)

    try:
        stats = analyzer.run(type1_output, type2_output, alpha=alpha, correction=correction)
        return stats, type1_output, type2_output
    except Exception as e:
        if logger:
            logger.error(f"Kendall analysis failed: {e}")
            logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}, None, None