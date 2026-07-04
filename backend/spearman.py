"""
Spearman Correlation Analyzer for Gene-Gene Interactions.
Optimized version, vectorized matrix computation instead of pair-by-pair loops.
Flask-compatible.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pvalue_correction import correct
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
        """Vectorized clean, no iterrows. Only inspects the first two columns so
        a trailing separator / all-NaN extra column doesn't drop every row."""
        if chunk.shape[1] < 2:
            return chunk.iloc[0:0]
        mask = (
            chunk.iloc[:, 0].notna() & chunk.iloc[:, 1].notna() &
            ~chunk.iloc[:, 0].astype(str).str.strip().str.lower().isin(['nan', 'none', '']) &
            ~chunk.iloc[:, 1].astype(str).str.strip().str.lower().isin(['nan', 'none', ''])
        )
        return chunk[mask]

    def build_mappings(self):
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

        gene_counter = 0
        chunk_count  = 0

        for chunk in tqdm(chunk_reader, total=total_chunks, desc="Processing network"):
            chunk = self._clean_data(chunk)

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

    def construct_sparse_matrix(self):
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

        gene_map = self.gene_dict

        for chunk in tqdm(chunk_reader, total=total_chunks, desc="Building matrix"):
            chunk = self._clean_data(chunk)

            col0 = chunk.iloc[:, 0].str.strip()
            col1 = chunk.iloc[:, 1].str.strip()

            if chunk.shape[1] > 2:
                weights = pd.to_numeric(chunk.iloc[:, 2], errors='coerce').fillna(1.0).values
            else:
                weights = np.ones(len(chunk))

            r = col0.map(gene_map)
            c = col1.map(gene_map)

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
            self._update_progress(40, f"Matrix ready: {self.matrix.shape[0]} genes")
        else:
            raise ValueError("No valid data found to construct matrix")

def _calculate_spearman_batch(args):
    """
    Compute Spearman correlations for a contiguous row slice [i_start, i_end).
    Uses scipy.stats.spearmanr on a dense sub-matrix, one call per batch row,
    not one call per pair.
    """
    from scipy.sparse import csr_matrix as _csr
    from scipy.stats import spearmanr as _spearmanr

    matrix_data, i_start, i_end, n, threshold = args
    nd, indices, indptr = matrix_data
    matrix = _csr((nd, indices, indptr), shape=(n, n))

    sub = matrix[i_start:i_end].toarray()

    rows_out, cols_out, vals_out = [], [], []

    for local_i, i in enumerate(range(i_start, i_end)):
        vec_i = sub[local_i]

        j_range = np.arange(i + 1, n)
        if len(j_range) == 0:
            continue

        sub_j = matrix[j_range].toarray()

        combined = np.vstack([vec_i, sub_j])
        rho, _ = _spearmanr(combined.T)

        if np.isscalar(rho):
            if not np.isnan(rho) and abs(rho) >= threshold:
                rows_out.append(i)
                cols_out.append(int(j_range[0]))
                vals_out.append(float(rho))
        else:
            corrs = rho[0, 1:]
            for k, corr in enumerate(corrs):
                if not np.isnan(corr) and abs(corr) >= threshold:
                    rows_out.append(i)
                    cols_out.append(int(j_range[k]))
                    vals_out.append(float(corr))

    return rows_out, cols_out, vals_out

class SpearmanCorrelationAnalyzer(BaseGeneCorrelationAnalyzer):

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
        """Spearman rank correlation between every pair of ROWS of `matrix`; keep
        edges significant at `alpha` under the chosen multiple-testing `correction`
        ('none' | 'fdr_bh' | 'bonferroni'). Returns
        [(labels[i], labels[j], rho, p), ...]."""
        from scipy.stats import spearmanr
        dense = matrix.toarray()
        n = dense.shape[0]
        if n < 2 or dense.shape[1] < 2:
            return []
        rho, pval = spearmanr(dense, axis=1)
        rho  = np.asarray(rho, dtype=float)
        pval = np.asarray(pval, dtype=float)

        m = 0
        cand = []
        if rho.ndim == 0:
            v, p = float(rho), float(pval)
            if not (np.isnan(v) or np.isnan(p)):
                m = 1
                if p < alpha:
                    cand.append((0, 1, v, p))
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    v, p = rho[i, j], pval[i, j]
                    if np.isnan(v) or np.isnan(p):
                        continue
                    m += 1
                    if p < alpha:
                        cand.append((i, j, float(v), float(p)))

        kept = correct(cand, m, alpha, correction)
        return [(labels[i], labels[j], v, p) for (i, j, v, p) in kept]

    @staticmethod
    def _write_edges(edges, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("source\ttarget\tspearman_rho\tp_value\n")
            for a, b, v, p in edges:
                f.write(f"{a}\t{b}\t{v:.6f}\t{p:.6g}\n")

    def run(self, type1_output, type2_output=None, alpha=0.05, correction='fdr_bh'):
        """Compute BOTH one-mode Spearman projections, keeping edges significant at
        `alpha` under the chosen multiple-testing `correction`. type2_output=None →
        only the type-1 projection is written."""
        start = time.time()
        self.build_incidence()

        gene_labels    = list(self.gene_dict.keys())
        disease_labels = list(self.disease_dict.keys())

        self._log(f"Type-1 projection (Spearman over type-2 profiles, alpha={alpha}, correction={correction})...")
        self._update_progress(40, "Step 2/3: Type-1 projection...")
        type1_edges = self._project(self.incidence, gene_labels, alpha, correction)
        self._write_edges(type1_edges, type1_output)
        self._log(f"Type-1: {len(gene_labels)} nodes, {len(type1_edges)} edges")

        type2_edges = []
        if type2_output:
            self._log(f"Type-2 projection (Spearman over type-1 profiles, alpha={alpha}, correction={correction})...")
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
            'method': 'spearman',
            'columns': ['source', 'target', 'spearman_rho', 'p_value'],
            'dual_output': type2_output is not None,
            'execution_time': round(time.time() - start, 2),
        }

def run_spearman_analysis(input_file, output_dir, alpha=0.05, correction='fdr_bh',
                          chunksize=10000, logger=None, progress_callback=None):
    """Run Spearman analysis, produces BOTH one-mode projections, keeping edges
    significant at `alpha` under the chosen `correction`
    ('none' | 'fdr_bh' | 'bonferroni').
    Returns (stats_dict, type1_output, type2_output)."""
    task_id      = str(uuid.uuid4())[:8]
    type1_output = os.path.join(output_dir, f"spearman_{task_id}_type1.tsv")
    type2_output = os.path.join(output_dir, f"spearman_{task_id}_type2.tsv")

    analyzer = SpearmanCorrelationAnalyzer(input_file, chunksize=chunksize, logger=logger)
    if progress_callback:
        analyzer.set_progress_callback(progress_callback)

    try:
        stats = analyzer.run(type1_output, type2_output, alpha=alpha, correction=correction)
        return stats, type1_output, type2_output
    except Exception as e:
        if logger:
            logger.error(f"Spearman analysis failed: {e}")
            logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}, None, None