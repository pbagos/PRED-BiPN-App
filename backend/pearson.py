"""
Pearson Correlation Analyzer for Gene-Gene Interactions.
Computes Pearson correlation coefficients between genes.
Flask-compatible version.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.sparse import lil_matrix, csr_matrix
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
        self._col0_set          = set()
        self._col1_set          = set()
        self.matrix             = None
        self.correlation_matrix = None
        self.pvalue_matrix      = None
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
        """Auto-detects separator from the first line of the file."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            line = f.readline()
        if '\t' in line:   sep = '\t'
        elif ',' in line:  sep = ','
        elif ';' in line:  sep = ';'
        else:              sep = r'\s+'
        self._log(f"Detected separator: {repr(sep)}")
        return sep

    def _has_header(self, sep):
        """First-line check: if cols 0 and 1 are both non-numeric header
        keywords (source/target/from/to/node/gene/...) treat row as header."""
        kw = {'source', 'target', 'from', 'to', 'node', 'node1', 'node2',
              'gene', 'genes', 'gene1', 'gene2', 'disease', 'diseases',
              'src', 'dst', 'id', 'name', 'weight', 'score', 'value',
              'col1', 'col2', 'edge', 'index'}
        with open(self.filepath, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
        if not line:
            return False
        parts = line.split(sep if sep != r'\s+' else None)
        if len(parts) < 2:
            return False
        return parts[0].strip().lower() in kw and parts[1].strip().lower() in kw

    def _count_lines(self):
        """Count all lines in file (header=None, no line subtracted)."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

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
        """Build a local gene dict from one chunk, positional access only.
        Tracks which entities appear in column 0 vs column 1 so we can later
        decide if the input is bipartite or one-mode."""
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
            self._col0_set.add(g1)
            self._col1_set.add(g2)
        return local

    def build_mappings(self):
        """
        Reads file in chunks and builds gene → int index mappings.
        Uses positional access (iloc), no predefined column names required.
        """
        self._log("Building gene mappings...")
        self._update_progress(5, "Step 1/4: Building gene mappings...")

        sep          = self._detect_separator()
        total_lines  = self._count_lines()
        skip         = 1 if self._has_header(sep) else 0
        if skip:
            self._log("Header row detected; skipping line 1.")
        total_chunks = max(1, (total_lines - skip) // self.chunksize)

        chunk_reader = pd.read_csv(
            self.filepath, sep=sep, header=None,
            skiprows=skip,
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
            weight = 1.0
            if len(row) > 2:
                try:
                    w = float(row.iloc[2])
                    if not np.isnan(w) and w > 0:
                        weight = w
                except (ValueError, TypeError):
                    pass
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
        skip         = 1 if self._has_header(sep) else 0
        total_chunks = max(1, (total_lines - skip) // self.chunksize)

        chunk_reader = pd.read_csv(
            self.filepath, sep=sep, header=None,
            skiprows=skip,
            chunksize=self.chunksize, dtype=str,
            engine='python', skipinitialspace=True
        )

        all_rows, all_cols, all_data = [], [], []
        chunk_count = 0

        for chunk in tqdm(chunk_reader, total=total_chunks, desc="Building matrix"):
            r, c, d = self._build_sparse_chunk(chunk)
            all_rows.extend(r); all_cols.extend(c); all_data.extend(d)
            chunk_count += 1
            self._update_progress(
                25 + min(15, int(15 * chunk_count / total_chunks)),
                f"Matrix chunk {chunk_count}/{total_chunks}..."
            )

        if all_rows:
            n    = len(self.gene_dict)
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
            self._update_progress(40, f"Matrix ready: {self.matrix.shape[0]} genes")
        else:
            raise ValueError("No valid data found to construct matrix")

def _calculate_pearson_static(args):
    """Compute Pearson r for a batch of gene pairs.

    `threshold` here is interpreted as a p-value cutoff: pairs with
    p_value >= threshold are dropped. Set threshold >= 1.0 to keep all.
    """
    from scipy.sparse import csr_matrix as _csr
    from scipy.stats import pearsonr as _pearsonr
    matrix_data, gene_pairs, threshold = args
    n, data, indices, indptr = matrix_data
    matrix = _csr((data, indices, indptr), shape=(n, n))

    results = []
    for i, j in gene_pairs:
        vec_i = matrix.getrow(i).toarray()[0]
        vec_j = matrix.getrow(j).toarray()[0]
        try:
            _pr = _pearsonr(vec_i, vec_j)
            corr_val = float(_pr[0])
            p_val    = float(_pr[1])
            if np.isnan(corr_val) or np.isnan(p_val):
                continue
            if p_val >= threshold:
                continue
            results.append((i, j, corr_val, p_val))
        except Exception:
            continue
    return results

class PearsonCorrelationAnalyzer(BaseGeneCorrelationAnalyzer):
    """Computes Pearson correlations between genes."""

    def compute_correlations(self, threshold=0.05, output_file=None):
        """
        Stream gene-gene Pearson correlations directly to `output_file`.
        Only pairs (i, j) where BOTH i and j appeared in column 0 (genes) are
        computed, gene-disease and disease-disease pairs are skipped.

        `threshold` is a p-value cutoff: pairs with p_value >= threshold
        are dropped before writing. Default 0.05 (classic significance level).
        Set threshold >= 1.0 to keep every pair.
        """
        self._log(f"Computing Pearson correlations (p-value cutoff < {threshold})...")
        self._update_progress(45, "Step 3/4: Computing Pearson correlations...")

        assert self.matrix is not None
        n = self.matrix.shape[0]

        self._log(f"[DIAG] Matrix shape: {self.matrix.shape}, nnz: {self.matrix.nnz}")
        self._log(f"[DIAG] gene_dict size: {len(self.gene_dict)}")
        self._log(f"[DIAG] _col0_set size: {len(self._col0_set)}")
        if self._col0_set:
            sample_col0 = list(self._col0_set)[:5]
            self._log(f"[DIAG] _col0_set sample: {sample_col0}")

        overlap = self._col0_set & self._col1_set
        smaller = min(len(self._col0_set), len(self._col1_set)) or 1
        overlap_ratio = len(overlap) / smaller
        if overlap_ratio >= 0.5:
            self._log(f"[DIAG] One-mode input detected (overlap={overlap_ratio:.2f}); pairing all nodes.")
            gene_indices = list(range(n))
        else:
            self._log(f"[DIAG] Bipartite input detected (overlap={overlap_ratio:.2f}); pairing col-0 entities only.")
            gene_indices = sorted(
                self.gene_dict[g] for g in self._col0_set if g in self.gene_dict
            )
            if not gene_indices:
                self._log("[DIAG] col-0 set empty; falling back to all indices.", 'warning')
                gene_indices = list(range(n))
        g_count = len(gene_indices)
        total_pairs = g_count * (g_count - 1) // 2
        self._log(f"Genes: {g_count}, total gene-gene pairs: {total_pairs}")

        if g_count > 0:
            sample_i = gene_indices[0]
            row_nnz = self.matrix.getrow(sample_i).nnz
            self._log(f"[DIAG] First gene (index {sample_i}) row nnz: {row_nnz}")

        if total_pairs == 0:
            self._update_progress(90, "No pairs to compute")
            return 0

        batch_size = max(200, min(2000, total_pairs // (cpu_count() * 20) or 200))

        matrix_csr  = self.matrix.tocsr()
        matrix_data = (n, matrix_csr.data, matrix_csr.indices, matrix_csr.indptr)

        def pair_gen():
            for ii in range(g_count):
                i = gene_indices[ii]
                for jj in range(ii + 1, g_count):
                    yield (i, gene_indices[jj])

        def args_gen():
            batch = []
            for pair in pair_gen():
                batch.append(pair)
                if len(batch) >= batch_size:
                    yield (matrix_data, batch, threshold)
                    batch = []
            if batch:
                yield (matrix_data, batch, threshold)

        keys = list(self.gene_dict.keys())
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True) if output_file else None

        written = 0
        processed = 0
        original_cb            = self.progress_callback
        self.progress_callback = None

        f_out = open(output_file, 'w', encoding='utf-8') if output_file else None
        if f_out:
            f_out.write("source\ttarget\tweight\tp_value\n")

        def consume(iterable, parallel_label):
            nonlocal written, processed
            for batch_idx, batch_result in enumerate(iterable):
                for i, j, val, p in batch_result:
                    if f_out:
                        f_out.write(f"{keys[i]}\t{keys[j]}\t{val:.6f}\t{p:.6e}\n")
                        written += 1
                processed += batch_size
                if (batch_idx + 1) % 50 == 0:
                    if f_out:
                        f_out.flush()
                    pct = min(90, 50 + int(40 * processed / max(1, total_pairs)))
                    self._update_progress(pct, f"[{parallel_label}] ~{processed}/{total_pairs} pairs, {written} written")

        try:
            pool = None
            try:
                pool = Pool(cpu_count())
            except Exception as e:
                self._log(f"Pool init failed ({e}); running sequentially.", 'warning')

            if pool is not None:
                try:
                    with pool:
                        consume(
                            pool.imap_unordered(_calculate_pearson_static, args_gen()),
                            "parallel"
                        )
                except Exception as e:
                    self._log(f"Parallel pass failed ({e}); falling back to sequential.", 'warning')
                    if f_out:
                        f_out.close()
                    f_out = open(output_file, 'w', encoding='utf-8') if output_file else None
                    if f_out:
                        f_out.write("source\ttarget\tweight\tp_value\n")
                    written = 0
                    processed = 0
                    consume((_calculate_pearson_static(a) for a in args_gen()), "sequential")
            else:
                consume((_calculate_pearson_static(a) for a in args_gen()), "sequential")
        finally:
            self.progress_callback = original_cb
            if f_out:
                f_out.close()

        self._log(f"Pairs written: {written}")
        self._update_progress(95, f"Found {written} correlations")
        return written

    def save_results(self, output_file, threshold=0.0):
        """
        Saves results as TSV with columns: source, target, weight, p_value
        weight   = Pearson correlation r
        p_value  = two-sided p-value from scipy.stats.pearsonr
        """
        self._log(f"Saving results to {output_file}...")
        self._update_progress(95, "Step 4/4: Saving results...")

        keys        = list(self.gene_dict.keys())
        saved_pairs = 0

        assert self.correlation_matrix is not None
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("source\ttarget\tweight\tp_value\n")
            for i in range(self.correlation_matrix.shape[0]):
                for j in range(i + 1, self.correlation_matrix.shape[1]):
                    corr = self.correlation_matrix[i, j]
                    if corr != 0 and abs(corr) >= threshold:
                        p = self.pvalue_matrix[i, j] if self.pvalue_matrix is not None else float('nan')
                        f.write(f"{keys[i]}\t{keys[j]}\t{corr:.6f}\t{p:.6e}\n")
                        saved_pairs += 1

        self._log(f"Saved {saved_pairs} pairs to {output_file}")
        self._update_progress(100, f"Done! Saved {saved_pairs} correlations")
        return saved_pairs

def _detect_sep(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readline()
    if '\t' in line: return '\t'
    if ',' in line:  return ','
    if ';' in line:  return ';'
    return r'\s+'

def _has_header_line(filepath, sep):
    kw = {'source', 'target', 'from', 'to', 'node', 'node1', 'node2', 'gene',
          'genes', 'gene1', 'gene2', 'src', 'dst', 'id', 'name', 'weight',
          'score', 'value', 'disease', 'diseases', 'col1', 'col2', 'edge', 'index'}
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
    if not line:
        return False
    parts = line.split(sep if sep != r'\s+' else None)
    if len(parts) < 2:
        return False
    return (parts[0].strip().lower() in kw and parts[1].strip().lower() in kw)

def _build_incidence_matrix(input_file, logger=None):
    """Load a bipartite edgelist and return (col0_labels, col1_labels, B).

    B is a dense float numpy array of shape (|col0|, |col1|), with B[i,j]
    holding the edge weight (default 1.0) between col0_labels[i] and col1_labels[j].
    Duplicates are deduplicated to first occurrence.
    """
    sep  = _detect_sep(input_file)
    skip = 1 if _has_header_line(input_file, sep) else 0
    if logger:
        logger.info(f"Loading incidence matrix from {input_file} (sep={repr(sep)}, skip_header={bool(skip)})")

    col0_to_idx = {}
    col1_to_idx = {}
    triples = []

    reader = pd.read_csv(
        input_file, sep=sep, header=None,
        skiprows=skip, dtype=str,
        engine='python', skipinitialspace=True,
        chunksize=50000,
    )
    seen = set()
    for chunk in reader:
        if chunk.shape[1] < 2:
            continue
        chunk = chunk[chunk.iloc[:, 0].notna() & chunk.iloc[:, 1].notna()]
        for _, row in chunk.iterrows():
            u = str(row.iloc[0]).strip()
            v = str(row.iloc[1]).strip()
            if not u or not v or u.lower() in ('nan', 'none') or v.lower() in ('nan', 'none'):
                continue
            if (u, v) in seen:
                continue
            seen.add((u, v))
            w = 1.0
            if len(row) > 2:
                try:
                    val = float(row.iloc[2])
                    if not np.isnan(val) and val > 0:
                        w = val
                except (ValueError, TypeError):
                    pass
            if u not in col0_to_idx:
                col0_to_idx[u] = len(col0_to_idx)
            if v not in col1_to_idx:
                col1_to_idx[v] = len(col1_to_idx)
            triples.append((col0_to_idx[u], col1_to_idx[v], w))

    n_rows = len(col0_to_idx)
    n_cols = len(col1_to_idx)
    B = np.zeros((n_rows, n_cols), dtype=np.float32)
    for i, j, w in triples:
        B[i, j] = w
    if logger:
        logger.info(f"Incidence matrix: {n_rows} x {n_cols}, nonzeros={len(triples)}")

    col0_labels = [None] * n_rows
    for k, idx in col0_to_idx.items():
        col0_labels[idx] = k
    col1_labels = [None] * n_cols
    for k, idx in col1_to_idx.items():
        col1_labels[idx] = k
    return col0_labels, col1_labels, B

def _write_pearson_projection(M, labels, output_file, alpha=0.05,
                              correction='fdr_bh', logger=None):
    """Compute Pearson r and p-value for all row pairs of M and write the pairs
    that are significant at `alpha` under the chosen multiple-testing `correction`
    ('none' | 'fdr_bh' | 'bonferroni') to `output_file` as TSV:
    source target pearson_r p_value.

    Uses numpy.corrcoef vectorized; p-values computed in bulk via the
    t-distribution survival function.
    """
    header = "source\ttarget\tpearson_r\tp_value\n"
    n_rows, n_cols = M.shape
    if n_rows < 2 or n_cols < 3:
        if logger:
            logger.warning(f"Not enough rows ({n_rows}) or columns ({n_cols}) for Pearson")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(header)
        return 0

    if logger:
        logger.info(f"Computing {n_rows}x{n_rows} correlation matrix from {n_rows}x{n_cols} input...")
    R = np.corrcoef(M)

    iu_i, iu_j = np.triu_indices(n_rows, k=1)
    r_vec = R[iu_i, iu_j]

    valid = np.isfinite(r_vec)
    m = int(valid.sum())
    if m == 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(header)
        return 0

    df = n_cols - 2
    r_clipped = np.clip(r_vec[valid], -0.999999, 0.999999)
    t_stat    = r_clipped * np.sqrt(df / (1.0 - r_clipped * r_clipped))
    from scipy.stats import t as _tdist
    p_vec_valid = 2.0 * _tdist.sf(np.abs(t_stat), df)

    vi = iu_i[valid]
    vj = iu_j[valid]
    rv = r_vec[valid]

    below = np.where(p_vec_valid < alpha)[0]
    cand = [(int(vi[k]), int(vj[k]), float(rv[k]), float(p_vec_valid[k])) for k in below]
    kept = correct(cand, m, alpha, correction)

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)
        for a, b, r_, p_ in kept:
            f.write(f"{labels[a]}\t{labels[b]}\t{r_:.6f}\t{p_:.6e}\n")

    if logger:
        logger.info(f"Wrote {len(kept)} significant pairs (alpha={alpha}, correction={correction}) to {output_file}")
    return len(kept)

def run_bipartite_pearson_dual(input_file, node1_output, node2_output,
                               alpha=0.05, correction='fdr_bh',
                               logger=None, progress_callback=None):
    """Run dual bipartite Pearson projection in one pass.

    For input with iloc[0]=col0_entity, iloc[1]=col1_entity:
      - node1_output: pairs of col-0 entities (e.g. gene-gene if col0=gene)
      - node2_output: pairs of col-1 entities (e.g. disease-disease if col1=disease)

    Output schema for both files: source\\ttarget\\tpearson_r\\tp_value
    (only pairs significant at `alpha` under `correction` are kept.)
    """
    start = time.time()
    try:
        if progress_callback: progress_callback(5)
        col0_labels, col1_labels, B = _build_incidence_matrix(input_file, logger=logger)
        if progress_callback: progress_callback(30)

        n1 = _write_pearson_projection(B, col0_labels, node1_output, alpha, correction, logger)
        if progress_callback: progress_callback(65)

        n2 = _write_pearson_projection(B.T, col1_labels, node2_output, alpha, correction, logger)
        if progress_callback: progress_callback(100)

        return {
            'success':            True,
            'col0_entities':      len(col0_labels),
            'col1_entities':      len(col1_labels),
            'node1_pairs':        n1,
            'node2_pairs':        n2,
            'alpha':              alpha,
            'correction':         correction,
            'execution_time':     round(time.time() - start, 2),
            'method':             'pearson_bipartite_dual',
        }
    except Exception as e:
        if logger:
            logger.error(f"Bipartite Pearson failed: {e}")
            logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}

def run_pearson_analysis(input_file, output_dir, threshold=0.05, chunksize=10000,
                         logger=None, progress_callback=None):
    """
    Run Pearson correlation analysis, entry point for Flask.

    `threshold` is a p-value cutoff (default 0.05). Pairs with
    p_value >= threshold are dropped. Use threshold >= 1.0 to keep all.

    Returns:
        tuple: (stats_dict, output_filename)
    """
    start_time = time.time()

    analyzer = PearsonCorrelationAnalyzer(input_file, chunksize=chunksize, logger=logger)
    if progress_callback:
        analyzer.set_progress_callback(progress_callback)

    task_id         = str(uuid.uuid4())[:8]
    output_filename = os.path.join(output_dir, f"pearson_backbone_{task_id}.tsv")

    try:
        analyzer.build_mappings()
        analyzer.construct_sparse_matrix()
        saved_pairs = analyzer.compute_correlations(threshold=threshold, output_file=output_filename)

        elapsed = time.time() - start_time
        _m = analyzer.matrix
        n       = _m.shape[0] if _m is not None else 0
        max_p   = n * (n - 1) / 2 if n > 1 else 1

        stats = {
            'success':           True,
            'genes':             len(analyzer.gene_dict),
            'significant_pairs': saved_pairs,
            'threshold':         threshold,
            'execution_time':    round(elapsed, 2),
            'method':            'pearson',
            'nodes_in_graph':    n,
            'edges_in_graph':    analyzer.matrix.nnz if analyzer.matrix is not None else 0,
            'density':           round(saved_pairs / max_p, 6),
        }
        return stats, output_filename

    except Exception as e:
        if logger:
            logger.error(f"Pearson analysis failed: {e}")
            logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}, None