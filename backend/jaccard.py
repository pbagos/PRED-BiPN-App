"""
Jaccard Similarity for Gene Networks

Supports two input modes (auto-detected):
  - bipartite  : gene TAB disease          (2 columns, different entity types)
                 → gene-gene OR disease-disease projection
  - gene_gene  : geneA TAB geneB [weight]  (2-3 columns, same entity type)
                 → direct Jaccard on neighbour sets

Jaccard similarity = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

Output CSV:  source, target, weight
"""

import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, save_npz
from tqdm import tqdm
import os
import uuid
import time
import warnings
import gc

warnings.filterwarnings('ignore')


class JaccardSimilarityAnalyzer:

    def __init__(self, filepath=None, chunksize=10000, logger=None):
        self.filepath          = filepath
        self.chunksize         = chunksize
        self.logger            = logger
        self.gene_dict         = {}
        self.disease_dict      = {}
        self.matrix            = None
        self.one_mode_matrix   = None
        self.similarity_matrix = None
        self.progress_callback = None
        self.total_lines       = None

    # ------------------------------------------------------------------
    # Logging / progress
    # ------------------------------------------------------------------
    def log(self, message, level='info'):
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            print(f"[{level.upper()}] {message}")

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def update_progress(self, percent):
        if self.progress_callback:
            self.progress_callback(percent)

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    def _count_lines(self):
        assert self.filepath is not None
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _detect_separator(self):
        assert self.filepath is not None
        with open(self.filepath, 'r', encoding='utf-8') as f:
            line = f.readline()
        if '\t' in line:
            sep = '\t'
        elif ',' in line:
            sep = ','
        elif ';' in line:
            sep = ';'
        else:
            sep = r'\s+'
        self.log(f"Detected separator: {repr(sep)}")
        return sep

    def _detect_input_type(self):
        """
        Reads up to 200 rows and decides:
          'gene_gene'  — col-0 and col-1 share >10% of values (same entity type)
          'bipartite'  — col-0 and col-1 are disjoint entity sets
        """
        sep    = self._detect_separator()
        assert self.filepath is not None
        sample = pd.read_csv(
            self.filepath, sep=sep, header=None,
            nrows=200, dtype=str,
            engine='python', skipinitialspace=True,
            usecols=[0, 1]
        )
        col0    = set(sample.iloc[:, 0].str.strip().dropna())
        col1    = set(sample.iloc[:, 1].str.strip().dropna())
        overlap = col0 & col1
        itype   = 'gene_gene' if len(col0) > 0 and len(overlap) / len(col0) > 0.10 \
                  else 'bipartite'
        self.log(f"Input type: {itype}  (overlap {len(overlap)}/{len(col0)})")
        return itype

    # ------------------------------------------------------------------
    # Cleaning — positional access only
    # ------------------------------------------------------------------
    def _clean_data(self, chunk):
        chunk = chunk.dropna()
        chunk = chunk[~chunk.iloc[:, 0].astype(str).str.contains(
            'nan|None', case=False, na=False)]
        chunk = chunk[~chunk.iloc[:, 1].astype(str).str.contains(
            'nan|None', case=False, na=False)]
        return chunk

    # ------------------------------------------------------------------
    # BIPARTITE path
    # ------------------------------------------------------------------
    def build_mappings_bipartite(self):
        self.log("Building gene & disease mappings (bipartite)...")
        self.update_progress(5)
        self.total_lines = self._count_lines()
        total_chunks     = (self.total_lines // self.chunksize) + 1
        sep              = self._detect_separator()
        gene_counter = disease_counter = chunk_count = 0

        for chunk in tqdm(
            pd.read_csv(self.filepath, sep=sep, header=None,  # type: ignore[arg-type]
                        chunksize=self.chunksize, dtype=str,
                        engine='python', skipinitialspace=True,
                        usecols=[0, 1]),
            total=total_chunks, desc="Processing network"
        ):
            chunk = self._clean_data(chunk)
            for _, row in chunk.iterrows():
                gene    = str(row.iloc[0]).strip()
                disease = str(row.iloc[1]).strip()
                if not gene or not disease:
                    continue
                if gene not in self.gene_dict:
                    self.gene_dict[gene] = gene_counter
                    gene_counter += 1
                if disease not in self.disease_dict:
                    self.disease_dict[disease] = disease_counter
                    disease_counter += 1
            chunk_count += 1
            self.update_progress(5 + int(25 * chunk_count / total_chunks))
            del chunk; gc.collect()

        self.log(f"Genes: {len(self.gene_dict)}, Diseases: {len(self.disease_dict)}")
        self.update_progress(30)

    def construct_sparse_matrix_bipartite(self):
        self.log("Constructing bipartite matrix (genes × diseases)...")
        self.update_progress(30)
        assert self.total_lines is not None
        total_chunks = (self.total_lines // self.chunksize) + 1
        sep          = self._detect_separator()
        rows, cols, data = [], [], []
        chunk_count  = 0

        for chunk in tqdm(
            pd.read_csv(self.filepath, sep=sep, header=None,  # type: ignore[arg-type]
                        chunksize=self.chunksize, dtype=str,
                        engine='python', skipinitialspace=True,
                        usecols=[0, 1]),
            total=total_chunks, desc="Building matrix"
        ):
            chunk = self._clean_data(chunk)
            for _, row in chunk.iterrows():
                g = str(row.iloc[0]).strip()
                d = str(row.iloc[1]).strip()
                if g in self.gene_dict and d in self.disease_dict:
                    rows.append(self.gene_dict[g])
                    cols.append(self.disease_dict[d])
                    data.append(1)
            chunk_count += 1
            self.update_progress(30 + int(20 * chunk_count / total_chunks))
            if chunk_count % 10 == 0: gc.collect()

        if rows:
            self.matrix = csr_matrix(
                (data, (rows, cols)),
                shape=(len(self.gene_dict), len(self.disease_dict))
            )
            save_npz("sparse_matrix.npz", self.matrix)
            self.log(f"Matrix: {self.matrix.shape}, nnz={self.matrix.nnz}")
            del rows, cols, data; gc.collect()
            self.update_progress(50)
        else:
            raise ValueError("No valid data found to construct bipartite matrix")

    # ------------------------------------------------------------------
    # GENE-GENE path
    # ------------------------------------------------------------------
    def build_mappings_gene_gene(self):
        self.log("Building gene mappings (gene-gene)...")
        self.update_progress(5)
        self.total_lines = self._count_lines()
        total_chunks     = (self.total_lines // self.chunksize) + 1
        sep              = self._detect_separator()
        gene_counter = chunk_count = 0

        for chunk in tqdm(
            pd.read_csv(self.filepath, sep=sep, header=None,  # type: ignore[arg-type]
                        chunksize=self.chunksize, dtype=str,
                        engine='python', skipinitialspace=True,
                        usecols=[0, 1]),
            total=total_chunks, desc="Processing network"
        ):
            chunk = self._clean_data(chunk)
            for _, row in chunk.iterrows():
                for g in (str(row.iloc[0]).strip(), str(row.iloc[1]).strip()):
                    if g and g not in self.gene_dict:
                        self.gene_dict[g] = gene_counter
                        gene_counter += 1
            chunk_count += 1
            self.update_progress(5 + int(25 * chunk_count / total_chunks))
            del chunk; gc.collect()

        self.log(f"Unique genes: {len(self.gene_dict)}")
        self.update_progress(30)

    def construct_sparse_matrix_gene_gene(self):
        """
        Symmetric adjacency matrix.
        Weight from iloc[2] if present, else 1.0.
        """
        self.log("Constructing gene-gene adjacency matrix...")
        self.update_progress(30)
        assert self.total_lines is not None
        total_chunks = (self.total_lines // self.chunksize) + 1
        sep          = self._detect_separator()
        rows, cols, data = [], [], []
        chunk_count  = 0

        for chunk in tqdm(
            pd.read_csv(self.filepath, sep=sep, header=None,  # type: ignore[arg-type]
                        chunksize=self.chunksize, dtype=str,
                        engine='python', skipinitialspace=True),
            total=total_chunks, desc="Building matrix"
        ):
            chunk = self._clean_data(chunk)
            for _, row in chunk.iterrows():
                u = str(row.iloc[0]).strip()
                v = str(row.iloc[1]).strip()
                try:
                    w = float(row.iloc[2]) if len(row) > 2 else 1.0
                except (ValueError, TypeError):
                    w = 1.0
                if not u or not v:
                    continue
                try:
                    i, j = self.gene_dict[u], self.gene_dict[v]
                    rows += [i, j]; cols += [j, i]; data += [w, w]
                except KeyError:
                    continue
            chunk_count += 1
            self.update_progress(30 + int(20 * chunk_count / total_chunks))
            if chunk_count % 10 == 0: gc.collect()

        if rows:
            n = len(self.gene_dict)
            self.matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
            self.matrix.setdiag(0)
            self.matrix.eliminate_zeros()
            self.log(f"Adjacency: {self.matrix.shape}, nnz={self.matrix.nnz}")
            del rows, cols, data; gc.collect()
            self.update_progress(50)
        else:
            raise ValueError("No valid data found to construct gene-gene matrix")

    # ------------------------------------------------------------------
    # Jaccard  (shared for both paths)
    # ------------------------------------------------------------------
    def compute_jaccard_similarity(self, mode='gene', threshold=0.0):
        """
        Computes Jaccard similarity based on sets of neighbors.
        Intersection = A * A^T
        Union = deg(u) + deg(v) - Intersection
        """
        self.log(f"Computing Jaccard (threshold={threshold})...")
        self.update_progress(55)

        if self.matrix is None:
            raise ValueError("Matrix not constructed yet.")

        if mode == 'disease':
            A = self.matrix.T.tocsr()
        else:
            A = self.matrix.copy()

        # Free original matrices to save memory
        self.matrix = None
        gc.collect()

        # Jaccard is based on sets (binary existence of edge), so we binarize the adjacency matrix
        A.data = np.ones_like(A.data)
        
        self.log("Computing intersections (A * A^T)...")
        intersection_matrix = A.dot(A.T)
        intersection_matrix.setdiag(0)
        intersection_matrix.eliminate_zeros()
        
        # Array of neighbor counts for each node
        degrees = np.array(A.sum(axis=1)).flatten()
        
        del A
        gc.collect()
        
        self.log("Computing Jaccard similarities...")
        coo = intersection_matrix.tocoo()
        del intersection_matrix
        gc.collect()
        
        # We only need upper triangle for undirected similarity to avoid double processing
        mask = coo.row < coo.col
        row = coo.row[mask]
        col = coo.col[mask]
        inter = coo.data[mask]
        del coo
        gc.collect()
        
        # union = |A| + |B| - |A n B|
        union = degrees[row] + degrees[col] - inter
        
        sim = np.zeros_like(inter, dtype=float)
        valid = union > 0
        sim[valid] = inter[valid] / union[valid]
        
        # Apply threshold
        thresh_mask = sim >= threshold
        row = row[thresh_mask]
        col = col[thresh_mask]
        sim = sim[thresh_mask]
        
        # Symmetrize the final matrix
        rows = np.concatenate([row, col])
        cols = np.concatenate([col, row])
        data = np.concatenate([sim, sim])
        
        n = len(degrees)
        self.similarity_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
        self.log(f"Non-zero similarities: {self.similarity_matrix.nnz}")
        self.update_progress(90)
        return self.similarity_matrix

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_results(self, output_file, mode='gene', batch_size=10000):
        keys = list(self.disease_dict.keys()) if mode == 'disease' \
               else list(self.gene_dict.keys())
        self.log(f"Saving → {output_file}")
        assert self.similarity_matrix is not None
        coo = self.similarity_matrix.tocoo()  # type: ignore[union-attr]

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("source\ttarget\tweight\n")
            batch = []
            for idx in range(len(coo.data)):
                if coo.row[idx] < coo.col[idx]:
                    batch.append(
                        f"{keys[coo.row[idx]]}\t{keys[coo.col[idx]]}\t{coo.data[idx]:.6f}\n"
                    )
                if len(batch) >= batch_size:
                    f.writelines(batch); batch = []; gc.collect()
            if batch:
                f.writelines(batch)

        self.log("Results saved.")
        self.update_progress(100)

    # ------------------------------------------------------------------
    # run_analysis  — auto-detects and routes
    # ------------------------------------------------------------------
    def run_analysis(self, input_file, output_file, mode='gene', threshold=0.0):
        import psutil
        start = time.time()
        mem   = lambda: psutil.Process().memory_info().rss / 1024 / 1024

        self.filepath = input_file
        m0 = mem()

        try:
            itype = self._detect_input_type()

            if itype == 'gene_gene':
                self.log("--- Gene-gene path ---")
                self.build_mappings_gene_gene();           m1 = mem()
                self.construct_sparse_matrix_gene_gene();  m2 = mem()
                self.compute_jaccard_similarity(mode='gene_gene', threshold=threshold)
                m3 = mem()
                save_mode = 'gene'

            else:
                self.log("--- Bipartite path ---")
                self.build_mappings_bipartite();           m1 = mem()
                self.construct_sparse_matrix_bipartite();  m2 = mem()
                self.compute_jaccard_similarity(mode=mode, threshold=threshold)
                m3 = mem()
                save_mode = mode

            self.save_results(output_file, mode=save_mode)

            elapsed = time.time() - start
            assert self.similarity_matrix is not None
            n_sim   = self.similarity_matrix.shape[0]  # type: ignore[union-attr]
            max_p   = n_sim * (n_sim - 1) / 2 or 1
            density = (self.similarity_matrix.nnz / 2) / max_p

            stats = {
                'success':          True,
                'input_type':       itype,
                'genes':            len(self.gene_dict),
                'diseases':         len(self.disease_dict),
                'similarity_pairs': self.similarity_matrix.nnz // 2,
                'density':          round(density, 6),
                'execution_time':   round(elapsed, 2),
                'peak_memory_mb':   round(max(m1, m2, m3), 2),
                'mode':             mode,
                'threshold':        threshold,
                'method':           'jaccard',
            }
            self.log(f"Done in {elapsed:.1f}s | peak {stats['peak_memory_mb']:.0f} MB")
            return stats

        except Exception as e:
            self.log(f"Error: {e}", 'error')
            import traceback; traceback.print_exc()
            return {'success': False, 'error': str(e)}


# ------------------------------------------------------------------
# Flask helper
# ------------------------------------------------------------------
def run_jaccard_analysis(input_file, output_dir, mode='gene', threshold=0.0,
                         chunksize=10000, logger=None, progress_callback=None):
    task_id         = str(uuid.uuid4())
    output_filename = os.path.join(output_dir, f"jaccard_{mode}_{task_id}.tsv")
    analyzer = JaccardSimilarityAnalyzer(
        filepath=input_file, chunksize=chunksize, logger=logger
    )
    if progress_callback:
        analyzer.set_progress_callback(progress_callback)
    stats = analyzer.run_analysis(input_file, output_filename, mode=mode, threshold=threshold)
    return stats, output_filename


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, psutil
    parser = argparse.ArgumentParser(description='Jaccard Similarity Analysis')
    parser.add_argument('input_file')
    parser.add_argument('--mode',      default='gene', choices=['gene', 'disease'])
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--chunksize', type=int,   default=10000)
    parser.add_argument('--output',    default=None)
    args = parser.parse_args()

    if not args.output:
        args.output = f"jaccard_{args.mode}_results.tsv"

    analyzer = JaccardSimilarityAnalyzer(filepath=args.input_file, chunksize=args.chunksize)
    stats    = analyzer.run_analysis(args.input_file, args.output,
                                     mode=args.mode, threshold=args.threshold)
    if stats['success']:
        print(f"\n=== Successful ===")
        print(f"Input type      : {stats['input_type']}")
        print(f"Genes           : {stats['genes']}")
        print(f"Diseases        : {stats['diseases']}")
        print(f"Similarity pairs: {stats['similarity_pairs']}")
        print(f"Density         : {stats['density']}")
        print(f"Peak memory     : {stats['peak_memory_mb']} MB")
        print(f"Execution time  : {stats['execution_time']} s")
        print(f"Output          : {args.output}")
    else:
        print(f"Error: {stats.get('error', 'Unknown')}")
