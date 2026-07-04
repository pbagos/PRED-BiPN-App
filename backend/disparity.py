import pandas as pd
from scipy.stats import beta
import warnings
import os
from tqdm import tqdm
import psutil
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")

def _filter_rows_batch(args):
    """Module-level worker: filter a batch of (u, v, weight) rows."""
    rows, strength, degree, alpha_val = args
    results = []
    for u, v, weight in rows:
        su = strength.get(u, 0)
        sv = strength.get(v, 0)
        if su == 0 or sv == 0:
            continue
        ku = degree.get(u, 1)
        kv = degree.get(v, 1)
        pij_u = weight / su
        aij_u = beta.sf(pij_u, 1, ku - 1) if ku > 1 else 1.0
        pij_v = weight / sv
        aij_v = beta.sf(pij_v, 1, kv - 1) if kv > 1 else 1.0
        aij = min(aij_u, aij_v)
        if aij < alpha_val:
            results.append((u, v, weight, aij))
    return results

class DisparityFilter:
    """
    Disparity Filter for weighted network backbone extraction.
    Automatic separator and header detection.
    Column access is always positional (iloc[0], iloc[1], iloc[2]).
    """

    def __init__(self, alpha=0.05, chunksize=100000, n_workers=1):
        self.alpha = alpha
        self.chunksize = chunksize
        self.beta_cache = {}
        self.n_workers = n_workers

    def _detect_separator(self, filename):
        with open(filename, 'r') as f:
            first_line = f.readline().strip()
        separators = {'\t': first_line.count('\t'), ',': first_line.count(','),
                      ';': first_line.count(';'), ' ': first_line.count(' ')}
        sep = max(separators, key=lambda k: separators[k])
        return sep if separators[sep] > 0 else '\t'

    def _get_beta_sf(self, p, k):
        if k <= 1:
            return 1.0
        if (p, k) not in self.beta_cache:
            self.beta_cache[(p, k)] = beta.sf(p, 1, k - 1)
        return self.beta_cache[(p, k)]

    def _process_chunk(self, chunk, strength, degree):
        results = []
        for _, row in chunk.iterrows():
            u = str(row.iloc[0]).strip()
            v = str(row.iloc[1]).strip()
            weight = row.iloc[2]
            try:
                if strength.get(u, 0) == 0 or strength.get(v, 0) == 0:
                    continue
                pij_u = weight / strength[u]
                alpha_ij_u = self._get_beta_sf(pij_u, degree[u])
                pij_v = weight / strength[v]
                alpha_ij_v = self._get_beta_sf(pij_v, degree[v])
                alpha_ij = min(alpha_ij_u, alpha_ij_v)
                if alpha_ij < self.alpha:
                    results.append((u, v, weight, alpha_ij))
            except Exception:
                continue
        return results

    def _calculate_network_stats(self, filename, sep):
        print("  Calculating network statistics...")
        strength = {}
        degree = {}

        with open(filename) as f:
            total_rows = sum(1 for line in f) - 1

        chunk_reader = pd.read_csv(filename, sep=sep, header=0,
                                   chunksize=self.chunksize)

        for chunk in tqdm(chunk_reader, total=total_rows / self.chunksize,
                          desc="Processing network"):
            chunk.iloc[:, 2] = pd.to_numeric(chunk.iloc[:, 2], errors='coerce')
            chunk = chunk.dropna(subset=[chunk.columns[2]])

            for _, row in chunk.iterrows():
                u = str(row.iloc[0]).strip()
                v = str(row.iloc[1]).strip()
                weight = float(row.iloc[2])

                if not u or not v or u == 'nan' or v == 'nan':
                    continue

                strength[u] = strength.get(u, 0) + weight
                strength[v] = strength.get(v, 0) + weight
                degree[u] = degree.get(u, 0) + 1
                degree[v] = degree.get(v, 0) + 1

        return strength, degree

    def run(self, input_file, output_file, progress_callback=None):
        """Main entry point, called from Flask server."""
        print(f"\n=== Disparity Filter ===")
        print(f"Input: {input_file}, Alpha: {self.alpha}")

        sep = self._detect_separator(input_file)
        sample = pd.read_csv(input_file, sep=sep, header=0, nrows=1)
        print(f"  Columns detected: {list(sample.columns)}")

        strength, degree = self._calculate_network_stats(input_file, sep)
        print(f"  Found {len(degree):,} nodes")

        if progress_callback:
            progress_callback(50)

        print(f"  Applying disparity filter ({self.n_workers} worker(s))...")
        with open(input_file) as f:
            total_rows = sum(1 for line in f) - 1

        all_rows = []
        chunk_reader = pd.read_csv(input_file, sep=sep, header=0,
                                   chunksize=self.chunksize)
        for chunk in tqdm(chunk_reader, total=total_rows / self.chunksize,
                          desc="Reading edges"):
            chunk.iloc[:, 2] = pd.to_numeric(chunk.iloc[:, 2], errors='coerce')
            chunk = chunk.dropna(subset=[chunk.columns[2]])
            for _, row in chunk.iterrows():
                u = str(row.iloc[0]).strip()
                v = str(row.iloc[1]).strip()
                w = float(row.iloc[2])
                if u and v and u != 'nan' and v != 'nan':
                    all_rows.append((u, v, w))

        if psutil.virtual_memory().percent > 90:
            warnings.warn("High memory usage! Consider reducing chunksize.")

        batch_size = max(1000, len(all_rows) // max(1, self.n_workers * 4))
        batches = [
            (all_rows[k:k + batch_size], strength, degree, self.alpha)
            for k in range(0, len(all_rows), batch_size)
        ]
        print(f"  Processing {len(all_rows):,} edges in {len(batches)} batches")

        with open(output_file, 'w') as f_out:
            f_out.write("source\ttarget\tweight\talpha\n")
            if self.n_workers > 1 and len(batches) > 1:
                with Pool(processes=self.n_workers) as pool:
                    for batch_result in tqdm(
                        pool.imap(_filter_rows_batch, batches),
                        total=len(batches), desc="Filtering edges"
                    ):
                        for u, v, weight, a in batch_result:
                            f_out.write(f"{u}\t{v}\t{weight}\t{a:.6f}\n")
            else:
                for batch in tqdm(batches, desc="Filtering edges"):
                    for u, v, weight, a in _filter_rows_batch(batch):
                        f_out.write(f"{u}\t{v}\t{weight}\t{a:.6f}\n")

        if progress_callback:
            progress_callback(100)

        print(f"  Saved to: {output_file}")

def run_disparity_filter(input_file, output_file, alpha=0.05,
                         chunksize=100000, n_workers=1, progress_callback=None):
    """Compatibility function for Flask server."""
    df = DisparityFilter(alpha=alpha, chunksize=chunksize, n_workers=n_workers)
    df.run(input_file, output_file, progress_callback=progress_callback)

if __name__ == "__main__":
    import sys
    input_file  = sys.argv[1] if len(sys.argv) > 1 else "network.tsv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "backbone.tsv"
    alpha       = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

    f = DisparityFilter(alpha=alpha)
    f.run(input_file, output_file)