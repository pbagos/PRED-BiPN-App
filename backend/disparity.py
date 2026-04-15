import pandas as pd
from scipy.stats import beta
import warnings
import os
from tqdm import tqdm
import psutil

warnings.filterwarnings("ignore")


class DisparityFilter:
    """
    Disparity Filter for weighted network backbone extraction.
    Automatic separator and header detection.
    Column access is always positional (iloc[0], iloc[1], iloc[2]).
    """

    def __init__(self, alpha=0.05, chunksize=100000):
        self.alpha = alpha
        self.chunksize = chunksize
        self.beta_cache = {}

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
        """Main entry point — called from Flask server."""
        print(f"\n=== Disparity Filter ===")
        print(f"Input: {input_file}, Alpha: {self.alpha}")

        sep = self._detect_separator(input_file)
        sample = pd.read_csv(input_file, sep=sep, header=0, nrows=1)
        print(f"  Columns detected: {list(sample.columns)}")

        # Pass 1: node strengths and degrees
        strength, degree = self._calculate_network_stats(input_file, sep)
        print(f"  Found {len(degree):,} nodes")

        if progress_callback:
            progress_callback(50)

        # Pass 2: filter edges
        print("  Applying disparity filter...")
        with open(input_file) as f:
            total_rows = sum(1 for line in f) - 1

        with open(output_file, 'w') as f_out:
            f_out.write("source\ttarget\tweight\talpha\n")

            chunk_reader = pd.read_csv(input_file, sep=sep, header=0,
                                       chunksize=self.chunksize)

            for chunk in tqdm(chunk_reader, total=total_rows / self.chunksize,
                              desc="Filtering edges"):
                chunk.iloc[:, 2] = pd.to_numeric(chunk.iloc[:, 2], errors='coerce')
                chunk = chunk.dropna(subset=[chunk.columns[2]])

                results = self._process_chunk(chunk, strength, degree)

                for u, v, weight, a in results:
                    f_out.write(f"{u}\t{v}\t{weight}\t{a:.6f}\n")

                if psutil.virtual_memory().percent > 90:
                    warnings.warn("High memory usage! Consider reducing chunksize.")

        if progress_callback:
            progress_callback(100)

        print(f"  Saved to: {output_file}")


def run_disparity_filter(input_file, output_file, alpha=0.05,
                         chunksize=100000, progress_callback=None):
    """Compatibility function for Flask server."""
    df = DisparityFilter(alpha=alpha, chunksize=chunksize)
    df.run(input_file, output_file, progress_callback=progress_callback)


if __name__ == "__main__":
    import sys
    input_file  = sys.argv[1] if len(sys.argv) > 1 else "network.tsv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "backbone.tsv"
    alpha       = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

    f = DisparityFilter(alpha=alpha)
    f.run(input_file, output_file)