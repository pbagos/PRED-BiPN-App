"""
Simple Weighted One-Mode Projection for Bipartite Gene-Disease Networks.

Weight = number of shared neighbours:
    gene-gene     : weight(g1, g2) = |diseases(g1) ∩ diseases(g2)|
    disease-disease: weight(d1, d2) = |genes(d1) ∩ genes(d2)|

Reads any file format using positional access, no predefined column names:
    iloc[0] → gene
    iloc[1] → disease

Output CSV columns: source, target, weight
"""

import os
import time
import traceback
from collections import defaultdict

import pandas as pd
from input_format import has_header_row
import psutil
from tqdm import tqdm

def _detect_separator(filepath: str) -> str:
    """Auto-detects separator from the first line of the file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readline()
    if '\t' in line:
        return '\t'
    if ',' in line:
        return ','
    if ';' in line:
        return ';'
    return r'\s+'

def _count_lines(filepath: str) -> int:
    """Count total lines (header=None, no line subtracted)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

class OneModeProjectionSparse:
    """
    Simple weighted one-mode projection from a bipartite gene-disease file.

    weight(u, v) = |neighbours(u) ∩ neighbours(v)|

    Example:
        proj = OneModeProjectionSparse('data.txt', logger=app.logger)
        proj.build_mappings()
        proj.compute_gene_edges('gene_edges.csv')
        proj.compute_disease_edges('disease_edges.csv')
    """

    def __init__(self, filepath: str, chunk_size: int = 50000, logger=None):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.logger = logger

        self.gene_to_diseases: defaultdict = defaultdict(set)
        self.disease_to_genes: defaultdict = defaultdict(set)
        self.total_rows = 0

    def _log(self, message, level='info'):
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            print(message)

    def build_mappings(self, progress_callback=None):
        """
        Reads the bipartite file in chunks.
        Uses positional access (iloc[0], iloc[1]), no column names needed.
        Populates gene_to_diseases and disease_to_genes.
        """
        sep = _detect_separator(self.filepath)
        total_lines = _count_lines(self.filepath)
        total_chunks = max(1, total_lines // self.chunk_size)

        self._log(f"Separator: {repr(sep)} | lines: {total_lines} | chunks: {total_chunks}")

        chunk_reader = pd.read_csv(
            self.filepath, sep=sep, header=None, skiprows=int(has_header_row(self.filepath, sep)),
            chunksize=self.chunk_size, dtype=str,
            engine='python', skipinitialspace=True,
            usecols=[0, 1]
        )

        for chunk_idx, chunk in enumerate(tqdm(chunk_reader, total=total_chunks, desc="Processing network")):
            chunk = chunk[chunk.iloc[:, 0].notna() & chunk.iloc[:, 1].notna()]
            chunk = chunk[~chunk.iloc[:, 0].astype(str).str.strip().str.lower().isin(['nan', 'none', ''])]
            chunk = chunk[~chunk.iloc[:, 1].astype(str).str.strip().str.lower().isin(['nan', 'none', ''])]

            for _, row in chunk.iterrows():
                u = str(row.iloc[0]).strip()
                v = str(row.iloc[1]).strip()
                if not u or not v:
                    continue
                self.gene_to_diseases[u].add(v)
                self.disease_to_genes[v].add(u)

            self.total_rows += len(chunk)
            mem = psutil.virtual_memory().percent
            self._log(f"  {self.total_rows} rows processed | Memory: {mem}%")

            if progress_callback:
                progress = int((chunk_idx + 1) / total_chunks * 50)
                progress_callback(progress)

        self._log(
            f"Mappings ready: {len(self.gene_to_diseases)} genes, "
            f"{len(self.disease_to_genes)} diseases, {self.total_rows} rows"
        )

    def compute_gene_edges(self, output_file: str, progress_callback=None) -> int:
        """
        Gene-gene one-mode projection.
        weight(g1, g2) = |diseases(g1) ∩ diseases(g2)|
        Output CSV: source, target, weight
        Returns number of edges written.
        """
        self._log(f"Computing gene-gene edges → {output_file}")
        genes = list(self.gene_to_diseases.keys())
        n = len(genes)
        total = n * (n - 1) // 2
        written = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("source\ttarget\tweight\n")
            with tqdm(total=total, desc="Gene-gene edges") as pbar:
                for i in range(n):
                    g1 = genes[i]
                    for j in range(i + 1, n):
                        g2 = genes[j]
                        shared = self.gene_to_diseases[g1] & self.gene_to_diseases[g2]
                        if shared:
                            f.write(f"{g1}\t{g2}\t{len(shared)}\n")
                            written += 1
                        pbar.update(1)

                    if progress_callback and i % 10 == 0:
                        progress = 50 + int((i + 1) / n * 25)
                        progress_callback(progress)

        self._log(f"Gene-gene edges written: {written}")
        return written

    def compute_disease_edges(self, output_file: str, progress_callback=None) -> int:
        """
        Disease-disease one-mode projection.
        weight(d1, d2) = |genes(d1) ∩ genes(d2)|
        Output CSV: source, target, weight
        Returns number of edges written.
        """
        self._log(f"Computing disease-disease edges → {output_file}")
        diseases = list(self.disease_to_genes.keys())
        n = len(diseases)
        total = n * (n - 1) // 2
        written = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("source\ttarget\tweight\n")
            with tqdm(total=total, desc="Disease-disease edges") as pbar:
                for i in range(n):
                    d1 = diseases[i]
                    for j in range(i + 1, n):
                        d2 = diseases[j]
                        shared = self.disease_to_genes[d1] & self.disease_to_genes[d2]
                        if shared:
                            f.write(f"{d1}\t{d2}\t{len(shared)}\n")
                            written += 1
                        pbar.update(1)

                    if progress_callback and i % 10 == 0:
                        progress = 75 + int((i + 1) / n * 25)
                        progress_callback(progress)

        self._log(f"Disease-disease edges written: {written}")
        return written

    def run(self, node1_output: str, node2_output: str, progress_callback=None) -> dict:
        """
        Runs all steps. Returns stats dict.
            node1_output : gene-gene     edges CSV
            node2_output : disease-disease edges CSV
        """
        start = time.time()
        try:
            self.build_mappings(progress_callback)
            gene_edges = self.compute_gene_edges(node1_output, progress_callback)
            disease_edges = self.compute_disease_edges(node2_output, progress_callback)
            return {
                'success': True,
                'genes': len(self.gene_to_diseases),
                'diseases': len(self.disease_to_genes),
                'total_rows': self.total_rows,
                'gene_edges': gene_edges,
                'disease_edges': disease_edges,
                'execution_time': round(time.time() - start, 2),
            }
        except Exception as e:
            self._log(f"Error: {e}", 'error')
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

def run_one_mode_projection(input_file: str,
                            node1_output: str,
                            node2_output: str,
                            chunk_size: int = 50000,
                            logger=None,
                            progress_callback=None) -> dict:
    """
    Flask entry point. Maintains compatibility with existing server call:
        run_one_mode_projection(input_path, node1_output, node2_output)

    Args:
        input_file        : bipartite gene-disease file (any separator, no header needed)
        node1_output      : path for gene-gene     edges CSV
        node2_output      : path for disease-disease edges CSV
        chunk_size        : rows per pandas chunk
        logger            : optional Flask/Python logger
        progress_callback : optional callable(percent 0-100)

    Returns:
        stats dict with 'success', counts, file basenames, execution_time
    """
    if progress_callback:
        progress_callback(5)

    proj = OneModeProjectionSparse(input_file, chunk_size=chunk_size, logger=logger)
    stats = proj.run(node1_output, node2_output, progress_callback=progress_callback)

    if progress_callback:
        progress_callback(100)

    if stats['success']:
        stats['node1_file'] = os.path.basename(node1_output)
        stats['node2_file'] = os.path.basename(node2_output)

    return stats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weighted One-Mode Projection")
    parser.add_argument('input_file', help="Bipartite input file (any separator)")
    parser.add_argument('--chunk', type=int, default=50000)
    parser.add_argument('--gene-out', default="weighted_gene_edges.csv")
    parser.add_argument('--disease-out', default="weighted_disease_edges.csv")
    args = parser.parse_args()

    proj = OneModeProjectionSparse(args.input_file, chunk_size=args.chunk)
    stats = proj.run(args.gene_out, args.disease_out)

    if stats['success']:
        print(f"\n=== Done ===")
        print(f"Genes:          {stats['genes']}")
        print(f"Diseases:       {stats['diseases']}")
        print(f"Total rows:     {stats['total_rows']}")
        print(f"Gene edges:     {stats['gene_edges']}")
        print(f"Disease edges:  {stats['disease_edges']}")
        print(f"Time:           {stats['execution_time']} s")
        print(f"Outputs:        {args.gene_out}  |  {args.disease_out}")
    else:
        print(f"Error: {stats.get('error')}")