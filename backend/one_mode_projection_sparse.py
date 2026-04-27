"""
One-Mode Projection for Bipartite Networks.
Uses sparse matrix multiplication — fast for large datasets.

Reads any file format using positional access (no predefined column names):
    iloc[0] → entity A (gene)
    iloc[1] → entity B (disease)

Output CSV columns: source, target
"""

import os
import numpy as np
from scipy import sparse
from tqdm import tqdm
import pandas as pd
import gc
import warnings
import time

warnings.filterwarnings("ignore")


class OneModeProjection:
    """
    Class for computing one-mode projections of bipartite networks.
    
    Attributes:
        chunk_size (int): Number of rows to read per chunk
        threshold (int): Minimum weight threshold for edges in output
        progress_callback (callable): Function to report progress (0-100)
        logger (logging.Logger): Logger for messages
    
    Example:
        projector = OneModeProjection(chunk_size=100000, threshold=0)
        result = projector.run(
            input_file="bipartite.csv",
            genes_output="gene_projection.csv",
            diseases_output="disease_projection.csv"
        )
    """
    
    def __init__(self, chunk_size=100000, threshold=0, 
                 progress_callback=None, logger=None):
        """
        Initialize OneModeProjection.
        
        Args:
            chunk_size (int): Number of rows to read per chunk
            threshold (int): Minimum weight threshold for edges in output
            progress_callback (callable): Function called with progress percentage
            logger (logging.Logger): Logger for messages
        """
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.progress_callback = progress_callback
        self.logger = logger
        self._progress_step = 0
        
    def _log(self, msg, level='info'):
        """Internal logging helper."""
        if self.logger:
            if level == 'info':
                self.logger.info(msg)
            elif level == 'error':
                self.logger.error(msg)
            elif level == 'warning':
                self.logger.warning(msg)
        else:
            print(msg)
    
    def _update_progress(self, percent):
        """Update progress if callback exists."""
        if self.progress_callback:
            self.progress_callback(percent)
    
    def _detect_separator(self, filepath):
        """Auto-detect file separator."""
        with open(filepath, 'r', encoding='utf-8') as f:
            line = f.readline()
        if '\t' in line:
            return '\t'
        if ',' in line:
            return ','
        if ';' in line:
            return ';'
        return r'\s+'
    
    def _count_lines(self, filepath):
        """Count total lines in file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    def _parse_data_chunked(self, filename):
        """
        Reads the bipartite file in chunks using positional access.
            iloc[0] → gene
            iloc[1] → disease
        
        Returns:
            tuple: (diseases list, genes list)
        """
        sep = self._detect_separator(filename)
        total_lines = self._count_lines(filename)
        total_chunks = max(1, total_lines // self.chunk_size)
        
        self._log(f"Separator: {repr(sep)} | lines: {total_lines} | chunks: {total_chunks}")
        
        chunk_reader = pd.read_csv(
            filename, sep=sep, header=None,
            chunksize=self.chunk_size, dtype=str,
            engine='python', skipinitialspace=True,
            usecols=[0, 1]  # only first 2 cols — ignores trailing separator/empty col
        )
        
        genes = []
        diseases = []
        
        for i, chunk in enumerate(tqdm(chunk_reader, total=total_chunks, desc="Processing network")):
            # Clean: remove NaN / 'nan' / 'None' / empty
            chunk = chunk[chunk.iloc[:, 0].notna() & chunk.iloc[:, 1].notna()]
            chunk = chunk[~chunk.iloc[:, 0].astype(str).str.strip().str.lower().isin(['nan', 'none', ''])]
            chunk = chunk[~chunk.iloc[:, 1].astype(str).str.strip().str.lower().isin(['nan', 'none', ''])]
            
            for _, row in chunk.iterrows():
                u = str(row.iloc[0]).strip()   # gene
                v = str(row.iloc[1]).strip()   # disease
                if not u or not v:
                    continue
                genes.append(u)
                diseases.append(v)
            
            # Update progress (10% of total for parsing)
            if total_chunks > 0:
                progress = (i + 1) / total_chunks * 10
                self._update_progress(progress)
        
        return diseases, genes
    
    def _create_sparse_matrix(self, diseases, genes):
        """
        Create sparse matrix from bipartite lists.
        
        Returns:
            tuple: (sparse_matrix, unique_diseases, unique_genes)
        """
        unique_diseases = sorted(list(set(diseases)))
        unique_genes = sorted(list(set(genes)))
        disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}
        gene_to_idx = {gene: idx for idx, gene in enumerate(unique_genes)}
        
        self._log(f"Creating matrix: {len(unique_diseases)} diseases × {len(unique_genes)} genes")
        
        sparse_matrix = sparse.dok_matrix(
            (len(unique_diseases), len(unique_genes)), dtype=np.int8
        )
        
        total_edges = len(diseases)
        for i, (disease, gene) in enumerate(tqdm(zip(diseases, genes), total=total_edges, desc="Building matrix")):
            sparse_matrix[disease_to_idx[disease], gene_to_idx[gene]] = 1
            
            # Update progress (20% of total for building matrix)
            if total_edges > 0:
                progress = 10 + (i + 1) / total_edges * 20
                self._update_progress(progress)
        
        return sparse_matrix.tocsr(), unique_diseases, unique_genes
    
    def _calculate_projections(self, sparse_matrix):
        """
        Compute one-mode projections via sparse matrix multiplication.
        
        Returns:
            tuple: (gene_projection, disease_projection)
        """
        self._log("Calculating gene projection...")
        gene_projection = sparse_matrix.T.dot(sparse_matrix)
        self._update_progress(60)
        
        self._log("Calculating disease projection...")
        disease_projection = sparse_matrix.dot(sparse_matrix.T)
        self._update_progress(70)
        
        return gene_projection, disease_projection
    
    def _write_edges(self, projection_matrix, labels, output_filename):
        """
        Writes upper-triangle edges as CSV with header: source, target
        Uses comma separator. Skips diagonal and edges with weight <= threshold.
        """
        coo = projection_matrix.tocoo()
        
        edge_count = 0
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("source\ttarget\n")
            for row, col, weight in zip(coo.row, coo.col, coo.data):
                if row < col and weight > self.threshold:
                    f.write(f"{labels[row]}\t{labels[col]}\n")
                    edge_count += 1
        
        self._log(f"Saved: {output_filename} with {edge_count} edges")
        return edge_count
    
    def run(self, input_file, genes_output, diseases_output):
        """
        Main entry point for one-mode projection.
        
        Args:
            input_file (str): Path to input bipartite file
            genes_output (str): Output path for gene projection CSV
            diseases_output (str): Output path for disease projection CSV
        
        Returns:
            dict: Result dictionary with:
                - success (bool): Whether processing succeeded
                - genes (int): Number of unique genes
                - diseases (int): Number of unique diseases
                - gene_edges (int): Number of edges in gene projection
                - disease_edges (int): Number of edges in disease projection
                - execution_time (float): Time taken in seconds
                - error (str): Error message if failed
        """
        start_time = time.time()
        
        try:
            self._update_progress(0)
            self._log("=" * 50)
            self._log("Starting One-Mode Projection")
            self._log(f"Input: {input_file}")
            self._log(f"Outputs: {genes_output}, {diseases_output}")
            
            # Step 1: Parse data
            self._log("Reading data...")
            diseases, genes = self._parse_data_chunked(input_file)
            self._log(f"Found {len(diseases)} entries | "
                     f"{len(set(genes))} genes | {len(set(diseases))} diseases")
            
            self._update_progress(10)
            
            # Step 2: Create sparse matrix
            sparse_matrix, unique_diseases, unique_genes = self._create_sparse_matrix(diseases, genes)
            del diseases, genes
            gc.collect()
            
            self._update_progress(30)
            
            # Step 3: Calculate projections
            gene_proj, disease_proj = self._calculate_projections(sparse_matrix)
            del sparse_matrix
            gc.collect()
            
            self._update_progress(70)
            
            # Step 4: Write outputs
            gene_edges = self._write_edges(gene_proj, unique_genes, genes_output)
            del gene_proj
            gc.collect()
            
            self._update_progress(85)
            
            disease_edges = self._write_edges(disease_proj, unique_diseases, diseases_output)
            del disease_proj
            gc.collect()
            
            self._update_progress(100)
            
            elapsed = round(time.time() - start_time, 2)
            self._log(f"Done in {elapsed}s")
            self._log(f"Gene projection: {len(unique_genes)} nodes, {gene_edges} edges")
            self._log(f"Disease projection: {len(unique_diseases)} nodes, {disease_edges} edges")
            self._log("=" * 50)
            
            return {
                'success': True,
                'genes': len(unique_genes),
                'diseases': len(unique_diseases),
                'gene_edges': gene_edges,
                'disease_edges': disease_edges,
                'execution_time': elapsed,
                'node1_file': os.path.basename(genes_output),
                'node2_file': os.path.basename(diseases_output),
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"One-mode projection failed: {str(e)}"
            self._log(error_msg, level='error')
            return {
                'success': False,
                'error': str(e)
            }


# ------------------------------------------------------------------
# Backward-compatible function (kept for existing code)
# ------------------------------------------------------------------
def run_one_mode_projection(input_file, genes_output, diseases_output,
                            chunk_size=100000, threshold=0,
                            logger=None, progress_callback=None):
    """
    Backward-compatible function wrapper for OneModeProjection class.
    
    Args:
        input_file (str): Path to input bipartite file
        genes_output (str): Output path for gene projection CSV
        diseases_output (str): Output path for disease projection CSV
        chunk_size (int): Number of rows to read per chunk
        threshold (int): Minimum weight threshold for edges in output
        logger (logging.Logger): Logger for messages
        progress_callback (callable): Function called with progress percentage
    
    Returns:
        dict: Result dictionary
    """
    projector = OneModeProjection(
        chunk_size=chunk_size,
        threshold=threshold,
        progress_callback=progress_callback,
        logger=logger
    )
    return projector.run(input_file, genes_output, diseases_output)


# ------------------------------------------------------------------
# Standalone CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="One-Mode Projection (sparse matrix)")
    parser.add_argument('input_file', help="Bipartite input file (any separator)")
    parser.add_argument('--chunk', type=int, default=100000)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--gene-out', default="gene_projection.csv")
    parser.add_argument('--disease-out', default="disease_projection.csv")
    args = parser.parse_args()
    
    run_one_mode_projection(
        args.input_file,
        args.gene_out,
        args.disease_out,
        chunk_size=args.chunk,
        threshold=args.threshold,
    )