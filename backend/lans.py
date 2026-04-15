"""
LANS (Locally Adaptive Network Sparsification) Method for Backbone Extraction
with Automatic Column Detection (Data-driven, no pattern matching)
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, triu
import multiprocessing as mp
from functools import partial
import psutil
import os
import gc
import time
from tqdm import tqdm
import warnings
from typing import Dict, List, Tuple, Union, Optional
import sys
import logging

# Import resource module only on Unix/Linux systems
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

warnings.filterwarnings('ignore')


class LANS:
    """
    Locally Adaptive Network Sparsification for backbone extraction
    with automatic column detection (data-driven).
    """
    
    def __init__(self, alpha=0.05, chunk_size=50000, n_cores=None, 
                 test_mode=False, max_rows=10000, max_memory=16000):
        """
        Initialize LANS extractor.
        
        Parameters
        ----------
        alpha : float, optional (default=0.05)
            Significance level for edge filtering
        chunk_size : int, optional (default=50000)
            Number of rows to process at once
        n_cores : int, optional (default=None)
            Number of CPU cores to use (auto-detect if None)
        test_mode : bool, optional (default=False)
            If True, process only max_rows rows
        max_rows : int, optional (default=10000)
            Maximum rows to process in test mode
        max_memory : int, optional (default=16000)
            Maximum memory usage in MB
        """
        self.alpha = alpha
        self.chunk_size = chunk_size
        self.n_cores = n_cores or max(1, mp.cpu_count() - 1)
        self.test_mode = test_mode
        self.max_rows = max_rows
        self.max_memory = max_memory
        
        # Column detection info
        self.detected_sep = None
        self.source_col_idx = None
        self.target_col_idx = None
        self.weight_col_idx = None
        self.has_header = None
        self.source_col_name = None
        self.target_col_name = None
        self.weight_col_name = None
        
        # Data storage
        self.matrix = None
        self.node_map = None
        self.fractional = None
        
    def _increase_memory_limit(self):
        """Attempts to increase memory limits (Linux/Unix only)."""
        if HAS_RESOURCE:
            try:
                import resource as _resource
                _resource.setrlimit(_resource.RLIMIT_AS, (_resource.RLIM_INFINITY, _resource.RLIM_INFINITY))  # type: ignore[attr-defined]
            except Exception as e:
                print(f"Warning: Could not increase memory limit: {e}")
        # On Windows, memory limits cannot be set this way - just continue
    
    def _memory_usage(self) -> float:
        """Returns current memory usage of the process in MB."""
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    
    def _check_memory(self):
        """Raises an error if memory usage exceeds max_memory."""
        if self._memory_usage() > self.max_memory:
            raise MemoryError(f"Memory limit exceeded {self.max_memory}MB")
    
    def _detect_separator(self, file_path):
        """Auto-detect the separator in the input file."""
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            
            separators = {
                '\t': first_line.count('\t'),
                ',': first_line.count(','),
                ';': first_line.count(';'),
                ' ': first_line.count(' ')
            }
            
            separators = {k: v for k, v in separators.items() if v > 0}
            
            if not separators:
                self.detected_sep = '\t'
            else:
                self.detected_sep = max(separators, key=lambda k: separators[k])
            
            return self.detected_sep
    
    def _detect_columns_by_data(self, file_path, sep):
        """
        Detect columns by reading the actual data values (no pattern matching).
        This method analyzes the data types to identify source, target, and weight columns.
        """
        # Read first few rows to analyze data types
        df_sample = pd.read_csv(file_path, sep=sep, nrows=10, header=None)
        
        # Analyze each column to determine its type
        string_cols = []
        numeric_cols = []
        
        for i in range(len(df_sample.columns)):
            col_data = df_sample.iloc[:, i]
            
            # Check if column contains strings (potential node identifiers)
            if col_data.dtype == 'object':
                # Check if it's not all numeric strings
                sample_vals = col_data.dropna().astype(str)
                if not sample_vals.str.isdigit().all():
                    string_cols.append(i)
            
            # Check if column contains numbers (potential weights)
            elif pd.api.types.is_numeric_dtype(col_data):
                numeric_cols.append(i)
        
        # Determine source and target columns (first two string columns)
        if len(string_cols) >= 2:
            self.source_col_idx = string_cols[0]
            self.target_col_idx = string_cols[1]
            self.source_col_name = f"Column_{self.source_col_idx}"
            self.target_col_name = f"Column_{self.target_col_idx}"
            print(f"  Detected source column: column {self.source_col_idx} (contains strings)")
            print(f"  Detected target column: column {self.target_col_idx} (contains strings)")
        
        elif len(string_cols) == 1:
            self.source_col_idx = string_cols[0]
            self.source_col_name = f"Column_{self.source_col_idx}"
            # Use the next column as target (prefer numeric if available)
            if len(numeric_cols) > 0 and numeric_cols[0] != self.source_col_idx:
                self.target_col_idx = numeric_cols[0]
                self.target_col_name = f"Column_{self.target_col_idx}"
                print(f"  Detected source column: column {self.source_col_idx} (contains strings)")
                print(f"  Using column {self.target_col_idx} as target (contains numbers)")
            else:
                # Fallback to next column
                self.target_col_idx = self.source_col_idx + 1 if self.source_col_idx + 1 < len(df_sample.columns) else 0
                self.target_col_name = f"Column_{self.target_col_idx}"
                print(f"  Detected source column: column {self.source_col_idx} (contains strings)")
                print(f"  Using column {self.target_col_idx} as target (fallback)")
        
        else:
            # No string columns found, use first two columns
            self.source_col_idx = 0
            self.target_col_idx = 1 if len(df_sample.columns) > 1 else 0
            self.source_col_name = f"Column_{self.source_col_idx}"
            self.target_col_name = f"Column_{self.target_col_idx}"
            print(f"  No string columns found, using columns {self.source_col_idx} and {self.target_col_idx}")
        
        # Determine weight column (first numeric column not used as source/target)
        for num_col in numeric_cols:
            if num_col != self.source_col_idx and num_col != self.target_col_idx:
                self.weight_col_idx = num_col
                self.weight_col_name = f"Column_{self.weight_col_idx}"
                print(f"  Detected weight column: column {self.weight_col_idx} (contains numbers)")
                break
        
        # If no weight column found, create default weight of 1
        if self.weight_col_idx is None:
            self.weight_col_idx = -1  # Will use default weight
            print(f"  No weight column detected, using default weight=1.0")
        
        return self.source_col_idx, self.target_col_idx, self.weight_col_idx
    
    def _detect_columns_with_header(self, file_path, sep):
        """
        Detect columns when file has header - still data-driven, no pattern matching.
        Uses first few rows to determine column types.
        """
        # Read sample with header
        df_sample = pd.read_csv(file_path, sep=sep, nrows=10)
        columns = df_sample.columns.tolist()
        
        # Analyze each column's data type
        string_cols = []
        numeric_cols = []
        
        for i, col in enumerate(columns):
            col_data = df_sample[col]
            
            # Check if column contains strings (potential node identifiers)
            if col_data.dtype == 'object':
                sample_vals = col_data.dropna().astype(str)
                if not sample_vals.str.isdigit().all():
                    string_cols.append(i)
                    self.source_col_name = col
                    self.target_col_name = col
            
            # Check if column contains numbers (potential weights)
            elif pd.api.types.is_numeric_dtype(col_data):
                numeric_cols.append(i)
                self.weight_col_name = col
        
        # Determine source and target columns
        if len(string_cols) >= 2:
            self.source_col_idx = string_cols[0]
            self.target_col_idx = string_cols[1]
            self.source_col_name = columns[self.source_col_idx]
            self.target_col_name = columns[self.target_col_idx]
            print(f"  Detected source column: '{self.source_col_name}' (index {self.source_col_idx})")
            print(f"  Detected target column: '{self.target_col_name}' (index {self.target_col_idx})")
        
        elif len(string_cols) == 1:
            self.source_col_idx = string_cols[0]
            self.source_col_name = columns[self.source_col_idx]
            
            # Use the next column as target
            if len(numeric_cols) > 0 and numeric_cols[0] != self.source_col_idx:
                self.target_col_idx = numeric_cols[0]
                self.target_col_name = columns[self.target_col_idx]
            else:
                self.target_col_idx = self.source_col_idx + 1 if self.source_col_idx + 1 < len(columns) else 0
                self.target_col_name = columns[self.target_col_idx]
            
            print(f"  Detected source column: '{self.source_col_name}' (index {self.source_col_idx})")
            print(f"  Using target column: '{self.target_col_name}' (index {self.target_col_idx})")
        
        else:
            # Use first two columns
            self.source_col_idx = 0
            self.target_col_idx = 1 if len(columns) > 1 else 0
            self.source_col_name = columns[self.source_col_idx]
            self.target_col_name = columns[self.target_col_idx]
            print(f"  Using columns {self.source_col_idx} and {self.target_col_idx} as source/target")
        
        # Determine weight column
        for num_col in numeric_cols:
            if num_col != self.source_col_idx and num_col != self.target_col_idx:
                self.weight_col_idx = num_col
                self.weight_col_name = columns[self.weight_col_idx]
                print(f"  Detected weight column: '{self.weight_col_name}' (index {self.weight_col_idx})")
                break
        
        if self.weight_col_idx is None:
            self.weight_col_idx = -1
            print(f"  No weight column detected, using default weight=1.0")
        
        return self.source_col_idx, self.target_col_idx, self.weight_col_idx
    
    def load_data(self, input_file):
        """
        Load data with automatic column detection.
        
        Parameters
        ----------
        input_file : str
            Path to input file
            
        Returns
        -------
        tuple
            (edges list, node_map)
        """
        # Step 1: Detect separator
        self._detect_separator(input_file)
        
        # Step 2: Determine if file has header by checking first line
        try:
            with open(input_file, 'r') as f:
                first_line = f.readline().strip()
                parts = first_line.split(self.detected_sep)
                # Check if first line contains words (not all numbers)
                has_words = any(p.isalpha() and len(p) > 0 for p in parts)
                self.has_header = has_words
                print(f"  Header detected: {self.has_header}")
        except:
            self.has_header = False
        
        # Step 3: Detect columns based on data
        if self.has_header:
            self._detect_columns_with_header(input_file, self.detected_sep)
        else:
            self._detect_columns_by_data(input_file, self.detected_sep)
        
        # Step 4: Build node map and edges
        print("\nPhase 1: Discovering nodes...")
        nodes = set()
        
        # First pass: discover all unique nodes
        for chunk in pd.read_csv(
            input_file, 
            sep=self.detected_sep, 
            header=0 if self.has_header else None,
            chunksize=self.chunk_size
        ):
            # Extract source and target columns
            source_data = chunk.iloc[:, self.source_col_idx].astype(str).str.strip()
            target_data = chunk.iloc[:, self.target_col_idx].astype(str).str.strip()
            
            nodes.update(source_data.unique())
            nodes.update(target_data.unique())
            
            self._check_memory()
            
            if self.test_mode and len(nodes) >= self.max_rows:
                break
        
        # Create node to index mapping
        self.node_map = {node: idx for idx, node in enumerate(sorted(nodes))}
        n_nodes = len(self.node_map)
        print(f"Found {n_nodes:,} unique nodes | Memory: {self._memory_usage():.2f}MB")
        
        # Second pass: build matrix
        print("\nPhase 2: Building matrix...")
        matrix = lil_matrix((n_nodes, n_nodes), dtype='float32')
        rows_processed = 0
        
        for chunk in pd.read_csv(
            input_file,
            sep=self.detected_sep,
            header=0 if self.has_header else None,
            chunksize=self.chunk_size
        ):
            # Extract columns
            source_data = chunk.iloc[:, self.source_col_idx].astype(str).str.strip()
            target_data = chunk.iloc[:, self.target_col_idx].astype(str).str.strip()
            
            # Extract weight if available
            if self.weight_col_idx is not None and self.weight_col_idx >= 0:
                weight_data = pd.to_numeric(chunk.iloc[:, self.weight_col_idx], errors='coerce').fillna(0)
            else:
                weight_data = pd.Series([1.0] * len(chunk))
            
            # Filter valid rows
            valid_mask = (source_data != '') & (target_data != '') & (weight_data > 0)
            valid_source = source_data[valid_mask]
            valid_target = target_data[valid_mask]
            valid_weight = weight_data[valid_mask]
            
            # Map to indices
            row_indices = [self.node_map.get(s, -1) for s in valid_source]
            col_indices = [self.node_map.get(t, -1) for t in valid_target]
            
            # Add to matrix (only upper triangular to avoid duplicates)
            for r, c, w in zip(row_indices, col_indices, valid_weight):
                if r >= 0 and c >= 0 and r != c:
                    if r < c:
                        matrix[r, c] += w
                    else:
                        matrix[c, r] += w
            
            rows_processed += len(chunk)
            self._check_memory()
            
            if self.test_mode and rows_processed >= self.max_rows:
                break
        
        # Convert to CSR for efficient operations
        self.matrix = matrix.tocsr()
        assert self.matrix is not None
        print(f"Matrix built ({self.matrix.shape[0]} nodes, {self.matrix.nnz} edges) | Memory: {self._memory_usage():.2f}MB")  # type: ignore[index]
        
        return self.matrix, self.node_map
    
    def _sparse_row_normalize(self, row_index: int) -> csr_matrix:
        """Normalizes a sparse matrix row so that the sum of its elements is 1."""
        assert self.matrix is not None
        row = csr_matrix(self.matrix[row_index, :])  # type: ignore[index]
        row_sum = row.sum()
        return row / row_sum if row_sum > 0 else csr_matrix(row.shape)
    
    def _process_node(self, i):
        """Worker function to normalize one node row."""
        row = self._sparse_row_normalize(i)
        return (i, row) if row.nnz > 0 else None
    
    def calculate_fractional_weights(self):
        """Calculates the normalized (fractional) edge weights for each node."""
        print("\nCalculating fractional weights...")
        assert self.matrix is not None
        n_nodes = self.matrix.shape[0]  # type: ignore[index]
        fractional = lil_matrix(self.matrix.shape, dtype='float32')  # type: ignore[union-attr]
        
        with mp.Pool(processes=self.n_cores) as pool:
            tasks = list(range(n_nodes))
            results = list(tqdm(
                pool.imap(self._process_node, tasks, chunksize=100),
                total=n_nodes,
                desc="Parallel processing"
            ))
        
        for result in results:
            if result:
                i, row = result
                fractional[i] = row
        
        self.fractional = fractional.tocsr()
        return self.fractional
    
    def _extract_backbone_chunk(self, args):
            nodes, alpha = args
            edges = []
            
            for i in nodes:
                assert self.fractional is not None
                row = csr_matrix(self.fractional[i, :])  # type: ignore[index]
                if row.nnz == 0:
                    continue
                
                weights = row.data.tolist()
                n_neighbors = len(weights)
                
                for j, p_ij in zip(row.indices, row.data):
                    if i < j:
                        # ECDF: ποσοστό γειτόνων με p <= p_ij
                        ecdf = sum(1 for w in weights if w <= p_ij) / n_neighbors
                        # Κράτα αν 1 - ECDF(p_ij) < alpha
                        if (1 - ecdf) < alpha:
                            assert self.matrix is not None
                            edges.append((i, j, self.matrix[i, j]))  # type: ignore[index]
            
            return edges
    
    def extract_backbone(self):
        """Extracts the network backbone based on significance level."""
        print(f"\nExtracting network backbone (α={self.alpha})...")
        assert self.matrix is not None
        n_nodes = self.matrix.shape[0]  # type: ignore[index]
        edges = []
        
        # Split nodes into chunks for parallel processing
        step = max(1, n_nodes // (self.n_cores * 4) + 1)
        chunks = [(list(range(i, min(i + step, n_nodes))), self.alpha)
                  for i in range(0, n_nodes, step)]
        
        with mp.Pool(processes=self.n_cores) as pool:
            for result in tqdm(
                pool.imap_unordered(self._extract_backbone_chunk, chunks),
                total=len(chunks),
                desc="Processing nodes"
            ):
                edges.extend(result)
                self._check_memory()
        
        return edges
    
    def save_results(self, edges, output_file="backbone_results.tsv", gephi_file="backbone_gephi.tsv"):
        """Saves the backbone results to files."""
        print("\nSaving results...")
        assert self.node_map is not None
        reverse_map = {idx: node for node, idx in self.node_map.items()}
        
        # Save as CSV with source, target, weight
        df_results = []
        for i, j, w in edges:
            df_results.append([reverse_map[i], reverse_map[j], w])
        
        df = pd.DataFrame(df_results, columns=['source', 'target', 'weight'])
        df.to_csv(output_file, index=False, sep='\t')
        
        # Save Gephi format (same as TSV)
        df.to_csv(gephi_file, index=False, sep='\t')
        
        print(f"Results saved to:")
        print(f"  - {output_file}")
        print(f"  - {gephi_file}")
        print(f"Total edges exported: {len(edges):,}")
        
        return df
    
    def get_column_info(self):
        """Returns detected column information."""
        return {
            'separator': self.detected_sep,
            'has_header': self.has_header,
            'source_column': self.source_col_name,
            'source_column_index': self.source_col_idx,
            'target_column': self.target_col_name,
            'target_column_index': self.target_col_idx,
            'weight_column': self.weight_col_name,
            'weight_column_index': self.weight_col_idx if (self.weight_col_idx is not None and self.weight_col_idx >= 0) else None,
            'nodes': len(self.node_map) if self.node_map else 0,
            'edges': self.matrix.nnz if self.matrix is not None else 0
        }
    
    def run(self, input_file, output_file=None, logger=None):
        """
        Main method to run LANS backbone extraction.
        
        Parameters
        ----------
        input_file : str
            Path to input file
        output_file : str, optional
            Path to output file (default: backbone_results.csv)
        logger : logging.Logger, optional
            Logger for progress messages
            
        Returns
        -------
        dict
            Statistics about the extraction process
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        
        start_time = time.time()
        stats: dict = {'success': True}
        
        try:
            # Increase memory limit
            self._increase_memory_limit()
            
            # Load data with automatic column detection
            logger.info("Loading data with automatic column detection...")
            self.load_data(input_file)
            
            col_info = self.get_column_info()
            stats['column_info'] = col_info
            logger.info(f"Detected columns: {col_info}")
            
            # Calculate fractional weights
            logger.info("Calculating fractional weights...")
            self.calculate_fractional_weights()
            
            # Extract backbone
            logger.info(f"Extracting backbone (α={self.alpha})...")
            edges = self.extract_backbone()
            
            # Save results
            if output_file is None:
                output_file = "backbone_results.tsv"
            
            df_results = self.save_results(edges, output_file)
            
            # Statistics
            assert self.matrix is not None
            stats['edges_original'] = self.matrix.nnz  # type: ignore[union-attr]
            stats['edges_backbone'] = len(edges)
            stats['retention_rate'] = (len(edges) / self.matrix.nnz * 100) if self.matrix.nnz > 0 else 0  # type: ignore[union-attr]
            stats['alpha'] = self.alpha
            stats['total_time'] = time.time() - start_time
            
            logger.info(f"\nResults:")
            logger.info(f"  Original edges: {self.matrix.nnz:,}")  # type: ignore[union-attr]
            logger.info(f"  Backbone edges: {len(edges):,}")
            logger.info(f"  Retention rate: {stats['retention_rate']:.2f}%")
            logger.info(f"  Total time: {stats['total_time']:.2f}s")
            
            return stats, df_results
            
        except Exception as e:
            stats['success'] = False
            stats['error'] = str(e)
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return stats, None


# Factory function for easy instantiation
def create_lans_extractor(alpha=0.05, chunk_size=50000, n_cores=None, 
                         test_mode=False, max_rows=10000):
    """
    Factory function to create a LANS extractor.
    
    Parameters
    ----------
    alpha : float, optional (default=0.05)
        Significance level
    chunk_size : int, optional (default=50000)
        Number of rows to process at once
    n_cores : int, optional (default=None)
        Number of CPU cores
    test_mode : bool, optional (default=False)
        If True, run in test mode
    max_rows : int, optional (default=10000)
        Maximum rows in test mode
        
    Returns
    -------
    LANS
    """
    return LANS(
        alpha=alpha,
        chunk_size=chunk_size,
        n_cores=n_cores,
        test_mode=test_mode,
        max_rows=max_rows
    )


# Backward compatibility function
def run_lans(input_file, output_file, alpha=0.05, test_mode=False, 
             source_col=None, target_col=None, weight_col=None, logger=None):
    """
    Legacy function to run LANS extraction.
    
    Parameters
    ----------
    input_file : str
        Path to input file
    output_file : str
        Path to output file
    alpha : float, optional (default=0.05)
        Significance level
    test_mode : bool, optional (default=False)
        If True, run in test mode
    source_col : str or int, optional
        Source column (auto-detected if None)
    target_col : str or int, optional
        Target column (auto-detected if None)
    weight_col : str or int, optional
        Weight column (auto-detected if None)
    logger : logging.Logger, optional
        Logger for progress messages
    """
    extractor = LANS(alpha=alpha, test_mode=test_mode)
    
    # If columns are specified, we need to modify the detection
    if source_col is not None and target_col is not None:
        # Store the specified columns
        extractor.source_col_idx = source_col if isinstance(source_col, int) else 0
        extractor.target_col_idx = target_col if isinstance(target_col, int) else 1
        extractor.weight_col_idx = weight_col if isinstance(weight_col, int) else 2 if weight_col else -1
        extractor.has_header = True  # Assume header if columns are named
    
    stats, df = extractor.run(input_file, output_file, logger=logger)
    return output_file


if __name__ == '__main__':
    mp.freeze_support()
    input_file = "weighted_disease_edges_one_mode.tsv"
    output_file = "lans_backbone_results.tsv"
    
    # Auto-detect columns
    extractor = LANS(alpha=0.05, test_mode=False)
    stats, df = extractor.run(input_file, output_file)
    
    print(f"\nColumn info: {extractor.get_column_info()}")