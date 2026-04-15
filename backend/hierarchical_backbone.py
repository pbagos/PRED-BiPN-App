"""
Hierarchical Backbone Extractor - Optimized implementation for bipartite networks
with automatic column detection and CSV output with source, target, weight columns
"""

import pandas as pd
import numpy as np
import time
import networkx as nx
from NEMtropy import BipartiteGraph
from scipy.sparse import csr_matrix, coo_matrix
import gc
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class HierarchicalBackboneExtractor:
    """
    Optimized Hierarchical Backbone Extractor for bipartite networks.
    
    Automatically detects column names and separators.
    Outputs CSV files with standardized columns: source, target, weight
    
    Parameters
    ----------
    z_threshold : float, optional (default=1.64)
        Z-score threshold for statistical significance (p < 0.05 corresponds to 1.64)
    alpha_percentile : float, optional (default=95)
        Percentile threshold for hierarchical strength (alpha)
    chunk_size : int, optional (default=100000)
        Size of chunks for processing large files
    n_jobs : int, optional (default=None)
        Number of parallel jobs. If None, uses cpu_count()
    source_col : str or int, optional (default=None)
        Column name or index for source nodes. If None, auto-detected.
    target_col : str or int, optional (default=None)
        Column name or index for target nodes. If None, auto-detected.
    sep : str, optional (default=None)
        Separator. If None, auto-detected.
    """
    
    def __init__(self, z_threshold=1.64, alpha_percentile=95, chunk_size=100000, 
                 n_jobs=None, source_col=None, target_col=None, sep=None):
        self.z_threshold = z_threshold
        self.alpha_percentile = alpha_percentile
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs or cpu_count()
        self.source_col = source_col
        self.target_col = target_col
        self.sep = sep
        
        # Internal state variables
        self.bipartite_matrix = None
        self.node_A_list = None
        self.node_B_list = None
        self.lambdas = None
        self.etas = None
        self.source_col_name = None
        self.target_col_name = None
        self.detected_sep = None
        
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
            
            # Remove zero-count separators
            separators = {k: v for k, v in separators.items() if v > 0}
            
            if not separators:
                self.detected_sep = '\t'
            else:
                # Choose the separator with the highest count
                self.detected_sep = max(separators, key=lambda k: separators[k])
            
            return self.detected_sep
    
    def _detect_columns(self, file_path):
        """
        Auto-detect source and target columns.
        Common patterns: ['source', 'target'], ['node1', 'node2'], 
        ['gene', 'disease'], ['col1', 'col2'], or first two columns.
        """
        # Read header
        df_sample = pd.read_csv(file_path, sep=self.detected_sep, nrows=0)
        columns = df_sample.columns.tolist()
        
        # Common column name patterns for bipartite networks
        source_patterns = ['source', 'from', 'node1', 'gene1', 'gene', 'protein', 
                           'col1', 'column1', 'left', 'row', 'start', 'id1', 'id_a']
        target_patterns = ['target', 'to', 'node2', 'gene2', 'disease', 'disease_id',
                           'col2', 'column2', 'right', 'col', 'end', 'id2', 'id_b']
        
        source_col = None
        target_col = None
        
        # Try to find by name
        for col in columns:
            col_lower = str(col).lower().strip()
            if col_lower in source_patterns:
                source_col = col
            if col_lower in target_patterns:
                target_col = col
        
        # If not found by name, use first two columns
        if source_col is None and len(columns) >= 2:
            source_col = columns[0]
            target_col = columns[1]
        elif source_col is None or target_col is None:
            # Fallback: first column is source, second is target
            source_col = columns[0]
            target_col = columns[1] if len(columns) > 1 else columns[0]
        
        self.source_col_name = source_col
        self.target_col_name = target_col
        
        return source_col, target_col
    
    def _process_chunk_load(self, chunk):
        """Processes a chunk of the input file to identify unique nodes."""
        node_A = chunk[self.source_col_name].astype(str).unique()
        node_B = chunk[self.target_col_name].astype(str).unique()
        return (set(node_A), set(node_B))
    
    def _build_csr_chunk(self, chunk, node_A_to_idx, node_B_to_idx, shape):
        """Builds a sparse matrix chunk from a portion of the bipartite network data."""
        mapped_rows = chunk[self.source_col_name].map(node_A_to_idx)
        mapped_cols = chunk[self.target_col_name].map(node_B_to_idx)
        
        valid_mask = ~mapped_rows.isna() & ~mapped_cols.isna()
        rows = mapped_rows[valid_mask].astype(int)
        cols = mapped_cols[valid_mask].astype(int)
        data = np.ones(len(rows), dtype=np.int8)
        
        return coo_matrix((data, (rows, cols)), shape=shape)
    
    def load_bipartite_network(self, file_path):
        """Loads a bipartite network with automatic column and separator detection."""
        start_time = time.time()
        
        # Step 1: Detect separator
        if self.sep is None:
            self._detect_separator(file_path)
        else:
            self.detected_sep = self.sep
        
        # Step 2: Detect columns
        if self.source_col is None or self.target_col is None:
            self._detect_columns(file_path)
        else:
            self.source_col_name = self.source_col
            self.target_col_name = self.target_col
        
        # Read file in chunks with detected separator
        chunks = pd.read_csv(file_path, sep=self.detected_sep, header=0, 
                            chunksize=self.chunk_size)
        
        # First pass: find unique nodes using parallel processing
        with Pool(self.n_jobs) as pool:
            results = pool.map(self._process_chunk_load, chunks)
        
        node_A_set, node_B_set = set(), set()
        for a_set, b_set in results:
            node_A_set.update(a_set)
            node_B_set.update(b_set)
        
        self.node_A_list = sorted(node_A_set, key=str)
        self.node_B_list = sorted(node_B_set, key=str)
        
        node_A_to_idx = {node: idx for idx, node in enumerate(self.node_A_list)}
        node_B_to_idx = {node: idx for idx, node in enumerate(self.node_B_list)}
        
        # Second pass: build CSR matrix in parallel
        chunks = pd.read_csv(file_path, sep=self.detected_sep, header=0, 
                            chunksize=self.chunk_size)
        shape = (len(self.node_A_list), len(self.node_B_list))
        
        build_csr_partial = partial(self._build_csr_chunk, 
                                    node_A_to_idx=node_A_to_idx, 
                                    node_B_to_idx=node_B_to_idx, 
                                    shape=shape)
        
        with Pool(self.n_jobs) as pool:
            coo_matrices = pool.map(build_csr_partial, chunks)
        
        from scipy.sparse import csr_matrix as _csr
        _summed = sum(coo_matrices)
        self.bipartite_matrix = _summed.tocsr() if hasattr(_summed, 'tocsr') else _csr(_summed)  # type: ignore[union-attr]
        
        elapsed = time.time() - start_time
        return elapsed
    
    def apply_bicm(self):
        """Applies the Bipartite Configuration Model (BiCM)."""
        start_time = time.time()
        
        assert self.bipartite_matrix is not None
        coo = self.bipartite_matrix.tocoo()
        edgelist = np.column_stack((coo.row, coo.col))
        
        B = BipartiteGraph()
        B.set_edgelist(edgelist.astype(np.int64))
        B.solve_tool(model="BiCM")
        
        self.lambdas = B.x  # Nodes on side A (rows)
        self.etas = B.y     # Nodes on side B (cols)
        
        # Clean up
        del B, edgelist, coo
        gc.collect()
        
        elapsed = time.time() - start_time
        return elapsed
    
    def _process_z_score_chunk(self, chunk, node_params, other_side_params):
        """Calculates Z-scores for a chunk of edges."""
        rows, cols, data = chunk
        
        sum_other_side_sq = np.sum(other_side_params**2)
        sum_other_side_cube = np.sum(other_side_params**3)
        
        row_params = node_params[rows]
        col_params = node_params[cols]
        
        expected = row_params * col_params * sum_other_side_sq
        variance = expected + (row_params * col_params * sum_other_side_cube)
        std_dev = np.sqrt(variance)
        std_dev[std_dev == 0] = 1e-10
        
        z_scores = (data - expected) / std_dev
        mask = z_scores > self.z_threshold
        
        return (rows[mask], cols[mask], data[mask], z_scores[mask])
    
    def project_and_filter(self, projection_side='A'):
        """
        Projects the bipartite network to a one-mode network and filters edges based on Z-scores.
        
        Parameters
        ----------
        projection_side : str, optional (default='A')
            Which side to project onto ('A' or 'B')
            
        Returns
        -------
        filtered_coo : coo_matrix
            Filtered projected network as COO sparse matrix
        node_list : list
            List of node names for the projected side
        """
        start_time = time.time()
        
        # Project to one-mode
        assert self.bipartite_matrix is not None
        if projection_side == 'A':
            projected = self.bipartite_matrix.dot(self.bipartite_matrix.T)
            node_list = self.node_A_list
            node_params = self.lambdas
            other_side_params = self.etas
        else:
            projected = self.bipartite_matrix.T.dot(self.bipartite_matrix)
            node_list = self.node_B_list
            node_params = self.etas
            other_side_params = self.lambdas
        
        # Convert to COO and remove self-loops
        projected_coo = projected.tocoo()
        upper_mask = projected_coo.row < projected_coo.col
        projected_coo = coo_matrix(
            (projected_coo.data[upper_mask],
             (projected_coo.row[upper_mask], projected_coo.col[upper_mask])),
            shape=projected_coo.shape
        )
        
        if projected_coo.nnz == 0:
            return projected_coo, node_list, time.time() - start_time
        
        # Prepare chunks for parallel Z-score computation
        chunk_size = max(100000, projected_coo.nnz // (self.n_jobs * 10))
        chunks = []
        for i in range(0, projected_coo.nnz, chunk_size):
            chunks.append((
                projected_coo.row[i:i+chunk_size],
                projected_coo.col[i:i+chunk_size],
                projected_coo.data[i:i+chunk_size]
            ))
        
        # Compute Z-scores in parallel
        process_z_partial = partial(self._process_z_score_chunk,
                                    node_params=node_params,
                                    other_side_params=other_side_params)
        
        with Pool(self.n_jobs) as pool:
            results = pool.map(process_z_partial, chunks)
        
        # Combine results
        filtered_rows = np.concatenate([r[0] for r in results]) if results else np.array([])
        filtered_cols = np.concatenate([r[1] for r in results]) if results else np.array([])
        filtered_data = np.concatenate([r[2] for r in results]) if results else np.array([])
        
        filtered_coo = coo_matrix(
            (filtered_data, (filtered_rows, filtered_cols)),
            shape=projected_coo.shape
        )
        
        elapsed = time.time() - start_time
        return filtered_coo, node_list, elapsed
    
    def _process_degree_chunk(self, chunk, num_nodes):
        """Calculates degrees for a chunk of edges."""
        rows, data = chunk
        chunk_degrees = np.zeros(num_nodes)
        np.add.at(chunk_degrees, rows, data)
        return chunk_degrees
    
    def compute_degrees(self, filtered_coo, num_nodes):
        """Computes degrees from filtered sparse matrix in parallel."""
        rows_upper = filtered_coo.row
        data_upper = filtered_coo.data
        rows_lower = filtered_coo.col
        data_lower = filtered_coo.data
        
        all_rows = np.concatenate([rows_upper, rows_lower])
        all_data = np.concatenate([data_upper, data_lower])
        
        chunk_size = max(100000, len(all_rows) // self.n_jobs)
        chunks = []
        for i in range(0, len(all_rows), chunk_size):
            chunks.append((
                all_rows[i:i+chunk_size],
                all_data[i:i+chunk_size]
            ))
        
        process_degree_partial = partial(self._process_degree_chunk, num_nodes=num_nodes)
        
        with Pool(self.n_jobs) as pool:
            results = pool.map(process_degree_partial, chunks)
        
        degrees = sum(results)
        return degrees
    
    def _process_alpha_chunk(self, chunk, degrees, k_max):
        """Calculates hierarchical strength (alpha) for a chunk of edges."""
        rows, cols, data = chunk
        
        row_degrees = degrees[rows] + 1e-10
        col_degrees = degrees[cols] + 1e-10
        
        P_uv = data / col_degrees
        P_vu = data / row_degrees
        
        normalization = np.minimum(row_degrees, col_degrees) / k_max
        alpha = normalization * (P_uv - P_vu)
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        
        return alpha
    
    def compute_hierarchical_strength(self, filtered_coo, degrees):
        """Computes hierarchical strength (alpha) for all edges in parallel."""
        if filtered_coo.nnz == 0:
            return np.array([])
        
        k_max = degrees.max()
        
        chunk_size = max(100000, filtered_coo.nnz // self.n_jobs)
        chunks = []
        for i in range(0, filtered_coo.nnz, chunk_size):
            chunks.append((
                filtered_coo.row[i:i+chunk_size],
                filtered_coo.col[i:i+chunk_size],
                filtered_coo.data[i:i+chunk_size]
            ))
        
        process_alpha_partial = partial(self._process_alpha_chunk, degrees=degrees, k_max=k_max)
        
        with Pool(self.n_jobs) as pool:
            alpha_chunks = pool.map(process_alpha_partial, chunks)
        
        alpha = np.concatenate(alpha_chunks)
        return alpha
    
    def _process_edge_chunk(self, chunk, node_names):
        """Converts a chunk of edge indices back to node names."""
        rows, cols, alphas = chunk
        return [(node_names[r], node_names[c], a) for r, c, a in zip(rows, cols, alphas)]
    
    def build_hierarchical_network(self, filtered_coo, alpha, node_names):
        """Constructs the hierarchical network by filtering edges based on alpha threshold."""
        if filtered_coo.nnz == 0 or len(alpha) == 0:
            return pd.DataFrame(columns=['source', 'target', 'weight']), 0
        
        alpha_threshold = np.percentile(np.abs(alpha), self.alpha_percentile)
        
        # Forward edges (u -> v) where alpha > threshold
        mask = alpha > alpha_threshold
        edges_rows, edges_cols, edges_alphas = filtered_coo.row[mask], filtered_coo.col[mask], alpha[mask]
        
        chunk_size = max(100000, len(edges_rows) // self.n_jobs)
        chunks = []
        for i in range(0, len(edges_rows), chunk_size):
            chunks.append((
                edges_rows[i:i+chunk_size],
                edges_cols[i:i+chunk_size],
                edges_alphas[i:i+chunk_size]
            ))
        
        process_edge_partial = partial(self._process_edge_chunk, node_names=node_names)
        
        with Pool(self.n_jobs) as pool:
            edge_chunks = pool.map(process_edge_partial, chunks)
        
        edges_flat = [edge for chunk in edge_chunks for edge in chunk]
        df_hierarchical = pd.DataFrame(edges_flat, columns=['source', 'target', 'weight'])
        
        # Reverse edges (v -> u) where alpha < -threshold
        mask_reverse = alpha < -alpha_threshold
        rev_edges_rows, rev_edges_cols, rev_edges_alphas = filtered_coo.row[mask_reverse], filtered_coo.col[mask_reverse], alpha[mask_reverse]
        
        rev_edges_flat = []
        for r, c, a in zip(rev_edges_rows, rev_edges_cols, rev_edges_alphas):
            rev_edges_flat.append((node_names[c], node_names[r], -a))
        
        df_hierarchical_reverse = pd.DataFrame(rev_edges_flat, columns=['source', 'target', 'weight'])
        
        # Combine and ensure positive weights
        df_hierarchical = pd.concat([df_hierarchical, df_hierarchical_reverse], ignore_index=True)
        df_hierarchical['weight'] = np.abs(df_hierarchical['weight'])
        
        return df_hierarchical, alpha_threshold
    
    def remove_redundant_edges(self, df_hierarchical):
        """Applies transitive reduction to remove redundant edges."""
        if df_hierarchical.empty:
            return pd.DataFrame(columns=['source', 'target', 'weight']), 0
        
        start_time = time.time()
        
        G = nx.DiGraph()
        edge_weights = {}
        
        chunk_size = 100000
        for i in range(0, len(df_hierarchical), chunk_size):
            chunk = df_hierarchical.iloc[i:i+chunk_size]
            for _, row in chunk.iterrows():
                G.add_edge(row.source, row.target)
                current_weight = edge_weights.get((row.source, row.target), 0)
                edge_weights[(row.source, row.target)] = max(current_weight, row.weight)
        
        # Perform transitive reduction
        G_reduced = nx.transitive_reduction(G)
        
        # Rebuild DataFrame with standardized column names
        cleaned_edges = [(u, v, edge_weights[(u, v)]) for u, v in G_reduced.edges()]
        df_cleaned = pd.DataFrame(cleaned_edges, columns=['source', 'target', 'weight'])
        
        elapsed = time.time() - start_time
        return df_cleaned, elapsed
    
    def get_column_info(self):
        """Returns detected column information."""
        return {
            'separator': self.detected_sep,
            'source_column': self.source_col_name,
            'target_column': self.target_col_name,
            'nodes_A': len(self.node_A_list) if self.node_A_list else 0,
            'nodes_B': len(self.node_B_list) if self.node_B_list else 0
        }
    
    def run(self, input_file, output_file_A, output_file_B=None, logger=None):
        """
        Main method to run the complete hierarchical backbone extraction.
        
        Parameters
        ----------
        input_file : str
            Path to input bipartite network file
        output_file_A : str
            Path for output backbone on side A (CSV with source, target, weight columns)
        output_file_B : str, optional
            Path for output backbone on side B (CSV with source, target, weight columns)
        logger : logging.Logger, optional
            Logger for progress messages
            
        Returns
        -------
        dict
            Statistics about the extraction process
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        
        stats = {
            'success': True,
            'side_A': {},
            'side_B': {},
            'total_time': 0,
            'column_info': {}
        }
        
        total_start = time.time()
        
        try:
            # Step 1: Load bipartite network
            logger.info("[1/7] Loading bipartite network...")
            load_time = self.load_bipartite_network(input_file)
            
            # Log detected columns
            col_info = self.get_column_info()
            stats['column_info'] = col_info
            logger.info(f"  Detected separator: '{col_info['separator']}'")
            logger.info(f"  Source column: '{col_info['source_column']}'")
            logger.info(f"  Target column: '{col_info['target_column']}'")
            logger.info(f"  Nodes in partition A: {col_info['nodes_A']}")
            logger.info(f"  Nodes in partition B: {col_info['nodes_B']}")
            assert self.bipartite_matrix is not None
            logger.info(f"  Loaded in {load_time:.2f}s: {self.bipartite_matrix.shape}, NNZ={self.bipartite_matrix.nnz}")
            stats['load_time'] = load_time
            
            # Step 2: Apply BiCM
            logger.info("[2/7] Applying BiCM...")
            bicm_time = self.apply_bicm()
            logger.info(f"  BiCM completed in {bicm_time:.2f}s")
            stats['bicm_time'] = bicm_time
            
            # Process side A
            logger.info("[3/7] Processing side A...")
            filtered_coo_A, node_list_A, proj_time_A = self.project_and_filter('A')
            logger.info(f"  Side A projection: {filtered_coo_A.nnz} edges after filtering")
            
            if filtered_coo_A.nnz > 0:
                logger.info("[4/7] Computing hierarchical strength (side A)...")
                degrees_A = self.compute_degrees(filtered_coo_A, len(node_list_A) if node_list_A is not None else 0)
                alpha_A = self.compute_hierarchical_strength(filtered_coo_A, degrees_A)
                
                logger.info("[5/7] Building hierarchical network (side A)...")
                df_hierarchical_A, alpha_thresh_A = self.build_hierarchical_network(filtered_coo_A, alpha_A, node_list_A)
                
                logger.info("[6/7] Removing redundant edges (side A)...")
                df_cleaned_A, red_time_A = self.remove_redundant_edges(df_hierarchical_A)
                
                # Save as CSV with source, target, weight columns
                df_cleaned_A.to_csv(output_file_A, index=False, sep='\t')
                
                stats['side_A'] = {
                    'edges_before_filter': self.bipartite_matrix.nnz if self.bipartite_matrix is not None else 0,
                    'edges_projected': filtered_coo_A.nnz,
                    'edges_backbone': len(df_cleaned_A),
                    'alpha_threshold': float(alpha_thresh_A),
                    'alpha_percentile': self.alpha_percentile,
                    'z_threshold': self.z_threshold,
                    'projection_time': proj_time_A,
                    'reduction_time': red_time_A
                }
                logger.info(f"  Side A: {len(df_cleaned_A)} edges in final backbone")
            else:
                # Save empty CSV with standard columns
                pd.DataFrame(columns=['source', 'target', 'weight']).to_csv(output_file_A, index=False, sep='\t')
                stats['side_A'] = {'edges_backbone': 0, 'message': 'No significant edges found'}
                logger.info("  Side A: No significant edges found")
            
            # Process side B if requested
            if output_file_B:
                logger.info("[7/7] Processing side B...")
                filtered_coo_B, node_list_B, proj_time_B = self.project_and_filter('B')
                logger.info(f"  Side B projection: {filtered_coo_B.nnz} edges after filtering")
                
                if filtered_coo_B.nnz > 0:
                    degrees_B = self.compute_degrees(filtered_coo_B, len(node_list_B) if node_list_B is not None else 0)
                    alpha_B = self.compute_hierarchical_strength(filtered_coo_B, degrees_B)
                    
                    df_hierarchical_B, alpha_thresh_B = self.build_hierarchical_network(filtered_coo_B, alpha_B, node_list_B)
                    df_cleaned_B, red_time_B = self.remove_redundant_edges(df_hierarchical_B)
                    
                    # Save as CSV with source, target, weight columns
                    df_cleaned_B.to_csv(output_file_B, index=False, sep='\t')
                    
                    stats['side_B'] = {
                        'edges_projected': filtered_coo_B.nnz,
                        'edges_backbone': len(df_cleaned_B),
                        'alpha_threshold': float(alpha_thresh_B),
                        'projection_time': proj_time_B,
                        'reduction_time': red_time_B
                    }
                    logger.info(f"  Side B: {len(df_cleaned_B)} edges in final backbone")
                else:
                    # Save empty CSV with standard columns
                    pd.DataFrame(columns=['source', 'target', 'weight']).to_csv(output_file_B, index=False, sep='\t')
                    stats['side_B'] = {'edges_backbone': 0, 'message': 'No significant edges found'}
                    logger.info("  Side B: No significant edges found")
            
            stats['total_time'] = time.time() - total_start
            logger.info(f"Completed in {stats['total_time']:.2f}s")
            
        except Exception as e:
            stats['success'] = False
            stats['error'] = str(e)
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        return stats


# Factory function for easy instantiation
def create_hierarchical_extractor(z_threshold=1.64, alpha_percentile=95, chunk_size=100000, 
                                  n_jobs=None, source_col=None, target_col=None, sep=None):
    """
    Factory function to create a HierarchicalBackboneExtractor instance.
    
    Parameters
    ----------
    z_threshold : float, optional (default=1.64)
        Z-score threshold for statistical significance
    alpha_percentile : float, optional (default=95)
        Percentile threshold for hierarchical strength
    chunk_size : int, optional (default=100000)
        Size of chunks for processing
    n_jobs : int, optional
        Number of parallel jobs
    source_col : str or int, optional
        Column name or index for source nodes (auto-detected if None)
    target_col : str or int, optional
        Column name or index for target nodes (auto-detected if None)
    sep : str, optional
        Separator (auto-detected if None)
    
    Returns
    -------
    HierarchicalBackboneExtractor
    """
    return HierarchicalBackboneExtractor(
        z_threshold=z_threshold,
        alpha_percentile=alpha_percentile,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        source_col=source_col,
        target_col=target_col,
        sep=sep
    )