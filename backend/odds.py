"""
Odds Ratio Calculator Class with Memory Optimization
Calculates odds ratios, confidence intervals, and p-values for node pairs based on shared references.
Uses positional column access: iloc[0]=source, iloc[1]=target, iloc[2]=weight
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2, fisher_exact
import os
import logging
from collections import defaultdict
import itertools
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OddsRatioCalculator:
    """
    A class to calculate odds ratios for node pairs based on shared references.
    Optimized for memory efficiency to handle large datasets.
    Uses positional column access: iloc[0]=source, iloc[1]=target, iloc[2]=weight
    """
    
    def __init__(self, chunk_size=50000, batch_size=100000):
        """
        Initialize the OddsRatioCalculator.
        
        Args:
            chunk_size (int): Size of chunks for reading input file
            batch_size (int): Size of batches for processing pairs
        """
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.total_references = 0
        self.node_counts = {}
        self.results = None
        self.reference_to_nodes = None

    def detect_separator(self, file_path):
        """Ανιχνεύει το διαχωριστικό του αρχείου"""
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        
        separators = {'\t': first_line.count('\t'), 
                      ',': first_line.count(','), 
                      ';': first_line.count(';'),
                      ' ': first_line.count(' ')}
        
        sep = max(separators, key=lambda k: separators[k])
        if separators[sep] == 0:
            sep = '\t'
        
        return sep
    
    def has_header(self, file_path, sep):
        """First line is a header only if BOTH columns 0 and 1 are exact
        header keywords. Substring matching incorrectly flagged real entity
        names like 'ALZHEIMER'S_DISEASE' (contains 'disease') as headers."""
        kw = {'source', 'target', 'from', 'to', 'node', 'gene', 'src', 'dst',
              'id', 'name', 'weight', 'score', 'reference', 'disease',
              'col1', 'col2', 'edge'}
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
        if not first_line:
            return False
        parts = first_line.split(sep)
        if len(parts) < 2:
            return False
        col0_is_kw = parts[0].strip().lower() in kw
        col1_is_kw = parts[1].strip().lower() in kw
        if col0_is_kw and col1_is_kw:
            return True
        # Fallback: if line 1 col-2 is non-numeric but line 2 col-2 is numeric,
        # line 1 is almost certainly a header.
        if second_line:
            p1 = parts[2].strip() if len(parts) > 2 else ''
            p2 = second_line.split(sep)
            p2_third = p2[2].strip() if len(p2) > 2 else ''
            try:
                float(p2_third)
                # line 2 col-2 parses; if line 1 col-2 doesn't, header
                try:
                    float(p1) if p1 else None
                    return False
                except ValueError:
                    return True
            except ValueError:
                pass
        return False
    
    def load_data_flexible(self, input_file, reference_column='col1', 
                          node_column=None, weight_column=None,
                          sep=None, header='infer', names=None, encoding='utf-8',
                          progress_callback=None):
        """
        Load and process the input data using POSITIONAL column access.
        Always uses: iloc[0] = source/reference, iloc[1] = target/node, iloc[2] = weight
        
        Args:
            input_file (str): Path to input file
            reference_column (str): Ignored - uses positional access
            node_column (str): Ignored - uses positional access
            weight_column (str): Ignored - uses positional access
            sep (str): Separator (auto-detect if None)
            header (str/int): Header row (0, None, 'infer')
            names (list): Column names if no header
            encoding (str): File encoding
            progress_callback (function): Callback for progress updates
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Reading input file: {input_file}")
            logger.info("Using POSITIONAL column access: iloc[0]=source, iloc[1]=target, iloc[2]=weight")
            
            # Auto-detect separator
            if sep is None:
                sep = self.detect_separator(input_file)
                logger.info(f"  Detected separator: {repr(sep)}")
            
            # Auto-detect header
            if header == 'infer':
                has_header_row = self.has_header(input_file, sep)
                header = 0 if has_header_row else None
                logger.info(f"  Detected header: {header if header == 0 else 'No header'}")
            
            # Count total rows for progress bar (excluding header if present)
            with open(input_file, 'r', encoding=encoding) as f:
                total_rows = sum(1 for _ in f)
                if header == 0:
                    total_rows -= 1
            
            # Read file without column names - use positional access
            chunk_reader = pd.read_csv(
                input_file, 
                sep=sep, 
                header=header,  # type: ignore[arg-type] 
                names=names,
                chunksize=self.chunk_size, 
                encoding=encoding,
                dtype=str  # Read everything as string initially
            )
            
            # Calculate number of chunks for progress bar
            num_chunks = (total_rows // self.chunk_size) + 1
            
            # Initialize data structures
            self.reference_to_nodes = defaultdict(set)
            self.node_counts = defaultdict(int)
            references = set()
            
            processed_rows = 0
            
            # Process chunks with progress bar
            for chunk in tqdm(chunk_reader, total=num_chunks, desc="Processing network"):
                # Convert weight column to numeric
                if chunk.shape[1] > 2:
                    weight_vals = pd.to_numeric(chunk.iloc[:, 2], errors='coerce')
                else:
                    weight_vals = pd.Series([1.0] * len(chunk), index=chunk.index)
                
                # Process each row using POSITIONAL access
                for idx in chunk.index:
                    # POSITIONAL ACCESS: iloc[0] = source/reference, iloc[1] = target/node
                    u = str(chunk.loc[idx, chunk.columns[0]]).strip()  # Source/Reference
                    v = str(chunk.loc[idx, chunk.columns[1]]).strip()  # Target/Node
                    
                    # Get weight using the same index
                    w = weight_vals.loc[idx] if chunk.shape[1] > 2 else 1.0
                    
                    # Skip invalid rows
                    if pd.isna(w) or w <= 0:
                        continue
                    
                    if not u or not v or u == 'nan' or v == 'nan':
                        continue
                    
                    # Store reference -> nodes mapping.
                    # Only count v once per unique reference u, otherwise
                    # duplicate (u,v) rows inflate node_counts and break the
                    # 2x2 table arithmetic (b = node_counts[n1] - a, etc.).
                    references.add(u)
                    if v not in self.reference_to_nodes[u]:
                        self.reference_to_nodes[u].add(v)
                        self.node_counts[v] += 1

                    processed_rows += 1
                
                # Update progress if callback provided
                if progress_callback:
                    progress = min(100, int((processed_rows / total_rows) * 100))
                    progress_callback(progress)
                
                # Log progress
                if processed_rows % (self.chunk_size * 10) == 0:
                    logger.info(f"  Processed {processed_rows} rows...")
            
            self.total_references = len(references)
            
            logger.info(f"Total references: {self.total_references}")
            logger.info(f"Total nodes: {len(self.node_counts)}")
            logger.info(f"Total rows processed: {processed_rows}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_reference_batch(self, refs_batch):
        """
        Process a batch of references to create node pairs.
        
        Args:
            refs_batch: List of (ref, nodes) tuples
            
        Returns:
            List of node pairs with shared counts
        """
        pair_counts = defaultdict(int)
        
        for ref, nodes in refs_batch:
            nodes_list = list(nodes)
            if len(nodes_list) > 1:
                for i in range(len(nodes_list)):
                    for j in range(i + 1, len(nodes_list)):
                        node1 = min(nodes_list[i], nodes_list[j])
                        node2 = max(nodes_list[i], nodes_list[j])
                        pair_counts[(node1, node2)] += 1
        
        return [(node1, node2, count) for (node1, node2), count in pair_counts.items()]
    
    def generate_pairs_in_batches(self):
        """
        Generate node pairs in batches to manage memory.
        Yields batches of pairs for processing.
        """
        assert self.reference_to_nodes is not None
        ref_items = list(self.reference_to_nodes.items())
        total_refs = len(ref_items)
        
        logger.info(f"Generating pairs from {total_refs} references in batches...")
        
        for i in range(0, total_refs, self.batch_size):
            batch = ref_items[i:i + self.batch_size]
            pairs = self._process_reference_batch(batch)
            
            if pairs:
                logger.info(f"Batch {i//self.batch_size + 1}: Generated {len(pairs)} pairs")
                yield pairs
            
            del batch
    
    def calculate_odds_ratios_streaming(self, min_shared=1, min_expected=5, 
                                        output_file=None, max_pairs_to_return=None,
                                        progress_callback=None):
        """
        Calculate odds ratios in streaming mode to minimize memory usage.
        
        Args:
            min_shared (int): Minimum number of shared references
            min_expected (int): Minimum expected value for chi-square
            output_file (str): If provided, save results directly to CSV file
            max_pairs_to_return (int): Maximum number of pairs to return in memory
            progress_callback (function): Callback for progress updates
            
        Returns:
            pd.DataFrame or None: Results if max_pairs_to_return is set, otherwise None
        """
        results_buffer = []
        processed = 0
        skipped = 0
        buffer_size = 10000
        
        # Count total references for progress
        total_refs = len(self.reference_to_nodes) if self.reference_to_nodes is not None else 0
        total_batches = (total_refs // self.batch_size) + 1
        
        logger.info("Calculating odds ratios in streaming mode...")
        
        # Open output file if provided
        f_out = None
        if output_file:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            f_out = open(output_file, 'w', encoding='utf-8')
            # Write TSV header with all fields
            f_out.write("source\ttarget\tweight\todds_ratio\tci_lower\tci_upper\tp_value\tshared_references\tnode1_total\tnode2_total\n")
        
        try:
            batch_num = 0
            
            for pairs_batch in self.generate_pairs_in_batches():
                batch_num += 1
                
                for node1, node2, a in pairs_batch:
                    
                    if a < min_shared:
                        skipped += 1
                        continue
                    
                    b = self.node_counts[node1] - a
                    c = self.node_counts[node2] - a
                    d = self.total_references - (a + b + c)
                    
                    if b <= 0 or c <= 0 or d <= 0:
                        skipped += 1
                        continue
                    
                    total = a + b + c + d
                    expected = (a + b) * (a + c) / total
                    
                    if expected < min_expected:
                        skipped += 1
                        continue
                    
                    try:
                        # Calculate odds ratio
                        or_value = (a * d) / (b * c)
                        
                        # Calculate confidence interval
                        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
                        log_or = np.log(or_value)
                        ci_lower = np.exp(log_or - 1.96 * se)
                        ci_upper = np.exp(log_or + 1.96 * se)
                        
                        # Calculate chi-square
                        chi2_stat = ((a * d - b * c) ** 2) * total / ((a + b) * (a + c) * (b + d) * (c + d))
                        p_value = 1 - chi2.cdf(chi2_stat, df=1)
                        
                        # Create result row with all fields
                        result_row = {
                            'source': node1,
                            'target': node2,
                            'weight': a,  # Shared references as weight
                            'odds_ratio': or_value,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'p_value': p_value,
                            'shared_references': a,
                            'node1_total': self.node_counts[node1],
                            'node2_total': self.node_counts[node2]
                        }
                        
                        # Write to CSV if output_file provided
                        if f_out:
                            f_out.write(f"{node1}\t{node2}\t{a}\t{or_value:.6f}\t{ci_lower:.6f}\t{ci_upper:.6f}\t{p_value:.6e}\t{a}\t{self.node_counts[node1]}\t{self.node_counts[node2]}\n")

                        # Keep in memory if needed
                        elif max_pairs_to_return and len(results_buffer) < max_pairs_to_return:
                            results_buffer.append(result_row)

                        processed += 1
                        if f_out and processed % buffer_size == 0:
                            f_out.flush()
                            logger.info(f"Processed {processed} pairs, skipped {skipped}")
                        
                    except Exception as e:
                        logger.warning(f"Error calculating for pair {node1}-{node2}: {e}")
                        skipped += 1
                
                # Update progress
                if progress_callback:
                    progress = min(100, int((batch_num / total_batches) * 100))
                    progress_callback(progress)
                
                del pairs_batch
            
            logger.info(f"Final - Processed: {processed}, Skipped: {skipped}")
            
            if results_buffer:
                self.results = pd.DataFrame(results_buffer)
                return self.results
            else:
                self.results = None
                return None
                
        finally:
            if f_out:
                f_out.close()
    
    def run_full_analysis_flexible(self, input_file, reference_column='col1', 
                                   node_column=None, weight_column=None,
                                   sep=None, header='infer', names=None,
                                   output_file=None, p_threshold=0.05, or_threshold=1.0,
                                   return_top_n=100, progress_callback=None):
        """
        Run complete odds ratio analysis with POSITIONAL column access.
        Always uses: iloc[0]=source, iloc[1]=target, iloc[2]=weight
        
        Args:
            input_file (str): Path to input file
            reference_column (str): Ignored - uses positional access
            node_column (str): Ignored - uses positional access
            weight_column (str): Ignored - uses positional access
            sep (str): Separator (auto-detect if None)
            header (str/int): Header row (0, None, 'infer')
            names (list): Column names if no header
            output_file (str): Path to output CSV file
            p_threshold (float): P-value threshold for significance
            or_threshold (float): Odds ratio threshold
            return_top_n (int): Number of top pairs to return in response
            progress_callback (function): Callback for progress updates
            
        Returns:
            dict: Results dictionary with statistics
        """
        # Load data with positional access
        if not self.load_data_flexible(
            input_file, 
            reference_column=reference_column,
            node_column=node_column,
            weight_column=weight_column,
            sep=sep,
            header=header,
            names=names,
            progress_callback=progress_callback
        ):
            return {'success': False, 'message': 'Failed to load data'}
        
        # Calculate odds ratios and save directly to CSV
        self.calculate_odds_ratios_streaming(
            output_file=output_file,
            min_shared=1,
            min_expected=5,
            progress_callback=progress_callback
        )
        
        # Read statistics from output file
        stats = self._calculate_stats_from_file(output_file, p_threshold, or_threshold)
        
        # Get top pairs
        top_pairs = self._get_top_pairs_from_file(output_file, return_top_n)
        
        return {
            'success': True,
            'total_references': self.total_references,
            'total_nodes': len(self.node_counts),
            'total_pairs': stats.get('total_pairs', 0),
            'significant_pairs': stats.get('significant_pairs', 0),
            'stats': stats,
            'top_pairs': top_pairs,
            'output_file': output_file
        }
    
    def _calculate_stats_from_file(self, file_path, p_threshold=0.05, or_threshold=1.0):
        """
        Calculate statistics from the output CSV file.
        """
        stats = {
            'total_pairs': 0,
            'significant_pairs': 0,
            'or_values': [],
            'p_values': []
        }
        
        try:
            # Read file in chunks — output is TSV
            sep = '\t' if str(file_path).endswith('.tsv') else ','
            for chunk in pd.read_csv(file_path, sep=sep, chunksize=10000):
                stats['total_pairs'] += len(chunk)
                stats['significant_pairs'] += sum((chunk['p_value'] < p_threshold) & 
                                                 (chunk['odds_ratio'] >= or_threshold))
                
                if len(stats['or_values']) < 10000:
                    stats['or_values'].extend(chunk['odds_ratio'].head(10000 - len(stats['or_values'])).tolist())
                if len(stats['p_values']) < 10000:
                    stats['p_values'].extend(chunk['p_value'].head(10000 - len(stats['p_values'])).tolist())
            
            if stats['or_values']:
                stats['or_mean'] = np.mean(stats['or_values'])
                stats['or_median'] = np.median(stats['or_values'])
                stats['or_std'] = np.std(stats['or_values'])
                stats['or_min'] = min(stats['or_values'])
                stats['or_max'] = max(stats['or_values'])
            
            if stats['p_values']:
                stats['p_min'] = min(stats['p_values'])
                stats['p_max'] = max(stats['p_values'])
            
            del stats['or_values']
            del stats['p_values']
            
        except Exception as e:
            logger.error(f"Error calculating stats from file: {e}")
        
        return stats
    
    def _get_top_pairs_from_file(self, file_path, n=100):
        """
        Stream the output CSV in chunks, keep the top-n by odds_ratio in a heap.
        Avoids loading the entire file into memory.
        """
        import heapq
        try:
            heap = []  # min-heap of (odds_ratio, idx, row_dict)
            counter = 0
            sep = '\t' if str(file_path).endswith('.tsv') else ','
            for chunk in pd.read_csv(file_path, sep=sep, chunksize=50000):
                for _, row in chunk.iterrows():
                    item = (
                        float(row['odds_ratio']),
                        counter,
                        {
                            'source': row['source'],
                            'target': row['target'],
                            'weight': row['weight'],
                            'odds_ratio': float(row['odds_ratio']),
                            'p_value': float(row['p_value']),
                        }
                    )
                    counter += 1
                    if len(heap) < n:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
            return [r for _, _, r in sorted(heap, key=lambda x: -x[0])]
        except Exception as e:
            logger.error(f"Error getting top pairs: {e}")
            return []
    
    # ------------------------------------------------------------------
    # Bipartite dual projection: gene-gene AND disease-disease in one pass
    # ------------------------------------------------------------------
    def _load_bipartite(self, input_file, sep=None, encoding='utf-8'):
        """Load a bipartite gene-disease edgelist once, building BOTH:
            col0_to_col1  (e.g. gene -> set of diseases)
            col1_to_col0  (e.g. disease -> set of genes)
        Returns (col0_to_col1, col1_to_col0, n_total_rows).
        """
        if sep is None:
            sep = self.detect_separator(input_file)
        has_hdr = self.has_header(input_file, sep)
        header_arg = 0 if has_hdr else None
        logger.info(f"_load_bipartite: sep={repr(sep)}, header={'yes' if has_hdr else 'no'}")

        col0_to_col1 = defaultdict(set)
        col1_to_col0 = defaultdict(set)
        n_rows = 0

        reader = pd.read_csv(
            input_file, sep=sep, header=header_arg,  # type: ignore[arg-type]
            chunksize=self.chunk_size, dtype=str,
            encoding=encoding, engine='python', skipinitialspace=True,
        )
        for chunk in reader:
            if chunk.shape[1] < 2:
                continue
            for idx in chunk.index:
                u = str(chunk.iloc[:, 0].loc[idx]).strip()
                v = str(chunk.iloc[:, 1].loc[idx]).strip()
                if not u or not v or u.lower() in ('nan', 'none') or v.lower() in ('nan', 'none'):
                    continue
                if v not in col0_to_col1[u]:
                    col0_to_col1[u].add(v)
                    col1_to_col0[v].add(u)
                    n_rows += 1
        logger.info(f"Loaded {n_rows} unique edges, {len(col0_to_col1)} col-0 entities, {len(col1_to_col0)} col-1 entities")
        return col0_to_col1, col1_to_col0, n_rows

    def _compute_projection(self, ref_to_nodes, output_file,
                            min_shared=1, min_expected=None,
                            p_threshold=0.05, or_threshold=1.0):
        """For a given reference->nodes mapping, compute odds-ratio pairs
        between nodes that share references using Fisher's exact test
        (statistically valid for sparse contingency tables — unlike
        chi-square which requires expected cell counts >= 5).

        Write 4-column TSV: source  target  odds_ratio  p_value
        `min_expected` is accepted for API compatibility but ignored
        (Fisher's exact has no min-count requirement).
        Returns number of pairs written.
        """
        # node_counts[v] = number of unique references containing v
        node_counts = defaultdict(int)
        for _ref, nodes in ref_to_nodes.items():
            for v in nodes:
                node_counts[v] += 1
        total_refs = len(ref_to_nodes)

        # Aggregate pair counts across ALL references (single dict, correct
        # across batches — avoids the cross-batch bug of the original code).
        pair_counts = defaultdict(int)
        for _ref, nodes in ref_to_nodes.items():
            if len(nodes) < 2:
                continue
            nodes_list = sorted(nodes)
            for i in range(len(nodes_list)):
                ni = nodes_list[i]
                for j in range(i + 1, len(nodes_list)):
                    pair_counts[(ni, nodes_list[j])] += 1

        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        written = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("source\ttarget\tweight\tpvalue\n")
            for (n1, n2), a in pair_counts.items():
                if a < min_shared:
                    continue
                b = node_counts[n1] - a
                c = node_counts[n2] - a
                d = total_refs - (a + b + c)
                # Fisher's exact handles zero cells, but b,c,d must be >= 0
                if b < 0 or c < 0 or d < 0:
                    continue
                try:
                    _res = fisher_exact(
                        [[a, b], [c, d]], alternative='two-sided'
                    )
                    p_val = float(_res[1])  # type: ignore[arg-type]
                    if np.isnan(p_val):
                        continue
                    # Haldane-Anscombe corrected OR (finite even with zero cells)
                    or_val = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
                    # Significance filter: only keep positively associated, significant pairs
                    if p_val >= p_threshold or or_val <= or_threshold:
                        continue
                    f.write(f"{n1}\t{n2}\t{or_val:.6f}\t{p_val:.6e}\n")
                    written += 1
                except Exception as e:
                    logger.warning(f"Skipped pair {n1}-{n2}: {e}")
        logger.info(f"Wrote {written} pairs to {output_file}")
        return written

    def run_dual_bipartite_analysis(self, input_file,
                                    col0_pairs_output,   # gene-gene (shared diseases, if col0=gene)
                                    col1_pairs_output,   # disease-disease (shared genes, if col0=gene)
                                    sep=None, encoding='utf-8',
                                    min_shared=1, min_expected=5,
                                    p_threshold=0.05, or_threshold=1.0,
                                    progress_callback=None):
        """Run BOTH projections of a bipartite input in a single pass.

        Convention given input file with iloc[0]=A, iloc[1]=B:
          - col0_pairs_output: pairs of A-entities, weighted by shared B's
            (i.e. use B as reference, A as node)
          - col1_pairs_output: pairs of B-entities, weighted by shared A's
            (i.e. use A as reference, B as node)

        For a gene-disease file with iloc[0]=gene, iloc[1]=disease:
          col0_pairs_output → gene-gene (shared diseases)
          col1_pairs_output → disease-disease (shared genes)
        """
        try:
            col0_to_col1, col1_to_col0, _n = self._load_bipartite(
                input_file, sep=sep, encoding=encoding
            )
            if progress_callback:
                progress_callback(40)

            # gene-gene: nodes are col-0 entities, references are col-1 entities
            n_col0_pairs = self._compute_projection(
                ref_to_nodes=col1_to_col0,
                output_file=col0_pairs_output,
                min_shared=min_shared, min_expected=min_expected,
                p_threshold=p_threshold, or_threshold=or_threshold,
            )
            if progress_callback:
                progress_callback(70)

            # disease-disease: nodes are col-1 entities, references are col-0 entities
            n_col1_pairs = self._compute_projection(
                ref_to_nodes=col0_to_col1,
                output_file=col1_pairs_output,
                min_shared=min_shared, min_expected=min_expected,
                p_threshold=p_threshold, or_threshold=or_threshold,
            )
            if progress_callback:
                progress_callback(100)

            return {
                'success': True,
                'col0_entities': len(col0_to_col1),
                'col1_entities': len(col1_to_col0),
                'col0_pairs':    n_col0_pairs,
                'col1_pairs':    n_col1_pairs,
                'col0_output':   col0_pairs_output,
                'col1_output':   col1_pairs_output,
            }
        except Exception as e:
            import traceback
            logger.error(f"Dual bipartite analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}

    # Για συμβατότητα με παλιό κώδικα
    def run_full_analysis(self, input_file, reference_column='col1',
                         output_file=None, p_threshold=0.05, or_threshold=1.0,
                         return_top_n=100):
        """Παλιά μέθοδος για συμβατότητα"""
        return self.run_full_analysis_flexible(
            input_file=input_file,
            reference_column=reference_column,
            output_file=output_file,
            p_threshold=p_threshold,
            or_threshold=or_threshold,
            return_top_n=return_top_n,
            sep='\t',
            header='infer',  # Auto-detect header
            names=None
        )