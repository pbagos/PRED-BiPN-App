"""
Odds Ratio Calculator Class with Memory Optimization
Calculates odds ratios, confidence intervals, and p-values for node pairs based on shared references.
Uses positional column access: iloc[0]=source, iloc[1]=target, iloc[2]=weight
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2
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
        """Ελέγχει αν η πρώτη γραμμή είναι header"""
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        
        parts = first_line.split(sep)
        
        # Αν η πρώτη γραμμή περιέχει κείμενο που μοιάζει με header
        header_keywords = ['source', 'target', 'from', 'to', 'node', 'gene', 
                          'weight', 'value', 'score', 'name', 'id', 'disease',
                          'reference', 'col1', 'col2']
        
        for part in parts:
            part_lower = part.lower()
            if any(keyword in part_lower for keyword in header_keywords):
                return True
        
        # Έλεγχος αν η δεύτερη γραμμή έχει αριθμούς
        try:
            f.seek(0)
            second_line = f.readline().strip()  # Διάβασε ξανά την πρώτη
            second_line = f.readline().strip()  # Διάβασε τη δεύτερη
            
            if second_line:
                second_parts = second_line.split(sep)
                # Αν η δεύτερη γραμμή έχει αριθμητικά βάρη, πιθανά η πρώτη είναι header
                if len(second_parts) > 2:
                    try:
                        float(second_parts[2])
                        return True
                    except:
                        pass
        except:
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
                    
                    # Store reference -> nodes mapping
                    references.add(u)
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
                            
                            processed += 1
                            if processed % buffer_size == 0:
                                f_out.flush()
                                logger.info(f"Processed {processed} pairs, skipped {skipped}")
                        
                        # Keep in memory if needed
                        elif max_pairs_to_return and len(results_buffer) < max_pairs_to_return:
                            results_buffer.append(result_row)
                        
                        processed += 1
                        
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
            # Read CSV file in chunks
            for chunk in pd.read_csv(file_path, chunksize=10000):
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
        Get top n pairs by odds ratio from CSV file.
        """
        try:
            df = pd.read_csv(file_path)
            if len(df) > n * 10:
                df = df.nlargest(n * 10, 'odds_ratio')
            
            df = df.nlargest(n, 'odds_ratio')
            return df[['source', 'target', 'weight', 'odds_ratio', 'p_value']].head(n).to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting top pairs: {e}")
            return []
    
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