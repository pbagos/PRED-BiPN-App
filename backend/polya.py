"""
Polya Filter - Αριθμητικά Σταθερή Υλοποίηση για Εξαγωγή Backbone
Με αυτόματη ανίχνευση, POSITIONAL ACCESS και CSV output με source, target
"""

import numpy as np
import scipy.special as sp
from typing import Tuple, Dict, Union, List, Optional
import os
import time
from statsmodels.stats.multitest import multipletests
import pandas as pd
from tqdm import tqdm
import gc
import logging


class PolyaFilter:
    """
    Pólya Filter: Αριθμητικά σταθερή υλοποίηση για εξαγωγή backbone.
    Χρησιμοποιεί positional access (iloc[0], iloc[1], iloc[2]) χωρίς προκαθορισμένα ονόματα στηλών.
    """
    
    def __init__(self, significance_level: float = 0.05, method: str = 'fdr', logger: Optional[logging.Logger] = None):
        """
        Αρχικοποίηση του Polya Filter.
        
        Args:
            significance_level (float): Επίπεδο σημαντικότητας (α_u)
            method (str): Μέθοδος πολλαπλών συγκρίσεων ('bonferroni' ή 'fdr')
            logger (logging.Logger): Optional logger instance
        """
        self.alpha_u = significance_level
        self.method = method.lower()
        self.logger = logger or logging.getLogger(__name__)
        self.adj_dict = {}
        self.node_props = {}
        self.stats = {}
        
        # Column detection info
        self.detected_sep = None
        self.has_header = None
        self.num_columns = None
        
        if self.method not in ['bonferroni', 'fdr']:
            raise ValueError("Η μέθοδος διόρθωσης πρέπει να είναι 'bonferroni' ή 'fdr'")
    
    def _log_message(self, message: str, level: str = 'info'):
        """Log message to both console and logger if available."""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
        else:
            print(message)
    
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
    
    def _detect_header(self, file_path, sep):
        """Detect if file has header by analyzing first line."""
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                parts = first_line.split(sep)
                
                # Check if first line contains words that might be headers
                header_likely = False
                for part in parts:
                    try:
                        float(part)
                    except ValueError:
                        header_likely = True
                        break
                
                self.has_header = header_likely
                self.num_columns = len(parts)
                
                return self.has_header
        except:
            self.has_header = False
            return False
    
    def _log_polya_pmf(self, w: int, k: int, s: int, a: float) -> float:
        """
        Υπολογίζει τον ΛΟΓΑΡΙΘΜΟ της συνάρτησης πιθανότητας P(w|k,s,a).
        Αριθμητικά σταθερή μέθοδος.
        """
        if w < 0 or w > s or s <= 0 or k <= 1 or a <= 0:
            return -np.inf
            
        alpha = 1.0 / a
        beta = (k - 1.0) / a
        
        try:
            # log(binomial(s, w)) = lgamma(s+1) - lgamma(w+1) - lgamma(s-w+1)
            log_binom = sp.gammaln(s + 1) - sp.gammaln(w + 1) - sp.gammaln(s - w + 1)
            
            # log(B(w + alpha, s - w + beta)) - log(B(alpha, beta))
            log_beta_ratio = sp.betaln(w + alpha, s - w + beta) - sp.betaln(alpha, beta)
            
            return log_binom + log_beta_ratio
        except ValueError:
            return -np.inf
    
    def polya_pvalue(self, w: float, k: int, s: float, a: float) -> float:
        """
        Υπολογίζει την p-value π_P = P(X >= w) αυστηρά ως log-άθροισμα (Log-Sum-Exp).
        """
        w_int = int(round(w))
        s_int = int(round(s))
        
        if k == 1:
            return 0.0 if w_int > 0 else 1.0
        if w_int <= 0:
            return 1.0
        if w_int > s_int:
            return 0.0
            
        log_probs = []
        for x in range(w_int, s_int + 1):
            log_prob = self._log_polya_pmf(x, k, s_int, a)
            if log_prob > -np.inf:
                log_probs.append(log_prob)
        
        if not log_probs:
            return 0.0
            
        log_max = max(log_probs)
        s_exp = sum(np.exp(log_p - log_max) for log_p in log_probs)
        
        log_cumulative_prob = log_max + np.log(s_exp)
        p_value = np.exp(log_cumulative_prob)
        
        return min(max(p_value, 0.0), 1.0)

    def compute_link_significance(self, w: float, node1: Tuple[int, float], 
                                  node2: Tuple[int, float], a: float) -> Tuple[float, float]:
        """Υπολογισμός p-values από τις δύο οπτικές γωνίες των κόμβων."""
        k1, s1 = node1
        k2, s2 = node2
        
        p1 = self.polya_pvalue(w, k1, s1, a)
        p2 = self.polya_pvalue(w, k2, s2, a)
            
        return p1, p2
    
    def load_network_data(self, file_path: str, directed: bool = False, chunksize: int = 100000):
        """
        Φορτώνει δεδομένα από αρχείο χρησιμοποιώντας POSITIONAL ACCESS (iloc[0], iloc[1], iloc[2]).
        
        Args:
            file_path (str): Διαδρομή προς το αρχείο εισόδου
            directed (bool): Αν το δίκτυο είναι κατευθυνόμενο
            chunksize (int): Μέγεθος chunk για ανάγνωση
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Το αρχείο {file_path} δεν βρέθηκε.")
        
        # Step 1: Detect separator
        self._detect_separator(file_path)
        self._log_message(f"  Detected separator: {repr(self.detected_sep)}")
        
        # Step 2: Detect header
        self._detect_header(file_path, self.detected_sep)
        self._log_message(f"  Detected header: {'Yes' if self.has_header else 'No'}")
        
        # Step 3: Count rows for progress bar
        self._log_message("  Counting rows...")
        total_rows = 0
        with open(file_path, 'r') as f:
            if self.has_header:
                next(f)
            for _ in f:
                total_rows += 1
        self._log_message(f"  Total rows: {total_rows:,}")
        
        # Step 4: FIRST PASS - Read all edges and compute node properties
        self._log_message("\n" + "="*60)
        self._log_message("FIRST PASS: Computing node properties")
        self._log_message("="*60)
        
        node_properties = {}
        edge_list = []
        
        chunk_reader = pd.read_csv(
            file_path,
            sep=self.detected_sep,
            header=0 if self.has_header else None,
            chunksize=chunksize,
            dtype=str
        )
        
        total_chunks = (total_rows // chunksize) + 1
        
        for chunk in tqdm(chunk_reader, total=total_chunks, desc="  Processing"):
            for _, row in chunk.iterrows():
                # POSITIONAL ACCESS: iloc[0], iloc[1], iloc[2]
                u = str(row.iloc[0]).strip()
                v = str(row.iloc[1]).strip()
                
                # Try to get weight from iloc[2], default to 1.0 if not available
                if len(row) > 2:
                    try:
                        w = float(row.iloc[2])
                    except (ValueError, TypeError):
                        w = 1.0
                else:
                    w = 1.0
                
                # Skip invalid rows
                if not u or not v or u == 'nan' or v == 'nan' or w <= 0:
                    continue
                
                # Skip self-loops
                if u == v:
                    continue
                
                # Normalize for undirected networks
                if not directed and u > v:
                    u, v = v, u
                
                edge_list.append((u, v, w))
                
                # Initialize node properties
                if u not in node_properties:
                    node_properties[u] = [0, 0.0]  # [degree, strength]
                if not directed and v not in node_properties:
                    node_properties[v] = [0, 0.0]
                
                node_properties[u][0] += 1
                node_properties[u][1] += w
                
                if not directed and u != v:
                    node_properties[v][0] += 1
                    node_properties[v][1] += w
        
        # Convert to tuples
        self.node_props = {node: (int(props[0]), props[1]) 
                          for node, props in node_properties.items() if props[0] > 0}
        
        self._log_message(f"\n  Found {len(self.node_props):,} unique nodes")
        
        # Step 5: SECOND PASS - Build adjacency dictionary with unique edges
        self._log_message("\n" + "="*60)
        self._log_message("SECOND PASS: Building adjacency dictionary")
        self._log_message("="*60)
        
        self.adj_dict = {}
        processed_edges = set()
        
        for u, v, w in tqdm(edge_list, desc="  Building edges"):
            edge_key = (u, v)
            
            if edge_key in processed_edges:
                continue
            
            self.adj_dict[edge_key] = w
            processed_edges.add(edge_key)
        
        self._log_message(f"\n  Unique edges: {len(self.adj_dict):,}")
        
        self.stats['input_edges'] = len(edge_list)
        self.stats['unique_edges'] = len(self.adj_dict)
        self.stats['nodes'] = len(self.node_props)
        
        return len(self.adj_dict)

    def extract_backbone(self, a: float = 1.0, directed: bool = False) -> List[Tuple]:
        """
        Εξάγει τη ραχοκοκαλιά (backbone) εφαρμόζοντας τη διόρθωση πολλαπλών συγκρίσεων.
        """
        if not self.adj_dict:
            raise ValueError("Τα δεδομένα του δικτύου δεν έχουν φορτωθεί.")
            
        num_links = len(self.adj_dict)
        num_tests = num_links if directed else 2 * num_links  # L
        
        p_values = []
        edges_data = []
        
        self._log_message("\n" + "="*60)
        self._log_message(f"CALCULATING P-VALUES (L = {num_tests:,})")
        self._log_message(f"  Parameter a = {a}")
        self._log_message("="*60)
        
        # Υπολογισμός p-values για όλες τις ακμές
        for (i, j), weight in tqdm(self.adj_dict.items(), desc="  Computing p-values"):
            if i not in self.node_props or j not in self.node_props: 
                continue
                
            node_i = self.node_props[i]
            node_j = self.node_props[j]
            
            p1, p2 = self.compute_link_significance(weight, node_i, node_j, a)
            
            p_values.append(min(p1, p2))
            edges_data.append(((i, j), weight, p1, p2))
            
        self.stats['total_p_values'] = len(p_values)
        
        # Εφαρμογή διόρθωσης πολλαπλών συγκρίσεων
        self._log_message("\n" + "="*60)
        self._log_message(f"MULTIPLE COMPARISON CORRECTION - Method: {self.method.upper()}")
        self._log_message("="*60)
        
        rejected = np.array([], dtype=bool)
        if not p_values:
            return []

        if self.method == 'bonferroni':
            thresh = self.alpha_u / num_tests
            rejected = np.array(p_values) < thresh
            self._log_message(f"  Bonferroni threshold (α_u/L): {thresh:.6e}")
            self._log_message(f"  Significant edges: {sum(rejected):,}")
            self.stats['threshold'] = thresh
            
        elif self.method == 'fdr':
            if not p_values: 
                return []
            rejected, p_corrected, _, _ = multipletests(p_values, alpha=self.alpha_u, method='fdr_bh')
            self._log_message(f"  FDR (Benjamini/Hochberg) applied.")
            self._log_message(f"  Significant edges: {sum(rejected):,}")
            self.stats['significant_fdr'] = sum(rejected)
        
        # Δημιουργία λίστας με τις σημαντικές ακμές
        backbone_edges = []
        for idx, (edge, weight, p1, p2) in enumerate(edges_data):
            if rejected[idx]:
                i, j = edge
                r_ratio = self.compute_r_ratio(weight, self.node_props[i], self.node_props[j])
                min_p = min(p1, p2)
                backbone_edges.append((i, j, weight, min_p, p1, p2, r_ratio))
        
        self.stats['backbone_edges'] = len(backbone_edges)
        self.stats['retained_percentage'] = (len(backbone_edges) / num_links) * 100 if num_links > 0 else 0
                
        self._log_message(f"\n  Backbone edges: {len(backbone_edges):,} / {num_links:,} ({self.stats['retained_percentage']:.2f}%)")
        return backbone_edges
    
    def compute_r_ratio(self, w: float, node1_props: Tuple[int, float], 
                        node2_props: Tuple[int, float]) -> float:
        """
        Υπολογίζει τον λόγο r = max(r_i, r_j) (Eq. 4).
        """
        r_values = []
        
        k1, s1 = node1_props
        if s1 > 0: 
            r_values.append((w / s1) * k1)
            
        k2, s2 = node2_props
        if s2 > 0: 
            r_values.append((w / s2) * k2)
            
        return max(r_values) if r_values else 0

    def save_results(self, backbone_edges: List[Tuple], output_file: str):
        """
        Αποθήκευση αποτελεσμάτων σε CSV αρχείο με στήλες: source, target, weight, p_value.
        """
        if not backbone_edges:
            self._log_message("  No results to save.")
            return
        
        try:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            
            self._log_message("\n" + "="*60)
            self._log_message("SAVING RESULTS")
            self._log_message("="*60)
            
            # Save as CSV with standard columns
            df_results = []
            for node1, node2, w, min_p, p1, p2, r in backbone_edges:
                df_results.append([node1, node2, w, min_p])
            
            df = pd.DataFrame(df_results, columns=['source', 'target', 'weight', 'p_value'])
            df.to_csv(output_file, index=False, sep='\t')
            
            self._log_message(f"  Main results: {output_file}")
            self._log_message(f"     Columns: source, target, weight, p_value")
            self._log_message(f"     Total edges: {len(backbone_edges):,}")
            
            # Also save detailed results with all statistics
            detailed_file = output_file.replace('.tsv', '_detailed.tsv')
            df_detailed = pd.DataFrame(backbone_edges, 
                                        columns=['source', 'target', 'weight', 
                                                'p_value_min', 'p_value_node1', 
                                                'p_value_node2', 'ratio_r'])
            df_detailed.to_csv(detailed_file, index=False, sep='\t')
            self._log_message(f"  Detailed results: {detailed_file}")
            
            # Save statistics summary
            stats_file = output_file.replace('.tsv', '_stats.txt')
            self.save_summary_stats(stats_file)
            
            self.stats['output_file'] = output_file
            self.stats['detailed_file'] = detailed_file
            
        except Exception as e:
            self._log_message(f"  Error saving results: {e}", 'error')
    
    def save_summary_stats(self, output_file: str):
        """Αποθήκευση στατιστικών σύνοψης."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("POLYA FILTER - ANALYSIS STATISTICS\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("EXECUTION PARAMETERS\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Correction method: {self.method.upper()}\n")
                f.write(f"  Significance level (α): {self.alpha_u}\n")
                f.write(f"  Separator: {repr(self.detected_sep)}\n")
                f.write(f"  Header: {'Yes' if self.has_header else 'No'}\n\n")
                
                f.write("NETWORK STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Input edges (with duplicates): {self.stats.get('input_edges', 0):,}\n")
                f.write(f"  Unique edges: {self.stats.get('unique_edges', 0):,}\n")
                f.write(f"  Nodes: {self.stats.get('nodes', 0):,}\n")
                f.write(f"  Total p-values calculated: {self.stats.get('total_p_values', 0):,}\n\n")
                
                f.write("BACKBONE RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Backbone edges: {self.stats.get('backbone_edges', 0):,}\n")
                f.write(f"  Retention percentage: {self.stats.get('retained_percentage', 0):.2f}%\n")
                
                if 'significant_fdr' in self.stats:
                    f.write(f"  Significant edges (FDR): {self.stats['significant_fdr']}\n")
                if 'threshold' in self.stats:
                    f.write(f"  Bonferroni threshold: {self.stats['threshold']:.6e}\n\n")
                
                f.write("EXECUTION TIME\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Total time: {self.stats.get('total_time', 0):.2f} seconds\n\n")
                
                f.write("OUTPUT FILES\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Main results: {self.stats.get('output_file', 'N/A')}\n")
                f.write(f"  Detailed results: {self.stats.get('detailed_file', 'N/A')}\n")
                f.write(f"  Statistics: {output_file}\n")
                
            self._log_message(f"  Statistics saved to: {output_file}")
            
        except Exception as e:
            self._log_message(f"  Error saving statistics: {e}", 'error')
    
    def get_column_info(self):
        """Returns information about detected columns."""
        return {
            'separator': self.detected_sep,
            'has_header': self.has_header,
            'num_columns': self.num_columns,
            'source_column': 'Column 0 (iloc[0]) - first column',
            'target_column': 'Column 1 (iloc[1]) - second column',
            'weight_column': 'Column 2 (iloc[2]) - third column (or default=1.0)',
            'nodes': self.stats.get('nodes', 0),
            'edges': self.stats.get('unique_edges', 0)
        }
    
    def run_full_analysis(self, input_file: str, output_file: str,
                          a: float = 1.0, directed: bool = False,
                          chunksize: int = 100000) -> Dict:
        """
        Εκτελεί πλήρη ανάλυση Polya Filter με POSITIONAL ACCESS.
        
        Args:
            input_file (str): Διαδρομή προς το αρχείο εισόδου
            output_file (str): Διαδρομή για το αρχείο εξόδου (CSV)
            a (float): Παράμετρος Polya (a=1.0 για Disparity Filter)
            directed (bool): Αν το δίκτυο είναι κατευθυνόμενο
            chunksize (int): Μέγεθος chunk για ανάγνωση
            
        Returns:
            Dict: Λεξικό με αποτελέσματα και στατιστικά
        """
        start_time = time.time()
        
        self._log_message("\n" + "="*60)
        self._log_message("POLYA FILTER - BACKBONE EXTRACTION")
        self._log_message("="*60)
        self._log_message(f"  Input file: {input_file}")
        self._log_message(f"  Output file: {output_file}")
        self._log_message(f"  Method: {self.method.upper()}")
        self._log_message(f"  Significance level: {self.alpha_u}")
        self._log_message(f"  Parameter a: {a}")
        self._log_message(f"  Directed: {'Yes' if directed else 'No'}")
        self._log_message("="*60)
        
        try:
            # Load data with positional access
            self.load_network_data(input_file, directed=directed, chunksize=chunksize)
            
            # Extract backbone
            backbone_edges = self.extract_backbone(a, directed)
            
            # Save results as CSV with source, target, weight, p_value
            self.save_results(backbone_edges, output_file)
            
            total_time = time.time() - start_time
            self.stats['total_time'] = total_time
            
            self._log_message("\n" + "="*60)
            self._log_message("ANALYSIS COMPLETED SUCCESSFULLY")
            self._log_message("="*60)
            self._log_message(f"  Total time: {total_time:.2f} seconds")
            self._log_message(f"  Results saved to: {output_file}")
            self._log_message("="*60 + "\n")
            
            return {
                'success': True,
                'input_file': input_file,
                'output_file': output_file,
                'edges_input': self.stats.get('unique_edges', 0),
                'edges_backbone': self.stats.get('backbone_edges', 0),
                'retained_percentage': self.stats.get('retained_percentage', 0),
                'nodes': self.stats.get('nodes', 0),
                'method': self.method,
                'alpha': self.alpha_u,
                'a_parameter': a,
                'total_time': total_time,
                'column_info': self.get_column_info(),
                'stats': self.stats
            }
            
        except Exception as e:
            self._log_message(f"\nERROR IN ANALYSIS: {e}", 'error')
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }


# Factory function with logger support
def create_polya_filter(significance_level: float = 0.05, 
                        method: str = 'fdr', 
                        logger: Optional[logging.Logger] = None) -> PolyaFilter:
    """
    Create and return a PolyaFilter instance.
    
    Args:
        significance_level (float): Significance level (α_u)
        method (str): Multiple comparison method ('bonferroni' or 'fdr')
        logger (logging.Logger): Optional logger instance
        
    Returns:
        PolyaFilter: Configured PolyaFilter instance
    """
    return PolyaFilter(
        significance_level=significance_level,
        method=method,
        logger=logger
    )


# Backward compatibility function
def run_polya_filter(input_file: str, output_file: str, 
                     alpha: float = 0.05, method: str = 'fdr',
                     a: float = 1.0, directed: bool = False,
                     logger: Optional[logging.Logger] = None) -> Dict:
    """
    Run Polya filter analysis.
    
    Args:
        input_file (str): Path to input file
        output_file (str): Path to output file (CSV)
        alpha (float): Significance level
        method (str): Correction method ('bonferroni' or 'fdr')
        a (float): Polya parameter
        directed (bool): Whether network is directed
        logger (logging.Logger): Optional logger instance
        
    Returns:
        Dict: Analysis results
    """
    filter_instance = create_polya_filter(
        significance_level=alpha,
        method=method,
        logger=logger
    )
    
    return filter_instance.run_full_analysis(
        input_file=input_file,
        output_file=output_file,
        a=a,
        directed=directed
    )


# --- Application ---
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Polya Filter - Backbone Extraction from Networks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_file', help='Input file (TSV/CSV)')
    parser.add_argument('output_file', help='Output file (CSV with columns source, target, weight, p_value)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level (default: 0.05)')
    parser.add_argument('--method', choices=['bonferroni', 'fdr'], default='fdr', 
                       help='Correction method (default: fdr)')
    parser.add_argument('--a', type=float, default=1.0, 
                       help='Polya parameter (a=1.0 = Disparity Filter, default: 1.0)')
    parser.add_argument('--directed', action='store_true', 
                       help='Directed network (default: undirected)')
    parser.add_argument('--chunksize', type=int, default=100000, 
                       help='Chunk size for reading (default: 100000)')
    
    args = parser.parse_args()
    
    # Create filter
    polya_filter = create_polya_filter(
        significance_level=args.alpha,
        method=args.method
    )
    
    # Run analysis
    results = polya_filter.run_full_analysis(
        input_file=args.input_file,
        output_file=args.output_file,
        a=args.a,
        directed=args.directed,
        chunksize=args.chunksize
    )
    
    if results['success']:
        print("\nSUMMARY")
        print("-" * 40)
        print(f"  Nodes: {results['nodes']:,}")
        print(f"  Input edges: {results['edges_input']:,}")
        print(f"  Backbone edges: {results['edges_backbone']:,}")
        print(f"  Retention: {results['retained_percentage']:.2f}%")
        print(f"  Time: {results['total_time']:.2f} sec")
        print(f"\n  Results: {results['output_file']}")
        print(f"  Detailed: {results['output_file'].replace('.tsv', '_detailed.tsv')}")
        print(f"  Statistics: {results['output_file'].replace('.tsv', '_stats.txt')}")