import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, save_npz
from collections import defaultdict
import gc
import time
import csv
import os
import uuid
import pandas as pd
from typing import Optional
from multiprocessing import Pool, cpu_count


def _resource_allocation_gene_batch(args):
    """Compute Resource Allocation similarities for a slice of gene indices."""
    i_start, i_end, genes, neighbors_dict, degrees, threshold = args
    results = []
    n = len(genes)
    for i in range(i_start, i_end):
        gene1 = genes[i]
        nb1 = neighbors_dict.get(gene1)
        if nb1 is None:
            continue
        for j in range(i + 1, n):
            gene2 = genes[j]
            nb2 = neighbors_dict.get(gene2)
            if nb2 is None:
                continue
            common = nb1 & nb2
            if not common:
                continue
            weight = sum(1.0 / degrees[c] for c in common if degrees.get(c, 0) > 0)
            if weight >= threshold:
                results.append((gene1, gene2, weight))
    return results

class GeneSimilarityResourceAllocation:
    """
    Κλάση για υπολογισμό γονιδιακών ομοιοτήτων χρησιμοποιώντας Resource Allocation index.
    Υποστηρίζει οποιοδήποτε tab-separated αρχείο bipartite μορφής (node1\\tnode2).
    
    Resource Allocation index: s(u,v) = Σ 1/|Γ(z)| for z in [Γ(u) ∩ Γ(v)]
    όπου Γ(u) είναι οι γείτονες του κόμβου u.
    """
    
    def __init__(self, sample_size=1000, logger=None, n_workers=1):
        """
        Αρχικοποίηση της κλάσης.

        Args:
            sample_size (int): Αριθμός γραμμών για επεξεργασία
            logger: Logger object για καταγραφή μηνυμάτων
            n_workers (int): Number of parallel worker processes
        """
        self.sample_size = sample_size
        self.logger = logger
        self.n_workers = n_workers
        self.filepath: Optional[str] = None
        self.chunksize = 10000
        self.total_lines = 0
        self.gene_dict: dict = {}
        self.disease_dict: dict = {}
        self.gene_to_diseases = None
        self.disease_to_genes = None
        self.gene_index = None
        self.disease_index = None
        self.bipartite_graph = None
        self.G = None
        self.edges_to_export = None
        self.progress_callback = None
        self.has_header = True
        self.delimiter = '\t'
        
    def log(self, message, level='info'):
        """Βοηθητική μέθοδος για logging"""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'debug':
                self.logger.debug(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def set_progress_callback(self, callback):
        """
        Ορισμός callback function για ενημέρωση προόδου.
        
        Args:
            callback: Συνάρτηση που δέχεται ένα ποσοστό (0-100)
        """
        self.progress_callback = callback
    
    def update_progress(self, percent):
        """Ενημέρωση προόδου αν υπάρχει callback"""
        if self.progress_callback:
            self.progress_callback(percent)
    
    def _clean_data(self, chunk):
        """
        Internal method for cleaning a chunk of data.
        Uses positional access — column names are irrelevant.
        """
        chunk = chunk.dropna()
        chunk = chunk[~chunk.iloc[:, 0].astype(str).str.strip().str.lower().isin(['nan', 'none', ''])]
        chunk = chunk[~chunk.iloc[:, 1].astype(str).str.strip().str.lower().isin(['nan', 'none', ''])]
        return chunk


    def _count_lines(self) -> int:
        """Count total lines in the file."""
        assert self.filepath is not None
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _detect_header(self):
        """Returns 0 if file has a header row, None otherwise."""
        with open(self.filepath, 'r', encoding='utf-8') as f:  # type: ignore[arg-type]
            first = f.readline().strip().split('\t')
        # If any token in the first row is non-numeric → treat as header
        for token in first:
            try:
                float(token)
            except ValueError:
                if any(c.isalpha() for c in token):
                    return 0
        return None

    def _iter_chunks(self):
        """Central chunk-reader: positional columns, auto header detection."""
        header_row = self._detect_header()
        return pd.read_csv(
            self.filepath,  # type: ignore[arg-type]
            sep='\t',
            header=header_row,   # 0 → skip header; None → no header
            dtype=str,
            chunksize=self.chunksize,
            on_bad_lines='skip',
            engine='python',
        )
    

    def build_mappings(self):
        self.log("Building gene and disease mappings...")
        self.update_progress(5)

        self.total_lines = self._count_lines()
        total_chunks = (self.total_lines // self.chunksize) + 1

        gene_counter, disease_counter = 0, 0
        chunk_count = 0

        for chunk in self._iter_chunks():              # ← central reader
            chunk = self._clean_data(chunk)

            for _, row in chunk.iterrows():
                gene    = str(row.iloc[0]).strip()     # ← positional
                disease = str(row.iloc[1]).strip()     # ← positional
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
            del chunk
            gc.collect()

        self.log(f"Total genes: {len(self.gene_dict)}")
        self.log(f"Total diseases: {len(self.disease_dict)}")
        self.update_progress(30)
    
    def safe_split(self, line):
        """
        Ασφαλής διαχωρισμός γραμμής.
        
        Args:
            line (str): Η γραμμή προς διαχωρισμό
            
        Returns:
            list: Τα μέρη της γραμμής
        """
        parts = line.split(self.delimiter)
        return [p.strip() for p in parts if p.strip()]
    
    def process_data(self, filename):
        """
        Επεξεργάζεται το αρχείο με γονίδια και ασθένειες.
        Χρησιμοποιεί positional access (iloc[0], iloc[1]) χωρίς προκαθορισμένα
        ονόματα στηλών — λειτουργεί με οποιοδήποτε tab-separated αρχείο.

        Args:
            filename (str): Μονοπάτι προς το αρχείο εισόδου
        """
        self.log(f"Starting data processing from {filename}")
        self.update_progress(10)

        # Ανίχνευση header
        self.has_header = bool(self._detect_header())
        header_row = 0 if self.has_header else None
        if self.has_header:
            self.log("Header detected, will skip first line")

        gene_to_diseases = defaultdict(set)
        disease_to_genes = defaultdict(set)
        gene_index = {}
        disease_index = {}
        line_count = 0

        chunksize = min(10_000, self.sample_size)

        try:
            chunk_reader = pd.read_csv(
                filename,
                sep=self.delimiter,
                header=header_row,   # None → no header; 0 → first row is header
                dtype=str,           # everything as string — no type assumptions
                chunksize=chunksize,
                on_bad_lines='skip', # skip malformed rows silently
                engine='python',     # supports single-char and multi-char separators
            )

            for chunk in chunk_reader:
                if line_count >= self.sample_size:
                    break

                for _, row in chunk.iterrows():
                    if line_count >= self.sample_size:
                        break

                    # ── Positional access: column names are irrelevant ──────────
                    try:
                        node1 = str(row.iloc[0]).strip()
                        node2 = str(row.iloc[1]).strip()
                    except IndexError:
                        continue  # row has fewer than 2 columns → skip

                    if not node1 or not node2:
                        continue
                    # Skip rows where either value looks like a missing-value marker
                    if node1.lower() in ('nan', 'none', '') or node2.lower() in ('nan', 'none', ''):
                        continue

                    if node1 not in gene_index:
                        gene_index[node1] = len(gene_index)
                    if node2 not in disease_index:
                        disease_index[node2] = len(disease_index)

                    gene_to_diseases[node1].add(node2)
                    disease_to_genes[node2].add(node1)
                    line_count += 1

            self.gene_to_diseases = gene_to_diseases
            self.disease_to_genes = disease_to_genes
            self.gene_index = gene_index
            self.disease_index = disease_index

            self.log(
                f"Processed {len(gene_index)} genes and {len(disease_index)} "
                f"diseases from {line_count} edges"
            )
            self.update_progress(30)

        except Exception as e:
            self.log(f"Error processing data: {str(e)}", 'error')
            import traceback
            traceback.print_exc()
            raise
    
    def build_bipartite_graph(self):
        """
        Κατασκευάζει bipartite γράφο από τα δεδομένα.
        
        Returns:
            networkx.Graph: Bipartite graph
        """
        if self.gene_to_diseases is None or self.disease_to_genes is None:
            raise ValueError("Data not processed. Call process_data() first.")
        
        self.log("Building bipartite graph...")
        self.update_progress(40)
        
        B = nx.Graph()
        
        # Add gene nodes (bipartite=0)
        gene_count = 0
        for gene in self.gene_to_diseases:
            B.add_node(gene, bipartite=0)
            gene_count += 1
        
        # Add disease nodes (bipartite=1)
        disease_count = 0
        for disease in self.disease_to_genes:
            B.add_node(disease, bipartite=1)
            disease_count += 1
        
        # Add edges
        edge_count = 0
        for gene, diseases in self.gene_to_diseases.items():
            for disease in diseases:
                B.add_edge(gene, disease)
                edge_count += 1
        
        self.bipartite_graph = B
        self.log(f"Bipartite graph built with {gene_count} gene nodes, {disease_count} disease nodes, and {edge_count} edges")
        self.update_progress(50)
        
        return self.bipartite_graph
    

    def construct_sparse_matrix(self):
            self.log("Constructing bipartite sparse matrix (genes x diseases)...")
            self.update_progress(30)

            total_chunks = (self.total_lines // self.chunksize) + 1
            rows, cols, data = [], [], []
            chunk_count = 0

            for chunk in self._iter_chunks():              # ← central reader
                chunk = self._clean_data(chunk)

                for _, row in chunk.iterrows():
                    gene    = str(row.iloc[0]).strip()     # ← positional
                    disease = str(row.iloc[1]).strip()     # ← positional
                    if not gene or not disease:
                        continue
                    if gene in self.gene_dict and disease in self.disease_dict:
                        rows.append(self.gene_dict[gene])
                        cols.append(self.disease_dict[disease])
                        data.append(1)

                chunk_count += 1
                self.update_progress(30 + int(20 * chunk_count / total_chunks))
                if chunk_count % 10 == 0:
                    gc.collect()

            if rows and cols and data:
                self.matrix = csr_matrix(
                    (data, (rows, cols)),
                    shape=(len(self.gene_dict), len(self.disease_dict))
                )
                save_npz("sparse_matrix.npz", self.matrix)
                self.log(f"Sparse matrix shape: {self.matrix.shape}, nnz: {self.matrix.nnz}")
                del rows, cols, data
                gc.collect()
                self.update_progress(50)
            else:
                raise ValueError("No valid data found to construct matrix")
    
    def calculate_similarities(self, threshold=0.5):
        """
        Υπολογίζει Resource Allocation similarities μεταξύ γονιδίων.
        
        Resource Allocation formula: s(u,v) = Σ 1/degree(z) for common neighbors z
        
        Args:
            threshold (float): Κατώφλι ομοιότητας
            
        Returns:
            tuple: (NetworkX graph, list of edges)
        """
        if self.bipartite_graph is None:
            raise ValueError("Bipartite graph not built. Call build_bipartite_graph() first.")
        
        if self.gene_index is None:
            raise ValueError("Gene index not available. Call process_data() first.")
        
        self.log(f"Calculating Resource Allocation similarities with threshold {threshold}...")
        self.update_progress(60)
        
        genes = list(self.gene_index.keys())
        self.G = nx.Graph()
        self.edges_to_export = []

        degrees = dict(self.bipartite_graph.degree())
        neighbors_dict = {g: set(self.bipartite_graph.neighbors(g)) for g in genes}

        n = len(genes)
        self.log(f"  {n} genes, {n*(n-1)//2:,} pairs, {self.n_workers} worker(s)")

        chunk = max(1, n // max(1, self.n_workers * 4))
        batches = [
            (i_start, min(i_start + chunk, n), genes, neighbors_dict, degrees, threshold)
            for i_start in range(0, n, chunk)
        ]

        if self.n_workers > 1 and len(batches) > 1:
            with Pool(processes=self.n_workers) as pool:
                for batch_result in pool.imap_unordered(_resource_allocation_gene_batch, batches):
                    for gene1, gene2, weight in batch_result:
                        self.G.add_edge(gene1, gene2, weight=weight)
                        self.edges_to_export.append((gene1, gene2, weight))
        else:
            for batch in batches:
                for gene1, gene2, weight in _resource_allocation_gene_batch(batch):
                    self.G.add_edge(gene1, gene2, weight=weight)
                    self.edges_to_export.append((gene1, gene2, weight))

        self.log(f"Graph constructed with {len(self.G.nodes())} nodes and {len(self.G.edges())} edges")
        self.update_progress(100)

        return self.G, self.edges_to_export
    
    def export_edges(self, output_filename, threshold):
        """
        Εξάγει τα αποτελέσματα σε CSV αρχείο.
        
        Args:
            output_filename (str): Μονοπάτι για το output αρχείο
            threshold (float): Το κατώφλι που χρησιμοποιήθηκε
        """
        if self.edges_to_export is None:
            raise ValueError("No edges to export. Run calculate_similarities() first.")
        
        self.log(f"Exporting {len(self.edges_to_export)} edges to {output_filename}")
        
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['source', 'target', 'weight'])  # ← μόνο αυτές οι 3 στήλες
            
            for edge in self.edges_to_export:
                writer.writerow([edge[0], edge[1], f"{edge[2]:.6f}"])
    
    def run_analysis(self, input_filename, output_filename, threshold=0.5):
        """
        Εκτελεί πλήρη ανάλυση από την αρχή μέχρι το τέλος.
        
        Args:
            input_filename (str): Αρχείο εισόδου
            output_filename (str): Αρχείο εξόδου
            threshold (float): Κατώφλι ομοιότητας
            
        Returns:
            dict: Στατιστικά αποτελεσμάτων
        """
        start_time = time.time()
        
        try:
            # Βήμα 1: Επεξεργασία δεδομένων
            self.process_data(input_filename)
            
            # Βήμα 2: Κατασκευή bipartite γράφου
            self.build_bipartite_graph()
            
            # Βήμα 3: Υπολογισμός ομοιοτήτων
            self.calculate_similarities(threshold)
            
            # Βήμα 4: Εξαγωγή αποτελεσμάτων
            self.export_edges(output_filename, threshold)
            
            # Στατιστικά
            execution_time = time.time() - start_time
            
            # Υπολογισμός density για τον γράφο ομοιοτήτων
            density = 0
            if self.G is not None and len(self.G.nodes()) > 1:
                density = round(2 * len(self.G.edges()) / (len(self.G.nodes()) * (len(self.G.nodes()) - 1)), 4)
            
            # Υπολογισμός μέσου βάρους
            avg_weight = 0
            if self.edges_to_export and len(self.edges_to_export) > 0:
                avg_weight = sum(w for _, _, w in self.edges_to_export) / len(self.edges_to_export)
            
            stats = {
                'success': True,
                'genes': len(self.gene_index) if self.gene_index is not None else 0,
                'diseases': len(self.disease_index) if self.disease_index is not None else 0,
                'bipartite_nodes': len(self.bipartite_graph.nodes()) if self.bipartite_graph is not None else 0,
                'bipartite_edges': len(self.bipartite_graph.edges()) if self.bipartite_graph is not None else 0,
                'nodes_in_graph': len(self.G.nodes()) if self.G is not None else 0,
                'edges_in_graph': len(self.G.edges()) if self.G is not None else 0,
                'density': density,
                'avg_weight': round(avg_weight, 4),
                'execution_time': round(execution_time, 2),
                'threshold': threshold,
                'method': 'resource-allocation',
                'sample_size': self.sample_size
            }
            
            self.log(f"Analysis completed in {execution_time:.2f} seconds")
            return stats
            
        except Exception as e:
            self.log(f"Error in analysis: {str(e)}", 'error')
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_graph_as_dataframe(self):
        """
        Επιστρέφει το γράφο ομοιοτήτων ως pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame με στήλες Gene1, Gene2, Similarity_Score
        """
        if self.edges_to_export is None:
            return pd.DataFrame(columns=['Gene1', 'Gene2', 'Similarity_Score'])
        
        return pd.DataFrame(self.edges_to_export, columns=['Gene1', 'Gene2', 'Similarity_Score'])
    
    def get_graph_as_json(self):
        """
        Επιστρέφει το γράφο ομοιοτήτων ως JSON για visualization.
        
        Returns:
            dict: JSON representation του γράφου
        """
        if self.G is None:
            return {'nodes': [], 'edges': []}
        
        nodes = [{'id': node, 'label': node} for node in self.G.nodes()]
        edges = [
            {
                'source': u, 
                'target': v, 
                'weight': self.G[u][v]['weight']
            } 
            for u, v in self.G.edges()
        ]
        
        return {'nodes': nodes, 'edges': edges}
    
    def get_bipartite_graph_stats(self):
        """
        Επιστρέφει στατιστικά για τον bipartite γράφο.
        
        Returns:
            dict: Στατιστικά bipartite γράφου
        """
        if self.bipartite_graph is None:
            return {}
        
        # Ξεχωριστά στατιστικά για genes και diseases
        gene_nodes = [n for n, attr in self.bipartite_graph.nodes(data=True) if attr.get('bipartite') == 0]
        disease_nodes = [n for n, attr in self.bipartite_graph.nodes(data=True) if attr.get('bipartite') == 1]
        
        # Υπολογισμός μέσου βαθμού
        degrees = dict(self.bipartite_graph.degree())
        avg_gene_degree = 0
        if gene_nodes:
            avg_gene_degree = sum(degrees[g] for g in gene_nodes) / len(gene_nodes)
        
        avg_disease_degree = 0
        if disease_nodes:
            avg_disease_degree = sum(degrees[d] for d in disease_nodes) / len(disease_nodes)
        
        return {
            'total_nodes': len(self.bipartite_graph.nodes()),
            'gene_nodes': len(gene_nodes),
            'disease_nodes': len(disease_nodes),
            'total_edges': len(self.bipartite_graph.edges()),
            'avg_gene_degree': round(avg_gene_degree, 2),
            'avg_disease_degree': round(avg_disease_degree, 2)
        }

def save_results(self, output_file, mode='gene', batch_size=10000):
    """
    Saves the similarity results to a CSV file in batches.
    Columns: source, target, weight
    """
    keys = list(self.gene_dict.keys()) if mode == 'gene' else list(self.disease_dict.keys())
    
    self.log(f"Saving results to {output_file}")
    
    coo = self.similarity_matrix.tocoo()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("source\ttarget\tweight\n")
        
        batch = []
        for idx in range(len(coo.data)):
            if coo.row[idx] < coo.col[idx]:  # write each pair once
                batch.append(
                    f"{keys[coo.row[idx]]}\t{keys[coo.col[idx]]}\t{coo.data[idx]:.6f}\n"
                )
            
            if len(batch) >= batch_size:
                f.writelines(batch)
                batch = []
                gc.collect()
        
        if batch:
            f.writelines(batch)
    
    self.log(f"Results saved to {output_file}")
    self.update_progress(100)

# Συνάρτηση-βοηθός για εύκολη κλήση από το Flask
def run_resource_allocation_similarity(input_file, output_dir, threshold=0.5, sample_size=1000, logger=None, progress_callback=None, n_workers=1):
    """
    Βοηθητική συνάρτηση για εκτέλεση της ανάλυσης Resource Allocation από Flask.
    
    Args:
        input_file (str): Μονοπάτι προς το αρχείο εισόδου
        output_dir (str): Φάκελος για αποθήκευση αποτελεσμάτων
        threshold (float): Κατώφλι ομοιότητας
        sample_size (int): Αριθμός γραμμών για επεξεργασία
        logger: Logger object
        progress_callback: Callback για πρόοδο
        
    Returns:
        tuple: (stats_dict, output_filename)
    """
    # Δημιουργία μοναδικού ονόματος αρχείου
    task_id = str(uuid.uuid4())
    output_filename = os.path.join(output_dir, f"resource_allocation_{task_id}.tsv")
    
    # Δημιουργία instance της κλάσης
    analyzer = GeneSimilarityResourceAllocation(sample_size=sample_size, logger=logger, n_workers=n_workers)
    
    if progress_callback:
        analyzer.set_progress_callback(progress_callback)
    
    # Εκτέλεση ανάλυσης
    stats = analyzer.run_analysis(input_file, output_filename, threshold)
    
    return stats, output_filename


# Για άμεση εκτέλεση από command line
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Gene Similarity Resource Allocation Analysis')
    parser.add_argument('input_file', help='Input file path (bipartite format: gene\\tdisease)')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold')
    parser.add_argument('--sample-size', type=int, default=1000, help='Number of lines to process')
    
    args = parser.parse_args()
    
    analyzer = GeneSimilarityResourceAllocation(sample_size=args.sample_size)
    stats = analyzer.run_analysis(args.input_file, args.output_file, args.threshold)
    
    if stats['success']:
        print(f"\n=== Analysis Successful ===")
        print(f"Genes: {stats['genes']}")
        print(f"Diseases: {stats['diseases']}")
        print(f"Bipartite graph: {stats['bipartite_nodes']} nodes, {stats['bipartite_edges']} edges")
        print(f"Similarity graph: {stats['nodes_in_graph']} nodes, {stats['edges_in_graph']} edges")
        print(f"Density: {stats['density']}")
        print(f"Average weight: {stats['avg_weight']}")
        print(f"Execution time: {stats['execution_time']} seconds")
        print(f"Output saved to: {args.output_file}")
    else:
        print(f"Error: {stats.get('error', 'Unknown error')}")