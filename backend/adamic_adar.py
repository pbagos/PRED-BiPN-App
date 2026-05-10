import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict
import time
import csv
import os
import uuid
import pandas as pd
from multiprocessing import Pool, cpu_count


def _adamic_adar_gene_batch(args):
    """Compute Adamic-Adar similarities for a slice of gene indices (i_start..i_end)."""
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
            weight = sum(1.0 / np.log(degrees[c]) for c in common if degrees.get(c, 1) > 1)
            if weight >= threshold:
                results.append((gene1, gene2, weight))
    return results

class GeneSimilarityAdamicAdar:
    """
    Κλάση για υπολογισμό γονιδιακών ομοιοτήτων χρησιμοποιώντας Adamic-Adar index.
    Υποστηρίζει οποιοδήποτε tab-separated αρχείο bipartite μορφής (node1\tnode2).
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
    
    def detect_header(self, filename):
        """Ανιχνεύει διαχωριστικό. Πάντα επιστρέφει True (header=0)."""
        with open(filename, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        separators = {'\t': first_line.count('\t'), ',': first_line.count(','),
                    ';': first_line.count(';'), ' ': first_line.count(' ')}
        sep = max(separators, key=lambda k: separators[k])
        self.delimiter = sep if separators[sep] > 0 else '\t'

        sample = pd.read_csv(filename, sep=self.delimiter, header=0, nrows=1)
        self.log(f"  Columns detected: {list(sample.columns)}")
        return True  # header=0 πάντα
    
    def process_data(self, filename):
            self.log(f"Starting data processing from {filename}")
            self.update_progress(10)

            gene_to_diseases = defaultdict(set)
            disease_to_genes = defaultdict(set)
            gene_index = {}
            disease_index = {}
            line_count = 0

            try:
                # Διάβασε με whitespace delimiter και συγκεκριμένα 3 στήλες
                df = pd.read_csv(
                    filename,
                    sep='\t',          # ή ',' ανάλογα με το αρχείο σου
                    header=0,
                    nrows=self.sample_size,
                    names=['source', 'target', 'weight'],  # ορισμός στηλών
                    skiprows=1         # παράλειψη header
                )

                self.log(f"Columns: {list(df.columns)}")
                self.log(f"First row: {df.iloc[0].tolist()}")

                for _, row in df.iterrows():
                    node1 = str(row['source']).strip()
                    node2 = str(row['target']).strip()
                    # Το weight αγνοείται — το ξαναυπολογίζουμε με Adamic-Adar

                    if not node1 or not node2 or node1 == 'nan' or node2 == 'nan':
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

                self.log(f"Processed {len(gene_index)} type1 nodes, "
                        f"{len(disease_index)} type2 nodes, {line_count} edges")
                self.update_progress(30)

            except Exception as e:
                self.log(f"Error: {str(e)}", 'error')
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
    
    def calculate_similarities(self, threshold=0.5):
        """
        Υπολογίζει Adamic-Adar similarities μεταξύ γονιδίων.
        
        Args:
            threshold (float): Κατώφλι ομοιότητας
            
        Returns:
            tuple: (NetworkX graph, list of edges)
        """
        if self.bipartite_graph is None:
            raise ValueError("Bipartite graph not built. Call build_bipartite_graph() first.")
        
        if self.gene_index is None:
            raise ValueError("Gene index not available. Call process_data() first.")
        
        self.log(f"Calculating Adamic-Adar similarities with threshold {threshold}...")
        self.update_progress(60)
        
        genes = list(self.gene_index.keys())
        self.G = nx.Graph()
        self.edges_to_export = []

        degrees = dict(self.bipartite_graph.degree())
        neighbors_dict = {g: set(self.bipartite_graph.neighbors(g)) for g in genes}

        n = len(genes)
        self.log(f"  {n} genes, {n*(n-1)//2:,} pairs, {self.n_workers} worker(s)")

        # Split outer gene index range across workers
        chunk = max(1, n // max(1, self.n_workers * 4))
        batches = [
            (i_start, min(i_start + chunk, n), genes, neighbors_dict, degrees, threshold)
            for i_start in range(0, n, chunk)
        ]

        if self.n_workers > 1 and len(batches) > 1:
            with Pool(processes=self.n_workers) as pool:
                for batch_result in pool.imap_unordered(_adamic_adar_gene_batch, batches):
                    for gene1, gene2, weight in batch_result:
                        self.G.add_edge(gene1, gene2, weight=weight)
                        self.edges_to_export.append((gene1, gene2, weight))
        else:
            for batch in batches:
                for gene1, gene2, weight in _adamic_adar_gene_batch(batch):
                    self.G.add_edge(gene1, gene2, weight=weight)
                    self.edges_to_export.append((gene1, gene2, weight))

        self.log(f"Graph constructed with {len(self.G.nodes())} nodes and {len(self.G.edges())} edges")
        self.update_progress(100)

        return self.G, self.edges_to_export
    
    def export_edges(self, output_filename, threshold):
        if self.edges_to_export is None:
            raise ValueError("No edges to export.")

        self.log(f"Exporting {len(self.edges_to_export)} edges to {output_filename}")

        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['source', 'target', 'weight'])
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
            
            stats = {
                'success': True,
                'genes': len(self.gene_index) if self.gene_index is not None else 0,
                'diseases': len(self.disease_index) if self.disease_index is not None else 0,
                'bipartite_nodes': len(self.bipartite_graph.nodes()) if self.bipartite_graph is not None else 0,
                'bipartite_edges': len(self.bipartite_graph.edges()) if self.bipartite_graph is not None else 0,
                'nodes_in_graph': len(self.G.nodes()) if self.G is not None else 0,
                'edges_in_graph': len(self.G.edges()) if self.G is not None else 0,
                'density': density,
                'execution_time': round(execution_time, 2),
                'threshold': threshold,
                'method': 'adamic-adar',
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
        
        return {
            'total_nodes': len(self.bipartite_graph.nodes()),
            'gene_nodes': len(gene_nodes),
            'disease_nodes': len(disease_nodes),
            'total_edges': len(self.bipartite_graph.edges())
        }


# Συνάρτηση-βοηθός για εύκολη κλήση από το Flask
def run_adamic_adar_similarity(input_file, output_dir, threshold=0.5, sample_size=1000, logger=None, progress_callback=None, n_workers=1):
    """
    Βοηθητική συνάρτηση για εκτέλεση της ανάλυσης Adamic-Adar από Flask.
    
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
    output_filename = os.path.join(output_dir, f"adamic_adar_{task_id}.tsv")
    
    # Δημιουργία instance της κλάσης
    analyzer = GeneSimilarityAdamicAdar(sample_size=sample_size, logger=logger, n_workers=n_workers)
    
    if progress_callback:
        analyzer.set_progress_callback(progress_callback)
    
    # Εκτέλεση ανάλυσης
    stats = analyzer.run_analysis(input_file, output_filename, threshold)
    
    return stats, output_filename


# Για άμεση εκτέλεση από command line
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Gene Similarity Adamic-Adar Analysis')
    parser.add_argument('input_file', help='Input file path (bipartite format: gene\tdisease)')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold')
    parser.add_argument('--sample-size', type=int, default=1000, help='Number of lines to process')
    
    args = parser.parse_args()
    
    analyzer = GeneSimilarityAdamicAdar(sample_size=args.sample_size)
    stats = analyzer.run_analysis(args.input_file, args.output_file, args.threshold)
    
    if stats['success']:
        print(f"\n=== Analysis Successful ===")
        print(f"Genes: {stats['genes']}")
        print(f"Diseases: {stats['diseases']}")
        print(f"Bipartite graph: {stats['bipartite_nodes']} nodes, {stats['bipartite_edges']} edges")
        print(f"Similarity graph: {stats['nodes_in_graph']} nodes, {stats['edges_in_graph']} edges")
        print(f"Density: {stats['density']}")
        print(f"Execution time: {stats['execution_time']} seconds")
        print(f"Output saved to: {args.output_file}")
    else:
        print(f"Error: {stats.get('error', 'Unknown error')}")