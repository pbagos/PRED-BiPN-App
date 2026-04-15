import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
import csv
import os
import uuid
import pandas as pd

class GeneSimilarityCosine:
    """
    Κλάση για υπολογισμό γονιδιακών ομοιοτήτων χρησιμοποιώντας cosine similarity.
    Υποστηρίζει οποιοδήποτε tab-separated αρχείο χωρίς απαιτήσεις σε ονόματα στηλών.
    """
    
    def __init__(self, sample_size=1000, logger=None):
        """
        Αρχικοποίηση της κλάσης.
        
        Args:
            sample_size (int): Αριθμός γραμμών για επεξεργασία
            logger: Logger object για καταγραφή μηνυμάτων
        """
        self.sample_size = sample_size
        self.logger = logger
        self.gene_to_diseases = None
        self.disease_to_genes = None
        self.gene_index = None
        self.disease_index = None
        self.sparse_adj = None
        self.G = None
        self.edges_to_export = None
        self.progress_callback = None
        self.input_format = None  # 'bipartite' or 'weighted'
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
    
    def detect_file_format_and_header(self, filename):
        """
        Ανιχνεύει διαχωριστικό και αριθμό στηλών.
        Η πρώτη γραμμή γίνεται header αυτόματα (header=0).
        """
        with open(filename, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

        # Ανίχνευση διαχωριστικού
        separators = {'\t': first_line.count('\t'), ',': first_line.count(','),
                    ';': first_line.count(';'), ' ': first_line.count(' ')}
        sep = max(separators, key=lambda k: separators[k])
        self.delimiter = sep if separators[sep] > 0 else '\t'

        # Διάβασμα με header=0 — η πρώτη γραμμή γίνεται ονόματα στηλών
        sample = pd.read_csv(filename, sep=self.delimiter, header=0, nrows=2)
        self.has_header = True  # πάντα True με header=0

        num_cols = sample.shape[1]
        if num_cols == 2:
            file_format = 'bipartite'
            self.log(f"Detected bipartite format (2 columns): {list(sample.columns)}")
        elif num_cols >= 3:
            file_format = 'weighted'
            self.log(f"Detected weighted format ({num_cols} columns): {list(sample.columns)}")
        else:
            raise ValueError(f"Unknown format: {num_cols} columns")

        return file_format, self.has_header, self.delimiter
    
    def safe_split(self, line, delimiter='\t'):
        """
        Ασφαλής διαχωρισμός γραμμής ανεξάρτητα από τον αριθμό των tabs.
        
        Args:
            line (str): Η γραμμή προς διαχωρισμό
            delimiter (str): Ο διαχωριστής
            
        Returns:
            list: Τα μέρη της γραμμής
        """
        parts = line.split(delimiter)
        return [p.strip() for p in parts if p.strip()]
    
    def process_data(self, filename):
        """
        Επεξεργάζεται το αρχείο ανάλογα με τη μορφή του.
        
        Args:
            filename (str): Μονοπάτι προς το αρχείο εισόδου
        """
        self.log(f"Starting data processing from {filename}")
        self.update_progress(10)
        
        try:
            # Ανίχνευση μορφής αρχείου και header
            self.input_format, self.has_header, self.delimiter = self.detect_file_format_and_header(filename)
            
            if self.input_format == 'bipartite':
                self._process_bipartite_file(filename)
            else:  # weighted format
                self._process_weighted_file(filename)
            
            self.update_progress(30)
            
        except Exception as e:
            self.log(f"Error processing data: {str(e)}", 'error')
            import traceback
            traceback.print_exc()
            raise
    
    def _process_bipartite_file(self, filename):
        """Positional access: iloc[0]=node1, iloc[1]=node2."""
        gene_to_diseases = defaultdict(set)
        disease_to_genes = defaultdict(set)
        gene_index = {}
        disease_index = {}
        line_count = 0

        # header=0: η πρώτη γραμμή παραλείπεται αυτόματα ως header
        df = pd.read_csv(filename, sep=self.delimiter, header=0,
                        nrows=self.sample_size)

        for _, row in df.iterrows():
            node1 = str(row.iloc[0]).strip()
            node2 = str(row.iloc[1]).strip()

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
    
    def _process_weighted_file(self, filename):
        """Positional access: iloc[0]=node1, iloc[1]=node2, iloc[2]=weight."""
        gene_to_genes = defaultdict(set)
        gene_index = {}
        weights = {}
        line_count = 0

        df = pd.read_csv(filename, sep=self.delimiter, header=0,
                        nrows=self.sample_size)

        # Μετατροπή 3ης στήλης σε numeric
        df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')

        for _, row in df.iterrows():
            node1 = str(row.iloc[0]).strip()
            node2 = str(row.iloc[1]).strip()

            if not node1 or not node2 or node1 == 'nan' or node2 == 'nan':
                continue

            w = 1.0
            if pd.notna(row.iloc[2]):
                w = float(row.iloc[2])

            if node1 not in gene_index:
                gene_index[node1] = len(gene_index)
            if node2 not in gene_index:
                gene_index[node2] = len(gene_index)

            gene_to_genes[node1].add(node2)
            gene_to_genes[node2].add(node1)
            weights[(node1, node2)] = w
            weights[(node2, node1)] = w
            line_count += 1

        self.gene_to_diseases = {g: {f"dummy_{i}"}
                                for i, g in enumerate(gene_index)}
        self.disease_to_genes = {f"dummy_{i}": {g}
                                for i, g in enumerate(gene_index)}
        self.gene_index = gene_index
        self.disease_index = {f"dummy_{i}": i for i in range(len(gene_index))}
        self.precomputed_weights = weights
        self.log(f"Processed {len(gene_index)} nodes, {line_count} edges")
    
    def build_sparse_matrix(self):
        """
        Κατασκευάζει sparse adjacency matrix από τα δεδομένα.
        
        Returns:
            scipy.sparse.csr_matrix: Sparse adjacency matrix
        """
        if self.gene_index is None:
            raise ValueError("Data not processed. Call process_data() first.")
        
        self.log("Building sparse matrix...")
        self.update_progress(40)
        
        num_genes = len(self.gene_index)
        
        if self.input_format == 'bipartite':
            # Κλασική περίπτωση: genes × diseases
            assert self.disease_index is not None and self.gene_to_diseases is not None
            num_diseases = len(self.disease_index)
            adj_matrix = lil_matrix((num_genes, num_diseases), dtype=np.int8)
            gene_index = self.gene_index
            disease_index = self.disease_index
            for gene, diseases in self.gene_to_diseases.items():
                for disease in diseases:
                    if gene in gene_index and disease in disease_index:
                        adj_matrix[gene_index[gene], disease_index[disease]] = 1
            
            self.sparse_adj = adj_matrix.tocsr()
            
        else:  # weighted format
            # Για weighted network, δημιουργούμε adjacency matrix με τα βάρη
            adj_matrix = lil_matrix((num_genes, num_genes), dtype=np.float32)
            
            # Βάζουμε 1 στη διαγώνιο για να μην είναι μηδενικά τα διανύσματα
            for i in range(num_genes):
                adj_matrix[i, i] = 1.0
            
            for (gene1, gene2), weight in self.precomputed_weights.items():
                if gene1 in self.gene_index and gene2 in self.gene_index:
                    i = self.gene_index[gene1]
                    j = self.gene_index[gene2]
                    adj_matrix[i, j] = weight
            
            self.sparse_adj = adj_matrix.tocsr()
        
        sparse_adj = self.sparse_adj
        assert sparse_adj is not None
        shape = sparse_adj.shape
        self.log(f"Sparse matrix built: {shape[0]}x{shape[1]}")  # type: ignore[index]
        self.update_progress(50)
        
        return self.sparse_adj
    
    def calculate_similarities(self, threshold=0.5):
        """
        Υπολογίζει cosine similarities μεταξύ γονιδίων.
        
        Args:
            threshold (float): Κατώφλι ομοιότητας (0-1)
            
        Returns:
            tuple: (NetworkX graph, list of edges)
        """
        if self.sparse_adj is None:
            raise ValueError("Sparse matrix not built. Call build_sparse_matrix() first.")
        
        self.log(f"Calculating cosine similarities with threshold {threshold}...")
        self.update_progress(60)
        
        assert self.gene_index is not None
        genes = list(self.gene_index.keys())
        self.G = nx.Graph()
        self.edges_to_export = []
        
        # Efficient sparse cosine similarity
        cosine_sim = cosine_similarity(csr_matrix(self.sparse_adj))
        
        # Αφαίρεση της διαγωνίου (ομοιότητα με τον εαυτό τους)
        np.fill_diagonal(cosine_sim, 0)
        
        # Βρες pairs πάνω από το threshold
        rows, cols = np.where(cosine_sim >= threshold)
        
        for i, j in zip(rows, cols):
            if i < j:  # Αποφυγή διπλότυπων
                gene1 = genes[i]
                gene2 = genes[j]
                weight = cosine_sim[i, j]
                
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
            
            # Βήμα 2: Κατασκευή sparse matrix
            self.build_sparse_matrix()
            
            # Βήμα 3: Υπολογισμός ομοιοτήτων
            self.calculate_similarities(threshold)
            
            # Βήμα 4: Εξαγωγή αποτελεσμάτων
            self.export_edges(output_filename, threshold)
            
            # Στατιστικά - ΔΙΟΡΘΩΜΕΝΟΣ ΕΛΕΓΧΟΣ
            execution_time = time.time() - start_time
            memory_usage = self.sparse_adj.data.nbytes / 1024 if self.sparse_adj is not None else 0
            
            stats = {
                'success': True,
                'nodes_type1': len(self.gene_index) if self.gene_index is not None else 0,
                'nodes_type2': len(self.disease_index) if hasattr(self, 'disease_index') and self.disease_index is not None else 0,
                'nodes_in_graph': len(self.G.nodes()) if self.G is not None else 0,
                'edges_in_graph': len(self.G.edges()) if self.G is not None else 0,
                'execution_time': round(execution_time, 2),
                'memory_usage_kb': round(memory_usage, 2),
                'threshold': threshold,
                'method': 'cosine',
                'input_format': self.input_format,
                'density': round(2 * len(self.G.edges()) / (len(self.G.nodes()) * (len(self.G.nodes()) - 1)), 4) if self.G is not None and len(self.G.nodes()) > 1 else 0
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
        Επιστρέφει το γράφο ως pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame με στήλες Node1, Node2, Similarity_Score
        """
        if self.edges_to_export is None:
            return pd.DataFrame(columns=['Node1', 'Node2', 'Similarity_Score'])
        
        return pd.DataFrame(self.edges_to_export, columns=['Node1', 'Node2', 'Similarity_Score'])
    
    def get_graph_as_json(self):
        """
        Επιστρέφει το γράφο ως JSON για visualization.
        
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


# Συνάρτηση-βοηθός για εύκολη κλήση από το Flask
def run_gene_similarity_cosine(input_file, output_dir, threshold=0.5, sample_size=1000, logger=None, progress_callback=None):
    """
    Βοηθητική συνάρτηση για εκτέλεση της ανάλυσης από Flask.
    
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
    output_filename = os.path.join(output_dir, f"cosine_similarity_{task_id}.tsv")
    
    # Δημιουργία instance της κλάσης
    analyzer = GeneSimilarityCosine(sample_size=sample_size, logger=logger)
    
    if progress_callback:
        analyzer.set_progress_callback(progress_callback)
    
    # Εκτέλεση ανάλυσης
    stats = analyzer.run_analysis(input_file, output_filename, threshold)
    
    return stats, output_filename


# Για άμεση εκτέλεση από command line
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Gene Similarity Cosine Analysis')
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold (0-1)')
    parser.add_argument('--sample-size', type=int, default=1000, help='Number of lines to process')
    
    args = parser.parse_args()
    
    analyzer = GeneSimilarityCosine(sample_size=args.sample_size)
    stats = analyzer.run_analysis(args.input_file, args.output_file, args.threshold)
    
    if stats['success']:
        print(f"Analysis successful!")
        print(f"Nodes: {stats['nodes_in_graph']}")
        print(f"Edges: {stats['edges_in_graph']}")
        print(f"Density: {stats['density']}")
        print(f"Execution time: {stats['execution_time']} seconds")
    else:
        print(f"Error: {stats.get('error', 'Unknown error')}")