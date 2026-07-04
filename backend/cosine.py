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
    Class for computing gene similarities using cosine similarity.
    Supports any tab-separated file with no column-name requirements.
    """

    def __init__(self, sample_size=None, logger=None):
        """
        Initialize the class.

        Args:
            sample_size: Deprecated/ignored, kept for API compatibility.
                         The full file is always processed (row cap removed).
            logger: Logger object for logging messages
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
        self.input_format = None
        self.has_header = True
        self.delimiter = '\t'

    def log(self, message, level='info'):
        """Helper method for logging"""
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
        Set callback function for progress updates.

        Args:
            callback: Function that accepts a percentage (0-100)
        """
        self.progress_callback = callback

    def update_progress(self, percent):
        """Update progress if a callback exists"""
        if self.progress_callback:
            self.progress_callback(percent)

    def detect_file_format_and_header(self, filename):
        """
        Detects the delimiter and number of columns.
        The first row automatically becomes the header (header=0).
        """
        with open(filename, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

        separators = {'\t': first_line.count('\t'), ',': first_line.count(','),
                    ';': first_line.count(';'), ' ': first_line.count(' ')}
        sep = max(separators, key=lambda k: separators[k])
        self.delimiter = sep if separators[sep] > 0 else '\t'

        sample = pd.read_csv(filename, sep=self.delimiter, header=0, nrows=2)
        self.has_header = True

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
        Safely split a line regardless of the number of tabs.

        Args:
            line (str): The line to split
            delimiter (str): The delimiter

        Returns:
            list: The parts of the line
        """
        parts = line.split(delimiter)
        return [p.strip() for p in parts if p.strip()]

    def process_data(self, filename):
        """
        Processes the file according to its format.

        Args:
            filename (str): Path to the input file
        """
        self.log(f"Starting data processing from {filename}")
        self.update_progress(10)

        try:
            _fmt, self.has_header, self.delimiter = self.detect_file_format_and_header(filename)
            self.input_format = 'bipartite'
            self._process_bipartite_file(filename)

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

        df = pd.read_csv(filename, sep=self.delimiter, header=0)

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

        df = pd.read_csv(filename, sep=self.delimiter, header=0)

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
        Builds a sparse adjacency matrix from the data.

        Returns:
            scipy.sparse.csr_matrix: Sparse adjacency matrix
        """
        if self.gene_index is None:
            raise ValueError("Data not processed. Call process_data() first.")

        self.log("Building sparse matrix...")
        self.update_progress(40)

        num_genes = len(self.gene_index)

        if self.input_format == 'bipartite':
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

        else:
            adj_matrix = lil_matrix((num_genes, num_genes), dtype=np.float32)

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
        self.log(f"Sparse matrix built: {shape[0]}x{shape[1]}")
        self.update_progress(50)

        return self.sparse_adj

    def _similarity_edges(self, matrix, labels, threshold):
        """Cosine similarity between the ROWS of `matrix`.
        Returns (edges, graph) where edges = [(label_i, label_j, score), ...]
        keeping only i<j pairs with score >= threshold."""
        sim = cosine_similarity(csr_matrix(matrix))
        np.fill_diagonal(sim, 0)
        rows, cols = np.where(sim >= threshold)
        G = nx.Graph()
        edges = []
        for i, j in zip(rows, cols):
            if i < j:
                a, b, w = labels[i], labels[j], float(sim[i, j])
                G.add_edge(a, b, weight=w)
                edges.append((a, b, w))
        return edges, G

    @staticmethod
    def _write_edges_file(edges, output_filename):
        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['source', 'target', 'weight'])
            for a, b, w in edges:
                writer.writerow([a, b, f"{w:.6f}"])

    def run_analysis(self, input_filename, type1_output, type2_output, threshold=0.5):
        """Full analysis producing BOTH one-mode projections of the bipartite
        input:
            type1_output → type-1 nodes (col 0, e.g. genes), similar if they
                           share type-2 neighbours  (cosine over the rows)
            type2_output → type-2 nodes (col 1, e.g. diseases), similar if they
                           share type-1 neighbours  (cosine over the columns)
        For a 3-column WEIGHTED (already one-mode) input there is only one
        projection, so type2_output is left unwritten.
        """
        start_time = time.time()

        try:
            self.process_data(input_filename)
            self.build_sparse_matrix()

            assert self.gene_index is not None and self.sparse_adj is not None
            self.update_progress(60)

            type1_labels = list(self.gene_index.keys())
            type1_edges, self.G = self._similarity_edges(self.sparse_adj, type1_labels, threshold)
            self._write_edges_file(type1_edges, type1_output)
            self.edges_to_export = type1_edges
            self.log(f"Type-1 projection: {len(type1_labels)} nodes, {len(type1_edges)} edges")

            type2_edges = None
            if self.input_format == 'bipartite' and type2_output:
                assert self.disease_index is not None
                type2_labels = list(self.disease_index.keys())
                type2_edges, _ = self._similarity_edges(self.sparse_adj.T, type2_labels, threshold)
                self._write_edges_file(type2_edges, type2_output)
                self.log(f"Type-2 projection: {len(type2_labels)} nodes, {len(type2_edges)} edges")

            self.update_progress(100)
            execution_time = time.time() - start_time

            stats = {
                'success': True,
                'nodes_type1': len(self.gene_index),
                'nodes_type2': len(self.disease_index) if self.disease_index is not None else 0,
                'edges_type1': len(type1_edges),
                'edges_type2': len(type2_edges) if type2_edges is not None else 0,
                'edges_in_graph': len(self.G.edges()) if self.G is not None else 0,
                'execution_time': round(execution_time, 2),
                'threshold': threshold,
                'method': 'cosine',
                'input_format': self.input_format,
                'dual_output': self.input_format == 'bipartite',
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
        Returns the graph as a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns Node1, Node2, Similarity_Score
        """
        if self.edges_to_export is None:
            return pd.DataFrame(columns=['Node1', 'Node2', 'Similarity_Score'])

        return pd.DataFrame(self.edges_to_export, columns=['Node1', 'Node2', 'Similarity_Score'])

    def get_graph_as_json(self):
        """
        Returns the graph as JSON for visualization.

        Returns:
            dict: JSON representation of the graph
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

def run_gene_similarity_cosine(input_file, output_dir, threshold=0.5, sample_size=None, logger=None, progress_callback=None):
    """
    Flask helper. Returns (stats_dict, type1_output, type2_output).
    type2_output is None for a weighted (already one-mode) input.
    sample_size is accepted for API compatibility but no longer caps rows.
    """
    task_id = str(uuid.uuid4())
    type1_output = os.path.join(output_dir, f"cosine_similarity_{task_id}_type1.tsv")
    type2_output = os.path.join(output_dir, f"cosine_similarity_{task_id}_type2.tsv")

    analyzer = GeneSimilarityCosine(sample_size=sample_size, logger=logger)
    if progress_callback:
        analyzer.set_progress_callback(progress_callback)

    stats = analyzer.run_analysis(input_file, type1_output, type2_output, threshold)
    if not stats.get('success'):
        return stats, None, None
    if analyzer.input_format != 'bipartite':
        type2_output = None
    return stats, type1_output, type2_output

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