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
    Class for computing gene similarities using the Resource Allocation index.
    Supports any tab-separated file in bipartite format (node1\\tnode2).

    Resource Allocation index: s(u,v) = Σ 1/|Γ(z)| for z in [Γ(u) ∩ Γ(v)]
    where Γ(u) are the neighbors of node u.
    """

    def __init__(self, sample_size=None, logger=None, n_workers=1):
        """
        Initialize the class.

        Args:
            sample_size: Deprecated/ignored, full file is processed (cap removed).
            logger: Logger object for logging messages
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

    def _clean_data(self, chunk):
        """
        Internal method for cleaning a chunk of data.
        Uses positional access, column names are irrelevant.
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
        with open(self.filepath, 'r', encoding='utf-8') as f:
            first = f.readline().strip().split('\t')
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
            self.filepath,
            sep='\t',
            header=header_row,
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

        for chunk in self._iter_chunks():
            chunk = self._clean_data(chunk)

            for _, row in chunk.iterrows():
                gene    = str(row.iloc[0]).strip()
                disease = str(row.iloc[1]).strip()
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
        Safely split a line.

        Args:
            line (str): The line to split

        Returns:
            list: The parts of the line
        """
        parts = line.split(self.delimiter)
        return [p.strip() for p in parts if p.strip()]

    def process_data(self, filename):
        """
        Processes the file with genes and diseases.
        Uses positional access (iloc[0], iloc[1]) without predefined
        column names, works with any tab-separated file.

        Args:
            filename (str): Path to the input file
        """
        self.log(f"Starting data processing from {filename}")
        self.update_progress(10)

        self.filepath = filename

        header_row = self._detect_header()
        self.has_header = header_row is not None
        if self.has_header:
            self.log("Header detected, will skip first line")

        gene_to_diseases = defaultdict(set)
        disease_to_genes = defaultdict(set)
        gene_index = {}
        disease_index = {}
        line_count = 0

        chunksize = 10_000

        try:
            chunk_reader = pd.read_csv(
                filename,
                sep=self.delimiter,
                header=header_row,
                dtype=str,
                chunksize=chunksize,
                on_bad_lines='skip',
                engine='python',
            )

            for chunk in chunk_reader:
                for _, row in chunk.iterrows():
                    try:
                        node1 = str(row.iloc[0]).strip()
                        node2 = str(row.iloc[1]).strip()
                    except IndexError:
                        continue

                    if not node1 or not node2:
                        continue
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
        Builds a bipartite graph from the data.

        Returns:
            networkx.Graph: Bipartite graph
        """
        if self.gene_to_diseases is None or self.disease_to_genes is None:
            raise ValueError("Data not processed. Call process_data() first.")

        self.log("Building bipartite graph...")
        self.update_progress(40)

        B = nx.Graph()

        gene_count = 0
        for gene in self.gene_to_diseases:
            B.add_node(gene, bipartite=0)
            gene_count += 1

        disease_count = 0
        for disease in self.disease_to_genes:
            B.add_node(disease, bipartite=1)
            disease_count += 1

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

            for chunk in self._iter_chunks():
                chunk = self._clean_data(chunk)

                for _, row in chunk.iterrows():
                    gene    = str(row.iloc[0]).strip()
                    disease = str(row.iloc[1]).strip()
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

    def _compute_edges(self, nodes, threshold):
        """Resource-Allocation similarity among `nodes` (one side of the
        bipartite graph). Each common neighbour contributes 1/degree.
        Returns (edges, graph)."""
        assert self.bipartite_graph is not None
        degrees = dict(self.bipartite_graph.degree())
        neighbors_dict = {node: set(self.bipartite_graph.neighbors(node)) for node in nodes}

        n = len(nodes)
        self.log(f"  {n} nodes, {n*(n-1)//2:,} pairs, {self.n_workers} worker(s)")

        chunk = max(1, n // max(1, self.n_workers * 4))
        batches = [
            (i_start, min(i_start + chunk, n), nodes, neighbors_dict, degrees, threshold)
            for i_start in range(0, n, chunk)
        ]

        G = nx.Graph()
        edges = []
        if self.n_workers > 1 and len(batches) > 1:
            with Pool(processes=self.n_workers) as pool:
                for batch_result in pool.imap_unordered(_resource_allocation_gene_batch, batches):
                    for a, b, w in batch_result:
                        G.add_edge(a, b, weight=w)
                        edges.append((a, b, w))
        else:
            for batch in batches:
                for a, b, w in _resource_allocation_gene_batch(batch):
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

    def run_analysis(self, input_filename, type1_output, type2_output=None, threshold=0.5):
        """Full analysis producing BOTH one-mode projections of the bipartite
        input (type-1 = col 0, type-2 = col 1). If type2_output is None only
        the type-1 projection is written."""
        start_time = time.time()

        try:
            self.process_data(input_filename)
            self.build_bipartite_graph()
            self.update_progress(60)

            assert self.gene_index is not None and self.disease_index is not None

            type1_nodes = list(self.gene_index.keys())
            type1_edges, self.G = self._compute_edges(type1_nodes, threshold)
            self._write_edges_file(type1_edges, type1_output)
            self.edges_to_export = type1_edges
            self.log(f"Type-1 projection: {len(type1_nodes)} nodes, {len(type1_edges)} edges")

            type2_edges = None
            if type2_output:
                type2_nodes = list(self.disease_index.keys())
                type2_edges, _ = self._compute_edges(type2_nodes, threshold)
                self._write_edges_file(type2_edges, type2_output)
                self.log(f"Type-2 projection: {len(type2_nodes)} nodes, {len(type2_edges)} edges")

            self.update_progress(100)
            execution_time = time.time() - start_time

            stats = {
                'success': True,
                'nodes_type1': len(self.gene_index),
                'nodes_type2': len(self.disease_index),
                'edges_type1': len(type1_edges),
                'edges_type2': len(type2_edges) if type2_edges is not None else 0,
                'bipartite_nodes': len(self.bipartite_graph.nodes()) if self.bipartite_graph is not None else 0,
                'bipartite_edges': len(self.bipartite_graph.edges()) if self.bipartite_graph is not None else 0,
                'execution_time': round(execution_time, 2),
                'threshold': threshold,
                'method': 'resource-allocation',
                'dual_output': type2_output is not None,
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
        Returns the similarity graph as a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns Gene1, Gene2, Similarity_Score
        """
        if self.edges_to_export is None:
            return pd.DataFrame(columns=['Gene1', 'Gene2', 'Similarity_Score'])

        return pd.DataFrame(self.edges_to_export, columns=['Gene1', 'Gene2', 'Similarity_Score'])

    def get_graph_as_json(self):
        """
        Returns the similarity graph as JSON for visualization.

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

    def get_bipartite_graph_stats(self):
        """
        Returns statistics for the bipartite graph.

        Returns:
            dict: Bipartite graph statistics
        """
        if self.bipartite_graph is None:
            return {}

        gene_nodes = [n for n, attr in self.bipartite_graph.nodes(data=True) if attr.get('bipartite') == 0]
        disease_nodes = [n for n, attr in self.bipartite_graph.nodes(data=True) if attr.get('bipartite') == 1]

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
            if coo.row[idx] < coo.col[idx]:
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

def run_resource_allocation_similarity(input_file, output_dir, threshold=0.5, sample_size=1000, logger=None, progress_callback=None, n_workers=1):
    """
    Helper function to run the Resource Allocation analysis from Flask.

    Args:
        input_file (str): Path to the input file
        output_dir (str): Folder to save results
        threshold (float): Similarity threshold
        sample_size (int): Number of rows to process
        logger: Logger object
        progress_callback: Callback for progress

    Returns:
        tuple: (stats_dict, output_filename)
    """
    task_id = str(uuid.uuid4())
    type1_output = os.path.join(output_dir, f"resource_allocation_{task_id}_type1.tsv")
    type2_output = os.path.join(output_dir, f"resource_allocation_{task_id}_type2.tsv")

    analyzer = GeneSimilarityResourceAllocation(sample_size=sample_size, logger=logger, n_workers=n_workers)
    if progress_callback:
        analyzer.set_progress_callback(progress_callback)

    stats = analyzer.run_analysis(input_file, type1_output, type2_output, threshold)
    if not stats.get('success'):
        return stats, None, None
    return stats, type1_output, type2_output

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Gene Similarity Resource Allocation Analysis')
    parser.add_argument('input_file', help='Input file path (bipartite format: gene\\tdisease)')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold')
    parser.add_argument('--sample-size', type=int, default=1000, help='Number of lines to process')

    args = parser.parse_args()

    analyzer = GeneSimilarityResourceAllocation(sample_size=args.sample_size)
    stats = analyzer.run_analysis(args.input_file, args.output_file, threshold=args.threshold)

    if stats['success']:
        print(f"\n=== Analysis Successful ===")
        print(f"Type-1 nodes: {stats['nodes_type1']}")
        print(f"Type-2 nodes: {stats['nodes_type2']}")
        print(f"Bipartite graph: {stats['bipartite_nodes']} nodes, {stats['bipartite_edges']} edges")
        print(f"Type-1 projection edges: {stats['edges_type1']}")
        print(f"Execution time: {stats['execution_time']} seconds")
        print(f"Output saved to: {args.output_file}")
    else:
        print(f"Error: {stats.get('error', 'Unknown error')}")