import argparse
import datetime
import gc
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import beta, norm
from tqdm import tqdm

warnings.filterwarnings('ignore')

class NoiseCorrectedBackbone:
    def __init__(self, input_file, output_file, delta=1.64, alpha_prior=1.0, one_sided=False, max_workers=None, limit=None, directed=False):
        self.input_file = input_file
        self.output_file = output_file
        self.delta = delta
        self.alpha_prior = alpha_prior
        self.one_sided = one_sided
        self.max_workers = max_workers
        self.limit = limit
        self.directed = directed
        self.df = None
        self.source_col = None
        self.target_col = None
        self.weight_col = None

    def detect_file_format(self, file_path, sample_size=5):
        """Detects the file format: delimiter, header, columns"""
        with open(file_path, 'r') as f:
            first_lines = [f.readline().strip() for _ in range(sample_size)]

        first_line = first_lines[0]
        separators = {'\t': first_line.count('\t'),
                      ',': first_line.count(','),
                      ';': first_line.count(';'),
                      ' ': first_line.count(' ')}

        sep = max(separators, key=lambda k: separators[k])
        if separators[sep] == 0:
            sep = '\t'

        has_header = False
        parts = first_line.split(sep)

        header_keywords = ['source', 'target', 'from', 'to', 'node', 'gene',
                          'weight', 'value', 'score', 'name', 'id', 'disease']

        for part in parts:
            part_lower = part.lower()
            if any(keyword in part_lower for keyword in header_keywords):
                has_header = True
                break

        return {
            'sep': sep,
            'has_header': has_header,
            'num_columns': len(parts),
            'sample_header': parts if has_header else None
        }

    def load_data_flexible(self, source_col=None, target_col=None, weight_col=None,
                          sep=None, header: 'str | int | None' = 'infer', names=None, encoding='utf-8'):
        """Load input data from file with flexible column specification."""
        print("Loading data...")

        if sep is None or header == 'infer':
            format_info = self.detect_file_format(self.input_file)

            if sep is None:
                sep = format_info['sep']
                print(f"  Detected separator: {repr(sep)}")

            if header == 'infer':
                header = 0 if format_info['has_header'] else None
                print(f"  Detected header: {header}")

        try:
            self.df = pd.read_csv(
                self.input_file,
                sep=sep,
                header=header,
                names=names,
                nrows=self.limit,
                on_bad_lines='skip',
                engine='python',
                encoding=encoding
            )
            print(f"  Loaded {len(self.df)} rows with columns: {list(self.df.columns)}")

        except Exception as e:
            print(f"Error loading file {self.input_file}: {e}")
            return False

        if self.df.empty:
            print("Error: File is empty")
            return False

        if source_col is None:
            possible_source = ['source', 'Source', 'from', 'From', 'node1', 'Node1',
                              'gene1', 'Gene1', 'u', 'U', 'src', 'Src', 'gene', 'Gene']
            for col in possible_source:
                if col in self.df.columns:
                    source_col = col
                    break
            if source_col is None:
                source_col = self.df.columns[0]
            print(f"  Auto-detected source column: '{source_col}'")

        if target_col is None:
            possible_target = ['target', 'Target', 'to', 'To', 'node2', 'Node2',
                              'gene2', 'Gene2', 'v', 'V', 'dst', 'Dst', 'disease', 'Disease']
            for col in possible_target:
                if col in self.df.columns:
                    target_col = col
                    break
            if target_col is None:
                target_col = self.df.columns[1] if len(self.df.columns) > 1 else self.df.columns[0]
            print(f"  Auto-detected target column: '{target_col}'")

        if weight_col is None:
            possible_weight = ['weight', 'Weight', 'w', 'value', 'Value',
                              'score', 'Score', 'count', 'Count', 'shared', 'Shared']
            for col in possible_weight:
                if col in self.df.columns:
                    weight_col = col
                    break
            if weight_col is None and len(self.df.columns) > 2:
                weight_col = self.df.columns[2]
            if weight_col is not None:
                print(f"  Auto-detected weight column: '{weight_col}'")
            else:
                print(f"  No weight column detected. Using weight = 1 for all edges.")
                self.df['weight_temp'] = 1.0
                weight_col = 'weight_temp'

        self.source_col = source_col
        self.target_col = target_col
        self.weight_col = weight_col

        self.df = self.df.copy()
        self.df['_source'] = self.df[source_col].astype(str).str.strip()
        self.df['_target'] = self.df[target_col].astype(str).str.strip()

        self.df['_weight'] = pd.to_numeric(self.df[weight_col], errors='coerce')

        self.df = self.df.dropna(subset=['_source', '_target', '_weight'])
        self.df = self.df[self.df['_source'] != 'nan']
        self.df = self.df[self.df['_target'] != 'nan']
        self.df = self.df[self.df['_source'] != self.df['_target']]
        self.df = self.df[self.df['_weight'] > 0]

        print(f"  Cleaned data: {len(self.df)} valid edges")
        print(f"  Weight range: {self.df['_weight'].min():.2f} - {self.df['_weight'].max():.2f}")

        return True

    def calculate_edge_stats(self, row, N_total, alpha_prior, delta, one_sided=False, eps=1e-12):
        """
        Numerically-stable calculation of per-edge statistics.
        Returns a dict with extra fields: E_ij, L_ij, L_tilde, P_ij, V_Nij,
        V_L_tilde, significant, z_score
        """
        try:
            Nij = float(row['_weight'])
            Ni = float(row['N_i'])
            Nj = float(row['N_j'])
            N = float(N_total)

            if Ni <= 0 or Nj <= 0 or N <= 0:
                out = {
                    '_source': row['_source'],
                    '_target': row['_target'],
                    '_weight': row['_weight'],
                    'N_i': Ni,
                    'N_j': Nj,
                    'E_ij': 0.0,
                    'L_ij': 0.0,
                    'L_tilde': 0.0,
                    'P_ij': 0.0,
                    'V_Nij': 0.0,
                    'V_L_tilde': 0.0,
                    'significant': False,
                    'z_score': 0.0
                }
                return out

            E_ij = (Ni * Nj) / (N + eps)

            alpha_post = Nij + alpha_prior
            beta_post = (N - Nij) + alpha_prior
            S = alpha_post + beta_post + eps

            P_ij = alpha_post / S

            V_Nij = N * P_ij * (1.0 - P_ij) * (S + N) / (S + 1.0 + eps)
            if not np.isfinite(V_Nij) or V_Nij < 0:
                V_Nij = 0.0

            E_safe = E_ij + eps
            kappa = 1.0 / E_safe
            dE_dnij = ((Nj + Ni) / (N + eps)) - ((Ni * Nj) / ((N + eps) ** 2))
            d_kappa = - dE_dnij / (E_safe * E_safe + eps)

            L_ij = Nij / E_safe
            L_tilde = (L_ij - 1.0) / (L_ij + 1.0 + eps)

            numerator = 2.0 * (kappa + Nij * d_kappa)
            denominator = (kappa * Nij + 1.0) ** 2 + eps
            V_L_tilde = V_Nij * ( (numerator / denominator) ** 2 )

            if not np.isfinite(V_L_tilde) or V_L_tilde <= 0:
                V_L_tilde = eps

            z_score = float(L_tilde / np.sqrt(V_L_tilde + eps))

            if one_sided:
                significant = bool(np.isfinite(z_score) and (z_score > delta))
            else:
                significant = bool(np.isfinite(z_score) and (abs(z_score) > delta))

            result = {
                '_source': row['_source'],
                '_target': row['_target'],
                '_weight': row['_weight'],
                'N_i': Ni,
                'N_j': Nj,
                'E_ij': float(E_ij),
                'L_ij': float(L_ij),
                'L_tilde': float(L_tilde),
                'P_ij': float(P_ij),
                'V_Nij': float(V_Nij),
                'V_L_tilde': float(V_L_tilde),
                'significant': significant,
                'z_score': float(z_score)
            }
            return result

        except Exception as e:
            result = {
                '_source': row['_source'],
                '_target': row['_target'],
                '_weight': row['_weight'],
                'N_i': row.get('N_i', 0),
                'N_j': row.get('N_j', 0),
                'significant': False,
                'z_score': 0.0,
                'E_ij': 0.0,
                'L_ij': 0.0,
                'L_tilde': 0.0,
                'P_ij': 0.0,
                'V_Nij': 0.0,
                'V_L_tilde': 0.0
            }
            return result

    def process_chunk(self, chunk, N_total, alpha_prior, delta, one_sided=False):
        results = []
        for _, row in chunk.iterrows():
            results.append(self.calculate_edge_stats(row, N_total, alpha_prior, delta, one_sided=one_sided))
        return pd.DataFrame(results)

    def noise_corrected_backbone(self):
        assert self.df is not None, "Data not loaded. Call load_data_flexible() first."
        df = self.df.copy()

        df['_weight'] = df['_weight'].astype('float64')

        print("Calculating node strengths...")
        if self.directed:
            node_strength_out = df.groupby('_source')['_weight'].sum()
            node_strength_in = df.groupby('_target')['_weight'].sum()
            N_total = float(df['_weight'].sum())
        else:
            # undirected: one strength per node (both columns), and the symmetric
            # weight matrix counts every edge twice, so N.. = 2 * sum(w)
            strength = df.groupby('_source')['_weight'].sum().add(
                df.groupby('_target')['_weight'].sum(), fill_value=0)
            node_strength_out = node_strength_in = strength
            N_total = 2.0 * float(df['_weight'].sum())

        df['N_i'] = df['_source'].map(node_strength_out).astype('float64')
        df['N_j'] = df['_target'].map(node_strength_in).astype('float64')

        chunk_size = min(50000, max(1000, len(df) // max(1, (cpu_count()))))
        chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]

        workers = self.max_workers if self.max_workers is not None else min(cpu_count(), 8)
        process_func = partial(self.process_chunk, N_total=N_total, alpha_prior=self.alpha_prior,
                              delta=self.delta, one_sided=self.one_sided)

        results = []
        print(f"Processing {len(df):,} edges in {len(chunks)} chunk(s) using {workers} worker(s)")
        with Pool(processes=workers) as pool:
            for result in tqdm(pool.imap(process_func, chunks), total=len(chunks), desc="Processing edges"):
                results.append(result)

        final_df = pd.concat(results, ignore_index=True)
        return final_df

    def save_filtered_edges(self, backbone, params=None):
        try:
            out = backbone.rename(columns={
                '_source': 'source',
                '_target': 'target',
                '_weight': 'weight'
            })

            cols = ['source', 'target', 'weight']
            out[cols].to_csv(self.output_file, index=False, float_format='%.6f', sep='\t')
            print(f"\nSaved {len(backbone)} edges to: {self.output_file}")

        except Exception as e:
            print(f"Error saving: {e}")
            backbone.to_csv(self.output_file, index=False, sep='\t')

    def visualize_network_safely(self, backbone):
        try:
            G = nx.Graph()
            for _, row in backbone.iterrows():
                u = row['_source']
                v = row['_target']
                w = float(row['_weight'])
                G.add_edge(u, v, weight=w)
                G.nodes[u]['label'] = u
                G.nodes[u]['size'] = np.log1p(row.get('N_i', 0.0))
                G.nodes[v]['label'] = v
                G.nodes[v]['size'] = np.log1p(row.get('N_j', 0.0))

            num_nodes = len(G.nodes())
            num_edges = len(G.edges())
            print(f"\nNetwork contains {num_nodes:,} nodes and {num_edges:,} edges")

            nx.write_gexf(G, "backbone_network.gexf", encoding='utf-8', prettyprint=True, version='1.2draft')
            print("GEXF file saved with complete node/edge attributes")

            if num_nodes < 5000 and num_edges > 0:
                plt.figure(figsize=(10, 10))
                pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)

                if num_nodes < 200:
                    nx.draw_networkx(G, pos, with_labels=True, node_size=50, font_size=8, alpha=0.8, edge_color='gray')
                else:
                    nx.draw_networkx_nodes(G, pos, node_size=5, alpha=0.6)
                    nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.2)

                plt.title(f"Network Preview: {num_nodes} nodes, {num_edges} edges")
                plt.axis('off')
                plt.savefig("network_preview.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("Created preview image (network_preview.png)")

        except Exception as e:
            print(f"Visualization error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Noise-corrected backbone extraction for networks')
    parser.add_argument('--strict', action='store_true', help='Use strict filtering')
    parser.add_argument('--delta', type=float, default=3.09, help='Delta threshold (z)')
    parser.add_argument('--zthresh', type=float, default=4.0, help='Z-score threshold for strict filtering')
    parser.add_argument('--percentile', type=float, default=95, help='Node degree percentile for strict filtering')
    parser.add_argument('--output', type=str, default='filtered_edges.tsv', help='Output TSV filename')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of input rows processed')
    parser.add_argument('--input', type=str, required=True, help='Input file')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha prior for Beta')
    parser.add_argument('--one-sided', action='store_true', help='Use one-sided z test (positive L_tilde only)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')

    parser.add_argument('--source-col', default=None, help='Source column name or index')
    parser.add_argument('--target-col', default=None, help='Target column name or index')
    parser.add_argument('--weight-col', default=None, help='Weight column name or index')
    parser.add_argument('--sep', default=None, help='Separator (auto-detect if not specified)')
    parser.add_argument('--header', default='infer', help='Header row (0, None, infer)')
    parser.add_argument('--encoding', default='utf-8', help='File encoding')

    args = parser.parse_args()

    if args.header == 'None':
        header = None
    elif args.header == 'infer':
        header = 'infer'
    else:
        try:
            header = int(args.header)
        except:
            header = args.header

    backbone_extractor = NoiseCorrectedBackbone(
        input_file=args.input,
        output_file=args.output,
        delta=args.delta,
        alpha_prior=args.alpha,
        one_sided=args.one_sided,
        max_workers=args.workers,
        limit=args.limit
    )

    if backbone_extractor.load_data_flexible(
        source_col=args.source_col,
        target_col=args.target_col,
        weight_col=args.weight_col,
        sep=args.sep,
        header=header,
        encoding=args.encoding
    ):
        if args.strict:
            print("\nApplying STRICT backbone filtering")
            backbone = None
            params = {}
        else:
            nc_df = backbone_extractor.noise_corrected_backbone()
            backbone = nc_df[nc_df['significant']].copy()
            params = {
                'filter_type': 'standard',
                'delta': args.delta,
                'row_limit': args.limit,
                'alpha': args.alpha,
                'source_col': args.source_col,
                'target_col': args.target_col,
                'weight_col': args.weight_col
            }

        backbone_extractor.save_filtered_edges(backbone, params)

        if backbone is not None and len(backbone) > 0:
            print("\nStarting visualization...")
            backbone_extractor.visualize_network_safely(backbone)
        else:
            print("No edges passed the backbone filter")
    else:
        print("Failed to load data. Exiting.")

if __name__ == '__main__':
    main()