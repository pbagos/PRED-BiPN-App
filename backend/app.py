import os
import io
import uuid
import sys
import logging
import traceback
import threading
from datetime import datetime
from multiprocessing import Pool, cpu_count
from urllib.parse import unquote
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import base64

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

from disparity import run_disparity_filter
from one_mode_projection_sparse import run_one_mode_projection
from weighted_projection import run_one_mode_projection as run_weighted_projection
from doubly_stochastic import TrueMultiscaleBackbone
from hierarchical_backbone import HierarchicalBackboneExtractor
from high_salience_skeleton import HighSalienceSkeleton, run_hss
from hypergraph import Hypergraph
from lans import run_lans, LANS, create_lans_extractor
from mlf import MLFProcessor, run_mlf, create_mlf_processor
from mst import KruskalMST
from noise_corrected import NoiseCorrectedBackbone
from modularity_filter import ModularityFilter
from odds import OddsRatioCalculator
from polya import PolyaFilter, create_polya_filter
from cosine import GeneSimilarityCosine, run_gene_similarity_cosine
from adamic_adar import run_adamic_adar_similarity
from resource_allocation import run_resource_allocation_similarity
from jaccard import run_jaccard_analysis, run_jaccard_dual_analysis, JaccardSimilarityAnalyzer
from kendall import run_kendall_analysis
from spearman import run_spearman_analysis
from pearson import run_pearson_analysis, run_bipartite_pearson_dual
from edge_betweenness import run_girvan_newman_analysis
from global_threshold import run_global_threshold

frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_path, static_url_path='')

ENV = os.environ.get('FLASK_ENV', 'production')
DEBUG = ENV == 'development'
IS_RENDER = os.environ.get('RENDER', False)

if IS_RENDER:
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
else:
    app.config['UPLOAD_FOLDER'] = os.path.abspath('tmp')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if IS_RENDER:
    CORS(app, supports_credentials=True, origins=[
        "https://*.onrender.com"
    ])
else:
    CORS(app, supports_credentials=True)

def normalize_method_name(method_name):
    if not isinstance(method_name, str):
        return method_name

    trimmed_method_name = method_name.strip()

    if trimmed_method_name.lower() == 'marginall likelihood filter':
        return 'Marginal Likelihood Filter'

    return trimmed_method_name

task_store = {}
task_store_lock = threading.Lock()

class TaskLogHandler(logging.Handler):
    """Captures log records for a specific task into task_store."""
    def __init__(self, task_id):
        super().__init__()
        self.task_id = task_id
        self.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        with task_store_lock:
            if self.task_id in task_store:
                task_store[self.task_id]['logs'].append(msg)

def _execute_processing(data, task_id):
    """Run the full processing pipeline in a background thread."""
    handler = TaskLogHandler(task_id)
    app.logger.addHandler(handler)

    def log(msg, level='info'):
        with task_store_lock:
            if task_id in task_store:
                task_store[task_id]['logs'].append(f'[{level.upper()}] {msg}')

    with app.app_context():
        try:
            method = normalize_method_name(data.get('method'))
            safe_method = method.lower().replace(' ', '_').replace('/', '_')
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            alpha = float(data.get('alpha', 0.05))
            uploaded_filename = data.get('uploaded_file')
            result_files = {}
            response_data = None

            input_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
            log(f'Starting: {method}')
            log(f'Input file: {uploaded_filename}')

            if method.lower() == 'disparity filter':
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}.tsv')
                run_disparity_filter(input_path, output_path, alpha=alpha, n_workers=cpu_count(), progress_callback=None)
                result_files = {'result_file': os.path.basename(output_path)}

            elif method.lower() == 'one mode projection':
                node1_output = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node1.tsv')
                node2_output = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node2.tsv')
                omp_res = run_one_mode_projection(input_path, node1_output, node2_output)
                if isinstance(omp_res, dict) and not omp_res.get('success', True):
                    raise RuntimeError(omp_res.get('error', 'One mode projection failed'))
                result_files = {
                    'node1_file': os.path.basename(node1_output),
                    'node2_file': os.path.basename(node2_output)
                }

            elif method.lower() == "doubly stochastic filter":
                output_path = os.path.join(app.config["UPLOAD_FOLDER"], f'result_{safe_method}_{ts}.tsv')
                extractor = TrueMultiscaleBackbone(significance_level=alpha)
                extractor.extract_backbone(
                    edge_list_path=input_path,
                    output_path=output_path,
                    mode='strict',
                    nrows=200000,
                    logger=app.logger
                )
                result_files = {'result_file': os.path.basename(output_path)}

            elif method.lower() == "hierarchical backbone":
                base_path = os.path.join(app.config["UPLOAD_FOLDER"], f"result_{safe_method}_{ts}")
                output_A = base_path + "_A.tsv"
                output_B = base_path + "_B.tsv"
                extractor = HierarchicalBackboneExtractor(
                    z_threshold=float(data.get('z_threshold', 1.64)),
                    alpha_percentile=int(data.get('alpha_percentile', 95)),
                    chunk_size=100000,
                    n_jobs=cpu_count()
                )
                extractor.run(input_path, output_A, output_B, logger=app.logger)
                result_files = {
                    'node1_file': os.path.basename(output_A),
                    'node2_file': os.path.basename(output_B)
                }

            elif method.lower() == "high salience skeleton":
                output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"result_{safe_method}_{ts}.tsv")
                threshold = float(data.get('threshold', 0.9))
                test_mode = str(data.get('test_mode', 'false')).lower() == 'true'
                extractor = HighSalienceSkeleton(threshold=threshold, n_jobs=cpu_count())
                df_result = extractor.run(
                    input_file=input_path,
                    output_file=output_path,
                    test_mode=test_mode,
                    logger=app.logger
                )
                result_files = {'result_file': os.path.basename(output_path)}
                response_data = {
                    'success': True, 'task_id': task_id, 'files': result_files,
                    'stats': {
                        'edges_original': len(df_result) if not df_result.empty else 0,
                        'edges_filtered': len(df_result[df_result['salience'] >= threshold]) if not df_result.empty else 0,
                        'threshold': threshold, 'method': 'high_salience_skeleton'
                    }
                }

            elif method.lower() == "hypergraph projection":
                hg = Hypergraph(input_file=input_path)
                gene_content, disease_content = hg.run()
                col_info = hg.get_column_info()
                app.logger.info(f"Detected columns: {col_info}")
                node1_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{safe_method}_{ts}_node1.tsv")
                node2_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{safe_method}_{ts}_node2.tsv")
                with open(node1_path, "w", encoding="utf-8") as f:
                    f.write(gene_content)
                result_files = {"node1_file": os.path.basename(node1_path)}
                if disease_content:
                    with open(node2_path, "w", encoding="utf-8") as f:
                        f.write(disease_content)
                    result_files["node2_file"] = os.path.basename(node2_path)

            elif method == "Locally Adaptive Network Sparsification Filter":
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{safe_method}_{ts}.tsv")
                alpha = float(data.get('alpha', 0.05))
                test_mode = str(data.get('test_mode', 'false')).lower() == 'true'
                app.logger.info(f"Detecting columns for LANS in file: {input_path}")
                sample_df = pd.read_csv(input_path, sep=None, engine='python', nrows=1)
                all_cols = sample_df.columns.tolist()
                src_col = all_cols[0]; tgt_col = all_cols[1]
                wgt_col = all_cols[2] if len(all_cols) > 2 else None
                for col in all_cols:
                    col_lower = str(col).lower()
                    if col_lower in ['source', 'node1', 'gene1', 'from']: src_col = col
                    if col_lower in ['target', 'node2', 'gene2', 'to']: tgt_col = col
                    if col_lower in ['weight', 'value', 'score', 'shared_diseases']: wgt_col = col
                app.logger.info(f"Auto-mapped columns: Source={src_col}, Target={tgt_col}, Weight={wgt_col}")
                result_csv = run_lans(
                    input_file=input_path, output_file=output_path, alpha=alpha,
                    source_col=src_col, target_col=tgt_col, weight_col=wgt_col,
                    test_mode=test_mode, logger=app.logger
                )
                result_files = {"result_file": os.path.basename(output_path)}

            elif method.lower() == "marginal likelihood filter":
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{safe_method}_{ts}.tsv")
                alpha_val = float(data.get('alpha', 0.05))
                test_mode = str(data.get('test_mode', 'false')).lower() == 'true'
                n_edges, full_results_df = run_mlf(
                    input_path, output_path, alpha=alpha_val, test_mode=test_mode, logger=app.logger
                )
                result_files = {
                    'result_file': os.path.basename(output_path),
                    'full_results_file': os.path.basename(output_path.replace('.tsv', '_full.tsv'))
                }
                response_data = {
                    'success': True, 'task_id': task_id, 'files': result_files,
                    'stats': {
                        'total_edges': len(full_results_df),
                        'significant_edges': n_edges,
                        'retention_rate': (n_edges / len(full_results_df) * 100) if len(full_results_df) > 0 else 0,
                        'alpha': alpha_val, 'method': 'mlf', 'columns': ['source', 'target', 'weight', 'p_value']
                    }
                }

            elif method.lower() == "maximum spanning tree filter":
                kruskal = KruskalMST(test_mode=False)
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{safe_method}_{ts}.tsv")
                kruskal.run_and_save(input_path, output_path)
                result_files = {'result_file': os.path.basename(output_path)}

            elif method.lower() == "noise corrected filter":
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{safe_method}_{ts}.tsv")
                extractor = NoiseCorrectedBackbone(
                    input_file=input_path, output_file=output_path,
                    delta=3.09, alpha_prior=alpha, one_sided=False,
                    max_workers=cpu_count(), limit=200_000
                )
                if not extractor.load_data_flexible():
                    raise RuntimeError('Failed to load data for Noise Corrected Filter.')
                nc_df = extractor.noise_corrected_backbone()
                backbone = nc_df[nc_df['significant']].copy()
                if backbone.empty:
                    raise RuntimeError('No significant edges found.')
                params = {'filter_type': 'noise_corrected', 'delta': extractor.delta,
                          'alpha_prior': extractor.alpha_prior, 'row_limit': extractor.limit}
                extractor.save_filtered_edges(backbone, params)
                result_files = {'result_file': os.path.basename(output_path)}

            elif method.lower() == "modularity filter":
                mf = ModularityFilter(percentage=0.3, test_mode=False)
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{safe_method}_{ts}.tsv")
                stats = mf.run(input_path, output_path)
                if not stats['success']:
                    raise RuntimeError(stats.get('message', 'Modularity filter failed'))
                result_files = {'result_file': os.path.basename(output_path)}

            elif method.lower() == "odds ratio weighted projection":
                alpha        = float(data.get('alpha', 0.05))
                or_threshold = float(data.get('or_threshold', 1.0))
                correction   = data.get('correction', 'fdr_bh')
                app.logger.info(
                    f"Starting bipartite odds ratio analysis (alpha={alpha}, correction={correction}, OR>{or_threshold})"
                )
                output_node1 = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{safe_method}_{ts}_node1.tsv")
                output_node2 = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{safe_method}_{ts}_node2.tsv")
                calculator = OddsRatioCalculator(chunk_size=20000, batch_size=50000)
                results = calculator.run_dual_bipartite_analysis(
                    input_file=input_path,
                    col0_pairs_output=output_node1,
                    col1_pairs_output=output_node2,
                    sep=None,
                    alpha=alpha, or_threshold=or_threshold, correction=correction,
                    progress_callback=None
                )
                if not results['success']:
                    raise RuntimeError(results.get('error', 'Odds ratio analysis failed'))
                result_files = {
                    'node1_file': os.path.basename(output_node1),
                    'node2_file': os.path.basename(output_node2)
                }
                response_data = {
                    'success': True, 'task_id': task_id, 'files': result_files,
                    'stats': results
                }

            elif method.lower() == "polya filter":
                alpha = float(data.get('alpha', 0.05))
                a_param = float(data.get('a_parameter', 1.0))
                correction_method = data.get('correction_method', 'fdr')
                directed = str(data.get('directed', 'false')).lower() == 'true'
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{safe_method}_{ts}.tsv")
                polya_filter = create_polya_filter(significance_level=alpha, method=correction_method, logger=app.logger, n_workers=cpu_count())
                results = polya_filter.run_full_analysis(
                    input_file=input_path, output_file=output_path, a=a_param, directed=directed
                )
                if not results['success']:
                    raise RuntimeError(results.get('error', 'Polya filter failed'))
                result_files = {'result_file': os.path.basename(output_path)}
                backbone_count = results.get('edges_backbone', 0)
                warning = None
                if backbone_count == 0:
                    warning = (
                        f"No backbone edges survived the {correction_method.upper()} correction. "
                        "Try using ML estimation (a=-1), a higher alpha, or Bonferroni correction."
                    )
                    app.logger.warning(warning)
                response_data = {
                    'success': True, 'task_id': task_id, 'files': result_files,
                    'stats': results, 'warning': warning
                }

            elif method.lower() == "cosine similarity weighted projection":
                threshold = float(data.get('threshold', 0.5))
                app.logger.info(f"Starting Cosine Similarity analysis")
                stats, type1_out, type2_out = run_gene_similarity_cosine(
                    input_file=input_path, output_dir=app.config['UPLOAD_FOLDER'],
                    threshold=threshold, sample_size=None,
                    logger=app.logger, progress_callback=None
                )
                if not stats['success']:
                    raise RuntimeError(stats.get('error', 'Cosine similarity failed'))
                result_files = {}
                if type1_out and os.path.exists(type1_out):
                    new1 = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node1.tsv')
                    os.rename(type1_out, new1)
                    result_files['node1_file'] = os.path.basename(new1)
                if type2_out and os.path.exists(type2_out):
                    new2 = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node2.tsv')
                    os.rename(type2_out, new2)
                    result_files['node2_file'] = os.path.basename(new2)
                response_data = {'success': True, 'task_id': task_id, 'files': result_files, 'stats': stats}

            elif method.lower() == "adamic adar similarity weighted projection":
                threshold = float(data.get('threshold', 0.5))
                app.logger.info(f"Starting Adamic-Adar analysis")
                stats, type1_out, type2_out = run_adamic_adar_similarity(
                    input_file=input_path, output_dir=app.config['UPLOAD_FOLDER'],
                    threshold=threshold, sample_size=None,
                    logger=app.logger, progress_callback=None, n_workers=cpu_count()
                )
                if not stats['success']:
                    raise RuntimeError(stats.get('error', 'Adamic-Adar failed'))
                result_files = {}
                if type1_out and os.path.exists(type1_out):
                    new1 = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node1.tsv')
                    os.rename(type1_out, new1)
                    result_files['node1_file'] = os.path.basename(new1)
                if type2_out and os.path.exists(type2_out):
                    new2 = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node2.tsv')
                    os.rename(type2_out, new2)
                    result_files['node2_file'] = os.path.basename(new2)
                response_data = {'success': True, 'task_id': task_id, 'files': result_files, 'stats': stats}

            elif method.lower() == "resource allocation similarity weighted projection":
                threshold = float(data.get('threshold', 0.5))
                app.logger.info(f"Starting Resource Allocation analysis")
                stats, type1_out, type2_out = run_resource_allocation_similarity(
                    input_file=input_path, output_dir=app.config['UPLOAD_FOLDER'],
                    threshold=threshold, sample_size=None,
                    logger=app.logger, progress_callback=None, n_workers=cpu_count()
                )
                if not stats['success']:
                    raise RuntimeError(stats.get('error', 'Resource Allocation failed'))
                result_files = {}
                if type1_out and os.path.exists(type1_out):
                    new1 = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node1.tsv')
                    os.rename(type1_out, new1)
                    result_files['node1_file'] = os.path.basename(new1)
                if type2_out and os.path.exists(type2_out):
                    new2 = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node2.tsv')
                    os.rename(type2_out, new2)
                    result_files['node2_file'] = os.path.basename(new2)
                response_data = {'success': True, 'task_id': task_id, 'files': result_files, 'stats': stats}

            elif method.lower() == "jaccard similarity weighted projection":
                threshold = float(data.get('threshold', 0.0))
                chunksize = int(data.get('chunksize', 10000))
                app.logger.info(f"Starting Jaccard Similarity analysis")
                stats, output_gene, output_disease = run_jaccard_dual_analysis(
                    input_file=input_path, output_dir=app.config['UPLOAD_FOLDER'],
                    threshold=threshold, chunksize=chunksize,
                    logger=app.logger, progress_callback=None
                )
                if not stats['success']:
                    raise RuntimeError(stats.get('error', 'Jaccard projection failed'))
                result_files = {}
                if output_gene and os.path.exists(output_gene):
                    new_gene = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_gene.tsv')
                    os.rename(output_gene, new_gene)
                    result_files['gene_file'] = os.path.basename(new_gene)
                if output_disease and os.path.exists(output_disease):
                    new_disease = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_disease.tsv')
                    os.rename(output_disease, new_disease)
                    result_files['disease_file'] = os.path.basename(new_disease)
                response_data = {'success': True, 'task_id': task_id, 'files': result_files, 'stats': stats}

            elif method.lower() == "kendall correlation weighted projection":
                alpha = float(data.get('alpha', 0.05))
                correction = data.get('correction', 'fdr_bh')
                chunksize = int(data.get('chunksize', 10000))
                stats, type1_out, type2_out = run_kendall_analysis(
                    input_file=input_path, output_dir=app.config['UPLOAD_FOLDER'],
                    alpha=alpha, correction=correction, chunksize=chunksize,
                    logger=app.logger, progress_callback=None
                )
                if not stats['success']:
                    raise RuntimeError(stats.get('error', 'Kendall failed'))
                result_files = {}
                if type1_out and os.path.exists(type1_out):
                    new1 = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node1.tsv')
                    os.rename(type1_out, new1)
                    result_files['node1_file'] = os.path.basename(new1)
                if type2_out and os.path.exists(type2_out):
                    new2 = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node2.tsv')
                    os.rename(type2_out, new2)
                    result_files['node2_file'] = os.path.basename(new2)
                response_data = {'success': True, 'task_id': task_id, 'files': result_files, 'stats': stats}

            elif method.lower() == "spearman correlation weighted projection":
                alpha = float(data.get('alpha', 0.05))
                correction = data.get('correction', 'fdr_bh')
                chunksize = int(data.get('chunksize', 10000))
                stats, type1_out, type2_out = run_spearman_analysis(
                    input_file=input_path, output_dir=app.config['UPLOAD_FOLDER'],
                    alpha=alpha, correction=correction, chunksize=chunksize,
                    logger=app.logger, progress_callback=None
                )
                if not stats['success']:
                    raise RuntimeError(stats.get('error', 'Spearman failed'))
                result_files = {}
                if type1_out and os.path.exists(type1_out):
                    new1 = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node1.tsv')
                    os.rename(type1_out, new1)
                    result_files['node1_file'] = os.path.basename(new1)
                if type2_out and os.path.exists(type2_out):
                    new2 = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node2.tsv')
                    os.rename(type2_out, new2)
                    result_files['node2_file'] = os.path.basename(new2)
                response_data = {'success': True, 'task_id': task_id, 'files': result_files, 'stats': stats}

            elif method.lower() == "pearson correlation weighted projection":
                alpha = float(data.get('alpha', 0.05))
                correction = data.get('correction', 'fdr_bh')
                app.logger.info(f"Starting bipartite Pearson dual projection (alpha={alpha}, correction={correction})")
                node1_output = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node1.tsv')
                node2_output = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node2.tsv')
                stats = run_bipartite_pearson_dual(
                    input_file=input_path,
                    node1_output=node1_output,
                    node2_output=node2_output,
                    alpha=alpha, correction=correction,
                    logger=app.logger, progress_callback=None
                )
                if not stats['success']:
                    raise RuntimeError(stats.get('error', 'Pearson failed'))
                result_files = {
                    'node1_file': os.path.basename(node1_output),
                    'node2_file': os.path.basename(node2_output)
                }
                response_data = {'success': True, 'task_id': task_id, 'files': result_files, 'stats': stats}

            elif method.lower() == "edge betweenness":
                max_lines = int(data.get('max_lines', 1000))
                num_processes = int(data.get('num_processes', cpu_count() - 1))
                early_stopping = str(data.get('early_stopping', 'true')).lower() == 'true'
                min_communities = int(data.get('min_communities', 2))
                max_iterations = int(data.get('max_iterations', 20))
                app.logger.info(f"Starting Girvan-Newman Community Detection")
                stats, result_files = run_girvan_newman_analysis(
                    input_file=input_path, output_dir=app.config['UPLOAD_FOLDER'],
                    max_lines=max_lines, num_processes=num_processes,
                    early_stopping=early_stopping, min_communities=min_communities,
                    max_iterations=max_iterations, logger=app.logger, progress_callback=None
                )
                if not stats['success']:
                    raise RuntimeError(stats.get('error', 'Girvan-Newman failed'))
                renamed_files = {}
                for key, fname in result_files.items():
                    old_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                    if os.path.exists(old_path):
                        ext = os.path.splitext(fname)[1]
                        new_name = f'result_{safe_method}_{ts}_{key}{ext}'
                        os.rename(old_path, os.path.join(app.config['UPLOAD_FOLDER'], new_name))
                        renamed_files[key] = new_name
                    else:
                        renamed_files[key] = fname
                result_files = renamed_files
                response_data = {'success': True, 'task_id': task_id, 'files': result_files, 'stats': stats}

            elif method.lower() == "simple weighted projection":
                app.logger.info(f"Starting Simple Weighted Projection")
                node1_output = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node1_edges.tsv')
                node2_output = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}_node2_edges.tsv')
                run_weighted_projection(input_path, node1_output, node2_output)
                app.logger.info("One mode projection completed")
                if os.path.exists(node1_output) and os.path.exists(node2_output):
                    try:
                        df_gene = pd.read_csv(node1_output)
                        df_disease = pd.read_csv(node2_output)
                    except pd.errors.ParserError:
                        try:
                            df_gene = pd.read_csv(node1_output, sep='\t')
                            df_disease = pd.read_csv(node2_output, sep='\t')
                        except Exception:
                            df_gene = pd.read_csv(node1_output, sep=None, engine='python')
                            df_disease = pd.read_csv(node2_output, sep=None, engine='python')
                    result_files = {
                        'node1_file': os.path.basename(node1_output),
                        'node2_file': os.path.basename(node2_output)
                    }
                    response_data = {
                        'success': True, 'task_id': task_id, 'files': result_files,
                        'stats': {'gene_edges': len(df_gene), 'disease_edges': len(df_disease),
                                  'method': 'simple_weighted_projection'}
                    }
                else:
                    raise RuntimeError('Simple Weighted Projection failed - output files not created')

            elif method.lower() == "global threshold filter":
                col_index = int(data.get('col_index', 2))
                threshold = float(data.get('threshold', 0.0))
                direction = data.get('direction', 'gte')
                chunk_size = int(data.get('chunk_size', 50000))
                app.logger.info(f"Global Threshold Filter: col={col_index}, threshold={threshold}, direction={direction}")
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{safe_method}_{ts}.tsv')
                stats = run_global_threshold(
                    input_file=input_path, output_file=output_path,
                    col_index=col_index, threshold=threshold,
                    direction=direction, chunk_size=chunk_size,
                    logger=app.logger, progress_callback=None
                )
                if not stats['success']:
                    raise RuntimeError(stats.get('error', 'Global Threshold Filter failed'))
                result_files = {'result_file': os.path.basename(output_path)}
                response_data = {
                    'success': True, 'task_id': task_id, 'files': result_files, 'stats': stats
                }

            else:
                raise RuntimeError(f'Method "{method}" not recognized')

            final_result = response_data if response_data is not None else {
                'success': True, 'task_id': task_id, 'files': result_files
            }
            log(f'Processing complete. Output: {list(result_files.values())}')
            with task_store_lock:
                task_store[task_id]['status'] = 'done'
                task_store[task_id]['result'] = final_result

        except Exception as e:
            traceback.print_exc()
            log(f'Error: {str(e)}', 'error')
            with task_store_lock:
                task_store[task_id]['status'] = 'error'
                task_store[task_id]['error'] = str(e)
        finally:
            app.logger.removeHandler(handler)

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files.get('file')
    method = request.form.get('method')

    if not uploaded_file or not method:
        return jsonify({'success': False, 'message': 'Missing file or method'}), 400

    filename = f"{uuid.uuid4().hex}_{uploaded_file.filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(path)

    return jsonify({'success': True, 'uploaded_file': filename})

@app.route('/process', methods=['POST'])
def process_file():
    if request.is_json:
        raw = request.get_json()
    else:
        raw = dict(request.form)

    uploaded_filename = raw.get('uploaded_file') if isinstance(raw, dict) else None
    method = raw.get('method') if isinstance(raw, dict) else None

    if not uploaded_filename or not method:
        return jsonify({'success': False, 'message': 'File or method missing'}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER'],
                              uploaded_filename if isinstance(uploaded_filename, str) else uploaded_filename[0])
    if not os.path.exists(input_path):
        return jsonify({'success': False, 'message': 'Input file not found.'}), 404

    task_id = str(uuid.uuid4())
    with task_store_lock:
        task_store[task_id] = {'status': 'running', 'logs': [], 'result': None, 'error': None}

    data = {k: (v[0] if isinstance(v, list) else v) for k, v in raw.items()} if isinstance(raw, dict) else raw

    thread = threading.Thread(target=_execute_processing, args=(data, task_id), daemon=True)
    thread.start()

    return jsonify({'success': True, 'task_id': task_id, 'status': 'running'}), 202

@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    entry = task_store.get(task_id)
    if not entry:
        return jsonify({'status': 'not_found'}), 404
    with task_store_lock:
        return jsonify({
            'status': entry['status'],
            'logs': list(entry['logs']),
            'result': entry.get('result'),
            'error': entry.get('error')
        })

@app.route('/get_file_columns', methods=['GET'])
def get_file_columns():
    """Detect columns in uploaded file for Global Threshold Filter"""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'success': False, 'message': 'No filename provided'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': 'File not found'}), 404

    try:
        df = pd.read_csv(filepath, sep=None, engine='python', nrows=10)

        columns = []
        for i, col in enumerate(df.columns):
            col_type = 'unknown'
            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = 'numeric'
            else:
                col_type = 'text'
            columns.append({'index': i, 'name': col, 'type': col_type})

        return jsonify({'success': True, 'columns': columns})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/detect_columns', methods=['POST'])
def detect_columns():
    """Upload file and detect columns"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400

    try:
        df = pd.read_csv(file.stream, sep=None, engine='python', nrows=10)

        columns = []
        for i, col in enumerate(df.columns):
            col_type = 'unknown'
            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = 'numeric'
            else:
                col_type = 'text'
            columns.append({'index': i, 'name': col, 'type': col_type})

        return jsonify({'success': True, 'columns': columns})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/visualize/compare', methods=['POST'])
def compare_networks():
    """Compare two networks (before and after) - returns two separate images"""
    try:
        data = request.get_json()
        before_file = data.get('before_file')
        after_file = data.get('after_file')
        method = data.get('method')

        if not before_file or not after_file:
            return jsonify({'success': False, 'message': 'Both files are required'}), 400

        before_path = os.path.join(app.config['UPLOAD_FOLDER'], before_file)
        after_path = os.path.join(app.config['UPLOAD_FOLDER'], after_file)

        if not os.path.exists(before_path) or not os.path.exists(after_path):
            return jsonify({'success': False, 'message': 'One of the files was not found'}), 404

        try:
            df_before = pd.read_csv(before_path, sep=None, engine='python')
            df_after = pd.read_csv(after_path, sep=None, engine='python')
        except:
            df_before = pd.read_csv(before_path, sep='\t', engine='python')
            df_after = pd.read_csv(after_path, sep='\t', engine='python')

        G_before = nx.Graph()
        G_after = nx.Graph()

        for _, row in df_before.iterrows():
            src = str(row.iloc[0])
            tgt = str(row.iloc[1])
            G_before.add_edge(src, tgt)

        for _, row in df_after.iterrows():
            src = str(row.iloc[0])
            tgt = str(row.iloc[1])
            G_after.add_edge(src, tgt)

        # Vivid, high-contrast colormap with dark node outlines keeps low-degree
        # nodes visible instead of washing out to white like the old 'Blues' map.
        # Each image is scaled to its OWN degree range so both use the full colour
        # gradient; a colorbar legend maps colour -> node degree per image.
        CMAP = 'turbo'

        fig_before, ax_before = plt.subplots(1, 1, figsize=(8, 6))
        if G_before.number_of_nodes() > 0:
            pos_before = nx.spring_layout(G_before, k=1, iterations=30)
            degrees_before = dict(G_before.degree())
            node_colors_before = list(degrees_before.values())
            vmin_b, vmax_b = min(node_colors_before), max(node_colors_before)
            if vmin_b == vmax_b:
                vmax_b = vmin_b + 1

            nx.draw_networkx_edges(G_before, pos_before, alpha=0.35, edge_color='#4a4a4a', width=0.8, ax=ax_before)
            nodes_b = nx.draw_networkx_nodes(G_before, pos_before, node_color=node_colors_before,
                                  node_size=70, cmap=CMAP, vmin=vmin_b, vmax=vmax_b,
                                  edgecolors='#1a1a1a', linewidths=0.4, ax=ax_before)
            cbar_b = fig_before.colorbar(nodes_b, ax=ax_before, fraction=0.046, pad=0.04)
            cbar_b.set_label('Node degree')
            ax_before.set_title(f'Original Network\nNodes: {G_before.number_of_nodes()}, Edges: {G_before.number_of_edges()}')
        else:
            ax_before.text(0.5, 0.5, 'No data to display', ha='center', va='center')
        ax_before.axis('off')

        buffer_before = io.BytesIO()
        plt.savefig(buffer_before, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer_before.seek(0)
        image_before_base64 = base64.b64encode(buffer_before.getvalue()).decode()
        plt.close(fig_before)

        fig_after, ax_after = plt.subplots(1, 1, figsize=(8, 6))
        if G_after.number_of_nodes() > 0:
            pos_after = nx.spring_layout(G_after, k=1, iterations=30)
            degrees_after = dict(G_after.degree())
            node_colors_after = list(degrees_after.values())
            vmin_a, vmax_a = min(node_colors_after), max(node_colors_after)
            if vmin_a == vmax_a:
                vmax_a = vmin_a + 1

            nx.draw_networkx_edges(G_after, pos_after, alpha=0.35, edge_color='#4a4a4a', width=0.8, ax=ax_after)
            nodes_a = nx.draw_networkx_nodes(G_after, pos_after, node_color=node_colors_after,
                                  node_size=70, cmap=CMAP, vmin=vmin_a, vmax=vmax_a,
                                  edgecolors='#1a1a1a', linewidths=0.4, ax=ax_after)
            cbar_a = fig_after.colorbar(nodes_a, ax=ax_after, fraction=0.046, pad=0.04)
            cbar_a.set_label('Node degree')
            ax_after.set_title(f'Filtered Network\nNodes: {G_after.number_of_nodes()}, Edges: {G_after.number_of_edges()}')
        else:
            ax_after.text(0.5, 0.5, 'No data to display', ha='center', va='center')
        ax_after.axis('off')

        buffer_after = io.BytesIO()
        plt.savefig(buffer_after, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer_after.seek(0)
        image_after_base64 = base64.b64encode(buffer_after.getvalue()).decode()
        plt.close(fig_after)

        reduction = 0
        if G_before.number_of_edges() > 0:
            reduction = (G_before.number_of_edges() - G_after.number_of_edges()) / G_before.number_of_edges() * 100

        return jsonify({
            'success': True,
            'image_before': image_before_base64,
            'image_after': image_after_base64,
            'stats_before': {
                'nodes': G_before.number_of_nodes(),
                'edges': G_before.number_of_edges(),
                'density': nx.density(G_before) if G_before.number_of_nodes() > 0 else 0
            },
            'stats_after': {
                'nodes': G_after.number_of_nodes(),
                'edges': G_after.number_of_edges(),
                'density': nx.density(G_after) if G_after.number_of_nodes() > 0 else 0
            },
            'reduction': reduction
        })

    except Exception as e:
        app.logger.error(f"Comparison error: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/download', methods=['GET'])
def download_file():
    filename = request.args.get('file')
    if not filename:
        return jsonify({'success': False, 'message': 'No filename provided'}), 400

    filename = os.path.basename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': 'File not found'}), 404

    return send_file(filepath, as_attachment=True, download_name=filename)

ENV = os.environ.get('FLASK_ENV', 'production')

@app.route('/')
def serve_index():
    """Serve the frontend index page"""
    return send_file(os.path.join(frontend_path, 'login.html'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(debug=DEBUG, host='0.0.0.0', port=port)