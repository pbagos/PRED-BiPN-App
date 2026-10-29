"""
Edge-betweenness backbone for weighted undirected networks (PRED-BiPN).

The edge betweenness of an edge e is the number of shortest paths between
pairs of nodes that pass through e (Girvan & Newman 2002; computed with the
algorithm of Brandes 2001). Edges with high betweenness carry a large share
of the shortest paths in the network.

Backbone rule
-------------
Edge weights are similarities (larger = stronger), so they are turned into
lengths d_ij = 1 / w_ij before shortest paths are computed. Betweenness is
computed ONCE on the whole network, and an edge is kept if

    EB(e) >= factor * mean(EB)

With the default factor = 1 the backbone contains the edges that carry more
shortest paths than the average edge. Larger factors give sparser backbones.

Differences from the previous Girvan-Newman module (edge_betweenness.py)
----------------------------------------------------------------------
* the whole input file is used (no max_lines cap);
* edge weights are used (weighted=True by default);
* betweenness is computed once instead of being recomputed after every
  removal, so the method scales to large networks;
* the output is a backbone (the high-betweenness edges), not a community
  partition.

Speed
-----
If python-igraph is installed (pip install igraph) the computation runs in C
(about 12 s for 3,580 nodes / 140,899 edges on a 2-core machine). Otherwise
NetworkX is used, which gives the same backbone but is much slower on large
networks. For very large networks, sample_size > 0 estimates betweenness
from a random sample of source nodes (rescaled by n / sample_size).

Input : TSV/CSV with three columns (source, target, weight); header optional.
Output: TSV with columns source, target, weight, edge_betweenness.
"""
import logging
import time

import numpy as np
import pandas as pd

from input_format import detect_sep, has_header_row

try:
    import igraph as ig
    HAVE_IGRAPH = True
except ImportError:  # fall back to NetworkX
    HAVE_IGRAPH = False


def _read_edges(input_file, logger):
    sep = detect_sep(input_file)
    df = pd.read_csv(input_file, sep=sep, engine="python", header=None, dtype=str,
                     skiprows=int(has_header_row(input_file, sep)), keep_default_na=False)
    if df.shape[1] < 2:
        raise ValueError("The input file must have at least two columns (source, target[, weight]).")
    if df.shape[1] == 2:
        df[2] = "1"
    df = df.iloc[:, :3]
    df.columns = ["source", "target", "weight"]
    df["source"] = df["source"].str.strip()
    df["target"] = df["target"].str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df[(df.source != "") & (df.target != "") & (df.source != df.target) & df.weight.notna()]
    # undirected: A-B and B-A are the same edge
    a = np.where(df.source < df.target, df.source, df.target)
    b = np.where(df.source < df.target, df.target, df.source)
    df = pd.DataFrame({"source": a, "target": b, "weight": df.weight.values})
    df = df.groupby(["source", "target"], as_index=False)["weight"].max()
    logger.info(f"Loaded {len(df)} undirected edges")
    return df


def _betweenness_igraph(df, lengths, sample_size, seed):
    nodes = pd.unique(pd.concat([df.source, df.target], ignore_index=True))
    index = {v: i for i, v in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=list(zip(df.source.map(index), df.target.map(index))), directed=False)
    sources = None
    if sample_size and sample_size < len(nodes):
        rng = np.random.default_rng(seed)
        sources = rng.choice(len(nodes), size=sample_size, replace=False).tolist()
    eb = np.asarray(g.edge_betweenness(directed=False, weights=lengths, sources=sources), dtype=float)
    if sources is not None:
        eb *= len(nodes) / sample_size
    return eb


def _betweenness_networkx(df, lengths, sample_size, seed):
    import networkx as nx
    G = nx.Graph()
    for (s, t), d in zip(zip(df.source, df.target), (lengths if lengths is not None else [1.0] * len(df))):
        G.add_edge(s, t, length=d)
    k = sample_size if sample_size and sample_size < G.number_of_nodes() else None
    eb = nx.edge_betweenness_centrality(G, k=k, normalized=False, weight="length", seed=seed)
    vals = [eb.get((s, t), eb.get((t, s))) for s, t in zip(df.source, df.target)]
    return np.asarray(vals, dtype=float)


def run_edge_betweenness_backbone(input_file, output_file, factor=1.0, weighted=True,
                                  sample_size=0, seed=42, logger=None):
    """Extract the edge-betweenness backbone and write it to output_file.

    Returns a dict with summary statistics (success, nodes, edges, kept edges, ...).
    """
    logger = logger or logging.getLogger(__name__)
    t0 = time.time()
    try:
        df = _read_edges(input_file, logger)
        if weighted:
            bad = (df.weight <= 0).sum()
            if bad:
                logger.warning(f"{bad} edges with non-positive weight were removed (weighted mode needs w > 0)")
                df = df[df.weight > 0].reset_index(drop=True)
            lengths = (1.0 / df.weight.values).tolist()
        else:
            lengths = None
        if df.empty:
            raise ValueError("No valid edges in the input file.")

        backend = "igraph" if HAVE_IGRAPH else "networkx"
        logger.info(f"Computing edge betweenness with {backend} (weighted={weighted}, sample_size={sample_size or 'exact'})")
        if HAVE_IGRAPH:
            eb = _betweenness_igraph(df, lengths, sample_size, seed)
        else:
            eb = _betweenness_networkx(df, lengths, sample_size, seed)

        mean_eb = float(eb.mean())
        threshold = factor * mean_eb
        keep = eb >= threshold
        out = df.loc[keep, ["source", "target", "weight"]].copy()
        out["edge_betweenness"] = eb[keep]
        out = out.sort_values("edge_betweenness", ascending=False)
        out.to_csv(output_file, sep="\t", index=False)

        n_nodes = len(pd.unique(pd.concat([df.source, df.target], ignore_index=True)))
        stats = {
            "success": True,
            "method": "edge_betweenness_backbone",
            "backend": backend,
            "weighted": bool(weighted),
            "factor": float(factor),
            "sample_size": int(sample_size or 0),
            "nodes": int(n_nodes),
            "edges_input": int(len(df)),
            "edges_backbone": int(keep.sum()),
            "retained_percentage": float(100.0 * keep.mean()),
            "mean_betweenness": mean_eb,
            "threshold": float(threshold),
            "seconds": round(time.time() - t0, 2),
        }
        logger.info(f"Kept {stats['edges_backbone']} of {stats['edges_input']} edges "
                    f"({stats['retained_percentage']:.1f}%) in {stats['seconds']} s")
        return stats
    except Exception as e:  # reported back to the web interface
        logger.exception("Edge betweenness backbone failed")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Edge-betweenness backbone (PRED-BiPN)")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--factor", type=float, default=1.0, help="keep edges with EB >= factor x mean EB (default 1)")
    p.add_argument("--unweighted", action="store_true", help="ignore weights (all edge lengths = 1)")
    p.add_argument("--sample-size", type=int, default=0, help="number of sampled source nodes (0 = exact)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(run_edge_betweenness_backbone(args.input, args.output, factor=args.factor,
                                        weighted=not args.unweighted, sample_size=args.sample_size))
