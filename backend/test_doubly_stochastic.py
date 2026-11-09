"""Doubly stochastic backbone (Slater 2009): the scaled matrix W + epsilon is doubly
stochastic, and the backbone is the shortest prefix of the edges, in decreasing
scaled value, with the same (strongly) connected components as the input."""
import os
import tempfile

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from doubly_stochastic import EPSILON, TrueMultiscaleBackbone

G = nx.les_miserables_graph()   # weighted, undirected, connected
d = tempfile.mkdtemp()
src, out = os.path.join(d, 'in.tsv'), os.path.join(d, 'out.tsv')

# 1. Sinkhorn: every row and column of diag(r) (W + eps) diag(c) sums to 1
W = csr_matrix(nx.to_scipy_sparse_array(G, weight='weight'))
S = TrueMultiscaleBackbone._make_doubly_stochastic(W)
dense = W.toarray() + EPSILON   # the same scaling, done densely
rs, cs = np.ones(len(dense)), np.ones(len(dense))
for _ in range(1000):
    rs = 1 / (dense @ cs); cs = 1 / (dense.T @ rs)
M = rs[:, None] * dense * cs[None, :]
assert np.allclose(M.sum(0), 1) and np.allclose(M.sum(1), 1)
assert np.allclose(S.toarray()[W.toarray() > 0], M[W.toarray() > 0])
print('sinkhorn OK')


def check(graph, directed):
    pd.DataFrame([(u, v, w['weight']) for u, v, w in graph.edges(data=True)],
                 columns=['source', 'target', 'weight']).to_csv(src, sep='\t', index=False)
    TrueMultiscaleBackbone(directed=directed).extract_backbone(src, out)
    b = pd.read_csv(out, sep='\t')
    comps = nx.number_strongly_connected_components if directed else nx.number_connected_components
    H = (nx.DiGraph if directed else nx.Graph)(); H.add_nodes_from(graph)
    H.add_edges_from(zip(b.source, b.target))
    assert comps(H) == comps(graph)                      # same components as the input
    H.remove_edge(b.source.iloc[-1], b.target.iloc[-1])  # one edge fewer is not enough
    assert comps(H) > comps(graph)
    assert b.ds_value.is_monotonic_decreasing
    return len(b)


n_und = check(G, directed=False)
D = nx.DiGraph()
rng = np.random.default_rng(0)
for u, v, w in G.edges(data=True):   # a directed version: both directions, different weights
    D.add_edge(u, v, weight=w['weight']); D.add_edge(v, u, weight=w['weight'] * rng.uniform(0.5, 2))
n_dir = check(D, directed=True)
print(f'slater OK ({n_und} undirected, {n_dir} directed edges of {G.number_of_edges()} / {D.number_of_edges()})')
