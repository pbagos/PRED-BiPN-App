"""Closed-form modularity vitality == brute-force recomputation of Q."""
import networkx as nx
import numpy as np
from scipy import sparse
from networkx.algorithms.community import modularity

from modularity_filter import SparseModularityBackbone

G = nx.les_miserables_graph()
mb = SparseModularityBackbone(G)
part = nx.community.louvain_communities(G, weight='weight', seed=1)
cid = {n: c for c, nodes in enumerate(part) for n in nodes}
comm = np.array([cid[n] for n in mb.nodes])
n, C = mb.n_nodes, len(part)
K = np.asarray((mb.adj @ sparse.csr_matrix((np.ones(n), (np.arange(n), comm)), shape=(n, C))).todense())
k = K.sum(axis=1)
tot = np.bincount(comm, weights=k, minlength=C)
S_in = K[np.arange(n), comm].sum() / 2
vit = SparseModularityBackbone.vitality(K, k, tot, comm, S_in, k.sum() / 2)

Q = modularity(G, part, weight='weight')
for i, node in enumerate(mb.nodes):
    H = G.copy(); H.remove_node(node)
    sub = [c - {node} for c in part if c - {node}]
    assert abs(vit[i] - (Q - modularity(H, sub, weight='weight'))) < 1e-9, node
print('modularity vitality OK')

# the incremental removal loop picks the same nodes as brute-force greedy
from community import community_louvain
p = community_louvain.best_partition(G, weight='weight', random_state=42)
H, nodes = G.copy(), sorted(G.nodes(), key=str)
while H.number_of_nodes() > int(G.number_of_nodes() * 0.3):
    groups = {}
    for v in H:
        groups.setdefault(p[v], set()).add(v)
    q = modularity(H, list(groups.values()), weight='weight')
    best = None
    for v in nodes:
        if v not in H:
            continue
        H2 = H.copy(); H2.remove_node(v)
        sub = [g - {v} for g in groups.values() if g - {v}]
        d = abs(q - modularity(H2, sub, weight='weight')) if H2.number_of_edges() else 0.0
        if best is None or d < best[0] - 1e-12:
            best = (d, v)
    H.remove_node(best[1])
assert set(SparseModularityBackbone(G, seed=42).extract_backbone(0.3).nodes()) == set(H.nodes())
print('modularity backbone OK')
