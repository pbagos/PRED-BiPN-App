"""Modularity vitality backbone (Rajeh et al. 2022, Algorithm 1): the closed-form
vitality on the original network equals the brute-force Q(G) - Q(G - v), and the
backbone is the subgraph of the nodes with the largest |vitality|."""
import math

import networkx as nx
from community import community_louvain
from networkx.algorithms.community import modularity

from modularity_filter import SparseModularityBackbone

G = nx.les_miserables_graph()   # weighted
mb = SparseModularityBackbone(G, seed=42)
vit = mb.node_vitalities()

# brute force, with the same partition of the original network
part = community_louvain.best_partition(G, weight='weight', random_state=42)
groups = {}
for v, c in part.items():
    groups.setdefault(c, set()).add(v)
Q = modularity(G, list(groups.values()), weight='weight')
brute = {}
for v in G:
    H = G.copy(); H.remove_node(v)
    brute[v] = Q - modularity(H, [g - {v} for g in groups.values() if g - {v}], weight='weight')
    assert abs(vit[v] - brute[v]) < 1e-9, v
print('modularity vitality OK')

keep = math.ceil(0.3 * G.number_of_nodes())
top = sorted(sorted(G, key=str), key=lambda v: -abs(brute[v]))[:keep]
assert set(SparseModularityBackbone(G, seed=42).extract_backbone(0.3).nodes()) == set(top)
B = SparseModularityBackbone(G, seed=42).extract_backbone(0.3, keep_lcc=True)
assert nx.is_connected(B) and B.number_of_nodes() <= keep
print('modularity backbone OK')
