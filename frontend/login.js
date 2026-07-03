// ============================
// Global Variables
// ============================
// Same-origin: the Flask backend serves this page, so relative URLs work both
// locally (localhost:5050) and behind a proxy (Render/Heroku on 443).
const API_BASE = '';

const ALPHA_METHODS = [
    'disparity filter',
    'doubly stochastic filter',
    'locally adaptive network sparsification filter',
    'marginal likelihood filter'
];

// Per-method tunable parameters surfaced in the UI. Keys match what the backend
// reads via data.get(...), so no backend change is needed — the values just get
// sent in the /process request. Methods absent here use backend defaults.
const METHOD_PARAMS = {
    'cosine similarity weighted projection':            [{ key: 'threshold', label: 'Similarity threshold (0–1)', value: 0.5, min: 0, max: 1, step: 0.01 }],
    'adamic adar similarity weighted projection':       [{ key: 'threshold', label: 'Similarity threshold', value: 0.5, min: 0, step: 0.01 }],
    'resource allocation similarity weighted projection': [{ key: 'threshold', label: 'Similarity threshold', value: 0.5, min: 0, step: 0.01 }],
    'jaccard similarity weighted projection':           [{ key: 'threshold', label: 'Jaccard threshold (0–1)', value: 0.0, min: 0, max: 1, step: 0.01 }],
    'kendall correlation weighted projection':          [{ key: 'threshold', label: 'Correlation threshold (0–1)', value: 0.0, min: 0, max: 1, step: 0.01 }],
    'spearman correlation weighted projection':         [{ key: 'threshold', label: 'Correlation threshold (0–1)', value: 0.0, min: 0, max: 1, step: 0.01 }],
    'pearson correlation weighted projection':          [{ key: 'threshold', label: 'p-value threshold', value: 0.05, min: 0, max: 1, step: 0.001 }],
    'odds ratio weighted projection': [
        { key: 'p_threshold',  label: 'p-value threshold', value: 0.05, min: 0, max: 1, step: 0.001 },
        { key: 'or_threshold', label: 'Odds ratio threshold', value: 1.0, min: 0, step: 0.1 }
    ],
    'high salience skeleton':  [{ key: 'threshold', label: 'Salience threshold (0–1)', value: 0.9, min: 0, max: 1, step: 0.01 }],
    'hierarchical backbone': [
        { key: 'z_threshold',     label: 'Z-score threshold', value: 1.64, step: 0.01 },
        { key: 'alpha_percentile', label: 'Alpha percentile (1–100)', value: 95, min: 1, max: 100, step: 1 }
    ]
};

function renderDynamicParams(method) {
    const panel = document.getElementById('dynamicParamsPanel');
    if (!panel) return;
    const specs = METHOD_PARAMS[(method || '').toLowerCase()];
    if (!specs) {
        panel.style.display = 'none';
        panel.innerHTML = '';
        return;
    }
    panel.innerHTML = specs.map(s =>
        `<label class="param-label" for="dp_${s.key}">${s.label}:</label>` +
        `<input type="number" id="dp_${s.key}" class="param-input" value="${s.value}"` +
        `${s.min != null ? ` min="${s.min}"` : ''}${s.max != null ? ` max="${s.max}"` : ''} step="${s.step || 'any'}" />`
    ).join('');
    panel.style.display = 'block';
}

function collectDynamicParams(method) {
    const specs = METHOD_PARAMS[(method || '').toLowerCase()];
    const out = {};
    if (!specs) return out;
    for (const s of specs) {
        const el = document.getElementById('dp_' + s.key);
        if (el && el.value !== '' && !isNaN(parseFloat(el.value))) {
            out[s.key] = parseFloat(el.value);
        }
    }
    return out;
}

let latestTaskId = null;
let selectedMethod = null;
let lastUploadedFormData = null;
let currentMethod = null;
let currentResultFile = null;

const methodDescriptions = {
    "Disparity Filter": `
        <p>
        <strong>Disparity Filter</strong> aims to extract the <em>multiscale backbone</em> of a weighted bipartite network by retaining edges whose weights cannot be explained by random fluctuations.
        The method applies a <strong>local statistical comparison</strong> at each node, evaluating edge weights against a null model based on a Beta distribution. Edges with p-values lower 
        than the significance level α are preserved, ensuring that important connections—especially those involving weak nodes—are retained.
        </p>
        <p>
        This method needs a file (.csv or .tsv) with three columns (Source Target Weight) representing a weighted one mode projected undirected network. If your data is bipartite, with 2 columns, you can use the Simple
        Weighted Projection method first to create a file with 3 columns.
        </p>
        <br>
        <p>
        <em>Reference:</em> Serrano, M. Á., Boguñá, M., & Vespignani, A. (2009).
        </br>
        <small>PNAS, 106(16), 6483–6488</small>
        </p>
    `,
    "Doubly Stochastic Filter": `
        <p>
        This method is a two-stage algorithm used to extract the backbone of complex weighted and directed networks. 
        Its main goal is to simplify a dense network while preserving its most important and structurally meaningful connections.
        In the first stage, the method normalizes the original flow matrix through an iterative row and column scaling process until it becomes doubly stochastic, meaning that all row and column sums equal one. This normalization removes the influence of node size and total flow, allowing each edge weight to represent the relative importance of the connection rather than absolute volume.
        In the second stage, the network is reconstructed progressively. Starting from an empty graph containing all nodes, edges are added one by one in descending order of their normalized weights. The process continues until the graph becomes strongly connected, meaning that every node can reach every other node through directed paths. The set of edges required at this exact point defines the network backbone, representing the minimal set of connections necessary to maintain full connectivity.
        Overall, the method reveals the core structure and dominant flow pathways of the network, reduces analytical complexity, and ensures that all nodes are retained without applying arbitrary significance thresholds.
        </p>

        <br>
        <p>
        This method can be excecuted on a file (.csv or .tsv) with three columns (Source Target Weight) representing a weighted one mode projected undirected network. If your data is bipartite, with 2 columns, you can use the Simple Weighted Projection method first to create a file with 3 columns.
        </p>
        </br>
        <p>
        Reference: Slater, P. B. (2009). Proceedings of the National Academy of Sciences, 106(26). https://doi.org/10.1073/pnas.0904725106  
        </p>
    `,
    "Hierarchical Backbone": `
       <p>
        Hierarchical Backbone Extraction

       The method builds a hierarchical backbone by comparing observed node co-occurrences to a random bipartite model. 
       Significant and asymmetric associations are interpreted as hierarchical relationships,
       where highly general nodes dominate more specific ones, resulting in a directed acyclic graph.
       </p>
       
       <br>
       <p>
       To use this method, provide a file (.csv or .tsv) with two columns representing the bipartite network.
       </p>
       </br>
       <p>
       Reference: Jo, W. S., Park, J., Luhur, A., Kim, B. J., & Ahn, Y. (2020). arXiv.org. https://arxiv.org/abs/2002.07239
       </p>

    `,
    "High Salience Skeleton": `
        <p>
        High Salience Skeleton method extracts the most salient connections in a network based on their importance
        in maintaining the network's overall connectivity and information flow. It is a structure-based approach
        that identifies edges that frequently appear in shortest path trees rooted at each node.
        </p>
        <p>
        The method calculates the salience of each edge by counting how many times it appears in the set of shortest path trees.
        Edges with high salience values are considered crucial for the network's connectivity and are retained in the backbone,
        while those with low salience are pruned away. This results in a simplified network that preserves essential pathways.
        </p>
        <p>
        Overall, the High Salience Skeleton method provides a way to distill complex networks into their most important structures,
        facilitating analysis and visualization while retaining key functional properties.
        </p>
        <br>
        <p>
        This method needs a file (.csv or .tsv) with three columns (Source Target Weight) representing a weighted one mode projected undirected network. If your data is bipartite, with 2 columns, you can use the Simple
        Weighted Projection method first to create a file with 3 columns.
        </p>
        </br>
        <p>
        <em>Reference:</em> Grady, D., Tiemann, C. & Brockmann, D. (2012). Robust classification of salient links in complex networks. 
        <br>
        <small>Nature Communications, 3, 864. https://doi.org/10.1038/ncomms1847</small>
        </p>
        \
    `,
    "Hypergraph Projection": `
        <p>
        Hypergraphs are a generalization of traditional graphs, where hyperedges can connect more than two nodes, allowing the representation of higher-order interactions in complex systems. This makes them ideal for modeling social interactions, biological processes, or collaborative networks, where relationships are often group-based rather than pairwise. Compressing these higher-order interactions into standard pairwise edges can lead to loss of critical information.
        Hypergraphs provide a framework for advanced machine learning tasks, such as community detection (clustering), classification, and node embedding, often improving results when additional node attributes are incorporated.
        In practice, a bipartite network can be represented as a hypergraph, with nodes of one set forming hyperedges connecting nodes of the other set. Hypergraph techniques can then be applied to extract a one-mode projection, highlighting clusters or communities of related nodes. Visualizations often use circular layouts with nodes positioned around community centers, providing a clear representation of hypergraph structure [Young et al., 2021].
       </p>
        <br>
       <p>
       To use this method, provide a file (.csv or .tsv) with two columns representing the bipartite network.
       </p>
       </br>
       <p>
       Reference: Young, J., Petri, G., & Peixoto, T. P. (2021). Communications Physics, 4(1). https://doi.org/10.1038/s42005-021-00637-w
       </p>
    `,

    "Locally Adaptive Network Sparsification Filter": `
        The Locally Adaptive Network Sparsification (LANS) Filter is a nonparametric backbone extraction method that does not assume any prior distribution of edge weights. Instead, it uses the empirical cumulative distribution function (ECDF) of a node’s outgoing edge weights to evaluate the statistical significance of each edge.
        For each edge from node i to neighbor j with weight w_ij, the fractional weight is computed as:
        p_ij = w_ij / sum of weights of all edges from node i
        where the sum includes all neighbors of node i. The empirical CDF of p_ij is the fraction of edges from node i that have fractional weight less than or equal to p_ij.
        An edge is considered locally significant if:1 - ECDF(p_ij) < alpha
        where alpha is a predefined significance level. Significant edges are retained in the network backbone, while non-significant ones are removed.
        Because LANS evaluates the empirical distribution for each node individually, it adapts to local heterogeneity and is fully parallelizable, making it suitable for sparse and large-scale networks. Unlike disparity filters that rely on parametric assumptions, LANS preserves the true local structure and avoids bias from incorrectly modeled weight distributions [Foti et al., 2011].
        <br>
        <p>
        This method needs a file (.csv or .tsv) with three columns (Source Target Weight) representing a weighted undirected network. If your data is bipartite, with 2 columns, you can use the Simple
        Weighted Projection method first to create a file with 3 columns.
        </p>
        </br>
    `,
    "Marginal Likelihood Filter": `
        <p>
        Uses marginal likelihood estimation to determine the statistical significance of edges.
        Preserves edges that are statistically unlikely to occur by chance.
        </p>
        <br>
        <p>
        This method needs a file (.csv or .tsv) with three columns (Source Target Weight) representing a weighted one mode projected undirected network. If your data is bipartite, with 2 columns, you can use the Simple
        Weighted Projection method first to create a file with 3 columns.
        </p>
        </br>
        <p>
        <em>Reference:</em> Dianati, N. (2016). Unwinding the hairball graph: Pruning algorithms for weighted complex networks. 
        <br>
        <small>Physical Review E, 93, 012304. https://doi.org/10.1103/PhysRevE.93.012304</small>
        </p>
    `,
    "Maximum Spanning Tree Filter": `
        <p>
        The Maximal Spanning Tree (MST) is an optimization technique that identifies a subset of edges in a graph connecting all nodes with the maximum possible total weight while avoiding cycles. Its goal is to highlight the strongest interactions in the network while maintaining overall connectivity.
        Using MST allows focusing on the most important pairs in the network, effectively extracting the network backbone. The method typically relies on Kruskal’s algorithm:
        Sort all edges by descending weight.Initialize an empty tree 𝑇=∅. Iteratively add the next heaviest edge if it does not form a cycle. Stop when 𝑇 contains exactly ∣𝑉∣−1∣V∣−1 edges.
        The resulting MST preserves the strongest connections without introducing loops, producing a clear and simplified structure of the network [Camerini, 1978].
        </p>
        <p>
       <br>
        To use this method, provide a file (.csv or .tsv) with two columns representing the weighted one mode bipartite network.
       </p>
       </br>

        <p>
        Reference: Camerini, P. (1978). Information Processing Letters, 7(1), 10–14. https://doi.org/10.1016/0020-0190(78)90030-3
        </p>
    `,
    "Modularity Filter": `
        <p>
        This method is a network backbone extraction technique that preserves the community structure of a network. It reduces the network size by removing nodes that contribute least to the overall modularity. Modularity is a measure of the quality of community structure, comparing the density of connections within communities to that expected in a random network.
        Unlike simple methods based on edge weights (like disparity filter) or node degree, this method prioritizes the preservation of community structure. Nodes that are critical for community cohesion are considered the most important. Instead of removing edges, the method removes entire nodes.
        The contribution of each node to modularity (called modularity vitality) is calculated. This allows identifying and keeping important nodes that act as "bridges" between communities.
        The modularity Q of the whole network G is computed as:Q = (1 / (2 * |E|)) * sum over i,j of [A_ij - (k_i * k_j) / (2 * |E|)] * delta(c_i, c_j)
        where: A_ij is the adjacency matrix element between nodes i and j. k_i is the degree of node i (sum of its edge weights for weighted networks).|E| is the total number of edges (or total weight for weighted networks).
        c_i and c_j are the communities of nodes i and j. delta(c_i, c_j) = 1 if i and j are in the same community, 0 otherwise. For each node v, vitality is computed as: alpha(v) = Q(G) - Q(G \ v) where G \ v is the network without node v. Nodes are ranked by the absolute value of vitality. Both nodes that increase modularity (hubs) and those that decrease it (bridges) are considered equally. Nodes with the lowest vitality are removed iteratively until the network reaches the desired size. The remaining nodes form the backbone of the network.
        </p>
        <br>
        <p>
        This method needs a file (.csv or .tsv) with three columns (Source Target Weight) representing a weighted one mode projected undirected network. If your data is bipartite, with 2 columns, you can use the Simple
        Weighted Projection method first to create a file with 3 columns.
        </p>
        </br>
        <br>
        <p>
        Reference: Rajeh, S., Savonnet, M., Leclercq, E., & Cherifi, H. (2022). Modularity-based backbone extraction in weighted complex networks. arXiv.org. https://arxiv.org/abs/2201.12905
       </br>
        </p>
    `,
    "Noise Corrected Filter": `
    <p>
    The Noise-Corrected (NC) Filter is a statistical backbone extraction method that uses a Bayesian framework to evaluate the significance of edge weights in a network.
    It computes a transformed Lift score for each edge, measuring how unexpectedly strong the connection is compared to a null model. Edges are retained only if their score exceeds a threshold based on the estimated variance, filtering out noisy or insignificant connections while preserving important relationships.
    </p>
    <p>
    The method adapts to local heterogeneity, avoids zero-variance issues, and is suitable for weighted networks where standard parametric assumptions may not hold.
    Reference: Coscia, M., Giannotti, F., & Pedreschi, D. (2017). Network Backboning with Noisy Data. IEEE.
    </p>

    `,

    "Odds Ratio Weighted Projection": `
    <p>
    The odds ratio (OR) measures the association between an exposure and an outcome, calculated from a 2×2 contingency table as OR = (a*d)/(b*c). 
    Weighted projections can use metrics like the number of common neighbors as edge weights to analyze network structure.
    </p>
    <p>
    This method can be excecuted on a file (.csv or .tsv) with three columns (Source Target Weight) representing a weighted one mode projected undirected network. If your data is bipartite, with 2 columns, you can use the Simple Weighted Projection method first to create a file with 3 columns.
    </p>
    <p>
    Reference: Yanagawa, T., Fujii, Y., & Matsuoka, J. (1994). Generalized Mantel-Haenszel procedures for 2 × J tables. Environmental Health Perspectives, 102(suppl 8), 57–60. https://doi.org/10.1289/ehp.94102s857
    </p>

    `,
    "Polya Filter": `
        <p>
        The Pólya Filter is a statistical network filtering technique used to extract the most significant connections (backbone) of a network while ignoring weak or random links. Its goal is to distinguish links that are truly strong from those that appear due to network heterogeneity.

        The method is inspired by the Pólya urn model, a process with self-reinforcement: each time a “ball” (representing a link) is drawn, it is returned to the urn along with an additional ball of the same type. This means that frequently occurring links become more likely to appear again, reflecting the idea that strong nodes are more likely to form new strong connections.

        For each link in the network, the method calculates the probability of observing its weight (how often it appears) given the node’s degree (number of connections) and strength (sum of weights of connections). This probability is then converted into a p-value, measuring how unusual the link is compared to the expected random behavior.

        A multiple testing correction (e.g., Benjamini-Hochberg FDR) is applied to account for all links simultaneously. Links with statistically significant p-values are retained as the network backbone, representing the strongest and most informative connections [Marcaccioli et al., 2019].

        In essence, the Pólya Filter:

        Accounts for the heterogeneity of nodes in the network.

        Identifies statistically significant and reinforced connections.

        Produces a filtered network backbone highlighting the most relevant links.
        To use this method, provide a file (.csv or .tsv) with three columns (Source Target Weight) representing a weighted one mode projected undirected network. If your data is bipartite, with 2 columns, you can use the Simple Weighted Projection method first to create a file with 3 columns.
        </p>

        <p>
        
        </p>

        <p>
       Reference: Marcaccioli, R., Livan, G., Marcaccioli, R., & Livan, G. (2019). Nature Communications, 10(1), 745. https://doi.org/10.1038/s41467-019-08667-3
        </p>
    `,
     
    "Cosine Similarity Weighted Projection": `
    <p>
    <strong>Cosine similarity</strong> measures the similarity between two non-zero vectors by calculating the cosine of the angle between them. It determines whether two vectors point in roughly the same direction, regardless of their magnitude (Lahitani et al., 2016).
    </p>
    
    <p style="text-align: center; font-size: 1.1em;">
    <strong>cosine(θ) = (A · B) / (∥A∥ · ∥B∥)</strong>
    </p>
    
    <p>
    In gene-disease networks, each gene is represented as a binary vector across all diseases. The value indicates the proportion of shared diseases between gene pairs, ranging from 0 (no common diseases) to 1 (identical disease associations).
    </p>
    
    <p>
    <strong>Input format:</strong> Tab-separated file with:
    <ul>
        <li><strong>Bipartite (2 columns):</strong> Gene \t Disease</li>
        <li><strong>Weighted (3+ columns):</strong> Node1 \t Node2 \t Weight</li>
    </ul>
    The method automatically detects the format and header rows.
    </p>
    
    <p>
    <strong>Output:</strong> CSV file with gene pairs and their cosine similarity scores above the user-defined threshold.
    </p>
    
    <br>
    <p>
    <em>Reference:</em> Lahitani, A. R., et al. (2016). Cosine similarity to determine similarity measure. <em>ISemantic</em>, 1–6.
    <br>
    <small>DOI: 10.1109/citsm.2016.7577578</small>
    </p>
    `,
    "Adamic Adar Similarity Weighted Projection": `
    <p>
    <strong>Adamic-Adar index</strong> measures the similarity between two nodes by considering their common neighbors, giving greater weight to neighbors that are less connected (rarer). It was originally designed to predict connections in social networks (Adamic & Adar, 2003).
    </p>
    
    <p style="text-align: center; font-size: 1.1em;">
    <strong>s(u,v) = Σ 1 / log(|Γ(z)|)</strong>
    <br>
    <small>for z ∈ [Γ(u) ∩ Γ(v)]</small>
    </p>
    
    <p>
    where:
    <ul>
        <li><strong>Γ(u), Γ(v):</strong> Sets of neighbors of nodes u and v</li>
        <li><strong>Γ(u) ∩ Γ(v):</strong> Intersection of neighbors (common neighbors)</li>
        <li><strong>|Γ(z)|:</strong> Degree of common neighbor z</li>
    </ul>
    </p>
    
    <p>
    The expression <strong>1/log(|Γ(z)|)</strong> ensures that:
    <ul>
        <li>If |Γ(z)| is large (common/hub neighbor), contribution is small</li>
        <li>If |Γ(z)| is small (rare neighbor), contribution is large</li>
    </ul>
    </p>
    
    <p>
    In gene-disease networks, this index identifies gene pairs that share rare disease associations, potentially indicating stronger functional relationships than those sharing common, well-studied diseases.
    </p>
    
    <p>
    <strong>Input format:</strong> Tab-separated bipartite file with 2 columns: Gene \\t Disease
    </p>
    
    <p>
    <strong>Output:</strong> CSV file with gene pairs and their Adamic-Adar similarity scores above the user-defined threshold.
    </p>
    
    <br>
    <p>
    <em>Reference:</em> Adamic, L. A., & Adar, E. (2003). Friends and neighbors on the Web. <em>Social Networks</em>, 25(3), 211–230.
    <br>
    <small>DOI: 10.1016/S0378-8733(03)00009-1</small>
    </p>
    `,

    "Resource Allocation Similarity Weighted Projection": `
    <p>
    <strong>Resource Allocation Index</strong> measures similarity between nodes based on resource distribution and the probability that one node can connect to another through common neighbors (Stram et al., 2017).
    </p>
    
    <p style="text-align: center; font-size: 1.1em;">
    <strong>s(u,v) = Σ 1/|Γ(z)|</strong>
    <br>
    <small>for z ∈ [Γ(u) ∩ Γ(v)]</small>
    </p>
    
    <p>
    where:
    <ul>
        <li><strong>Γ(u), Γ(v):</strong> Sets of neighbors of nodes u and v</li>
        <li><strong>Γ(u) ∩ Γ(v):</strong> Common neighbors between u and v</li>
        <li><strong>|Γ(z)|:</strong> Degree of common neighbor z</li>
    </ul>
    </p>
    
    <p>
    The index is based on the concept of resource allocation in networks: a common neighbor with few connections (low degree) contributes more to similarity, as it has more resources to allocate to the connection between u and v. Conversely, highly connected hubs contribute less, as their resources are distributed among many neighbors.
    </p>
    
    <p>
    In gene-disease networks, this method identifies gene pairs that share rare disease associations, potentially indicating stronger functional relationships. The approach is particularly useful for detecting meaningful connections in sparse bipartite graphs where traditional similarity measures may overlook significant but infrequent associations.
    </p>
    
    <p>
    <strong>Input format:</strong> Tab-separated bipartite file with 2 columns: Gene \\t Disease
    </p>
    
    <p>
    <strong>Output:</strong> CSV file with gene pairs and their Resource Allocation similarity scores above the user-defined threshold.
    </p>
    
    <br>
    <p>
    <em>Reference:</em> Stram, R., Reuss, P., & Althoff, K. D. (2017). Weighted one mode projection of a bipartite graph as a local similarity measure. In <em>International Conference on Case-Based Reasoning (ICCBR)</em>, 375–389.
    <br>
    <small>DOI: 10.1007/978-3-319-61030-6_26</small>
    </p>
    
    `,
      "Jaccard Similarity Weighted Projection": `
        <p>
        <strong>Jaccard Similarity</strong> measures the similarity between two finite sets (A and B). It is defined as the ratio of the number of common neighbors between nodes i and j to the number of unique neighbors, i.e., the size of the union of the two sets (Bag et al., 2019).
        </p>
        
        <p style="text-align: center; font-size: 1.1em;">
        <strong>w<sub>ij</sub> = |U(i) ∩ V(j)| / |U(i) ∪ V(j)|</strong>
        </p>
        
        <p>
        where:
        <ul>
            <li><strong>U(i), V(j):</strong> Sets of neighbors of nodes i and j</li>
            <li><strong>|U(i) ∩ V(j)|:</strong> Number of common neighbors (intersection)</li>
            <li><strong>|U(i) ∪ V(j)|:</strong> Number of unique neighbors (union)</li>
        </ul>
        </p>
        
        <p>
        The Jaccard index ranges from 0 to 1, where:
        <ul>
            <li><strong>0:</strong> The sets have no common elements (completely dissimilar)</li>
            <li><strong>1:</strong> The sets are identical (maximum similarity)</li>
        </ul>
        </p>
        
        <p>
        In gene-disease networks, this metric identifies gene pairs that share a high proportion of their disease associations. It is particularly useful for finding functionally related genes that may be involved in similar biological processes or pathways.
        </p>
        
        <p>
        <strong>Input format:</strong> Tab-separated bipartite file with 2 columns: Gene \\t Disease
        </p>
        
        <p>
        <strong>Output:</strong> TSV file with gene pairs and their Jaccard similarity scores above the user-defined threshold.
        </p>
        
        <br>
        <p>
        <em>Reference:</em> Bag, S., Kumar, S. K., & Tiwari, M. K. (2019). An efficient recommendation generation using relevant Jaccard similarity. <em>Information Sciences</em>, 483, 53–64.
        <br>
        <small>DOI: 10.1016/j.ins.2019.01.023</small>
        </p>
    `,
    "Simple Weighted Projection": `
        <p>
        Basic projection method that creates weighted unipartite networks from bipartite data.
        Edge weights are based on shared connections in the bipartite network.
        </p>
    `,
    "Kendall Correlation Weighted Projection":`
    <p>
    <strong>Kendall's Tau (τ) correlation coefficient</strong> is a non-parametric measure of association 
    that evaluates the strength and direction of the relationship between two ranked variables. Unlike 
    Pearson correlation which measures linear relationships, Kendall's Tau assesses ordinal associations 
    based on the concordance and discordance of paired observations (El-Hashash & Shiekh, 2022).
    </p>
    
    <p style="text-align: center; font-size: 1.1em;">
    <strong>τ = (C - D) / (n(n-1)/2) = 2(C - D) / n(n-1)</strong>
    </p>
    
    <p>
    where:
    <ul>
        <li><strong>C:</strong> Number of concordant pairs (pairs with the same ranking order)</li>
        <li><strong>D:</strong> Number of discordant pairs (pairs with different ranking order)</li>
        <li><strong>n:</strong> Sample size (total number of pairs)</li>
    </ul>
    </p>
    
    <p>
    A pair (x<sub>i</sub>, y<sub>i</sub>) and (x<sub>j</sub>, y<sub>j</sub>) is considered:
    <ul>
        <li><strong>Concordant (C):</strong> If x<sub>i</sub> > x<sub>j</sub> and y<sub>i</sub> > y<sub>j</sub>, 
            or x<sub>i</sub> < x<sub>j</sub> and y<sub>i</sub> < y<sub>j</sub></li>
        <li><strong>Discordant (D):</strong> If x<sub>i</sub> > x<sub>j</sub> and y<sub>i</sub> < y<sub>j</sub>, 
            or x<sub>i</sub> < x<sub>j</sub> and y<sub>i</sub> > y<sub>j</sub></li>
    </ul>
    </p>
    
    <p>
    The coefficient τ ranges from -1 to +1:
    <ul>
        <li><strong>τ = +1:</strong> Perfect agreement (all pairs are concordant)</li>
        <li><strong>τ = 0:</strong> No association (concordant and discordant pairs are balanced)</li>
        <li><strong>τ = -1:</strong> Perfect disagreement (all pairs are discordant)</li>
    </ul>
    </p>
    
    <p>
    <strong>Key advantages of Kendall's Tau:</strong>
    <ul>
        <li><strong>Non-parametric:</strong> Makes no assumptions about the distribution of the data</li>
        <li><strong>Robust to outliers:</strong> Less sensitive to extreme values compared to Pearson correlation</li>
        <li><strong>Interpretation:</strong> Directly represents the probability of concordance minus discordance</li>
        <li><strong>Handles ties:</strong> Appropriate modifications exist for tied observations</li>
        <li><strong>Small sample reliability:</strong> More reliable than Spearman for small sample sizes</li>
    </ul>
    </p>
    
    <p>
    In gene-disease networks, Kendall's Tau is particularly useful for:
    <ul>
        <li>Identifying genes with similar disease association patterns across multiple conditions</li>
        <li>Detecting monotonic relationships between gene expression profiles</li>
        <li>Comparing ranked lists of disease associations without assuming linearity</li>
        <li>Providing robust correlation estimates in the presence of outliers or skewed data</li>
    </ul>
    </p>
    
    <p>
    <strong>Comparison with other correlation methods:</strong>
    <ul>
        <li><strong>Pearson:</strong> Measures linear relationships, sensitive to outliers, assumes normality</li>
        <li><strong>Spearman:</strong> Based on ranks but uses squared differences, computationally similar to Pearson on ranks</li>
        <li><strong>Kendall:</strong> Based on concordance/discordance, more interpretable probability, better for small samples</li>
    </ul>
    </p>
    
    <p>
    <strong>Input format:</strong> Tab-separated file with:
    <ul>
        <li><strong>Bipartite (2 columns):</strong> Gene \\t Disease (for gene-disease association analysis)</li>
        <li><strong>Weighted one-mode (3 columns):</strong> Gene1 \\t Gene2 \\t Weight (for pre-computed interactions)</li>
    </ul>
    The method automatically detects the format and constructs the appropriate gene-gene interaction matrix.
    </p>
    
    <p>
    <strong>Input:</strong> TSV/CSV file with 3 columns (Gene1, Gen2, Weight etc.) representing either a bipartite e network. The method will construct the necessary matrices to compute Kendall's Tau for node pairs.
    </p>
    
    <p>
    <strong>Output:</strong> TSV file with gene pairs and their Kendall correlation coefficients 
    above the user-defined threshold. Both positive and negative correlations are reported, allowing 
    identification of both similar and inversely related gene pairs.
    </p>
    
    <p>
    <em>Reference:</em> El-Hashash, E. F., & Shiekh, R. H. A. (2022). A comparison of the Pearson, Spearman rank and Kendall Tau correlation coefficients using quantitative variables. <em>Asian Journal of Probability and Statistics</em>, 20(3), 36–48.
    <small>DOI: 10.9734/ajpas/2022/v20i3425</small>
    </p>
    
    <p>
    <em>Additional Reference:</em> Kendall, M. G. (1938). A new measure of rank correlation. <em>Biometrika</em>, 30(1/2), 81–93.
    <br>
    <small>DOI: 10.2307/2332226</small>
    </p>
`,
"Spearman Correlation Weighted Projection":`
    <p>
    <strong>Spearman's rank correlation coefficient (ρ)</strong> is a non-parametric measure that 
    quantifies the monotonic relationship between two variables. It is used when the data are at least 
    ordinal and requires no assumptions about the distribution of the data. It is mainly used when the 
    data are not normally distributed and we want to measure non-linear monotonic relationships 
    (El-Hashash & Shiekh, 2022).
    </p>
    
    <p style="text-align: center; font-size: 1.2em;">
    <strong>ρ = 1 - (6Σdᵢ²) / (n(n² - 1))</strong>
    </p>
    
    <p>
    where:
    <ul>
        <li><strong>dᵢ = R<sub>xᵢ</sub> - R<sub>yᵢ</sub>:</strong> The difference between the ranks 
            of corresponding observations for the two variables. 
            If R<sub>xᵢ</sub> is the rank of the i-th observation of variable X 
            and R<sub>yᵢ</sub> is the rank of the i-th observation of variable Y.</li>
        <li><strong>Σdᵢ²:</strong> The sum of the squared differences in ranks for all observations.</li>
        <li><strong>n:</strong> The number of observations (or sample size), i.e., the number of data pairs.</li>
    </ul>
    </p>
    
    <p>
    The Spearman coefficient (ρ) ranges from -1 to +1:
    </p>
    
    <ul>
        <li><strong>ρ = +1:</strong> Perfect positive correlation (the ranks of the two variables are identical) - 
            when one variable increases, the other also increases in the same rank order.</li>
        <li><strong>ρ = -1:</strong> Perfect negative correlation (the ranks are inversely identical) - 
            when one variable increases, the other decreases in reverse rank order.</li>
        <li><strong>ρ = 0:</strong> No monotonic relationship between the two variables.</li>
    </ul>
    
    <p>
    <strong>Key characteristics of Spearman correlation:</strong>
    <ul>
        <li><strong>Non-parametric:</strong> Makes no assumptions about the distribution of the data</li>
        <li><strong>Monotonic relationships:</strong> Detects any monotonic relationship, not just linear</li>
        <li><strong>Robust to outliers:</strong> Less sensitive to extreme values than Pearson correlation</li>
        <li><strong>Ordinal data:</strong> Suitable for ordinal variables or continuous variables that don't meet normality assumptions</li>
        <li><strong>Computational efficiency:</strong> Faster than Kendall's Tau for large datasets</li>
    </ul>
    </p>
    
    <p>
    <strong>Comparison with other correlation methods:</strong>
    <ul>
        <li><strong>Pearson:</strong> Measures linear relationships only, sensitive to outliers, assumes normality</li>
        <li><strong>Spearman:</strong> Based on ranks, detects monotonic relationships, more robust to outliers</li>
        <li><strong>Kendall:</strong> Also based on ranks, different interpretation (probability), better for small samples</li>
    </ul>
    </p>
    
    <p>
    In gene-disease networks, Spearman correlation is particularly useful for:
    <ul>
        <li>Identifying genes with similar disease association patterns across multiple conditions</li>
        <li>Detecting monotonic relationships between gene expression levels and disease severity</li>
        <li>Comparing ranked lists of gene-disease associations from different studies</li>
        <li>Providing robust correlation estimates when data contains outliers or non-normal distributions</li>
    </ul>
    </p>
    
    <p>
    <strong>Input format:</strong> TSV file with:
    <ul>
        <li><strong>Bipartite (2 columns):</strong> Gene \\t Disease (for gene-disease association analysis)</li>
        <li><strong>Weighted one-mode (3 columns):</strong> Gene1 \\t Gene2 \\t Weight (for pre-computed interactions)</li>
    </ul>
    The method automatically detects the format and constructs the appropriate gene-gene interaction matrix.
    </p>
    
    <p>
    <strong>Output:</strong> TSV file with gene pairs and their Spearman correlation coefficients 
    above the user-defined threshold. Both positive and negative correlations are reported.
    </p>
    
    <br>
    <p>
    <em>Reference:</em> El-Hashash, E. F., & Shiekh, R. H. A. (2022). A comparison of the Pearson, Spearman rank and Kendall Tau correlation coefficients using quantitative variables. <em>Asian Journal of Probability and Statistics</em>, 20(3), 36–48.
    <br>
    <small>DOI: 10.9734/ajpas/2022/v20i3425</small>
    </p>
    
    <p>
    <em>Additional Reference:</em> Spearman, C. (1904). The proof and measurement of association between two things. <em>The American Journal of Psychology</em>, 15(1), 72–101.
    <br>
    <small>DOI: 10.2307/1412159</small>
    </p>
`,
"Pearson Correlation Weighted Projection":`
    <p>
    <strong>Pearson Correlation Coefficient (r)</strong> measures the linear relationship between two variables. 
    It is the most widely used correlation coefficient and quantifies the strength and direction of a linear association 
    between two continuous variables (El-Hashash & Shiekh, 2022).
    </p>
    
    <p style="text-align: center; font-size: 1.2em;">
    <strong>r(X,Y) = Σ((xᵢ - x̄)(yᵢ - ȳ)) / √(Σ(xᵢ - x̄)² · Σ(yᵢ - ȳ)²)</strong>
    </p>
    
    <p>
    where:
    <ul>
        <li><strong>x̄, ȳ:</strong> Means (averages) of variables X and Y</li>
        <li><strong>xᵢ, yᵢ:</strong> Individual observation values</li>
        <li><strong>n:</strong> Number of observations (sample size)</li>
        <li><strong>Σ((xᵢ - x̄)(yᵢ - ȳ)):</strong> Covariance between X and Y</li>
        <li><strong>√(Σ(xᵢ - x̄)² · Σ(yᵢ - ȳ)²):</strong> Product of the standard deviations of X and Y</li>
    </ul>
    </p>
    
    <p>
    The Pearson coefficient (r) ranges from -1 to +1:
    </p>
    
    <ul>
        <li><strong>r = +1:</strong> Perfect positive linear correlation - as one variable increases, the other increases in perfect linear fashion</li>
        <li><strong>r = -1:</strong> Perfect negative linear correlation - as one variable increases, the other decreases in perfect linear fashion</li>
        <li><strong>r = 0:</strong> No linear correlation between the variables</li>
    </ul>
    
    <p>
    The closer |r| is to 1, the stronger the linear relationship. Values between indicate:
    <ul>
        <li><strong>|r| > 0.7:</strong> Strong linear relationship</li>
        <li><strong>0.3 < |r| < 0.7:</strong> Moderate linear relationship</li>
        <li><strong>|r| < 0.3:</strong> Weak linear relationship</li>
    </ul>
    </p>
    
    <p>
    <strong>Key assumptions of Pearson correlation:</strong>
    <ul>
        <li><strong>Linearity:</strong> The relationship between variables should be linear</li>
        <li><strong>Normality:</strong> Variables should be approximately normally distributed</li>
        <li><strong>Homoscedasticity:</strong> Constant variance across the range of data</li>
        <li><strong>No outliers:</strong> Sensitive to extreme values that can distort results</li>
        <li><strong>Continuous data:</strong> Variables should be measured on interval or ratio scale</li>
    </ul>
    </p>
    
    <p>
    <strong>Comparison with other correlation methods:</strong>
    <ul>
        <li><strong>Pearson:</strong> Measures linear relationships, assumes normality, sensitive to outliers - best for normally distributed continuous data with linear relationships</li>
        <li><strong>Spearman:</strong> Based on ranks, detects monotonic relationships, robust to outliers - better for ordinal data or non-normal distributions</li>
        <li><strong>Kendall:</strong> Also based on ranks, different interpretation (probability), most robust - better for small samples with many ties</li>
    </ul>
    </p>
    
    <p>
    In gene-disease networks, Pearson correlation is particularly useful for:
    <ul>
        <li>Analyzing gene expression data that follows approximately normal distribution</li>
        <li>Detecting linear relationships between continuous biological measurements</li>
        <li>Identifying co-expressed genes in transcriptomic studies after appropriate normalization</li>
        <li>Building gene co-expression networks from microarray or RNA-seq data</li>
        <li>Quantifying relationships between protein concentrations or metabolite levels</li>
    </ul>
    </p>
    
    <p>
    <strong>Important limitations and considerations:</strong>
    <ul>
        <li>Pearson correlation only detects linear relationships - it may miss non-linear associations</li>
        <li>Outliers can dramatically affect the coefficient value</li>
        <li>Correlation does not imply causation</li>
        <li>If assumptions are violated, consider using Spearman or Kendall correlation instead</li>
        <li>Always visualize data with scatter plots before interpreting Pearson correlation</li>
    </ul>
    </p>
    
    <p>
    <strong>Input format:</strong> TSV file with:
    <ul>
        <li><strong>Bipartite (2 columns):</strong> Gene \\t Value (for continuous measurements like expression levels)</li>
        <li><strong>Weighted one-mode (3 columns):</strong> Gene1 \\t Gene2 \\t Weight (for pre-computed interactions)</li>
    </ul>
    The method automatically detects the format and constructs the appropriate gene-gene interaction matrix.
    </p>
    
    <p>
    <strong>Output:</strong> TSV file with gene pairs and their Pearson correlation coefficients 
    above the user-defined threshold. Both positive and negative correlations are reported, allowing 
    identification of both positively co-expressed and inversely correlated gene pairs.
    </p>
    
    <br>
    <p>
    <em>Reference:</em> El-Hashash, E. F., & Shiekh, R. H. A. (2022). A comparison of the Pearson, Spearman rank and Kendall Tau correlation coefficients using quantitative variables. <em>Asian Journal of Probability and Statistics</em>, 20(3), 36–48.
    <br>
    <small>DOI: 10.9734/ajpas/2022/v20i3425</small>
    </p>
    
    <p>
    <em>Additional Reference:</em> Pearson, K. (1895). Notes on regression and inheritance in the case of two parents. <em>Proceedings of the Royal Society of London</em>, 58, 240–242.
    <br>
    <small>DOI: 10.1098/rspl.1895.0041</small>
    </p>
`,
"Edge Betweenness": `
    <p>
    <strong>Girvan-Newman algorithm</strong> is a hierarchical method for detecting community structure in networks, 
    such as social or biological networks. Communities are defined as subsets of nodes with dense internal connections, 
    while connections between communities are sparser. This approach is particularly useful for understanding the 
    organization and function of complex systems. Unlike traditional hierarchical clustering methods that focus on 
    community cores, this algorithm focuses on their boundaries (Girvan & Newman, 2002).
    </p>
    
    <p>
    The method is based on the concept of <strong>edge betweenness</strong> to identify edges that act as "bridges" 
    between communities. The betweenness of a node i is defined as the number of shortest paths between pairs of 
    other nodes that pass through i. Similarly, edge betweenness is defined as the number of shortest paths between 
    all pairs of nodes in the network that pass through edge e. If there are multiple shortest paths between a pair 
    of nodes, each path receives equal weight so that the total weight sum of all paths equals one.
    </p>
    
    <p style="text-align: center; font-size: 1.2em;">
    <strong>B(e) = Σ<sub>s,t ∈ V</sub> σ<sub>st</sub>(e) / σ<sub>st</sub></strong>
    </p>
    
    <p>
    where:
    <ul>
        <li><strong>e</strong> is the edge under consideration</li>
        <li><strong>V</strong> is the set of all nodes in the network</li>
        <li><strong>σ<sub>st</sub></strong> is the total number of shortest paths between nodes s and t</li>
        <li><strong>σ<sub>st</sub>(e)</strong> is the number of shortest paths between s and t that pass through edge e</li>
    </ul>
    </p>
    
    <p>
    Edges connecting different communities tend to have high betweenness, as many shortest paths between different 
    groups necessarily pass through them. The Girvan-Newman algorithm for community detection is an iterative 
    process that progressively removes network edges (Girvan & Newman, 2002):
    </p>
    
    <ol>
        <li>Calculate betweenness for all edges in the network</li>
        <li>Identify the edge with the highest betweenness and remove it from the network</li>
        <li>Recalculate betweenness for all edges affected by the removal</li>
        <li>Repeat steps 2 and 3 until no edges remain in the network</li>
    </ol>
    
    <p>
    As edges are removed, the network breaks into separate components, which are considered communities. 
    The algorithm produces a hierarchical structure showing how nodes group into communities as edges are removed. 
    Betweenness must be recalculated at each step, as removing an edge can change the shortest paths and consequently 
    the betweenness of remaining edges.
    </p>
    
    <p>
    <strong>Key advantages:</strong>
    <ul>
        <li>Accurately identifies community boundaries, overcoming limitations of traditional methods that struggle with peripheral nodes</li>
        <li>Produces hierarchical community structure</li>
        <li>Does not require specifying the number of communities in advance</li>
        <li>Well-suited for networks with clear community structure</li>
    </ul>
    </p>
    
    <p>
    <strong>Main disadvantage:</strong>
    <ul>
        <li><strong>High computational complexity:</strong> The algorithm runs in O(m²n) time in the worst case (where m is the number of edges and n is the number of nodes), making it impractical for very large networks (Girvan & Newman, 2002)</li>
        <li>Betweenness must be recalculated after each edge removal, which is computationally expensive</li>
    </ul>
    </p>
    
    <p>
    <strong>Our optimized implementation features:</strong>
    <ul>
        <li><strong>Parallel processing:</strong> Uses multiple CPU cores for faster betweenness calculation</li>
        <li><strong>Adjacency matrix:</strong> Optimized data structures for faster computations</li>
        <li><strong>Early stopping:</strong> Stops when modularity stops improving to save computation time</li>
        <li><strong>Modularity tracking:</strong> Monitors network modularity to find optimal community structure</li>
        <li><strong>Configurable parameters:</strong> Control maximum iterations, minimum communities, and early stopping</li>
        <li><strong>Comprehensive visualization:</strong> Generates network plots with community coloring and modularity evolution</li>
    </ul>
    </p>
    
    <p>
    <strong>Parameters:</strong>
    <ul>
        <li><strong>Max lines:</strong> Maximum number of edges to load (for performance with large files)</li>
        <li><strong>Processes:</strong> Number of parallel processes to use (default: CPU count - 1)</li>
        <li><strong>Early stopping:</strong> Stop when modularity stops improving to save computation time</li>
        <li><strong>Min communities:</strong> Minimum number of communities to find</li>
        <li><strong>Max iterations:</strong> Maximum algorithm iterations</li>
    </ul>
    </p>
    
    <p>
    <strong>Input format:</strong> TSV file with:
    <ul>
        <li><strong>Unweighted (2 columns):</strong> Node1 \\t Node2</li>
        <li><strong>Weighted (3 columns):</strong> Node1 \\t Node2 \\t Weight</li>
    </ul>
    The method automatically detects the format and constructs the appropriate graph.
    </p>
    
    <p>
    <strong>Output:</strong>
    <ul>
        <li><strong>Communities file:</strong> Text file with all nodes organized by community</li>
        <li><strong>Network image:</strong> Visualization of the network with communities colored</li>
        <li><strong>Modularity image:</strong> Plot showing modularity evolution throughout the algorithm</li>
        <li><strong>Statistics:</strong> Number of communities, modularity score, runtime, and community summaries</li>
    </ul>
    </p>
    
    <p>
    In gene-disease networks, this method is particularly useful for:
    <ul>
        <li>Identifying functional modules of genes that work together</li>
        <li>Discovering disease-related gene clusters</li>
        <li>Understanding the hierarchical organization of biological processes</li>
        <li>Finding genes that act as bridges between different functional modules</li>
    </ul>
    </p>
    
    <br>
    <p>
    <em>Reference:</em> Girvan, M., & Newman, M. E. J. (2002). Community structure in social and biological networks. 
    <em>Proceedings of the National Academy of Sciences</em>, 99(12), 7821–7826.
    <br>
    <small>DOI: 10.1073/pnas.122653799</small>
    </p>
`,

    "One Mode Projection": `
        <p>
        Projects bipartite networks to one-mode networks.
        Creates connections between nodes of the same type based on shared connections.
        </p>
    `
};

// ============================
// Error Handling Functions
// ============================
function showFieldError(inputId, message) {
    const input = document.getElementById(inputId);
    if (!input) return;

    input.classList.add('error');
    let existingMessage = input.nextElementSibling;
    if (existingMessage && existingMessage.classList.contains('error-message')) {
        existingMessage.remove();
    }

    const errorSpan = document.createElement('div');
    errorSpan.className = 'error-message';
    errorSpan.textContent = message;
    input.insertAdjacentElement('afterend', errorSpan);
}

function clearFieldErrors() {
    document.querySelectorAll('input').forEach(input => {
        input.classList.remove('error');
        let errorElem = input.nextElementSibling;
        if (errorElem && errorElem.classList.contains('error-message')) {
            errorElem.remove();
        }
    });
}

// ============================
// Process Functions
// ============================
// Download Helper
// ============================
async function downloadFile(url, filename) {
    const response = await fetch(url, { 
        method: 'GET', 
        credentials: 'include' 
    });
    
    if (!response.ok) {
        throw new Error('File download failed');
    }
    
    const blob = await response.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
}

// ============================
// Method Selection Setup
// ============================
function setupChooseMethod() {
    const projectionsBtn = document.getElementById('projectionsBtn');
    const backboneBtn = document.getElementById('backboneBtn');
    const projectionsOptions = document.getElementById('projectionsOptions');
    const backboneOptions = document.getElementById('backboneOptions');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const previewBox = document.getElementById('filePreviewBox');
    const methodInfoText = document.getElementById("methodInfoText");
    const methodInfoContainer = document.getElementById("methodInfoContainer");
    const continueBtn = document.getElementById('continueBtn');
    const paramField = document.getElementById('paramField');
    const downloadSection = document.getElementById('downloadSection');
    const downloadBtn = document.getElementById('downloadBtn');
    const downloadMessage = document.getElementById('downloadMessage');

    continueBtn.style.display = 'none';

    function closeAllDropdowns() {
        if (projectionsOptions) projectionsOptions.style.display = 'none';
        if (backboneOptions) backboneOptions.style.display = 'none';
    }

    function onMethodSelected(item, sourceBtn) {
        selectedMethod = item.textContent.replace(/^\d+\.\s*/, '').trim();
        projectionsBtn.innerHTML = 'Bipartite Projections <span class="dropdown-arrow">▼</span>';
        backboneBtn.innerHTML = 'Backbone Extraction <span class="dropdown-arrow">▼</span>';
        sourceBtn.innerHTML = selectedMethod + ' <span class="dropdown-arrow">▼</span>';
        closeAllDropdowns();

        hideVisualization();
        hideAllScreens();

        previewBox.style.display = 'none';
        previewBox.textContent = '';
        fileInput.value = '';
        lastUploadedFormData = null;
        downloadSection.style.display = 'none';

        const description = methodDescriptions[selectedMethod];
        methodInfoText.innerHTML = description || `
            <p style="font-style: italic;">No description available for this method.</p>
        `;
        methodInfoContainer.style.display = 'block';

        const polyaPanel = document.getElementById('polyaPanel');
        if (ALPHA_METHODS.includes(selectedMethod.toLowerCase())) {
            paramField.style.display = 'block';
        } else {
            paramField.style.display = 'none';
        }
        if (polyaPanel) {
            polyaPanel.style.display = selectedMethod.toLowerCase() === 'polya filter' ? 'block' : 'none';
        }
        const globalThresholdPanel = document.getElementById('globalThresholdPanel');
        if (globalThresholdPanel) {
            if (selectedMethod.toLowerCase() === 'global threshold filter') {
                globalThresholdPanel.style.display = 'block';
                const uploadedFile = sessionStorage.getItem('uploaded_file');
                if (uploadedFile) {
                    fetch(`${API_BASE}/get_file_columns?filename=${uploadedFile}`)
                        .then(r => r.json())
                        .then(data => {
                            if (data.success && typeof updateThresholdPanel === 'function') {
                                updateThresholdPanel(data.columns);
                            }
                        })
                        .catch(() => {});
                }
            } else {
                globalThresholdPanel.style.display = 'none';
            }
        }

        renderDynamicParams(selectedMethod);
    }

    // Toggle projections dropdown
    projectionsBtn.addEventListener('click', e => {
        e.stopPropagation();
        const isOpen = projectionsOptions.style.display === 'block';
        closeAllDropdowns();
        if (!isOpen) projectionsOptions.style.display = 'block';
    });

    // Toggle backbone dropdown
    backboneBtn.addEventListener('click', e => {
        e.stopPropagation();
        const isOpen = backboneOptions.style.display === 'block';
        closeAllDropdowns();
        if (!isOpen) backboneOptions.style.display = 'block';
    });

    document.addEventListener('click', e => {
        if (!projectionsBtn.contains(e.target) && !projectionsOptions.contains(e.target) &&
            !backboneBtn.contains(e.target) && !backboneOptions.contains(e.target)) {
            closeAllDropdowns();
        }
    });

    document.querySelectorAll('#projectionsOptions li').forEach(item => {
        item.addEventListener('click', () => onMethodSelected(item, projectionsBtn));
    });

    document.querySelectorAll('#backboneOptions li').forEach(item => {
        item.addEventListener('click', () => onMethodSelected(item, backboneBtn));
    });

    // Upload button
    uploadBtn.addEventListener('click', () => {
        if (!selectedMethod) {
            alert('Please select a method first.');
            return;
        }
        fileInput.click();
    });

    fileInput.addEventListener('change', async event => {
        const file = event.target.files[0];
        methodInfoContainer.style.display = 'none';
        if (!file) {
            previewBox.style.display = 'none';
            continueBtn.style.display = 'none';
            lastUploadedFormData = null;
            return;
        }

        const allowedTypes = ['text/plain', 'text/csv'];
        const allowedExtensions = ['.txt', '.csv', '.tsv'];
        const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        
        if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
            alert('Please select a .txt, .csv, or .tsv file');
            fileInput.value = '';
            previewBox.style.display = 'none';
            continueBtn.style.display = 'none';
            lastUploadedFormData = null;
            return;
        }

        previewBox.style.display = 'block';
        previewBox.textContent = `📄 ${file.name}\n\nLoading content...`;
        
        const reader = new FileReader();
        reader.onload = e => {
            const content = e.target.result;
            // Show first few lines of the file
            const lines = content.split('\n').slice(0, 10).join('\n');
            previewBox.textContent = `📄 ${file.name}\n\nFirst 10 lines:\n${lines}`;
        };
        reader.readAsText(file);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('method', selectedMethod);

        // Αν η μέθοδος χρειάζεται alpha, στείλε και το alpha
        
        
        if (ALPHA_METHODS.includes(selectedMethod.toLowerCase())) {
            const alphaValue = document.getElementById('alphaInput')?.value || 0.05;
            formData.append('alpha', alphaValue);
        }

        lastUploadedFormData = formData;

        try {
            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData,
                credentials: 'include'
            });
            
            const data = await response.json();
            if (data.success) {
                sessionStorage.setItem('uploaded_file', data.uploaded_file);
                alert('File uploaded successfully! You can now click "Run Method" to process it.');
                continueBtn.style.display = 'inline-block';
            } else {
                alert(data.message || 'File upload error. Please check your file format.');
                continueBtn.style.display = 'none';
                lastUploadedFormData = null;
            }
        } catch (err) {
            alert('Network error during upload. Please check your connection.');
            continueBtn.style.display = 'none';
            lastUploadedFormData = null;
            console.error(err);
        }
    });

    // Continue button
    // Continue button
// Continue button
// Continue button
continueBtn.addEventListener('click', async () => {
    const uploadedFile = sessionStorage.getItem('uploaded_file');
    if (!selectedMethod) {
        alert('Please select a method first.');
        return;
    }
    if (!uploadedFile) {
        alert('Please upload a file first.');
        return;
    }
    const _mic = document.getElementById('methodInfoContainer');
    if (_mic) _mic.style.display = 'none';

    continueBtn.disabled = true;
    continueBtn.textContent = 'Processing...';
    downloadSection.style.display = 'none';

    // Show processing panel
    const processingPanel = document.getElementById('processingPanel');
    const procMethodLabel = document.getElementById('procMethodLabel');
    const terminalOutput = document.getElementById('terminalOutput');
    if (processingPanel) {
        if (procMethodLabel) procMethodLabel.textContent = selectedMethod;
        if (terminalOutput) terminalOutput.innerHTML = '';
        processingPanel.style.display = 'block';
    }

    function appendLog(msg) {
        if (!terminalOutput) return;
        const line = document.createElement('div');
        const lc = msg.toLowerCase();
        if (lc.startsWith('[error]')) line.className = 'log-error';
        else if (lc.startsWith('[warning]') || lc.startsWith('[warn]')) line.className = 'log-warn';
        else if (lc.startsWith('[debug]')) line.className = 'log-debug';
        else line.className = 'log-info';
        line.textContent = msg;
        terminalOutput.appendChild(line);
        terminalOutput.scrollTop = terminalOutput.scrollHeight;
    }

    try {
        let alphaValue = 0.05;
        if (ALPHA_METHODS.includes(selectedMethod.toLowerCase())) {
            const input = document.getElementById('alphaInput');
            if (input) alphaValue = parseFloat(input.value) || 0.05;
        }

        let requestData = {
            uploaded_file: uploadedFile,
            method: selectedMethod,
            alpha: alphaValue
        };

        if (selectedMethod.toLowerCase() === 'polya filter') {
            requestData.alpha = parseFloat(document.getElementById('polyaAlpha')?.value) || 0.05;
            requestData.a_parameter = parseFloat(document.getElementById('polyaA')?.value) ?? 1.0;
            requestData.correction_method = document.getElementById('polyaCorrection')?.value || 'fdr';
            requestData.directed = document.getElementById('polyaDirected')?.value || 'false';
        }

        if (selectedMethod.toLowerCase() === 'global threshold filter') {
            let params = null;
            const stored = sessionStorage.getItem('threshold_params');
            if (stored) {
                params = JSON.parse(stored);
            } else {
                // Fallback: read current panel values directly
                const colSel = document.getElementById('thresholdColumn');
                const dirSel = document.getElementById('thresholdDirection');
                const valIn  = document.getElementById('thresholdValue');
                if (colSel && dirSel && valIn && valIn.value !== '') {
                    const dirRaw = dirSel.value;
                    params = {
                        column: colSel.value === 'auto' ? 'auto' : parseInt(colSel.value),
                        direction: dirRaw === 'greater' ? 'gte' : dirRaw === 'less' ? 'lte' : dirRaw,
                        value: parseFloat(valIn.value)
                    };
                }
            }
            if (params && !isNaN(params.value)) {
                const direction = params.direction === 'greater' ? 'gte' :
                                  params.direction === 'less' ? 'lte' :
                                  params.direction;
                requestData = {
                    ...requestData,
                    col_index: params.column === 'auto' ? 2 : params.column,
                    threshold: params.value,
                    direction: direction,
                    chunk_size: 50000
                };
            } else {
                alert('Please enter a threshold value.');
                continueBtn.disabled = false;
                continueBtn.textContent = 'Run Method';
                if (processingPanel) processingPanel.style.display = 'none';
                return;
            }
        }

        // Merge any per-method tunable params (threshold, p_threshold, etc.)
        Object.assign(requestData, collectDynamicParams(selectedMethod));

        window.currentMethod = selectedMethod;

        // Start async processing — backend returns task_id immediately
        const startResp = await fetch(`${API_BASE}/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData),
            credentials: 'include'
        });
        const startData = await startResp.json();

        if (!startResp.ok) {
            throw new Error(startData.message || 'Failed to start processing');
        }

        const taskId = startData.task_id;
        appendLog(`[INFO] Task started: ${taskId}`);

        // Poll for task status
        let seenLogs = 0;
        await new Promise((resolve, reject) => {
            const timer = setInterval(async () => {
                try {
                    const pollResp = await fetch(`${API_BASE}/task_status/${taskId}`, {
                        credentials: 'include'
                    });
                    const pollData = await pollResp.json();

                    // Append any new log lines
                    const logs = pollData.logs || [];
                    for (let i = seenLogs; i < logs.length; i++) {
                        appendLog(logs[i]);
                    }
                    seenLogs = logs.length;

                    if (pollData.status === 'done') {
                        clearInterval(timer);
                        if (processingPanel) processingPanel.style.display = 'none';
                        resolve(pollData.result);
                    } else if (pollData.status === 'error') {
                        clearInterval(timer);
                        reject(new Error(pollData.error || 'Processing failed'));
                    }
                    // else still 'running' — keep polling
                } catch (pollErr) {
                    clearInterval(timer);
                    reject(pollErr);
                }
            }, 800);
        }).then((result) => {
            const data = result;
            const methodName = window.currentMethod || selectedMethod;
            const files = data.files || {};
            let resultFiles = [];

            if (files.result_file) {
                resultFiles.push({name: files.result_file, desc: `${methodName} Result`});
                if (files.communities_file) {
                    resultFiles.push({name: files.communities_file, desc: `${methodName} Communities`});
                }
            } else if (files.result) {
                resultFiles.push({name: files.result, desc: `${methodName} Result`});
            } else if (files.node1_file && files.node2_file) {
                resultFiles.push({name: files.node1_file, desc: `${methodName} - Node1 Projection`});
                resultFiles.push({name: files.node2_file, desc: `${methodName} - Node2 Projection`});
            } else if (files.gene_file && files.disease_file) {
                resultFiles.push({name: files.gene_file, desc: `${methodName} - Gene Projection`});
                resultFiles.push({name: files.disease_file, desc: `${methodName} - Disease Projection`});
            } else {
                for (const key in files) {
                    if (typeof files[key] === 'string' && files[key].length > 0) {
                        resultFiles.push({name: files[key], desc: `${methodName} Result`});
                        break;
                    }
                }
            }

            if (resultFiles.length === 0) {
                throw new Error('No result files found in the response.');
            }

            currentResultFile = resultFiles;

            sessionStorage.setItem('result_file', resultFiles[0].name);
            if (resultFiles.length >= 2) {
                sessionStorage.setItem('result_file_2', resultFiles[1].name);
            }

            downloadMessage.textContent = resultFiles.length === 1
                ? `Processing completed. Result: ${resultFiles[0].name}`
                : `Processing completed. ${resultFiles.length} result files generated.`;

            downloadSection.style.display = 'block';
            continueBtn.style.display = 'none';

            if (window.updateComparisonButtons) {
                window.updateComparisonButtons();
            }
        });

    } catch (error) {
        console.error('Error during processing:', error);
        const processingPanel = document.getElementById('processingPanel');
        const terminalOutput = document.getElementById('terminalOutput');
        if (terminalOutput) {
            const errLine = document.createElement('div');
            errLine.className = 'log-error';
            errLine.textContent = `[ERROR] ${error.message}`;
            terminalOutput.appendChild(errLine);
            terminalOutput.scrollTop = terminalOutput.scrollHeight;
        } else {
            alert(`Error: ${error.message}`);
        }
    } finally {
        continueBtn.disabled = false;
        continueBtn.textContent = 'Run Method';
    }
});

    // Download button
    if (downloadBtn) {
        downloadBtn.addEventListener('click', async () => {
            if (!currentResultFile || currentResultFile.length === 0) {
                alert('No result file to download.');
                return;
            }
            
            downloadBtn.disabled = true;
            downloadBtn.textContent = 'Downloading...';
            
            try {
                // Αν έχουμε πολλαπλά αρχεία, κατεβάζουμε όλα
                for (const fileInfo of currentResultFile) {
                    await downloadFile(
                        `${API_BASE}/download?file=${encodeURIComponent(fileInfo.name)}`,
                        fileInfo.name
                    );
                }
                
                // Show download complete message
                const completeMsg = document.createElement('div');
                completeMsg.style.cssText = `
                    margin-top: 10px;
                    padding: 10px;
                    background-color: rgba(76, 175, 80, 0.8);
                    color: white;
                    border-radius: 5px;
                    text-align: center;
                    font-family: monospace;
                `;
                completeMsg.textContent = 'Download completed successfully! You can now select a new method and upload another file.';
                methodPanel.appendChild(completeMsg);
                
                // Αφαίρεση του μηνύματος μετά από 5 δευτερόλεπτα
                setTimeout(() => {
                    if (completeMsg.parentNode) {
                        completeMsg.parentNode.removeChild(completeMsg);
                    }
                }, 5000);
                
                // Reset για νέα ανάλυση
                sessionStorage.removeItem('uploaded_file');
                fileInput.value = '';
                previewBox.style.display = 'none';
                previewBox.textContent = '';
                downloadSection.style.display = 'none';
                continueBtn.style.display = 'none';
                currentResultFile = null;
                
                // Reset method selection
                const pBtn2 = document.getElementById('projectionsBtn');
                const bBtn2 = document.getElementById('backboneBtn');
                if (pBtn2) pBtn2.innerHTML = 'Bipartite Projections <span class="dropdown-arrow">▼</span>';
                if (bBtn2) bBtn2.innerHTML = 'Backbone Extraction <span class="dropdown-arrow">▼</span>';
                selectedMethod = null;
                
            } catch (error) {
                alert(`Download error: ${error.message}`);
            } finally {
                downloadBtn.disabled = false;
                downloadBtn.textContent = 'Download Result';
            }
        });
    }
}


// ============================
// Guide Screen - Help & How To Use
// ============================
let isGuideVisible = false;

function showGuideContent() {
    const mainPanel = document.getElementById('mainPanel');
    const methodPanel = document.getElementById('methodPanel');
    const paramField = document.getElementById('paramField');
    const downloadSection = document.getElementById('downloadSection');
    const globalThresholdPanel = document.getElementById('globalThresholdPanel');
    const visualizationSection = document.getElementById('visualizationSection');
    const continueBtn = document.getElementById('continueBtn');
    const uploadBtn = document.getElementById('uploadBtn');

    // Κρύβουμε ΟΛΑ τα υπάρχοντα στοιχεία
    if (methodPanel) methodPanel.style.display = 'none';
    if (paramField) paramField.style.display = 'none';
    if (downloadSection) downloadSection.style.display = 'none';
    if (globalThresholdPanel) globalThresholdPanel.style.display = 'none';
    const polyaPanelGuide = document.getElementById('polyaPanel');
    if (polyaPanelGuide) polyaPanelGuide.style.display = 'none';
    const dpGuide = document.getElementById('dynamicParamsPanel');
    if (dpGuide) dpGuide.style.display = 'none';
    if (visualizationSection) visualizationSection.style.display = 'none';

    // Κρύβουμε και τα κουμπιά που μπορεί να είναι ορατά
    if (continueBtn) continueBtn.style.display = 'none';
    if (uploadBtn) uploadBtn.style.display = 'none';
    const methodDropdownGroup = document.getElementById('methodDropdownGroup');
    if (methodDropdownGroup) methodDropdownGroup.style.display = 'none';

    // Κρύβουμε το file preview box
    const previewBox = document.getElementById('filePreviewBox');
    if (previewBox) previewBox.style.display = 'none';
    
    // Κρύβουμε το method dropdown lists
    const projOpts = document.getElementById('projectionsOptions');
    if (projOpts) projOpts.style.display = 'none';
    const bkbOpts = document.getElementById('backboneOptions');
    if (bkbOpts) bkbOpts.style.display = 'none';
    
    // Κρύβουμε το method info container
    const methodInfoContainer = document.getElementById('methodInfoContainer');
    if (methodInfoContainer) methodInfoContainer.style.display = 'none';
    
    const existingGuide = document.getElementById('guideContent');
    if (existingGuide) existingGuide.remove();
    
    // Δημιουργία guide content
    const helpContent = document.createElement('div');
    helpContent.id = 'guideContent';
    helpContent.className = 'help-main-content';
    helpContent.innerHTML = `
        <div class="help-main-container">
            <div class="how-to-use-header" style="text-align:center; margin-bottom:2rem;">
                <h1>Guide</h1>
                <p>How to use PRED-BiPN &amp; Scientific Background</p>
            </div>
            <div class="how-to-use-steps">
                <div class="how-to-use-step"><div class="step-number">1</div><div class="step-content"><h3>Choose a method</h3><p>Click <strong>Bipartite Projections</strong> or <strong>Backbone Extraction</strong> to expand the method list and select one of the 23 available algorithms.</p><div class="step-tip"><strong>Tip:</strong> Method information appears automatically below the parameters once a method is selected.</div></div></div>
                <div class="how-to-use-step"><div class="step-number">2</div><div class="step-content"><h3>Upload your network file</h3><p>Click <strong>Upload file</strong> and select your .csv, .tsv, or .txt file. A preview of the first 10 rows appears automatically.</p><div class="step-tip"><strong>Supported:</strong> 2-column bipartite (Gene → Disease) or 3-column weighted (Source → Target → Weight).</div></div></div>
                <div class="how-to-use-step"><div class="step-number">3</div><div class="step-content"><h3>Configure parameters and execute</h3><p>Adjust parameters (e.g., significance level α), then click <strong>Run Method</strong>.</p><div class="step-tip"><strong>Default α = 0.05.</strong> Lower α → stricter filter → fewer edges.</div></div></div>
                <div class="how-to-use-step"><div class="step-number">4</div><div class="step-content"><h3>Compare and download results</h3><p>Once done: <strong>Compare Networks</strong> (original vs filtered), <strong>Download Result</strong> (TSV), or <strong>Export as PNG</strong>.</p></div></div>
                <div class="how-to-use-step"><div class="step-number">5</div><div class="step-content"><h3>Reset or start new analysis</h3><p>Use the <strong>Tool</strong> button in the top bar to reset and start fresh.</p></div></div>
            </div>
            <hr style="margin: 2rem 0; border-color: rgba(255,255,255,0.2);">

            <div class="help-main-header">
                <h1 class="help-main-title">PRED-BiPN: Scientific Documentation</h1>
            </div>
            
            <!-- Introduction -->
            <div class="help-main-section">
                <h3>⊡ Purpose & Scientific Background</h3>
                <p>
                    PRED-BiPN is a scientific software for 
                    network backbone extraction and analysis from bipartite networks. It was 
                    developed in the context of studying multifactorial diseases, where the rapid 
                    advancement of high-throughput techniques in Systems Biology has led to the 
                    accumulation of vast amounts of data, establishing network analysis as a 
                    crucial tool for understanding complex biological systems.
                </p>
            </div>

            <!-- Methodology Diagram -->
            <div class="help-main-section">
                <h3>⌗ Methodology Workflow</h3>
                <div class="methodology-diagram">
                    <img src="network.png" alt="PRED-BiPN Methodology Workflow" class="workflow-image">
                    <p class="image-caption">Figure: Network Backbone Extraction Workflow</p>
                </div>
                <p>
                    The tool integrates <strong>23 distinct backbone extraction methods</strong>, 
                    categorized into:
                </p>
                <div class="methods-grid">
                    <div class="method-category">
                        <h4>Statistical Methods</h4>
                        <ul>
                            <li>Disparity Filter</li>
                            <li>Marginal Likelihood Filter</li>
                            <li>Noise Corrected Filter3</li>
                            <li>Locally Adaptive Network Sparsifcation Filter</li>
                            <li>Polya Filter</li>
                        </ul>
                    </div>
                    <div class="method-category">
                        <h4>Structural Methods</h4>
                        <ul>
                            <li>MST (Maximum Spanning Tree)</li>
                            <li>Edge Betweenness (Girvan-Newman)</li>
                            <li>High Salience Skeleton</li>
                            <li>Global Threshold Filter</li>
                            <li>Modularity Filter</li>
                        </ul>
                    </div>
                    <div class="method-category">
                        <h4>Hybrid Methods</h4>
                        <ul>
                            <li>Doubly Stochastic Filter</li>
                            <li>Hierarchical Backbone</li>
                        </ul>
                    </div>
                    <div class="method-category">
                        <h4>Weighted Projection & Similarity</h4>
                        <ul>
                            <li>Cosine Similarity</li>
                            <li>Adamic-Adar Index</li>
                            <li>Jaccard Similarity</li>
                            <li>Pearson / Spearman / Kendall Correlation</li>
                            <li>Resource Allocation</li>
                            <li>Odds Ratio Weight</li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Data Sources -->
            <div class="help-main-section">
                <h3>⨀ Data Sources</h3>
                <p>
                    Method development and validation were performed using data from:
                </p>
                <ul>
                    <li><strong>NHGRI-EBI GWAS Catalog:</strong> Genome-wide association studies with statistically significant associations (p-value &lt; 5×10⁻⁸)</li>
                    <li><strong>NCBI OMIM (Online Mendelian Inheritance in Man):</strong> Authoritative database for monogenic disorders with expert curation</li>
                </ul>
                <p>
                    The combination of these sources ensures comprehensive coverage of both 
                    multifactorial and monogenic diseases.
                </p>
            </div>

            <!-- Theoretical Background -->
            <div class="help-main-section">
                <h3>⊞ Theoretical Background</h3>
                <p>
                    Bipartite networks B = (U, V, E) are characterized by:
                </p>
                <ul>
                    <li>Two distinct node sets U (e.g., genes) and V (e.g., diseases)</li>
                    <li>Edges connecting only nodes from different sets (E ⊆ U × V)</li>
                    <li>Support for weighted edge representation</li>
                </ul>
                <p>
                    One-mode projection transforms a bipartite network into a unipartite network, 
                    where two nodes are connected if they share a common neighbor in the original network.
                </p>
            </div>

            <!-- Evaluation Metrics -->
            <div class="help-main-section">
                <h3>Evaluation Metrics</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Density</h4>
                    </div>
                    <div class="metric-card">
                        <h4>Clustering Coefficient</h4>
                    </div>
                    <div class="metric-card">
                        <h4>Modularity</h4>
                    </div>
                    <div class="metric-card">
                        <h4>Diameter</h4>
                    </div>
                </div>
            </div>

            <!-- Biological Applications -->
            <div class="help-main-section">
                <h3>⤥ Biological Applications</h3>
                <div class="apps-grid">
                    <div class="app-card">
                        <span class="app-icon">⊺</span>
                        <h4>Community Detection</h4>
                        <p>Identification of functional gene groups collaborating in common biological pathways</p>
                    </div>
                    <div class="app-card">
                        <span class="app-icon">↯</span>
                        <h4>Comorbidity Analysis</h4>
                        <p>Recognition of diseases sharing common genetic backgrounds</p>
                    </div>
                    <div class="app-card">
                        <span class="app-icon">⌗</span>
                        <h4>Biomarker Discovery</h4>
                        <p>Identification of critical hub nodes connecting seemingly unrelated pathologies</p>
                    </div>
                    <div class="app-card">
                        <span class="app-icon">⤚</span>
                        <h4>Hierarchical Organization</h4>
                        <p>Construction of DAGs for understanding general/specific relationships</p>
                    </div>
                </div>
            </div>

            <!-- Key Findings -->
            <div class="help-main-section">
                <h3>⊡ Key Findings</h3>
                <div class="findings-list">
                    <div class="finding-item">
                        <span class="finding-marker">⊳</span>
                        <div>
                            <strong>Modularity Filter:</strong> Preserves dense community structures - ideal for studying comorbidities
                        </div>
                    </div>
                    <div class="finding-item">
                        <span class="finding-marker">⊳</span>
                        <div>
                            <strong>Polya Filter & MST:</strong> Produce highly sparse, tree-like skeletons - reveal hierarchical relationships
                        </div>
                    </div>
                    <div class="finding-item">
                        <span class="finding-marker">⊳</span>
                        <div>
                            <strong>Disparity Filter:</strong> Retains locally significant connections without undervaluing low-degree nodes
                        </div>
                    </div>
                    <div class="finding-item">
                        <span class="finding-marker">⊳</span>
                        <div>
                            <strong>SDSM:</strong> Statistically rigorous but may lead to elongated communication paths
                        </div>
                    </div>
                </div>
                <p class="conclusion">
                    <strong>There is no single optimal method for all scenarios.</strong> The choice of algorithm 
                    must be tailored to the specific biological question, whether it aims to detect distinct 
                    functional modules or to identify the core pathways of biological information flow.
                </p>
            </div>

            <!-- Selected Bibliography -->
            <div class="help-main-section">
                <h3>⌘ Selected Bibliography</h3>
                <ul class="bibliography-list">
                    <li>Serrano, M. Á., Boguñá, M., & Vespignani, A. (2009). <em>Extracting the multiscale backbone of complex weighted networks</em>. PNAS, 106(16), 6483–6488.</li>
                    <li>Dianati, N. (2016). <em>Unwinding the hairball graph: Pruning algorithms for weighted complex networks</em>. Physical Review E, 93, 012304.</li>
                    <li>Marcaccioli, R., Livan, G. (2019). <em>A Pólya urn approach to information filtering in complex networks</em>. Nature Communications, 10(1), 745.</li>
                    <li>Girvan, M., & Newman, M. E. J. (2002). <em>Community structure in social and biological networks</em>. PNAS, 99(12), 7821–7826.</li>
                    <li>Rajeh, S., Savonnet, M., Leclercq, E., & Cherifi, H. (2022). <em>Modularity-based backbone extraction in weighted complex networks</em>. arXiv:2201.12905.</li>
                    <li>Jo, W. S., Park, J., Luhur, A., Kim, B. J., & Ahn, Y. (2020). <em>Extracting hierarchical backbones from bipartite networks</em>. arXiv:2002.07239.</li>
                </ul>
            </div>

            <!-- Technical Specifications -->
            <div class="help-main-section">
                <h3>⨯ Technical Specifications</h3>
                <div class="tech-specs">
                    <div class="spec-item">
                        <strong>Languages:</strong> Python, JavaScript, HTML/CSS
                    </div>
                    <div class="spec-item">
                        <strong>Libraries:</strong> NetworkX, Pandas, SciPy, NumPy, Scikit-learn
                    </div>
                    <div class="spec-item">
                        <strong>Parallel Processing:</strong> Multiprocessing support for large-scale networks
                    </div>
                    <div class="spec-item">
                        <strong>Export Formats:</strong> TSV
                    </div>
                </div>
            </div>
            
            <button id="backToToolFromGuideBtn" class="back-to-tool-btn">← Back to Tool</button>
        </div>
    `;
    
    // Προσθέτουμε το guide content
    mainPanel.insertBefore(helpContent, methodPanel);
    
    isGuideVisible = true;
    
    // Event listener για το κουμπί επιστροφής
    const backBtn = document.getElementById('backToToolFromGuideBtn');
    if (backBtn) {
        backBtn.addEventListener('click', hideGuideContent);
    }
}

function hideGuideContent() {
    const mainPanel = document.getElementById('mainPanel');
    const methodPanel = document.getElementById('methodPanel');
    const paramField = document.getElementById('paramField');
    const downloadSection = document.getElementById('downloadSection');
    const helpContent = document.getElementById('guideContent');
    const continueBtn = document.getElementById('continueBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const previewBox = document.getElementById('filePreviewBox');
    
    // Αφαιρούμε το guide content
    if (helpContent) {
        helpContent.remove();
    }
    
    // Επαναφέρουμε όλα τα στοιχεία στην αρχική τους κατάσταση
    if (methodPanel) methodPanel.style.display = 'flex';
    if (continueBtn) continueBtn.style.display = 'inline-block';
    if (uploadBtn) uploadBtn.style.display = 'inline-block';
    const methodDropdownGroup = document.getElementById('methodDropdownGroup');
    if (methodDropdownGroup) methodDropdownGroup.style.display = 'flex';
    
    // Ελέγχουμε αν το param field πρέπει να είναι ορατό
    
    
    if (selectedMethod && ALPHA_METHODS.includes(selectedMethod.toLowerCase())) {
        if (paramField) paramField.style.display = 'block';
    } else {
        if (paramField) paramField.style.display = 'none';
    }
    const polyaPanelHide = document.getElementById('polyaPanel');
    if (polyaPanelHide) {
        polyaPanelHide.style.display = selectedMethod && selectedMethod.toLowerCase() === 'polya filter' ? 'block' : 'none';
    }

    if (downloadSection) downloadSection.style.display = 'none';
    if (previewBox) previewBox.style.display = 'none';

    isGuideVisible = false;
}

// Τροποποίηση του event listener για το Help button
function setupGuideButton() {
    const helpBtn = document.getElementById('helpBtn');
    
    if (helpBtn) {
        helpBtn.addEventListener('click', function() {
            if (isGuideVisible) {
                hideGuideContent();
            } else {
                showGuideContent();
            }
        });
    }

    const infoBtn = document.getElementById('infoBtn');
    if (infoBtn) {
        infoBtn.addEventListener('click', function() {
            if (isInfoVisible) {
                hideInfoContent();
            } else {
                if (isGuideVisible) hideGuideContent();
                showInfoContent();
            }
        });
    }
}

let isInfoVisible = false;

function showInfoContent() {
    const mainPanel = document.getElementById('mainPanel');
    const methodPanel = document.getElementById('methodPanel');
    const paramField = document.getElementById('paramField');
    const downloadSection = document.getElementById('downloadSection');
    const globalThresholdPanel = document.getElementById('globalThresholdPanel');
    const visualizationSection = document.getElementById('visualizationSection');
    const continueBtn = document.getElementById('continueBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const methodDropdownGroup = document.getElementById('methodDropdownGroup');
    const methodInfoContainer = document.getElementById('methodInfoContainer');
    const previewBox = document.getElementById('filePreviewBox');

    if (methodPanel) methodPanel.style.display = 'none';
    if (paramField) paramField.style.display = 'none';
    if (downloadSection) downloadSection.style.display = 'none';
    if (globalThresholdPanel) globalThresholdPanel.style.display = 'none';
    const polyaPanelInfo = document.getElementById('polyaPanel');
    if (polyaPanelInfo) polyaPanelInfo.style.display = 'none';
    const dpInfo = document.getElementById('dynamicParamsPanel');
    if (dpInfo) dpInfo.style.display = 'none';
    if (visualizationSection) visualizationSection.style.display = 'none';
    if (continueBtn) continueBtn.style.display = 'none';
    if (uploadBtn) uploadBtn.style.display = 'none';
    if (methodDropdownGroup) methodDropdownGroup.style.display = 'none';
    if (methodInfoContainer) methodInfoContainer.style.display = 'none';
    if (previewBox) previewBox.style.display = 'none';

    const existing = document.getElementById('infoContent');
    if (existing) existing.remove();

    const infoDiv = document.createElement('div');
    infoDiv.id = 'infoContent';
    infoDiv.className = 'help-main-content';
    infoDiv.innerHTML = `
        <div class="help-main-container">
            <div class="how-to-use-header" style="text-align:center; margin-bottom:2rem;">
                <h1>Credits</h1>
                <p>The people behind PRED-BiPN</p>
            </div>

            <div class="help-main-section">
                <h3>Development</h3>
                <ul class="bibliography-list">
                    <li><strong>Lead Developer:</strong> &nbsp;</li>
                    <li><strong>Backend & Algorithms:</strong> &nbsp;</li>
                    <li><strong>Frontend & UI:</strong> &nbsp;</li>
                </ul>
            </div>

            <div class="help-main-section">
                <h3>Research & Scientific Supervision</h3>
                <ul class="bibliography-list">
                    <li><strong>Principal Investigator:</strong> &nbsp;</li>
                    <li><strong>Scientific Advisor:</strong> &nbsp;</li>
                </ul>
            </div>

            <div class="help-main-section">
                <h3>Acknowledgements</h3>
                <p>&nbsp;</p>
            </div>

            <div class="help-main-section">
                <h3>Institution</h3>
                <p>&nbsp;</p>
            </div>

            <button id="backToToolFromInfoBtn" class="back-to-tool-btn">← Back to Tool</button>
        </div>
    `;

    mainPanel.insertBefore(infoDiv, methodPanel);
    isInfoVisible = true;

    const backBtn = document.getElementById('backToToolFromInfoBtn');
    if (backBtn) {
        backBtn.addEventListener('click', hideInfoContent);
    }
}

function hideInfoContent() {
    const mainPanel = document.getElementById('mainPanel');
    const methodPanel = document.getElementById('methodPanel');
    const paramField = document.getElementById('paramField');
    const downloadSection = document.getElementById('downloadSection');
    const continueBtn = document.getElementById('continueBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const methodDropdownGroup = document.getElementById('methodDropdownGroup');
    const infoDiv = document.getElementById('infoContent');

    if (infoDiv) infoDiv.remove();

    if (methodPanel) methodPanel.style.display = 'flex';
    if (continueBtn) continueBtn.style.display = 'inline-block';
    if (uploadBtn) uploadBtn.style.display = 'inline-block';
    if (methodDropdownGroup) methodDropdownGroup.style.display = 'flex';

    if (selectedMethod && ALPHA_METHODS.includes(selectedMethod.toLowerCase())) {
        if (paramField) paramField.style.display = 'block';
    } else {
        if (paramField) paramField.style.display = 'none';
    }
    const polyaPanelInfoHide = document.getElementById('polyaPanel');
    if (polyaPanelInfoHide) {
        polyaPanelInfoHide.style.display = selectedMethod && selectedMethod.toLowerCase() === 'polya filter' ? 'block' : 'none';
    }

    if (downloadSection) downloadSection.style.display = 'none';

    isInfoVisible = false;
}

// ============================
// Network Visualization - Inline Comparison
// ============================

// Δημιουργία της ενότητας οπτικοποίησης
function createVisualizationSection() {
    const mainPanel = document.getElementById('mainPanel');
    if (!mainPanel) return;
    
    // Έλεγχος αν υπάρχει ήδη
    if (document.getElementById('visualizationSection')) return;
    
    const visSection = document.createElement('div');
    visSection.id = 'visualizationSection';
    visSection.className = 'visualization-section';
    visSection.style.display = 'none';
    visSection.innerHTML = `
        <div class="vis-header">
            <div class="vis-title">
                <span class="vis-icon">⌗</span>
                <h3>Network Topology Comparison</h3>
            </div>
            <button id="closeVisSection" class="vis-close">×</button>
        </div>
        <div class="vis-body">
            <div id="visLoading" class="vis-loading">
                <div class="vis-spinner"></div>
                <p>Generating network visualization...</p>
            </div>
            <div id="visContent" style="display: none;">
                <div class="comparison-wrapper">
                    <div class="network-panel">
                        <div class="panel-header before">
                            <div class="panel-title">
                                <span class="panel-icon">⊡</span>
                                <span>Original Network</span>
                            </div>
                            <div class="panel-stats" id="statsBefore">-</div>
                        </div>
                        <div id="networkBefore" class="network-display"></div>
                        <div class="panel-metrics" id="metricsBefore">
                            <div class="metric-item">
                                <span class="metric-symbol">ω</span>
                                <span class="metric-value">-</span>
                            </div>
                        </div>
                    </div>
                    <div class="vs-divider">
                        <span>↯</span>
                    </div>
                    <div class="network-panel">
                        <div class="panel-header after">
                            <div class="panel-title">
                                <span class="panel-icon">⌗</span>
                                <span>Filtered Network</span>
                            </div>
                            <div class="panel-stats" id="statsAfter">-</div>
                        </div>
                        <div id="networkAfter" class="network-display"></div>
                        <div class="panel-metrics" id="metricsAfter">
                            <div class="metric-item">
                                <span class="metric-symbol">ω</span>
                                <span class="metric-value">-</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="comparison-footer">
                    <div id="reductionInfo" class="reduction-badge"></div>
                    <button id="exportVisBtn" class="vis-export-btn">
                        <span>⤥</span> Export
                    </button>
                </div>
            </div>
        </div>
    `;
    
    mainPanel.appendChild(visSection);
    
    // Event listener για κλείσιμο
    document.getElementById('closeVisSection')?.addEventListener('click', () => {
        visSection.style.display = 'none';
        const hideBtn = document.getElementById('hideComparisonBtn');
        if (hideBtn) hideBtn.style.display = 'none';
    });
    
    // Event listener για export
    document.getElementById('exportVisBtn')?.addEventListener('click', () => {
        const img = document.querySelector('#networkBefore img');
        if (img && img.src) {
            const link = document.createElement('a');
            link.download = `network_comparison_${new Date().toISOString().slice(0,19)}.png`;
            link.href = img.src;
            link.click();
        }
    });
}
// Εμφάνιση σύγκρισης στη σελίδα
// Εμφάνιση σύγκρισης στη σελίδα
async function showInlineComparison(beforeFile, afterFile, method) {
    console.log('showInlineComparison called with:', { beforeFile, afterFile, method });
    
    if (!beforeFile || !afterFile) {
        console.error('Missing files:', { beforeFile, afterFile });
        alert('Missing files for comparison. Please ensure both original and result files are available.');
        return;
    }
    
    // Δημιουργία της ενότητας αν δεν υπάρχει
    if (!document.getElementById('visualizationSection')) {
        createVisualizationSection();
    }
    
    const visSection = document.getElementById('visualizationSection');
    const visLoading = document.getElementById('visLoading');
    const visContent = document.getElementById('visContent');
    const statsBefore = document.getElementById('statsBefore');
    const statsAfter = document.getElementById('statsAfter');
    const reductionInfo = document.getElementById('reductionInfo');
    const networkBefore = document.getElementById('networkBefore');
    const networkAfter = document.getElementById('networkAfter');
    const metricsBefore = document.getElementById('metricsBefore');
    const metricsAfter = document.getElementById('metricsAfter');
    const hideBtn = document.getElementById('hideComparisonBtn');
    
    // Εμφάνιση της ενότητας
    visSection.style.display = 'block';
    visLoading.style.display = 'flex';
    visContent.style.display = 'none';
    
    // Εμφάνιση του hide button
    if (hideBtn) hideBtn.style.display = 'inline-block';
    
    // Scroll στην ενότητα
    visSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    try {
        const response = await fetch(`${API_BASE}/visualize/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                before_file: beforeFile,
                after_file: afterFile,
                method: method
            }),
            credentials: 'include'
        });
        
        const data = await response.json();
        console.log('Comparison response:', data);
        
        if (data.success) {
            // Ενημέρωση στατιστικών
            statsBefore.innerHTML = `${data.stats_before.nodes} nodes | ${data.stats_before.edges} edges`;
            statsAfter.innerHTML = `${data.stats_after.nodes} nodes | ${data.stats_after.edges} edges`;
            
            // Ενημέρωση μετρικών πυκνότητας
            if (metricsBefore) {
                metricsBefore.innerHTML = `
                    <div class="metric-item">
                        <span class="metric-symbol">ω</span>
                        <span class="metric-value">${data.stats_before.density.toFixed(4)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-symbol">δ</span>
                        <span class="metric-value">${(data.stats_before.edges / data.stats_before.nodes).toFixed(2)}</span>
                    </div>
                `;
            }
            
            if (metricsAfter) {
                metricsAfter.innerHTML = `
                    <div class="metric-item">
                        <span class="metric-symbol">ω</span>
                        <span class="metric-value">${data.stats_after.density.toFixed(4)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-symbol">δ</span>
                        <span class="metric-value">${(data.stats_after.edges / data.stats_after.nodes).toFixed(2)}</span>
                    </div>
                `;
            }
            
            // Ενημέρωση reduction
            const reduction = data.reduction.toFixed(1);
            const edgesRemoved = data.stats_before.edges - data.stats_after.edges;
            reductionInfo.innerHTML = `⤥ ${reduction}% · ${edgesRemoved} edges removed`;
            
            // Εμφάνιση εικόνων
            networkBefore.innerHTML = '';
            networkAfter.innerHTML = '';
            
            const imgBefore = document.createElement('img');
            imgBefore.src = `data:image/png;base64,${data.image_before}`;
            imgBefore.alt = 'Original Network';
            imgBefore.style.maxWidth = '100%';
            imgBefore.style.height = 'auto';
            imgBefore.style.borderRadius = '8px';
            imgBefore.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
            networkBefore.appendChild(imgBefore);
            
            const imgAfter = document.createElement('img');
            imgAfter.src = `data:image/png;base64,${data.image_after}`;
            imgAfter.alt = 'Filtered Network';
            imgAfter.style.maxWidth = '100%';
            imgAfter.style.height = 'auto';
            imgAfter.style.borderRadius = '8px';
            imgAfter.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
            networkAfter.appendChild(imgAfter);
            
            visLoading.style.display = 'none';
            visContent.style.display = 'block';
        } else {
            throw new Error(data.message || 'Visualization failed');
        }
    } catch (error) {
        console.error('Comparison error:', error);
        visLoading.innerHTML = `
            <div style="color: #c0392b; text-align: center; padding: 2rem;">
                <strong>Error:</strong> ${error.message}<br><br>
                Make sure the backend server is running on ${API_BASE}
            </div>
        `;
    }
}

// Προσθήκη κουμπιών για οπτικοποίηση στο download section
// Προσθήκη κουμπιών για οπτικοποίηση (ενσωματωμένα στο download section)
function setupComparisonButtons() {
    const compareBtn = document.getElementById('compareNetworksInlineBtn');
    const hideBtn = document.getElementById('hideComparisonBtn');
    
    // ponytail: compare button isn't in the DOM; return a no-op so the
    // destructure in DOMContentLoaded doesn't throw and abort init.
    if (!compareBtn) return { updateButtonsState: () => {} };

    // Ενημέρωση κατάστασης κουμπιών
    function updateButtonsState() {
        const uploadedFile = sessionStorage.getItem('uploaded_file');
        let resultFile = null;
        
        if (currentResultFile && currentResultFile.length > 0) {
            resultFile = currentResultFile[0].name;
        } else if (sessionStorage.getItem('result_file')) {
            resultFile = sessionStorage.getItem('result_file');
        }
        
        console.log('Update buttons - Uploaded:', uploadedFile);
        console.log('Update buttons - Result:', resultFile);
        
        if (uploadedFile && resultFile) {
            compareBtn.disabled = false;
            compareBtn.style.opacity = '1';
            compareBtn.style.cursor = 'pointer';
            compareBtn.style.pointerEvents = 'auto';
        } else {
            compareBtn.disabled = true;
            compareBtn.style.opacity = '0.5';
            compareBtn.style.cursor = 'not-allowed';
            compareBtn.style.pointerEvents = 'none';
        }
    }
    
    // Event listener για σύγκριση
    compareBtn.addEventListener('click', () => {
        if (compareBtn.disabled) {
            alert('Please wait for processing to complete.');
            return;
        }
        
        const currentUploaded = sessionStorage.getItem('uploaded_file');
        let currentResult = null;
        
        if (currentResultFile && currentResultFile.length > 0) {
            currentResult = currentResultFile[0].name;
        } else if (sessionStorage.getItem('result_file')) {
            currentResult = sessionStorage.getItem('result_file');
        }
        
        if (currentUploaded && currentResult) {
            showInlineComparison(currentUploaded, currentResult, selectedMethod);
            if (hideBtn) hideBtn.style.display = 'inline-block';
        } else {
            alert('Both original and result files are required for comparison.');
        }
    });
    
    // Event listener για απόκρυψη
    if (hideBtn) {
        hideBtn.addEventListener('click', () => {
            const visSection = document.getElementById('visualizationSection');
            if (visSection) {
                visSection.style.display = 'none';
                hideBtn.style.display = 'none';
            }
        });
        // Αρχικά κρυμμένο
        hideBtn.style.display = 'none';
    }
    
    // Αρχική ενημέρωση
    updateButtonsState();
    
    return { updateButtonsState };
}


// ============================
// Global Threshold Filter Functions
// ============================

let globalThresholdParams = {
    column: null,
    direction: 'greater',
    value: 0.5,
    columnNames: []
};

async function detectColumnsForThreshold(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE}/detect_columns`, {
            method: 'POST',
            body: formData,
            credentials: 'include'
        });
        
        const data = await response.json();
        if (data.success) {
            return data.columns;
        }
        return [];
    } catch (error) {
        console.error('Column detection error:', error);
        return [];
    }
}

function updateThresholdPanel(columns) {
    const columnSelect = document.getElementById('thresholdColumn');
    const columnPreview = document.getElementById('columnPreview');
    const detectedColumnsSpan = document.getElementById('detectedColumns');
    
    if (!columnSelect) return;
    
    // Clear existing options
    columnSelect.innerHTML = '<option value="auto">Auto-detect (first numeric column)</option>';
    
    // Add detected columns
    columns.forEach((col, idx) => {
        const option = document.createElement('option');
        option.value = idx;
        option.textContent = `${col.name} (${col.type})`;
        columnSelect.appendChild(option);
    });
    
    // Show preview
    if (columns.length > 0) {
        detectedColumnsSpan.textContent = columns.map(c => `${c.name} (${c.type})`).join(', ');
        columnPreview.style.display = 'block';
    } else {
        detectedColumnsSpan.textContent = 'No numeric columns detected';
        columnPreview.style.display = 'block';
    }
}

function setupGlobalThreshold() {
    const methodOptions = document.querySelectorAll('#projectionsOptions li, #backboneOptions li');
    const globalThresholdPanel = document.getElementById('globalThresholdPanel');
    const applyBtn = document.getElementById('applyThresholdBtn');
    const columnSelect = document.getElementById('thresholdColumn');
    const directionSelect = document.getElementById('thresholdDirection');
    const valueInput = document.getElementById('thresholdValue');
    
    // Detect when Global Threshold Filter is selected
    methodOptions.forEach(option => {
        option.addEventListener('click', () => {
            const methodText = option.textContent.toLowerCase();
            if (methodText.includes('global threshold')) {
                globalThresholdPanel.style.display = 'block';
                // Auto-detect columns from uploaded file
                const uploadedFile = sessionStorage.getItem('uploaded_file');
                if (uploadedFile) {
                    fetch(`${API_BASE}/get_file_columns?filename=${uploadedFile}`)
                        .then(r => r.json())
                        .then(data => {
                            if (data.success) {
                                updateThresholdPanel(data.columns);
                            }
                        });
                }
            } else {
                globalThresholdPanel.style.display = 'none';
            }
        });
    });
    
    // Apply threshold parameters - ΑΠΟΘΗΚΕΥΣΗ ΣΤΟ sessionStorage
    if (applyBtn) {
        applyBtn.addEventListener('click', () => {
            const selectedColumn = columnSelect.value;
            const direction = directionSelect.value;
            const value = parseFloat(valueInput.value);
            
            // Convert direction to backend format
            const backendDirection = direction === 'greater' ? 'gte' : 'lte';
            
            globalThresholdParams = {
                column: selectedColumn === 'auto' ? 'auto' : parseInt(selectedColumn),
                direction: backendDirection,
                value: value
            };
            
            // Store parameters in sessionStorage
            sessionStorage.setItem('threshold_params', JSON.stringify(globalThresholdParams));
            
            console.log('Threshold parameters saved:', globalThresholdParams);
            alert(`Threshold filter configured!\nColumn: ${selectedColumn === 'auto' ? 'Auto-detect' : columnSelect.options[columnSelect.selectedIndex].text}\nDirection: ${direction === 'greater' ? '≥' : '≤'} ${value}`);
        });
    }
}
// ============================
// Initialization
// ============================
// Initialization
// ============================

// ============================
// Reset to Main Page Function
function resetToMainPage() {
    console.log('Resetting to main page...');
    
    // 1. Καθαρισμός μεταβλητών
    selectedMethod = null;
    currentResultFile = null;
    currentMethod = null;
    lastUploadedFormData = null;
    
    // 2. Καθαρισμός sessionStorage
    sessionStorage.removeItem('uploaded_file');
    sessionStorage.removeItem('result_file');
    sessionStorage.removeItem('result_file_2');
    sessionStorage.removeItem('threshold_params');
    sessionStorage.removeItem('selected_method');
    
    // 3. Επαναφορά των dropdown buttons
    const pBtn = document.getElementById('projectionsBtn');
    const bBtn = document.getElementById('backboneBtn');
    if (pBtn) pBtn.innerHTML = 'Bipartite Projections <span class="dropdown-arrow">▼</span>';
    if (bBtn) bBtn.innerHTML = 'Backbone Extraction <span class="dropdown-arrow">▼</span>';
    
    // 4. Κλείσιμο των dropdowns αν είναι ανοιχτά
    const projOpts = document.getElementById('projectionsOptions');
    const bkbOpts = document.getElementById('backboneOptions');
    if (projOpts) projOpts.style.display = 'none';
    if (bkbOpts) bkbOpts.style.display = 'none';
    
    // 5. Κρύψιμο του parameter field
    const paramField = document.getElementById('paramField');
    if (paramField) {
        paramField.style.display = 'none';
    }
    
    // 6. Κρύψιμο του Global Threshold Panel
    const globalThresholdPanel = document.getElementById('globalThresholdPanel');
    if (globalThresholdPanel) {
        globalThresholdPanel.style.display = 'none';
    }
    const polyaPanelReset = document.getElementById('polyaPanel');
    if (polyaPanelReset) {
        polyaPanelReset.style.display = 'none';
    }
    const dpReset = document.getElementById('dynamicParamsPanel');
    if (dpReset) {
        dpReset.style.display = 'none';
    }
    
    // 7. Κρύψιμο του download section
    const downloadSection = document.getElementById('downloadSection');
    if (downloadSection) {
        downloadSection.style.display = 'none';
    }
    
    // 8. Κρύψιμο του visualization section
    const visualizationSection = document.getElementById('visualizationSection');
    if (visualizationSection) {
        visualizationSection.style.display = 'none';
    }
    
    // 9. Εμφάνιση του method panel
    const methodPanel = document.getElementById('methodPanel');
    if (methodPanel) {
        methodPanel.style.display = 'flex';
    }
    
    // 10. Επαναφορά του continue button
    const continueBtn = document.getElementById('continueBtn');
    if (continueBtn) {
        continueBtn.style.display = 'none';
        continueBtn.disabled = false;
        continueBtn.textContent = 'Run Method';
    }
    
    // 11. Καθαρισμός του file preview box
    const previewBox = document.getElementById('filePreviewBox');
    if (previewBox) {
        previewBox.style.display = 'none';
        previewBox.textContent = '';
    }
    
    // 12. Καθαρισμός του file input
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.value = '';
    }
    
    // 13. Κλείσιμο guide screen αν είναι ορατό
    if (isGuideVisible) {
        hideGuideContent();
    }
    if (isInfoVisible) {
        hideInfoContent();
    }
    
    // 15. Επαναφορά του method info container
    const methodInfoContainer = document.getElementById('methodInfoContainer');
    if (methodInfoContainer) {
        methodInfoContainer.style.display = 'none';
    }
    
    // 16. Ενημέρωση των κουμπιών σύγκρισης
    if (window.updateComparisonButtons) {
        window.updateComparisonButtons();
    }
    
    // 17. Καθαρισμός του download message
    const downloadMessage = document.getElementById('downloadMessage');
    if (downloadMessage) {
        downloadMessage.textContent = '';
    }
    
    // 18. Επαναφορά του alpha input
    const alphaInput = document.getElementById('alphaInput');
    if (alphaInput) {
        alphaInput.value = '0.05';
    }
    
    console.log('Reset completed. Application is ready for new analysis.');
    
    // Προαιρετικά: scroll στην κορυφή
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================
// Hide All Screens (Welcome, Help, HowToUse)
// ============================
function hideAllScreens() {
    if (isGuideVisible) {
        hideGuideContent();
    }
    if (isInfoVisible) {
        hideInfoContent();
    }
}
// ============================
// Hide Visualization Section
// ============================
function hideVisualization() {
    const visSection = document.getElementById('visualizationSection');
    const hideBtn = document.getElementById('hideComparisonBtn');
    
    if (visSection) {
        visSection.style.display = 'none';
    }
    if (hideBtn) {
        hideBtn.style.display = 'none';
    }
}


document.addEventListener('DOMContentLoaded', () => {
    // Setup method selection
    setupChooseMethod();
    
    // Setup Guide button
    setupGuideButton();
    
    // Setup comparison buttons
    const { updateButtonsState } = setupComparisonButtons();
    window.updateComparisonButtons = updateButtonsState;
    
    // Setup Global Threshold Panel
    setupGlobalThreshold();
    
    // Setup Main Page button
    const mainPageBtn = document.getElementById('mainPageBtn');
    if (mainPageBtn) {
        mainPageBtn.addEventListener('click', resetToMainPage);
    }
    
    console.log('PRED-BiPN application initialized');

    // Auto-hide the title after 5 seconds for better visibility
    setTimeout(() => {
        const dynamicText = document.getElementById('dynamic-text');
        if (dynamicText) {
            dynamicText.style.opacity = '0.3';
            dynamicText.style.transition = 'opacity 1s ease';
        }
    }, 5000);
});

