# PRED-BiPN: Bipartite Network Projection & Backbone Extraction Tool

A scientific tool for projecting bipartite data into monopartite networks and for pruning weighted networks down to their statistically significant backbone. It bundles 23 distinct methods and needs no Python installation or setup to run.

## Quick Start

### Windows
Double-click `RUN_APP.bat`

The script will:
1. Detect or automatically install Python 3.11 (no admin rights needed)
2. Create a virtual environment and install all dependencies (first run only)
3. Open your browser at http://localhost:5050

### macOS / Linux
```bash
chmod +x run_app.sh
./run_app.sh
```

## System Requirements

- Windows 10/11 (64-bit), or macOS / Linux
- Internet connection on first run, to download Python and dependencies if not already installed
- No Python pre-installation required

## How to Use

1. Start the app: double-click `RUN_APP.bat` (Windows) or run `./run_app.sh` (Mac/Linux)
2. Choose a method (Projections or Backbone extraction; the search box filters the 23 methods)
3. Add your edge list (CSV, TSV or TXT). The page reports the header row, whether the file is bipartite or one-mode, and the node counts, and warns when the file does not suit the method
4. Set the method's parameters and click Run; the log shows every parameter used
5. Download the result (TSV). A projection gives one network per node set, and "Use for backbone" filters one of them straight away; a backbone shows the nodes, edges and components kept, and Compare networks draws the input and the backbone side by side

## Input File Formats

### Bipartite Network (2 columns)
```csv
Gene,Disease
BRCA1,BreastCancer
TP53,Lymphoma
```

### Weighted Network (3 columns)
```csv
Source,Target,Weight
Gene1,Gene2,0.95
Gene1,Gene3,0.87
```

## Methods Available

The tool groups its 23 methods into two categories, matching the interface.

### Bipartite Projections
- Hypergraph Projection
- Odds Ratio Weighted Projection
- Cosine Similarity Weighted Projection
- Adamic Adar Similarity Weighted Projection
- Resource Allocation Similarity Weighted Projection
- Jaccard Similarity Weighted Projection
- Kendall Correlation Weighted Projection
- Spearman Correlation Weighted Projection
- Pearson Correlation Weighted Projection
- Simple Weighted Projection
- One Mode Projection

### Backbone Extraction
- Disparity Filter
- Doubly Stochastic Filter
- Hierarchical Backbone
- High Salience Skeleton
- Locally Adaptive Network Sparsification Filter
- Marginal Likelihood Filter
- Maximum Spanning Tree Filter
- Modularity Filter
- Noise Corrected Filter
- Global Threshold Filter
- Edge Betweenness
- Polya Filter

## Parameters & Output

Each method shows only the parameters it needs:

- Similarity methods: a similarity threshold; node pairs scoring below it are dropped. Cosine and Jaccard are bounded to the 0-1 range, and a value outside it is rejected with an error. Adamic-Adar and Resource Allocation are unbounded (their scores can exceed 1), so any non-negative threshold is allowed.
- Correlation methods (Pearson, Spearman, Kendall) and Odds Ratio: a significance level alpha (default 0.05) and a multiple-testing correction, Benjamini-Hochberg FDR (default), Bonferroni, or None. Only statistically significant pairs are kept, so these methods do not collapse into a near-complete graph on sparse data.
- Backbone filters: a significance level, z-score, or threshold appropriate to the method.
- Edge Betweenness: computes the betweenness of every edge once (Brandes 2001; Girvan & Newman 2002), with weights turned into lengths 1/w, and keeps edges with betweenness >= factor x mean (default factor 1). Options: use weights (yes/no) and a number of sampled source nodes (0 = exact). Install `igraph` for speed; without it NetworkX gives the same result more slowly.
- Noise Corrected, Doubly Stochastic and Polya: undirected by default (A-B and B-A are one edge), with a Directed option. Polya defaults to a = -1 (maximum-likelihood estimate of a).
- Noise Corrected (Coscia & Neffke 2017): computed exactly as the authors' `backboning.py` (binomial model, Beta prior fitted by the method of moments); an edge is kept if score - delta x sd > 0. Parameters: delta (default 3.09), directed (default no).
- Doubly Stochastic (Slater 2009): the weight matrix (symmetric when undirected) plus 0.0001 in every cell, as in the reference code, is scaled with Sinkhorn-Knopp until std(row sums) < 1e-12; edges are then added in decreasing order of value. Stopping rule `slater` (default): stop when the backbone has as many connected components as the input. Rule `strict` (a PRED-BiPN heuristic, not in Slater): per edge, (1) skip it if below alpha (the interface value, default 0.05), (2) add it, (3) stop if below the 20th percentile of all values, (4) every 1,000 edges stop if the nodes reached form one component. Parameters: rule, directed, alpha (strict only).
- Modularity Filter (Rajeh et al. 2022, Algorithm 1): weighted Louvain partition (seed, default 42); the modularity vitality of every node computed once on the original network; the backbone is the subgraph of the share of nodes with the largest |vitality| (default 0.30, rounded up; ties in alphabetical order). `keep_lcc` (default off) keeps only the largest connected component, as in the article; our projections are fragmented from the start, so it is off by default.
- Every run writes all its parameters to the log, so a result can be reproduced.
- Every method: a "First row is a header" option (auto-detect by default). Hierarchical Backbone needs the original two-column bipartite file and rejects a projection.

Results are written as tab-separated (TSV) files:

- Bipartite projections produce two files, one per node set (for example gene-gene and disease-disease).
- Similarity methods write `source  target  weight`.
- Correlation methods write `source  target  <statistic>  p_value`, where the statistic is `pearson_r`, `spearman_rho`, or `kendall_tau`; Odds Ratio writes `source  target  odds_ratio  p_value`.
- Backbone filters write `source  target  weight` (some add a method-specific column, such as the Disparity Filter's alpha).
- Edge Betweenness writes `source  target  weight  edge_betweenness`, sorted by betweenness.

## Troubleshooting

### App won't start
- Delete the `.venv` folder and run `RUN_APP.bat` again; it will rebuild automatically
- On Mac/Linux, make sure you ran `chmod +x run_app.sh`

### Browser doesn't open
- Manually go to `http://localhost:5050`

### Port 5050 already in use
Windows:
```cmd
netstat -ano | findstr :5050
taskkill /PID <PID> /F
```
Mac/Linux:
```bash
lsof -i :5050 && kill -9 <PID>
```

### File upload fails
- Ensure the file is CSV or TSV
- Bipartite: 2 columns (node type A, node type B)
- Weighted: 3 columns (Source, Target, Weight)

## Folder Structure

```
PRED-BiPN-App/
├── RUN_APP.bat          # Windows launcher (start here)
├── run_app.sh           # Mac/Linux launcher
├── backend/
│   ├── app.py           # Flask server
│   ├── *.py             # Analysis method modules
│   ├── requirements.txt
│   └── tmp/             # Temporary result files
├── frontend/
│   ├── login.html
│   ├── login.js
│   └── login.css
└── .venv/               # Auto-created on first run
```

## Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for deploying to Render, Heroku, or a VPS.

## Distribution

See [DISTRIBUTION.md](DISTRIBUTION.md) for sharing this app with other users.
