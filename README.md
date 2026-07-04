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
2. Upload your file in CSV or TSV format
3. Select a method from the 23 available algorithms
4. Set the method's parameters, then click Run Method
5. Download the results (TSV) when the run completes; for backbone methods you can also Compare Networks to view the original vs filtered network side by side

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

### Backbone Extraction
- Disparity Filter
- Doubly Stochastic Filter
- Hierarchical Backbone
- High Salience Skeleton
- Locally Adaptive Network Sparsification (LANS)
- Marginal Likelihood Filter (MLF)
- Modularity Filter
- Noise Corrected Filter
- Polya Filter
- Maximum Spanning Tree
- Edge Betweenness (Girvan-Newman)
- Global Threshold Filter

### Projection Methods
- One Mode Projection
- Simple Weighted Projection
- Odds Ratio Weighted Projection
- Hypergraph Projection

### Similarity / Correlation
- Cosine Similarity
- Adamic-Adar
- Resource Allocation
- Jaccard Similarity
- Pearson Correlation
- Spearman Correlation
- Kendall Correlation

## Parameters & Output

Each method shows only the parameters it needs:

- Similarity methods (Cosine, Jaccard, Adamic-Adar, Resource Allocation): a similarity threshold between 0 and 1. Node pairs scoring below the threshold are dropped.
- Correlation methods (Pearson, Spearman, Kendall) and Odds Ratio: a significance level alpha (default 0.05) and a multiple-testing correction, Benjamini-Hochberg FDR (default), Bonferroni, or None. Only statistically significant pairs are kept, so these methods do not collapse into a near-complete graph on sparse data.
- Backbone filters: a significance level, z-score, or threshold appropriate to the method.

Results are written as tab-separated (TSV) files:

- Bipartite projections produce two files, one per node set (for example gene-gene and disease-disease).
- Similarity methods write `source  target  weight`.
- Correlation methods write `source  target  <statistic>  p_value`, where the statistic is `pearson_r`, `spearman_rho`, or `kendall_tau`; Odds Ratio writes `source  target  odds_ratio  p_value`.
- Backbone filters write `source  target  weight` (some add a method-specific column, such as the Disparity Filter's alpha).

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
