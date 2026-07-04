# PRED-BiPN: Bipartite Network Projection & Backbone Extraction Tool

A scientific tool for network backbone extraction from bipartite networks using 23 distinct methods. No Python installation or setup is required to run it.

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
3. Select a method from the 23 backbone extraction algorithms
4. Run the analysis by clicking Start
5. Download the results when the run completes

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
