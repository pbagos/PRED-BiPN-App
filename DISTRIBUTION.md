# Distribution Guide

How to share PRED-BiPN with other users.

---

## What to Ship

You only need to share these files/folders:

```
PRED-BiPN-App/
├── RUN_APP.bat
├── run_app.sh
├── backend/
├── frontend/
├── Procfile
├── runtime.txt
└── README.md
```

**Do NOT include `.venv/`** — it is machine-specific (contains hardcoded paths). The launcher scripts create it automatically on the recipient's machine on first run.

---

## How to Share

### Option 1: Zip File (Recommended)
1. Right-click `PRED-BiPN-App` folder
2. Send to → Compressed (zipped) folder
3. Share the zip (approx. 2–5 MB without `.venv`)

### Option 2: Cloud Storage
Upload to Google Drive, Dropbox, or OneDrive and share the link.

### Option 3: GitHub Release
Attach the zip to a GitHub release.

---

## What Happens on First Run

When a user runs `RUN_APP.bat` on a fresh machine:

1. Script checks for a working Python 3 install
2. If not found: installs Python 3.11 automatically via `winget` or direct download from python.org (no admin rights required)
3. Creates `.venv` and installs all dependencies from `requirements.txt`
4. Launches the server and opens the browser

**Subsequent runs skip steps 1–3** and launch in seconds.

---

## User Instructions (copy-paste this)

### Windows
1. Extract `PRED-BiPN-App.zip`
2. Double-click `RUN_APP.bat`
3. On first run, wait a few minutes while dependencies install
4. Browser opens automatically at http://localhost:5050

### Mac / Linux
1. Extract `PRED-BiPN-App.zip`
2. Open Terminal in the folder
3. Run: `chmod +x run_app.sh && ./run_app.sh`
4. Browser opens at http://localhost:5050

---

## Troubleshooting for Recipients

| Problem | Solution |
|---|---|
| "Python not found" error | Script installs it automatically; if it fails, install Python 3.11 from https://www.python.org/downloads/ and re-run |
| App worked on one machine but not another | Delete the `.venv` folder and re-run the launcher — it rebuilds for the current machine |
| Port 5050 in use | Close the other app using port 5050 and try again |
| Slow first run | Normal — dependencies (~500MB) are installing once |

---

## Updating the App

1. Make changes to `backend/*.py` or `frontend/` files
2. Delete `.venv/` before re-zipping (it will be rebuilt on first run)
3. Zip and redistribute
