#!/usr/bin/env bash
# ============================================================
# PRED-BiPN Server Launcher  (Linux / macOS)
# Works on a bare OS - no pre-installed Python required
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================"
echo " PRED-BiPN Bipartite Network Analysis Tool"
echo "============================================"
echo ""

# ----------------------------------------------------------
# Helper: find a working Python 3.9+ interpreter
# ----------------------------------------------------------
find_python() {
    # 1. Existing venv
    if [ -x ".venv/bin/python" ]; then echo ".venv/bin/python"; return 0; fi

    # 2. Common commands on PATH
    for cmd in python3 python python3.11; do
        if command -v "$cmd" >/dev/null 2>&1; then
            if "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

# ----------------------------------------------------------
# Helper: install Python via the OS package manager
# ----------------------------------------------------------
install_python() {
    echo "[INFO] Python 3 not found. Attempting automatic install ..."
    echo ""

    if [ "$(uname)" = "Darwin" ]; then
        # macOS
        if command -v brew >/dev/null 2>&1; then
            echo "[INFO] Installing via Homebrew ..."
            brew install python@3.11
        else
            echo "[INFO] Installing Homebrew first ..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            # Add brew to PATH for this session
            if [ -x "/opt/homebrew/bin/brew" ]; then
                eval "$(/opt/homebrew/bin/brew shellenv)"
            elif [ -x "/usr/local/bin/brew" ]; then
                eval "$(/usr/local/bin/brew shellenv)"
            fi
            brew install python@3.11
        fi
    else
        # Linux
        if command -v apt-get >/dev/null 2>&1; then
            echo "[INFO] Installing via apt ..."
            sudo apt-get update -qq
            sudo apt-get install -y python3 python3-venv python3-pip
        elif command -v dnf >/dev/null 2>&1; then
            echo "[INFO] Installing via dnf ..."
            sudo dnf install -y python3 python3-pip
        elif command -v yum >/dev/null 2>&1; then
            echo "[INFO] Installing via yum ..."
            sudo yum install -y python3 python3-pip
        elif command -v pacman >/dev/null 2>&1; then
            echo "[INFO] Installing via pacman ..."
            sudo pacman -Sy --noconfirm python python-pip
        elif command -v zypper >/dev/null 2>&1; then
            echo "[INFO] Installing via zypper ..."
            sudo zypper install -y python3 python3-pip
        else
            echo "[ERROR] Could not detect package manager."
            echo "        Please install Python 3.11+ manually and re-run this script."
            exit 1
        fi
    fi
}

# ===========================================================
# Fast path: venv already exists → skip to launch
# ===========================================================
if [ -x ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
else
    # STEP 1 - Find Python
    PYTHON_EXE=$(find_python) || true

    # STEP 2 - Install if missing
    if [ -z "$PYTHON_EXE" ]; then
        install_python
        PYTHON_EXE=$(find_python) || true
        if [ -z "$PYTHON_EXE" ]; then
            echo "[ERROR] Python installation finished but cannot be found."
            echo "        Please open a new terminal and try again."
            exit 1
        fi
        echo "[OK]   Python installed!"
        echo ""
    fi

    echo "[OK]   Using Python: $PYTHON_EXE"

    # STEP 3 - Create venv & install deps (first run)
    if [ ! -x ".venv/bin/python" ]; then
        echo "[INFO] Creating virtual environment ..."
        "$PYTHON_EXE" -m venv .venv

        echo "[INFO] Installing dependencies (first run, may take a few minutes) ..."
        .venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1
        .venv/bin/python -m pip install -r backend/requirements.txt
        echo "[OK]   Dependencies installed."
        echo ""
    fi
fi

# ===========================================================
# STEP 4 - Launch
# ===========================================================
mkdir -p backend/tmp

echo "[OK]   Starting server ..."
echo ""
echo "  URL:  http://localhost:5050"
echo "  Press Ctrl+C to stop the server"
echo ""

# Open browser in background (best-effort, ignore failures)
( sleep 2 && \
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "http://localhost:5050"; \
  elif command -v open >/dev/null 2>&1; then open "http://localhost:5050"; fi \
) >/dev/null 2>&1 &

source .venv/bin/activate
cd backend
# PYTHONUTF8=1 ensures correct encoding on any locale
export PYTHONUTF8=1
"$SCRIPT_DIR/.venv/bin/python" app.py
