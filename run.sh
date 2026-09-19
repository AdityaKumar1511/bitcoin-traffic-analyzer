#!/usr/bin/env bash
# ==============================================================================
# Bitcoin Traffic Analyzer - Full Execution Script
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "    🔗 BITCOIN TRAFFIC ANALYZER (NTRO SIH26146) EXECUTION        "
echo "=================================================================="

# Check Python environment
if [ -d ".venv" ]; then
    PYTHON_EXEC=".venv/bin/python"
    STREAMLIT_EXEC=".venv/bin/streamlit"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
    STREAMLIT_EXEC="streamlit"
else
    echo "[-] Error: Python 3 not found. Please install Python 3.11+."
    exit 1
fi

# Ensure synthetic data exists
if [ ! -f "data/raw/synthetic_transactions.csv" ]; then
    echo "[+] Generating synthetic Bitcoin transaction dataset..."
    $PYTHON_EXEC data/synthetic_generator.py --output data/raw/synthetic_transactions.csv
fi

# Run Full End-to-End Analysis Pipeline
echo "[+] Executing end-to-end forensic analysis pipeline..."
$PYTHON_EXEC -m src.scoring --input data/raw/synthetic_transactions.csv --output data/processed

echo "[+] Analysis pipeline completed successfully!"
echo "[+] Processed leads saved to data/processed/scored_alerts.csv"

# If --no-dashboard flag is not passed, start Streamlit dashboard
if [ "$1" != "--no-dashboard" ]; then
    echo "=================================================================="
    echo "    🚀 Launching Streamlit Forensic Dashboard                    "
    echo "    URL: http://localhost:8501                                    "
    echo "=================================================================="
    $STREAMLIT_EXEC run dashboard/app.py --server.headless true
fi
