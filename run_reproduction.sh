#!/usr/bin/env bash
# ==============================================================================
# Immuno-Metabolic Stratification in Schizophrenia - Reproduction Pipeline
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo " Starting Full Epidemiological & Statistical Reproduction Pipeline"
echo "========================================================================"

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Error: python3 is not found in PATH."
    exit 1
fi

echo "[1/7] Running Phase 1 & 2: Data Audit and Preprocessing..."
"$PYTHON_BIN" scripts/01_audit_and_preprocess.py

echo "[2/7] Running Phase 3: Case-Control Descriptive Comparisons..."
"$PYTHON_BIN" scripts/02_case_control_descriptive.py

echo "[3/7] Running Phase 3: Primary Nested Poisson & Firth Logistic Models..."
"$PYTHON_BIN" scripts/03_primary_models.py

echo "[4/7] Running Phase 3: Multi-Marker Synergy & Additive Interaction (RERI)..."
"$PYTHON_BIN" scripts/04_synergy_and_interaction.py

echo "[5/7] Running Phase 3: Gaussian Mixture Modeling & Phenotyping..."
"$PYTHON_BIN" scripts/05_clustering_and_phenotyping.py

echo "[6/7] Running Phase 3: Sensitivity Audits & Incremental Discrimination..."
"$PYTHON_BIN" scripts/06_sensitivity_and_discrimination.py

echo "[7/7] Generating Publication Figures and Summary Tables..."
"$PYTHON_BIN" scripts/07_generate_figures_and_tables.py

echo "========================================================================"
echo " Reproduction pipeline completed successfully!"
echo " Results, figures, and tables are available in the 'output/' directory."
echo "========================================================================"
