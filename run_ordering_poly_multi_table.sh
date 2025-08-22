#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV=".venv_ordering_poly_multi_table"
OUTDIR="output_poly_multi_table"

$PYTHON -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install networkx pandas numpy requests matplotlib ollama

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not installed. Install from https://ollama.com"
  exit 1
fi

if ! pgrep -x "ollama" >/dev/null 2>&1; then
  echo "[*] Starting Ollama server…"
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  sleep 5
fi

echo "[*] Pulling models… (first time can be large)"
ollama pull llama3.2
ollama pull gemma3:12b

python ordering_experiment_poly_multi_table.py \
  --graphs 25 \
  --nodes 50 \
  --p 0.1 \
  --models "llama3.2,gemma3:12b" \
  --output-dir "$OUTDIR"

echo "[*] Done. See:"
echo "    $OUTDIR/results_table.csv        # matches your diagram (E, P, A% under UO/O/OE for Degree & BFS, per model)"
echo "    $OUTDIR/results_long.csv         # tidy (useful for plotting)"
echo "    $OUTDIR/transcript.jsonl         # prompts + LLM outputs + truths"
echo "    $OUTDIR/graph_0.png ... graph_24.png"

