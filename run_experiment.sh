#!/bin/bash
set -e

# Configure endpoint (adjust if you use OpenRouter or LM Studio instead of Ollama)
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"

# Run the experiment
python3 graph_ordering_eval.py \
  --num-graphs 25 \
  --nodes 50 \
  --ordering degree \
  --model-a "llama3.2" \
  --model-b "gemma-3-12b" \
  --seed 42

