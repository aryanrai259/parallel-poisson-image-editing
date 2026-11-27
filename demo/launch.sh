#!/usr/bin/env bash
# Start the demo at http://localhost:8501 (open it in the Windows browser when running in WSL).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi
if [ ! -f bench/data/square6.png ]; then
  python bench/prepare_data.py
fi
exec streamlit run demo/app.py \
  --server.headless true \
  --server.port "${PORT:-8501}" \
  --browser.gatherUsageStats false
