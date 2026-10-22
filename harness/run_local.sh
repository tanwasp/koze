#!/usr/bin/env bash
# Run the Kreol ASR harness on a Mac (Apple Silicon) or Linux box.
# First run: creates .venv and installs everything (a few minutes).
# Then opens http://127.0.0.1:7860 in your browser. Extra args go to app.py,
# e.g.  ./run_local.sh --device cpu   or   ./run_local.sh --mock
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname)" == "Darwin" ]]; then
  command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
  brew list libsndfile >/dev/null 2>&1 || brew install libsndfile
  brew list python@3.12 >/dev/null 2>&1 || brew install python@3.12
  PY="$(brew --prefix python@3.12)/bin/python3.12"
  # Let PyTorch run unsupported Apple-GPU ops on the CPU instead of failing.
  export PYTORCH_ENABLE_MPS_FALLBACK=1
else
  PY="$(command -v python3.12 || command -v python3)"
fi

if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python app.py "$@"
