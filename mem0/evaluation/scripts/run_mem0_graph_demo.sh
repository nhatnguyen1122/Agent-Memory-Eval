#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "run_mem0_graph_demo.sh is deprecated; launching the normal Mem0 demo instead." >&2
exec bash "${SCRIPT_DIR}/run_mem0_demo.sh"
