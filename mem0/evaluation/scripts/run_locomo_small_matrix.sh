#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_locomo_multihop_small_matrix.sh"
bash "${SCRIPT_DIR}/run_locomo_singlehop_small_matrix.sh"
