#!/usr/bin/env bash
set -euo pipefail

# Run normally to detach after launch. Pass --wait when the caller must remain
# attached until the five refits finish (for example, an automated agent).
regional_cv_result=/nas/soils/swapstress/releases/v03_20260729/evaluation/regional_cv_results_major_conus.csv
regional_cv_log=/nas/soils/swapstress/releases/v03_20260729/logs/regional_cv_major_conus.log

if pgrep -f '[s]wapstress.validation.regional_cv.*--evaluation-domain conus' >/dev/null; then
    echo "A CONUS regional-CV refit is already running."
    exit 1
fi

if [[ -e "${regional_cv_result}" ]]; then
    echo "Refusing to overwrite existing result: ${regional_cv_result}"
    exit 1
fi

nohup /home/dgketchum/.local/bin/uv run --project /home/dgketchum/code/swap-stress python -u -m swapstress.validation.regional_cv --model-dir /nas/soils/swapstress/models/direct_qrf_9km_global_pruned --output-dir /nas/soils/swapstress/releases/v03_20260729/evaluation --level major --evaluation-domain conus --n-estimators 250 --min-samples 100 --n-jobs -1 > "${regional_cv_log}" 2>&1 < /dev/null &
regional_cv_pid=$!

echo "PID: ${regional_cv_pid}"
echo "Monitor: tail -f ${regional_cv_log}"

if [[ "${1:-}" == "--wait" ]]; then
    wait "${regional_cv_pid}"
fi
