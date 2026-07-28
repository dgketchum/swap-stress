#!/usr/bin/env bash
#
# Reproduce a SWAP-Stress release, stage 00 through 08.
#
#   ./reproduce.sh --dry-run              resolve every path and config, run nothing
#   ./reproduce.sh --from 03 --to 06      run a slice of the chain
#   ./reproduce.sh                        run the whole thing
#
# Stages 00-02 rebuild the training table from the raw sources; 03-04 train and
# validate; 05-08 produce and package the gridded product. Stage 01 talks to
# Earth Engine and 05 is long-running, so the usual invocation is a slice, not
# the full chain. See docs/REPRODUCE.md for what each stage needs.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS="${REPO}/configs"

# The stage commands are console scripts installed by `uv sync`. Going through
# `uv run` means the chain works without activating the venv first; set
# RUNNER="" if the console scripts are already on PATH.
IFS=' ' read -r -a RUNNER <<< "${RUNNER-uv run}"

RELEASE="${RELEASE:-global_pruned_refresh_20260520}"
DATA_ROOT="${DATA_ROOT:-/nas/soils}"
MODEL_DIR="${MODEL_DIR:-${DATA_ROOT}/swapstress/models/direct_rf_9km_global_pruned}"
FIG_DIR="${FIG_DIR:-${REPO}/figs/descriptor}"

RELEASE_DIR="${RELEASE_DIR:-${DATA_ROOT}/swapstress/releases/${RELEASE}}"
INFERENCE_DIR="${INFERENCE_DIR:-${RELEASE_DIR}/inference}"   # Level 1
GAPFILL_DIR="${GAPFILL_DIR:-${RELEASE_DIR}/gapfill}"         # Level 2
PRODUCT_DIR="${PRODUCT_DIR:-${RELEASE_DIR}/product}"

# The deposit container. NetCDF drops the linear suction band by default, since
# it is an exact transform of the log band and compresses worst; a GeoTIFF run
# keeps it, because that form is for GIS users reading single bands.
CONTAINER="${CONTAINER:-netcdf}"
if [[ "${CONTAINER}" == "netcdf" ]]; then
  DROP_LINEAR_SUCTION="${DROP_LINEAR_SUCTION---drop-linear-suction}"
else
  DROP_LINEAR_SUCTION="${DROP_LINEAR_SUCTION-}"
fi

TRAIN_CONFIG="${TRAIN_CONFIG:-${CONFIGS}/train_9km_global_pruned.toml}"
PREDICT_CONFIG="${PREDICT_CONFIG:-${CONFIGS}/predict_9km_global_pruned.toml}"
GAPFILL_CONFIG="${GAPFILL_CONFIG:-${CONFIGS}/gapfill_9km_global_pruned.toml}"

DRY_RUN=""
FROM="00"
TO="08"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Stage numbers are zero-padded, so string comparison orders them correctly.
in_range() {
  [[ ! "$1" < "$FROM" && ! "$1" > "$TO" ]]
}

# Run stage $1 with the remaining arguments, unless the caller asked for a slice
# that excludes it.
stage() {
  local number="$1"; shift
  in_range "$number" || return 0
  echo ""
  echo "──────────────────────────────────────────────────────────────────────"
  echo "  stage $number: $*"
  echo "──────────────────────────────────────────────────────────────────────"
  "${RUNNER[@]}" "$@" ${DRY_RUN:+$DRY_RUN}
}

echo "release:    ${RELEASE}"
echo "data root:  ${DATA_ROOT}"
echo "model dir:  ${MODEL_DIR}"
echo "stages:     ${FROM}..${TO}${DRY_RUN:+  (dry run)}"

stage 00 swapstress-standardize --data-root "${DATA_ROOT}"

# Stage 01 in two steps: the export starts Earth Engine batch tasks that write
# to Cloud Storage, and the tables step needs those CSVs synced down first. A
# real run stops here until the tasks finish; a dry run walks straight through.
stage 01 swapstress-extract --step export --data-root "${DATA_ROOT}"
if [[ -z "$DRY_RUN" ]] && in_range 01; then
  echo ""
  echo "Earth Engine tasks submitted. Sync the exported CSVs to"
  echo "  ${DATA_ROOT}/swapstress/inference/global_features/<source>/"
  echo "before the tables step below can find them."
fi
stage 01 swapstress-extract --step tables --data-root "${DATA_ROOT}"

stage 02 swapstress-build-table --data-root "${DATA_ROOT}"

stage 03 swapstress-train --config "${TRAIN_CONFIG}" --output-dir "${MODEL_DIR}"

stage 04 swapstress-validate --model-dir "${MODEL_DIR}"

stage 05 swapstress-predict --config "${PREDICT_CONFIG}"

stage 06 swapstress-gapfill --config "${GAPFILL_CONFIG}"

# Level 2 is what gets released; deriving its per-pixel gapfill_flag needs the
# Level 1 rasters alongside, which is why both directories are passed. The
# deposit is time-stacked NetCDF, one file per year, without the redundant
# linear suction band; set CONTAINER=geotiff for the per-day GIS form.
stage 07 swapstress-package \
  --source-dir "${GAPFILL_DIR}" \
  --level1-dir "${INFERENCE_DIR}" \
  --output-dir "${PRODUCT_DIR}" \
  --level 2 \
  --container "${CONTAINER}" \
  ${DROP_LINEAR_SUCTION}

stage 08 swapstress-figures --output-dir "${FIG_DIR}"

echo ""
echo "Done${DRY_RUN:+ (dry run -- nothing was written)}."
