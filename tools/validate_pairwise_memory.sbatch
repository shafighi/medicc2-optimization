#!/bin/bash
#SBATCH --job-name=medicc2-pairwise-validate
#SBATCH --output=slurm-%x-%j.out
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/slurm_pairwise_validation_${SLURM_JOB_ID:-manual}}"

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

echo "Repository: $REPO_ROOT"
echo "Output directory: $OUT_DIR"
echo "Python: $PYTHON_BIN"
echo "MEDICC2_PAIRWISE_MODE=${MEDICC2_PAIRWISE_MODE:-forked}"
echo "MEDICC2_PAIRWISE_BATCH_SIZE=${MEDICC2_PAIRWISE_BATCH_SIZE:-3}"
echo "MEDICC2_PAIRWISE_WORKERS=${MEDICC2_PAIRWISE_WORKERS:-1}"

export MEDICC2_PAIRWISE_MODE="${MEDICC2_PAIRWISE_MODE:-forked}"
export MEDICC2_PAIRWISE_BATCH_SIZE="${MEDICC2_PAIRWISE_BATCH_SIZE:-3}"
export MEDICC2_PAIRWISE_WORKERS="${MEDICC2_PAIRWISE_WORKERS:-1}"

$PYTHON_BIN -m pytest medicc/test_medicc.py \
  -k "shorten_cn_strings or pairwise_distance_matrix_matches_previous_reference or pairwise_distance_matrix_forked_batches_match_previous_reference or calc_MED_distance_is_not_unbounded_lru_cache" \
  | tee "$OUT_DIR/pytest.log"

$PYTHON_BIN tools/pairwise_memory_probe.py --mode compare \
  | tee "$OUT_DIR/compare.json"

if [ -x "$TIME_BIN" ]; then
  "$TIME_BIN" -v "$PYTHON_BIN" tools/pairwise_memory_probe.py --mode run --impl reference \
    > "$OUT_DIR/reference.json" 2> "$OUT_DIR/reference.time"
  "$TIME_BIN" -v "$PYTHON_BIN" tools/pairwise_memory_probe.py --mode run --impl optimized \
    > "$OUT_DIR/optimized.json" 2> "$OUT_DIR/optimized.time"

  echo "Peak RSS summary:"
  grep -h "Maximum resident set size" "$OUT_DIR/reference.time" "$OUT_DIR/optimized.time" || true
else
  echo "Warning: $TIME_BIN not available; skipping peak RSS measurement" | tee "$OUT_DIR/time-warning.log"
  "$PYTHON_BIN" tools/pairwise_memory_probe.py --mode run --impl reference \
    | tee "$OUT_DIR/reference.json"
  "$PYTHON_BIN" tools/pairwise_memory_probe.py --mode run --impl optimized \
    | tee "$OUT_DIR/optimized.json"
fi

echo "Validation artifacts written to $OUT_DIR"
