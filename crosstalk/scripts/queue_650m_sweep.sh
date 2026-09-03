#!/bin/bash
# Keep the 650M sweep going until all 194 assays are scored.
# The run is resumable (completed assays are skipped), so a crash costs only the
# assay in flight. Restarts are capped so a genuine, repeatable failure stops the
# loop instead of spinning on it forever.
cd /Users/saanviiyer/Downloads/CALTECH/RESEARCH/crosstalk || exit 1
OUT=results/reliability_forecast_full.csv
for attempt in $(seq 1 40); do
  n=$(( $(wc -l < "$OUT") - 1 ))
  [ "$n" -ge 194 ] && { echo "COMPLETE: $n/194 assays"; exit 0; }
  echo "=== attempt $attempt, $n/194 done, $(date '+%H:%M:%S') ==="
  ./.venv-glm/bin/python -u scripts/run_reliability_forecast_full.py \
      --model facebook/esm2_t33_650M_UR50D --batch-tokens 3000 \
      --out "$OUT" </dev/null 2>&1
  after=$(( $(wc -l < "$OUT") - 1 ))
  if [ "$after" -le "$n" ]; then
    echo "no progress on attempt $attempt; backing off"
    sleep 60
  fi
done
echo "gave up after 40 attempts at $(( $(wc -l < "$OUT") - 1 ))/194"
