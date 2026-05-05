#!/usr/bin/env bash

set -u

# Sequential reproduction runner for scripts.md.
# Each run.py invocation writes its own stream log to <out-dir>/out.log.

if [[ -n "${STAR_RUNNER:-}" ]]; then
  read -r -a RUN_CMD <<< "${STAR_RUNNER}"
else
  RUN_CMD=(uv run run.py)
fi

total=0
failed=()

run_experiment() {
  local name="$1"
  shift
  total=$((total + 1))

  printf '\n[%03d] START %s\n' "${total}" "${name}"
  printf '      CMD:'
  printf ' %q' "$@"
  printf '\n'

  if "$@"; then
    printf '[%03d] OK    %s\n' "${total}" "${name}"
  else
    local status=$?
    printf '[%03d] ERROR %s (exit %d); continuing\n' "${total}" "${name}" "${status}" >&2
    failed+=("${name} (exit ${status})")
  fi
}

run_star() {
  run_experiment "$1" "${RUN_CMD[@]}" "${@:2}"
}

run_star "table1-small-main" \
  STAR sil tsp,cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --out-dir results/small-main

run_star "table1-medium-main" \
  STAR sil tsp,cvrp \
  --size medium \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 8 \
  --out-dir results/medium-main

run_star "table2-small-sampling" \
  STAR sil tsp,cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --no-STAR-memory \
  --out-dir results/small-sampling

run_star "table2-medium-sampling" \
  STAR sil tsp,cvrp \
  --size medium \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 8 \
  --no-STAR-memory \
  --out-dir results/medium-sampling

run_star "table2-small-recon" \
  STAR sil tsp,cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --no-STAR-memory \
  --no-refine \
  --out-dir results/small-recon

run_star "table2-medium-recon" \
  STAR sil tsp,cvrp \
  --size medium \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 8 \
  --no-STAR-memory \
  --no-refine \
  --out-dir results/medium-recon

# STAR_progress.csv in each out-dir records the full cost/gap curve for
# completed_iterations=0..100, so a single max-budget run replaces separate
# 10/20/40/60/100 reruns.
run_star "fig1-medium-star-100" \
  STAR sil tsp,cvrp \
  --size medium \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 8 \
  --out-dir results/scaling/medium-star-100

run_star "fig1-medium-nomem-100" \
  STAR sil tsp,cvrp \
  --size medium \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 8 \
  --no-STAR-memory \
  --out-dir results/scaling/medium-nomem-100

run_star "table3-ablation-full" \
  STAR sil tsp,cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --out-dir results/ablation/full

run_star "table3-ablation-no-memory" \
  STAR sil tsp,cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --no-STAR-memory \
  --out-dir results/ablation/no-memory

run_star "table3-ablation-no-refinement" \
  STAR sil tsp,cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --no-refine \
  --out-dir results/ablation/no-refinement

run_star "table4-memory-source" \
  STAR sil tsp,cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --STAR-memory-update-mode source \
  --out-dir results/memory/source

run_star "table4-memory-source-adv" \
  STAR sil tsp,cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --STAR-memory-update-mode source-advantage \
  --out-dir results/memory/source-adv

for scale in 0 0.25 0.5 1.0 2.0 4.0; do
  run_star "table5-memory-scale-${scale}" \
    STAR sil tsp,cvrp \
    --size small \
    --iterations 100 \
    --STAR-samples 100 \
    --min-new-edges 16 \
    --STAR-advantage-scale "${scale}" \
    --out-dir "results/memory/scale-${scale}"
done

for r in 4 8 16 24 32 64; do
  run_star "fig3-sens-r-${r}" \
    STAR sil tsp,cvrp \
    --size small \
    --iterations 100 \
    --STAR-samples 100 \
    --min-new-edges "${r}" \
    --out-dir "results/sens/r-${r}"
done

for k in 8 16 32 64 128; do
  run_star "fig3-sens-k-${k}" \
    STAR sil tsp,cvrp \
    --size small \
    --iterations 100 \
    --STAR-samples 100 \
    --min-new-edges 24 \
    --refine-k "${k}" \
    --out-dir "results/sens/k-${k}"
done

# run_star "table1-large-main" \
#   STAR sil tsp,cvrp \
#   --size large \
#   --iterations 100 \
#   --STAR-samples 100 \
#   --min-new-edges 4 \
#   --out-dir results/large-main

for backbone in nearest sil lehd; do
  run_star "table7-backbone-${backbone}-small" \
    STAR "${backbone}" tsp,cvrp \
    --size small \
    --iterations 100 \
    --STAR-samples 100 \
    --min-new-edges 24 \
    --out-dir "results/backbone/${backbone}/small"

  run_star "table8-backbone-${backbone}-medium" \
    STAR "${backbone}" tsp,cvrp \
    --size medium \
    --iterations 100 \
    --STAR-samples 100 \
    --min-new-edges 8 \
    --out-dir "results/backbone/${backbone}/medium"
done

printf '\nFinished %d experiments.\n' "${total}"
if (( ${#failed[@]} > 0 )); then
  printf 'Failed experiments:\n' >&2
  printf '  - %s\n' "${failed[@]}" >&2
  exit 1
fi

printf 'All experiments completed successfully.\n'
