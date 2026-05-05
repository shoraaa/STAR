````markdown
# Commands for Reproducing STAR+SiL Results

## Table 1 (`tab:star-results`) — Main STAR+SiL Results

```bash
uv run run.py STAR sil cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --out-dir results/small-main

uv run run.py STAR sil cvrp \
  --size medium \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 8 \
  --out-dir results/medium-main
  
  
uv run run.py STAR sil cvrp \
  --size large \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 4 \
  --out-dir results/large-main
````

---

## Table 2 (`tab:expanded-comparison`) — Comparison Baselines

The paper draws baselines from prior published work, including LKH-3, SiL PRC, DRHG, and others. Those rows are cited values, not re-run.

The STAR+SiL row comes from the Table 1 commands above.

### Frozen sampling

No STAR refinement or memory.

```bash
uv run run.py STAR sil cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --no-STAR-memory \
  --out-dir results/small-sampling

uv run run.py STAR sil cvrp \
  --size medium \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 8 \
  --no-STAR-memory \
  --out-dir results/medium-sampling
```

### Broad reconstruction

STAR memory disabled, refinement enabled.

```bash
uv run run.py STAR sil cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --no-STAR-memory \
  --no-refine \
  --out-dir results/small-recon

uv run run.py STAR sil cvrp \
  --size medium \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 8 \
  --no-STAR-memory \
  --no-refine \
  --out-dir results/medium-recon
```

---

## Figure 1 (`fig:scaling`) — Inference Scaling Curves

### STAR+SiL scaling

```bash
uv run run.py STAR sil tsp,cvrp \
  --size medium \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 8 \
  --out-dir results/scaling/medium-star-100
```

`results/scaling/medium-star-100/STAR_progress.csv` records
`completed_iterations=0..100`, so the 10/20/40/60/100 budget points are read
from one run instead of rerunning shorter prefixes.

### Sampling only / no memory

```bash
uv run run.py STAR sil tsp,cvrp \
  --size medium \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 8 \
  --no-STAR-memory \
  --out-dir results/scaling/medium-nomem-100
```

---

## Table 3 (`tab:components`) — Ablation on Small

```bash
uv run run.py STAR sil cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --out-dir results/ablation/full

uv run run.py STAR sil cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --no-STAR-memory \
  --out-dir results/ablation/no-memory

uv run run.py STAR sil cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --refine-k 0 \
  --out-dir results/ablation/no-refinement
```

---

## Table 4 (`tab:memory-design`) and Table 5 (`tab:lambda`) — Memory Variants on Small

### Memory update modes

```bash
uv run run.py STAR sil cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --STAR-memory-update-mode source \
  --out-dir results/memory/source

uv run run.py STAR sil cvrp \
  --size small \
  --iterations 100 \
  --STAR-samples 100 \
  --min-new-edges 24 \
  --STAR-memory-update-mode source-advantage \
  --out-dir results/memory/source-adv
```

### Memory strength / lambda sweep

```bash
for scale in 0 0.25 0.5 1.0 2.0 4.0; do
  uv run run.py STAR sil cvrp \
    --size small \
    --iterations 100 \
    --STAR-samples 100 \
    --min-new-edges 16 \
    --STAR-advantage-scale $scale \
    --out-dir results/memory/scale-$scale
done
```

---

## Figure 3 (`fig:sensitivity-rk`) — Scope `r` and Width `K` on Small

### Perturbation scope `r`

```bash
for r in 4 8 16 24 32 64; do
  uv run run.py STAR sil cvrp \
    --size small \
    --iterations 100 \
    --STAR-samples 100 \
    --min-new-edges $r \
    --out-dir results/sens/r-$r
done
```

### Refinement width `K`

```bash
for k in 8 16 32 64 128; do
  uv run run.py STAR sil cvrp \
    --size small \
    --iterations 100 \
    --STAR-samples 100 \
    --min-new-edges 24 \
    --refine-k $k \
    --out-dir results/sens/k-$k
done
```

---

## Table 7 (`tab:backbones`) and Table 8 (`tab:ood`) — Plug-and-Play Across Backbones

```bash
for backbone in nearest sil lehd; do
  uv run run.py STAR $backbone cvrp \
    --size small \
    --iterations 100 \
    --STAR-samples 100 \
    --min-new-edges 24 \
    --out-dir results/backbone/${backbone}/small

  uv run run.py STAR $backbone cvrp \
    --size medium \
    --iterations 100 \
    --STAR-samples 100 \
    --min-new-edges 8 \
    --out-dir results/backbone/${backbone}/medium
done
```

---

## Key Flags Mapped to Paper Parameters

|Paper parameter|CLI flag|
|---|---|
|Outer iterations `B`|`--iterations`|
|Proposals per iteration `S`|`--STAR-samples`|
|Min new-edge threshold|`--min-new-edges`|
|Perturbation size `r`|`--min-new-edges`|
|Refinement width `K`|`--refine-k`|
|Memory strength `λ`|`--STAR-advantage-scale`|
|Memory update mode|`--STAR-memory-update-mode`|
|Disable memory|`--no-STAR-memory`|
|Start mode|`--STAR-start-mode {random,cost,hybrid}`|
