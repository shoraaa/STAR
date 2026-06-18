# STAR: Scoped Test-time Adaptation and Refinement for Neural Routing Solvers

This repository contains the **STAR** (Scoped Test-time Adaptation and Refinement) method implementation and evaluation framework, built on top of the [Neural Routing Survey](https://github.com/CIAM-Group/NRS_Survey).

## STAR Method

STAR is a test-time adaptation framework for frozen neural routing policies that enables improved inference without weight updates. It combines three key components:

- **Context-preserving perturbation**: Bounded local exploration around the current solution
- **Advantage-credit memory**: Sparse edge reinforcement based on improvement signals
- **Scope-restricted refinement**: Targeted local search within perturbed regions

### Key Results

| Benchmark | Avg Gap | Solved |
|-----------|---------|--------|
| Small TSP (69 instances) | 0.57% | 69/69 |
| Medium TSP (109 instances) | 5.91% | 109/109 |

STAR improves over construction-only baselines (LEHD RRC1K, L2C-Insert) and complements specialized methods (SIL PRC1K, DRHG).

## Repository Structure

```
NRS-Survey-STAR/
├── STAR/              # STAR method implementation
│   ├── STAR.cpp      # C++ core with kd-tree acceleration
│   ├── core.py       # Python interface and STARStrategy
│   └── kd_tree.h    # KD-tree for efficient neighbor queries
├── survey/            # Original Neural Routing Survey
│   ├── NRS/          # Various NRS methods (LEHD, SIL, DRHG, etc.)
│   ├── heuristics/    # Baseline heuristics
│   └── tests/        # Test suite including STAR tests
├── references/         # Reference implementations
├── README.md          # This file
├── paper.tex          # STAR paper (LaTeX)
└── requirements.txt    # Dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NRS-Survey-STAR.git
cd NRS-Survey-STAR

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build STAR C++ extension
python setup.py build_ext --inplace
```

## Pretrained Assets

The pretrained checkpoints are not stored in normal Git because the full asset
set is large. The repo tracks `pretrained_assets_manifest.csv`, which records
the expected checkpoint paths, file sizes, and SHA-256 hashes, plus
`scripts/pretrained_assets.py`, which packs, unpacks, and verifies the archive.

On a machine that already has the pretrained files, create the transfer archive
from the repository root:

```bash
uv run python scripts/pretrained_assets.py verify
uv run python scripts/pretrained_assets.py pack --verify-first
```

This writes `pretrained-assets.tar.gz` in the repository root. The archive is
ignored by Git, so move it separately to the machine where you cloned the repo.
For example:

```bash
scp pretrained-assets.tar.gz user@new-machine:/path/to/NRS-Survey-STAR/
```

On the new machine, extract and verify it from the cloned repository root:

```bash
cd /path/to/NRS-Survey-STAR
uv run python scripts/pretrained_assets.py unpack --verify-after
```

If `uv` is not installed there, the helper also works with Python directly:

```bash
python scripts/pretrained_assets.py unpack --verify-after
python scripts/pretrained_assets.py verify
```

For the short standalone reference, see [`PRETRAINED_ASSETS.md`](PRETRAINED_ASSETS.md).

## Usage

### Running Original NRS Methods

The `original.py` wrapper is intended to run all listed NRS method entrypoints.
For a fresh clone to do that, the method source directories under `survey/NRS/`
must be committed to Git, and the large checkpoints/data files must be restored
with the pretrained-assets archive above.

```bash
uv run original.py
```

By default, the all-method run continues after individual method failures and
records each status in `results/original-*/original_results.csv`. To stop at
the first failed method:

```bash
uv run original.py --fail-fast
```

### Running STAR Inference

```python
from STAR.core import STARStrategy
from nrs_experiment.core import Instance

# Load your TSP/CVRP instance
instance = Instance(
    name="my-instance",
    problem="tsp",  # or "cvrp"
    coords={...},
    demands={...},
    capacity=None,  # for TSP
    bks_cost=None
)

# Create STAR strategy
star = STARStrategy(
    iterations=90,
    min_new_edges=16,
    refine_k=64,
    memory=True,
    memory_update_mode="advantage-introduced"
)

# Run on frozen policy
policy = YourNeuralPolicy()
initial_cost, final_cost, valid = star.run(instance, policy, rng)
```

### Evaluating on Benchmarks

```bash
# TSP evaluation
python neural_swap.py lehd_rrc,sil_prc lehd,sil tsp small

# CVRP evaluation
python neural_swap.py lehd_rrc,sil_prc lehd,sil cvrp small
```

## Original Survey

This work builds on the [Neural Routing Survey](https://github.com/CIAM-Group/NRS_Survey) paper:

> Ba, Y., Lin, X., Zhou, C., Zheng, R., Wang, Z., Liang, X., Lu, Z., Sun, J., Qian, Y., & Zhang, Q. (2026). Survey on Neural Routing Solvers. *arXiv preprint arXiv:2602.21761*.

The complete survey implementation and evaluation pipeline can be found in the [`survey/`](survey/) directory.

## Citation

**If this repository is helpful for your research, please cite:**

### STAR Method (when published)
```bibtex
@article{your2026star,
  title={STAR: Scoped Test-time Adaptation and Refinement for Neural Routing Solvers},
  author={Your Name and Collaborators},
  journal={arXiv preprint},
  year={2026}
}
```

### Original Survey
```bibtex
@article{ba2026survey,
  title={Survey on Neural Routing Solvers},
  author={Ba, Yunpeng and Lin, Xi and Zhou, Changliang and Zheng, Ruihao and Wang, Zhenkun and Liang, Xinyan and Lu, Zhichao and Sun, Jianyong and Qian, Yuhua and Zhang, Qingfu},
  journal={arXiv preprint arXiv:2602.21761},
  year={2026}
}
```

## License

The code can only be used for non-commercial purposes. Please contact the authors if you want to use this code for business matters.

Original survey code: Copyright (c) 2026 CIAM Group  
STAR method: Copyright (c) 2026 [Your Name/Group]

## Acknowledgements

The STAR implementation builds upon the code and evaluation pipeline from the [Neural Routing Survey](https://github.com/CIAM-Group/NRS_Survey). Thanks to the CIAM Group for their comprehensive survey and open-source implementation.

### Sources of Adopted Methods

See the [`survey/README.md`](survey/README.md) for the complete list of adopted methods and benchmarks.
