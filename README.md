# ARB: AttriReBoost

### The official implementation of **Attrireboost: A gradient-free propagation optimization method for cold start mitigation in attribute missing graphs**.

#### This repository provides PyTorch / PyG code to (1) **estimate missing node attributes** by propagation and (2) **evaluate downstream tasks** (classification) on standard graph benchmarks.

---

## ✨ What’s inside

- **ARB/APA model** for iterative, gradient‑free attribute propagation (`ARB/models/apa.py`)
- **GNN baselines** (GCN/GAT/GraphSAGE) and a simple **MLP** (`ARB/models/gnn.py`, `ARB/models/mlp.py`)
- **Datasets loader** for Planetoid/Amazon/Coauthor families (`ARB/data/data.py`)
- **Metrics** for classification and top‑k ranking (Recall@k, nDCG, RMSE, CORR) (`ARB/metrics.py`)
- **Validators** to reproduce paper‑style runs:
  - Estimation search & early‑stopping (`ARB/validators/estimation.py`)
  - Classification benchmarking (`ARB/validators/classification.py`)
  - SGD/analytical comparisons (`ARB/validators/sgd.py`)
- **Utilities** for parameter search & score recording (`ARB/utils.py`)

Directory tree:
```
ARB/
├── data/
│   ├── __init__.py
│   ├── data.py
│   └── plt.py
├── models/
│   ├── __init__.py
│   ├── apa.py
│   ├── gnn.py
│   └── mlp.py
├── validators/
│   ├── __init__.py
│   ├── classification.py
│   ├── estimation.py
│   ├── path_test.py
│   └── sgd.py
├── metrics.py
├── utils.py
├── plt.py
└── validator.py
```

> Minimal original README has been expanded into a complete guide based on the codebase structure.

---

## 📦 Environment

- Python ≥ 3.8 (3.9/3.10 also fine)
- PyTorch (CUDA or CPU)
- PyTorch Geometric & friends

Suggested install (pick CUDA version that matches your system; see PyTorch/PyG docs if needed):

```bash
# 1) Install PyTorch (example: CPU only)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Install PyG dependencies (torch-scatter/torch-sparse/torch-cluster/pyg)
#    For CUDA builds, see https://pytorch-geometric.readthedocs.io/
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# 3) General scientific stack
pip install numpy pandas scikit-learn scipy tqdm
```

> If wheels are not found for your CUDA/PyTorch combination, follow the official PyG instructions to select the proper extra index URLs.

---

## 📚 Datasets

`ARB/data/data.py` uses **PyTorch Geometric** to automatically download and prepare common benchmarks:

- **Planetoid**: `Cora`, `CiteSeer`, `PubMed`
- **Amazon**: `Computers`, `Photo`
- **Coauthor**: `CS`, `Physics`

The loader also creates train/val/test masks and supports both **binary/multi‑class** classification and **continuous** targets (for RMSE/CORR).

---

## 🚀 Quick Start

### 1) Clone & prepare
```bash
git clone https://github.com/limengran98/ARB.git
cd ARB
```

### 2) Run the default validator

The repository bundles a convenience entry in `ARB/validator.py`. You can invoke it as a module:

```bash
# Run estimation validator (example inside validator.py: Cora, early-stop, up to 30 iters)
python -m ARB.validator
```

This will search ARB hyper‑parameters on the specified dataset and print the best score and iteration. Edit `ARB/validator.py` to switch presets or datasets.

---

## 🧪 Reproducing the validators

You can call the validators directly to reproduce the paper‑style pipelines.

### Estimation (hyper‑param search + early stop)
```python
from ARB.validators import EstimationValidator

# grid/random search over {alpha, beta, gamma, num_iter} on one or many datasets
EstimationValidator.multi_run(
    file_name="combine_k10_le30",          # output scores file (npy/csv via utils.Scores)
    dataset_names=["cora"],                # or multiple: ["cora","citeseer","pubmed"]
    max_num_iter=30,
    early_stop=True,
    k_index=-1                             # see data/metrics for available metrics/ks
)
```

### Classification (downstream evaluation)
```python
from ARB.validators import ClassificationValidator

ClassificationValidator.run(
    file_name="class_k10_le30",
    est_scores_file_name="combine_k10_le30",   # plug in estimation results
    val_only_once=False,
    run_algos=["fp", "gcn", "gat", "sage"]     # see classification.py for options
)
```

### SGD / analytical comparisons
```python
from ARB.validators import SGDValidator
SGDValidator.compare(
    dataset_name="cora",
    apa=...,
    val_nodes=...,
    params=dict(alpha=0.6, beta=0.3, gamma=0.1),
    metric="Recall",
    k=10,
    epochs=200
)
```

Outputs are recorded with `utils.Scores` into `.npy` and `.csv` files for easy analysis.

---

## 🔧 Tips & Troubleshooting

- **PyG installation** is the most common hurdle. Ensure your PyTorch and CUDA versions match the downloaded wheels.
- For **CPU‑only** environments, install the CPU wheels for torch and PyG.
- If you see import errors for `torch_scatter`/`torch_sparse`, reinstall them after PyTorch is installed.
- Some example presets in `ARB/validator.py` are commented—uncomment and adjust to your experiments.

---

## 🖇️ Citation

If you find this work useful in your research, please cite the ARB paper.

```bibtex
@article{li2025attrireboost,
  title={Attrireboost: A gradient-free propagation optimization method for cold start mitigation in attribute missing graphs},
  author={Li, Mengran and Ding, Chaojun and Chen, Junzhou and Xing, Wenbin and Ye, Cong and Zhang, Ronghui and Zhuang, Songlin and Hu, Jia and Qiu, Tony Z and Gao, Huijun},
  journal={arXiv preprint arXiv:2501.00743},
  year={2025}
}
```
---

## 🤝 Acknowledgements

This project builds on the excellent **PyTorch** and **PyTorch Geometric** ecosystems.
