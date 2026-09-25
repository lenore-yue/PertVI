# PertVI: Disentangled Perturbation Variational Inference

[![PyPI version](https://img.shields.io/badge/pypi-v0.3.0-blue.svg)](https://pypi.org)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![scvi-tools](https://img.shields.io/badge/built%20with-scvi--tools-purple.svg)](https://scvi-tools.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PertVI** is a deep generative model built on `scvi-tools` designed specifically to disentangle conditional drug responses and genetic perturbation effects from intrinsic cell cycle dynamics and biological differentiation baselines in single-cell RNA-seq and Perturb-seq data.

Unlike general-purpose single-cell models, PertVI uses a hypothesis-driven, constrained architecture featuring **Split-Natural FiLM transitions**, a **bifurcated decoder head**, and **deterministic covariance orthogonality penalties**. It enables:
1. **Cell-Cycle Scrubbed States (`u_state`):** Isolates cellular differentiation, cancer state transitions, and metabolic baselines, completely cleaned of cell cycle confounding.
2. **Dedicated Cell-Cycle Manifolds (`u_cycle`):** Preserves continuous closed-loop cycle kinetics (G1 -> S -> G2/M) via hard gene masking and topological prior relaxation.
3. **Shielded Out-of-Distribution Bottlenecks (`v_ood`):** Captures rare, unexpected, or extreme perturbation-induced variance while strictly anchored to zero on control cells.
4. **Counterfactual Causal Scores (Coupled-CCS vs. State-CCS):** Quantifies true causal intervention effects by pushing shifted control manifolds through generative decoders.
5. **Path-Integrated Perturbation Momentum:** Calculates the average causal transcriptional drive along continuous developmental or activation trajectories, accelerated **>900×** via linear hidden-space operator factorization.

---

## 📑 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Installation](#-installation)
- [Quickstart: Raw Counts to Discovery](#-quickstart-raw-counts-to-discovery)
- [Biological Preprocessing Protocol](#-biological-preprocessing-protocol)
- [Downstream Analysis & Metric Decision Guide](#-downstream-analysis--metric-decision-guide)
  - [1. Empirical Latent State Displacement (ELSD)](#1-empirical-latent-state-displacement-elsd)
  - [2. Counterfactual Causal Scores (Coupled-CCS vs. State-CCS)](#2-counterfactual-causal-scores-coupled-ccs-vs-state-ccs)
  - [3. Path-Integrated Perturbation Momentum](#3-path-integrated-perturbation-momentum)
- [Batch Harmonization & Counterfactual Simulation](#-batch-harmonization--counterfactual-simulation)
- [Deep Learning Reference & Hyperparameters](#-deep-learning-reference--hyperparameters)
- [Hardware Sizing & Troubleshooting](#-hardware-sizing--troubleshooting)
- [Citation](#-citation)

---

## 🧬 Architecture Overview

PertVI partitions single-cell transcriptomic variance into three interpretable, mathematically orthogonal latent subspaces:

```
                      Raw Counts x in N^{B x G}
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
      Encoder_cycle        Encoder_state        Encoder_ood
     [q_m, q_v: 3d]       [q_m, q_v: 30d]     [q_m, q_v: 5-10d]
             │                   │                   │
      LayerNorm(u_cyc)    LayerNorm(u_state)    Dropout(p=0.3)
             │                   │                   │
             └─────────┬─────────┘                   │
                       ▼                             │
             SplitNaturalShiftFiLM                   │
          (Sample Embedding e_p in R^{16})           │
          z = (1 + gamma(e_p)) * LN(u) + beta(e_p)   │
                       │                             │
                       └──────────────┬──────────────┘
                                      ▼
                              BifurcatedDecoder
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
     Pathway A                    Pathway B                   Batch Bypass
  Linear Cycle Head            Dense Main Head              Linear Embedding
 (W_cyc * Mask_cyc)          (W_head * h_hidden)               (W_b * e_b)
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      ▼
                      Rate Synthesis: h_total in R^G
                       px_scale = Softmax(h_total)
                     px_rate  = exp(library) * px_scale
```

* **`u_cycle` (3–5 dims):** Minimal circular manifold ($S^1 \subset \mathbb{R}^3$) capturing cell cycle periodicity without overfitting.
* **`u_state` (20–30 dims):** Continuous representation of cell state, differentiation trajectories, and drug-free baseline biology.
* **`v_ood` (5–10 dims):** Constrained escape-hatch bottleneck. Anchored strictly to zero for unperturbed control cells via an $L_1$ penalty; absorbs unmodeled, extreme, or off-target perturbation variance.
* **`SplitNaturalShiftFiLM`:** Condition-dependent affine modulators with LayerNorm standardization and residual zero-initialization ($W_\gamma=0, W_\beta=0$), preventing manifold collapse.
* **`BifurcatedDecoder`:** Strictly enforces that cell cycle latents $z_{\text{cycle}}$ only regulate canonical cell cycle genes via a hard mask $M_{\text{cyc}}$, while non-cycle and state genes are modeled through the main dense network with a soft $L_1$ penalty on cycle connections.

---

## 🚀 Installation

Install PertVI in editable mode within your Python environment (Python $\ge$ 3.9):

```bash
git clone https://github.com/lenore-yue/PertVI.git
cd PertVI
pip install -e .
```

### Dependencies
- `python >= 3.9`
- `torch >= 2.0`
- `scvi-tools >= 1.0`
- `scanpy >= 1.9`
- `anndata >= 0.8`

---

## 💻 Quickstart: Raw Counts to Discovery

This self-contained, cut-and-paste runnable pipeline demonstrates the complete PertVI workflow:

```python
import scanpy as sc
import numpy as np
import pertvi

# 1. Load data with raw counts
adata = sc.datasets.pbmc3k()
adata.layers["counts"] = adata.X.copy()

# 2. Flag Cell Cycle genes (Mandatory in PertVI v3.0)
cycle_markers = ["PCNA", "MCM6", "TOP2A", "MKI67", "CDK1", "CCNB1"]
adata.var["is_cycle_gene"] = adata.var_names.isin(cycle_markers)

# 3. Setup Metadata & Binary Treatment Mask
adata.obs["batch"] = np.random.choice(["Batch_1", "Batch_2"], size=adata.n_obs)
adata.obs["perturbation"] = np.random.choice(["Control", "Drug_A", "Drug_B"], size=adata.n_obs)

# is_treated MUST be float: 0.0 for controls, 1.0 for treated
adata.obs["is_treated"] = (adata.obs["perturbation"] != "Control").astype(np.float32)

# 4. Register AnnData with PertVI
pertvi.PertVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="batch",
    sample_key="perturbation",
    treatment_mask_key="is_treated",
)

# 5. Initialize and Train Model
model = pertvi.PertVI(
    adata,
    n_latent_cycle=3,   # 3 dimensions capture G1/S/G2M ring topology
    n_latent_state=30,  # 30 dimensions capture biological state
    n_latent_ood=5,     # 5 dimensions provide a tight OOD bottleneck
    cycle_gene_mask_key="is_cycle_gene",
)

model.train(
    max_epochs=200,
    warmup_epochs=50,
    anneal_epochs=100,
    batch_size=256,
    early_stopping=True,
)

# 6. Extract Disentangled Latents
latents = model.get_latent_representation()
adata.obsm["X_u_cycle"] = latents["u_cycle"]
adata.obsm["X_u_state"] = latents["u_state"]
adata.obsm["X_v_ood"]   = latents["v_ood"]

# 7. Compute Counterfactual Causal Scores (Coupled-CCS)
gene_scores = pertvi.compute_ccs(
    adata,
    model,
    condition_key="batch",
    pert_key="perturbation",
    control_group="Control",
    inject_film=True,  # Coupled-CCS: combines latent shift + drug FiLM modulation
)

# 8. Save and Load Model
model.save("pertvi_trained_model", overwrite=True)
# loaded_model = pertvi.PertVI.load("pertvi_trained_model", adata=adata)
```

---

## 🔬 Biological Preprocessing Protocol

To ensure clean mathematical disentanglement without artifacts, follow these best practices:

### 1. Annotating Canonical Cell Cycle Genes
PertVI requires a boolean mask `adata.var['is_cycle_gene']`. Annotate using standard canonical marker sets (e.g. Tirosh 2016, Regev Lab, or Seurat `cc.genes`):

```python
# From Scanpy's built-in Regev Lab cell cycle gene set:
cycle_genes = sc.datasets.regev_lab_cell_cycle_genes()
s_genes = cycle_genes[:43]
g2m_genes = cycle_genes[43:]
all_cycle = list(set(s_genes + g2m_genes))

adata.var["is_cycle_gene"] = adata.var_names.isin(all_cycle)
print(f"Annotated {adata.var['is_cycle_gene'].sum()} cell cycle genes.")
```

### 2. Protecting Cycle Regulators During HVG Selection
Standard Highly Variable Gene (HVG) selection often filters out critical cyclins and checkpoint kinases with modest variance. Always retain the union of HVGs and cycle genes:

```python
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", layer="counts", subset=False)

# Retain the union of HVGs and cycle genes to preserve bifurcated decoder integrity:
adata = adata[:, (adata.var["highly_variable"] | adata.var["is_cycle_gene"])].copy()
```

### 3. Setting the Binary Treatment Mask (`is_treated`)
The treatment mask directly regulates PertVI's loss functions:
* **Unperturbed control cells** (`NTC`, `DMSO`, `Control`) **must strictly equal `0.0`** (float32).
* **Perturbed / treated cells** **must strictly equal `1.0`** (float32).

```python
control_labels = ["DMSO", "NTC", "Control"]
adata.obs["is_treated"] = np.where(
    adata.obs["perturbation"].isin(control_labels), 0.0, 1.0
).astype(np.float32)
```

> [!WARNING]
> Do not pass boolean strings (e.g. `"True"`/`"False"`) or mark all cells as `1.0`. Control cells must be `0.0` to trigger OOD $L_1$ zero-anchoring and enable shielded covariance constraints.

---

## 📊 Downstream Analysis & Metric Decision Guide

PertVI offers a specialized downstream suite for perturbation genomics. Select the metric appropriate for your biological inquiry:

| Metric | API Function | Mathematical Concept | Biological Application |
| :--- | :--- | :--- | :--- |
| **ELSD** (Empirical Latent State Displacement) | `compute_elsd` | `d_p(c) = E[u_p] - E[u_ctrl]` | Rapid exploratory screening, coarse compound clustering in latent space. |
| **Coupled-CCS** (Causal Pushforward w/ FiLM) | `compute_ccs(inject_film=True)` | `W_head · E[h(z_p) - h(z_ctrl)]` | **Primary metric**: total causal drug response (latent transition + drug FiLM modulation). |
| **State-CCS** (Pure State Pushforward) | `compute_ccs(inject_film=False)` | `W_head · E[h(z_state) - h(z_ctrl)]` | Isolates pure differentiation/state shift, decoupled from chemical stress or off-target toxicity. |
| **Linear Momentum** (Algorithm A, default) | `compute_momentum_score(style="linear")` | Factored path integral along `u_start` -> `u_end` | Quantifies average perturbation drive across discrete states (e.g. Rest $\to$ Stimulated). |
| **Trajectory Momentum** (Algorithms C & D) | `compute_momentum_score(style="trajectory-waypoint")` | Factored integration along pseudotime waypoints | Captures state-dependent kinetics, transition stalls, and developmental branch shifts. |
| **Trajectory Decomposition** | `compute_momentum_score(return_decomposition=True)` | Returns: Mean Drive (`gene`), TIV (`tiv`), LD (`ld`) | Detects **state-gated regulators** (perturbations whose causal effect is unlocked only at specific developmental stages). |

---

### 1. Empirical Latent State Displacement (ELSD)

Computes the empirical displacement of perturbed cells relative to unperturbed control cells in latent state space:

$$
d_p(c) = \mathbb{E}_{i \in \text{Cells}(p, c)} [u_i] - \mathbb{E}_{j \in \text{Cells}(\text{ctrl}, c)} [u_j]
$$

```python
elsd_dict = pertvi.compute_elsd(
    adata,
    condition_key="condition",
    pert_key="perturbation",
    control_group="Control",
    latent_key="X_u_state",
)
# Pairwise cosine alignment between perturbations:
alignment_dict = pertvi.compute_pairwise_alignment(elsd_dict, metric="cosine")
```

---

### 2. Counterfactual Causal Scores (Coupled-CCS vs. State-CCS)

Evaluates counterfactual potential outcomes by shifting the invariant control cell population along the estimated perturbation displacement:

$$
u_{\text{cf}} = u_{\text{ctrl}} + d_p(c)
$$

* **Coupled-CCS (`inject_film=True`)**:
  Applies condition-specific FiLM modulators to capture total causal response (state transition + drug-specific transcriptional cascade):

$$
\Delta Y_p^{\text{Coupled}} = W_{\text{head}} \left( \mathbb{E} \left[ h\left((1 + \gamma_p) \odot u_{\text{cf}} + \beta_p\right) \right] - \mathbb{E} \left[ h(z_{\text{ctrl}}) \right] \right)
$$

* **State-CCS (`inject_film=False`)**:
  Holds modulators fixed at control baseline to isolate pure state-transition effects decoupled from chemical stress or off-target toxicity:

$$
\Delta Y_p^{\text{State}} = W_{\text{head}} \left( \mathbb{E} \left[ h\left((1 + \gamma_{\text{ctrl}}) \odot u_{\text{cf}} + \beta_{\text{ctrl}}\right) \right] - \mathbb{E} \left[ h(z_{\text{ctrl}}) \right] \right)
$$

```python
# A. Gene-Level Coupled-CCS (Total Causal Response)
gene_ccs = pertvi.compute_ccs(
    adata,
    model,
    condition_key="condition",
    pert_key="perturbation",
    control_group="Control",
    inject_film=True,  # Coupled-CCS
)

# B. Canonical Pathway Projection with Negative Control Gene (NCG) Calibration
pathway_dict = {
    "TCR_Signaling": ["CD3D", "CD3E", "ZAP70", "LCP2", "LAT"],
    "NFkB_Activation": ["NFKB1", "NFKBIA", "REL", "TNFAIP3"],
}
pathway_ccs = pertvi.compute_ccs(
    adata,
    model,
    condition_key="condition",
    pert_key="perturbation",
    control_group="Control",
    pathway_dict=pathway_dict,
    calibrate_ncg=True,
    ncg_list=["NTC_1", "NTC_2", "NTC_3"],
)
```

---

### 3. Path-Integrated Perturbation Momentum

Calculates the average causal drive exerted by a perturbation along an activation or differentiation path across $T$ discretized states:

$$
M_p = \frac{1}{T} \sum_{t=1}^T \Big( \text{Decoder}(z_p(u_t)) - \text{Decoder}(z_{\text{ctrl}}(u_t)) \Big)
$$

Because the generative head $W_{\text{head}}$ is linear, PertVI factors the trajectory integration directly into 128-dimensional hidden representation space:

$$
M_p \equiv W_{\text{head}} \left( \frac{1}{T} \sum_{t=1}^T [h(z_p(u_t)) - h(z_{\text{ctrl}}(u_t))] \right) = W_{\text{head}} \overline{\Delta h}_p
$$

This hidden-space factorization achieves a **>900× speedup** and eliminates the need to materialize multi-gigabyte $(P \times T \times G)$ activation tensors.

```python
# Style 1: 'linear' (Algorithm A, default) - straight interpolation between centroids
momentum_linear = pertvi.compute_momentum_score(
    adata,
    model,
    style="linear",
    condition_key="condition",
    start_state="Rest",
    end_state="Stim8hr",
    control_group="Control",
)

# Style 2: 'trajectory-basic' (Algorithm C) - empirical pseudotime path
momentum_basic = pertvi.compute_momentum_score(
    adata,
    model,
    style="trajectory-basic",
    dpt_key="dpt_pseudotime",
    control_group="Control",
    n_bins=100,
)

# Style 3: 'trajectory-waypoint' (Algorithm D) with 3-Component Decomposition
momentum_decomp = pertvi.compute_momentum_score(
    adata,
    model,
    style="trajectory-waypoint",
    condition_key="condition",
    start_state="Rest",
    end_state="Stim8hr",
    dpt_key="dpt_pseudotime",
    control_group="Control",
    return_decomposition=True,
)

# Extract components:
mean_momentum = momentum_decomp["gene"]  # Mean causal drive
tiv_df = momentum_decomp["tiv"]          # Trajectory Interaction Variance (state-gated response)
ld_df = momentum_decomp["ld"]            # Landmark Divergence (end vs. start)
```

---

## 🧪 Batch Harmonization & Counterfactual Simulation

Use `model.get_normalized_expression()` for multi-plate batch harmonization and in silico perturbation experiments:

```python
# 1. Batch Harmonization (Removing Plate / Sequencing Batch Artifacts)
# Denoises gene expression while holding plate embedding fixed to "Plate_1":
harmonized_expr = model.get_normalized_expression(
    transform_batch="Plate_1",
    target_sum=10000,
)
adata.layers["batch_corrected_counts"] = np.log1p(harmonized_expr.values)

# 2. In Silico Counterfactual Drug Response Simulation
# Simulates what Resting control cells would express if treated with Drug_A in Plate_1:
resting_controls = adata[(adata.obs["condition"] == "Rest") & (adata.obs["perturbation"] == "Control")]
simulated_df = model.get_normalized_expression(
    adata=resting_controls,
    transform_sample="Drug_A",
    transform_batch="Plate_1",
)
```

---

## 🔬 Deep Learning Reference & Hyperparameters

### Mathematical Objective Function

PertVI optimizes a composite variational objective:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NB}} + \beta_{\text{cyc}}\text{KL}_{\text{cyc}} + \text{KL}_{\text{state}} + \text{KL}_{\text{ood}} + \text{KL}_{\ell} + \lambda_{\text{ood}} \mathcal{L}_{\text{ood}} + \lambda_{\text{cov}}^{\text{state}} \mathcal{L}_{\text{cov}}^{\text{state}} + \lambda_{\text{cov}}^{\text{ood}} \mathcal{L}_{\text{cov}}^{\text{ood}} + \lambda_{\text{soft}} \Vert W_{\text{head}} \odot M_{\text{cyc}} \Vert_1
$$

* **Deterministic Covariance Penalty on Posterior Means:** Covariance is evaluated strictly on centered encoder means $q_m$, eliminating stochastic sampling noise and stabilizing gradients.
* **Floor-to-Max Annealing on OOD $L_1$:** Scheduled from $\lambda = 5.0$ to $\lambda = 50.0$ (`floor_lambda_ood` to `max_lambda_ood`). Pins control cells to zero from the first epoch to prevent early manifold drift.
* **Topological Cycle Relaxation ($\beta_{\text{cyc}}=0.5$):** Relaxes the standard Gaussian prior on `u_cycle`, allowing unconstrained continuous ring formation.

---

### Hyperparameter Reference Table

| Parameter | Default | Recommended Range | Theoretical & Practical Rationale |
| :--- | :---: | :---: | :--- |
| `n_latent_cycle` | `3` | `2 – 4` | Minimal degrees of freedom to parameterize the $S^1$ cell-cycle ring manifold without redundant capacity. |
| `n_latent_state` | `30` | `20 – 50` | Expressive manifold capturing intrinsic cellular differentiation, cell type identity, and baseline dynamics. |
| `n_latent_ood` | `5` or `10` | `3 – 10` | Tightly constrained bottleneck capacity preventing OOD from acting as an information shortcut. |
| `max_lambda_adv_state`| `100.0` | `50.0 – 150.0` | Heavy penalty enforcing near-zero covariance between intrinsic biological state and cell cycle dynamics. |
| `max_lambda_adv_ood` | `5.0` | `1.0 – 10.0` | Gentle orthogonalization preventing perturbation residual effects from leaking into cell cycle phase. |
| `floor_lambda_ood` | `5.0` | `2.0 – 10.0` | Non-zero floor ensuring unperturbed control cells cannot drift away from 0 in early training epochs. |
| `max_lambda_ood` | `50.0` | `30.0 – 100.0` | Upper bound enforcing exact sparsity on control OOD representations. |
| `lambda_soft` | `50.0` | `20.0 – 100.0` | Suppresses cycle gene weights in the main head, routing cycle variance strictly through Pathway A. |
| `beta_cyc` | `0.5` | `0.3 – 0.7` | Topological relaxation on cycle prior encouraging continuous closed-loop cycle structures. |
| `warmup_epochs` | `50` | `20 – 50` | Phase 1: Pure reconstruction and prior learning before applying orthogonality and sparsity penalties. |
| `anneal_epochs` | `100` | `50 – 150` | Phase 2: Smooth linear ramp of covariance penalties to prevent gradient shock and optimization collapse. |
| `batch_size` | `256` | `128 – 512` | Batch size offering stable mini-batch covariance estimation while avoiding GPU memory overflow. |

---

## 🛠 Hardware Sizing & Troubleshooting

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA $\ge$ 11.8 (RTX 3090, RTX 4090, A100, H100).
- **VRAM Sizing**:
  - $\le 50,000$ cells / 3,000 genes: **8 GB VRAM** (`batch_size=256`)
  - $100,000+$ cells / 10,000 genes: **16–24 GB VRAM** (`batch_size=256` or `128`)
- **CPU Support**: Fully supported via PyTorch Lightning CPU fallback (`accelerator="cpu"`).

### Troubleshooting FAQ

| Exception / Symptom | Root Cause | Solution |
| :--- | :--- | :--- |
| `ValueError: Column 'is_cycle_gene' not found in adata.var` | Cycle gene mask not set before model initialization. | Set `adata.var['is_cycle_gene'] = adata.var_names.isin(cycle_genes)` prior to calling `PertVI(adata)`. |
| `ValueError: No control cells found matching control_group='NTC'` | Control string does not match `adata.obs[pert_key]`. | Verify `adata.obs[pert_key].unique()` and provide the exact matching string (e.g. `"Control"` or `"DMSO"`). |
| `CUDA Out of Memory` during training | Batch size exceeds available GPU memory. | In `model.train()`, reduce `batch_size=128` or `batch_size=64`. |
| `OOD Dimension Collapse` (`v_ood` is all zero) | `is_treated` was left as `0.0` for all cells. | Ensure treated cells are explicitly marked `1.0` in `adata.obs['is_treated']`. |
| `Insufficient control cells (< 5)` | Condition has fewer than `min_cells` unperturbed cells. | Lower `min_cells=3` or pool replicate control wells across batches. |

---

## 📖 Citation

TBD
