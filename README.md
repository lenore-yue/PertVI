# PertVI: Conditional Perturbation Variational Inference

PertVI is a deep generative model built on the `scvi-tools` framework, designed specifically to disentangle conditional drug responses and perturbation effects from intrinsic cellular biology in single-cell RNA-seq data.

Unlike general-purpose disentanglement models designed for atlas-level cell type discovery, PertVI is a highly constrained, hypothesis-driven architecture. It isolates how a cell's intrinsic differentiation state and cell cycle phase dictate its transcriptomic response to external perturbations.

## 🧬 Architecture Overview

PertVI partitions the standard monolithic VAE latent space into three specialized, biologically interpretable subspaces using targeted mathematical penalties (Covariance independence and Masked L2 constraints):

* **`u_cycle` (10 dims):** Isolates transcriptomic variance driven by G1, S, and G2M cell cycle phases.
* **`u_state` (20 dims):** Captures the pure intrinsic biological state (e.g., differentiation trajectory, metabolic baseline), completely scrubbed of cell cycle effects.
* **`v_ood` (20 dims):** The "Out-of-Distribution" escape hatch. Strictly constrained to zero for control cells, it captures rare, heterogeneous, or unexpected variance induced *only* by the perturbations.

The model utilizes a conditional Multi-Layer Perceptron (MLP) transition to calculate an expected, state-dependent shift ($z_{shift}$) based on the specific perturbation, allowing for the modeling of highly non-linear, context-specific drug mechanisms.

## 🚀 Installation

You can install PertVI locally in editable mode. Ensure you have your `conda` environment activated with `scvi-tools` and `scanpy` installed.

```bash
git clone [https://github.com/YOUR_USERNAME/PertVI.git](https://github.com/YOUR_USERNAME/PertVI.git)
cd PertVI
pip install -e .
💻 Quickstart
Python
import scanpy as sc
from pertvi import PertVI

# 1. Load and prep your data
adata = sc.read_h5ad("your_data.h5ad")

# 2. Setup the AnnData object for PertVI
PertVI.setup_anndata(
    adata, 
    layer="counts", 
    batch_key="plate_id", 
    sample_key="perturbation_id", # Perturbations are passed as the sample adapter
    treatment_mask_key="is_treated" # 0 for DMSO/Control, 1 for treated
)

# 3. Initialize and Train
model = PertVI(adata, n_latent_cycle=10, n_latent_state=20, n_latent_ood=20)
model.train(max_epochs=300, early_stopping=True)

# 4. Extract Disentangled Latent Spaces
latents = model.get_latent_representation()
adata.obsm["X_u_cycle"] = latents["u_cycle"]
adata.obsm["X_u_state"] = latents["u_state"]
adata.obsm["X_v_ood"] = latents["v_ood"]
📖 Citation
(Coming Soon)
