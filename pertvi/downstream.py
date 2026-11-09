"""
Downstream Analysis Module for PertVI Models.

This module provides tools for downstream analysis of trained PertVI models,
focusing on latent space perturbation geometry, empirical state displacements,
and condition-specific regulatory dynamics.
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
import torch


def compute_elsd(
    adata,
    condition_key: str = "condition",
    pert_key: str = "perturbed_gene_name",
    control_group: Union[str, List[str]] = "NTC",
    control_name: Optional[Union[str, List[str]]] = None,
    control: Optional[Union[str, List[str]]] = None,
    latent_key: str = "X_u_state",
    conditions: Optional[List[str]] = None,
    perturbations: Optional[List[str]] = None,
    min_cells: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Compute Empirical Latent State Displacement (ELSD) for each condition.

    The ELSD metric quantifies the perturbation effect as the empirical displacement
    of the perturbed cell population from the control cell population in a given
    representation space (latent or gene expression):

        d_p(c) = E_{i in Cells(p, c)} [v_i] - E_{j in Cells(control, c)} [v_j]

    Parameters
    ----------
    adata : AnnData
        Annotated single-cell dataset containing cell metadata and latent/gene representations.
    condition_key : str, default 'condition'
        Column name in adata.obs identifying experimental conditions / timepoints (e.g. 'Stim8hr', 'Stim48hr').
    pert_key : str, default 'perturbed_gene_name'
        Column name in adata.obs identifying genetic perturbations.
    control_group : str or list of str, default 'NTC'
        Identifier(s) for negative control / unperturbed cells in pert_key (e.g. 'NTC', 'control', ['NTC_1', 'NTC_2']).
    control_name : str or list of str, optional
        Alias for control_group for backwards compatibility.
    control : str or list of str, optional
        Convenience alias for control_group.
    latent_key : str, default 'X_u_state'
        Key in adata.obsm (e.g. 'X_u_state', 'X_z_state', 'X_z_final') or adata.layers
        (e.g. 'X_pertvi_log1p') to extract representation vectors.
    conditions : list of str, optional
        Specific conditions to analyze. If None, uses all unique conditions in adata.obs[condition_key].
    perturbations : list of str, optional
        Specific perturbations to compute. If None, uses all unique perturbations except control_group.
    min_cells : int, default 5
        Minimum number of cells required for a (perturbation, condition) pair to be computed.

    Returns
    -------
    dict of {str: pd.DataFrame}
        Dictionary mapping each condition name to a DataFrame of shape (n_features, n_perturbations),
        where each column represents the displacement vector d_p(c).
    """
    if condition_key not in adata.obs.columns:
        raise ValueError(f"condition_key '{condition_key}' not found in adata.obs.")
    if pert_key not in adata.obs.columns:
        raise ValueError(f"pert_key '{pert_key}' not found in adata.obs.")

    # Resolve control group identifier(s)
    if control is not None:
        control_group = control
    elif control_name is not None:
        control_group = control_name

    if isinstance(control_group, str):
        control_list = [control_group]
    else:
        control_list = list(control_group)

    # Retrieve representation data
    if latent_key in adata.obsm:
        rep_data = adata.obsm[latent_key]
        feature_names = [f"{latent_key}_{i}" for i in range(rep_data.shape[1])]
    elif latent_key in adata.layers:
        rep_data = adata.layers[latent_key]
        feature_names = list(adata.var_names)
    elif latent_key == "X":
        rep_data = adata.X
        feature_names = list(adata.var_names)
    else:
        raise KeyError(
            f"latent_key '{latent_key}' not found in adata.obsm, adata.layers, or adata.X."
        )

    all_conditions = adata.obs[condition_key].unique()
    if conditions is None:
        conditions = list(all_conditions)

    all_perts = adata.obs[pert_key].unique()
    if perturbations is None:
        perturbations = [p for p in all_perts if p not in control_list]

    elsd_by_condition: Dict[str, pd.DataFrame] = {}

    for cond in conditions:
        cond_mask = adata.obs[condition_key] == cond
        ctrl_mask = cond_mask & adata.obs[pert_key].isin(control_list)

        if ctrl_mask.sum() < min_cells:
            raise ValueError(
                f"Condition '{cond}' has insufficient control cells ({ctrl_mask.sum()} < {min_cells}) "
                f"matching control group {control_list} in column '{pert_key}'."
            )

        ctrl_rep = rep_data[ctrl_mask]
        if sparse.issparse(ctrl_rep):
            ctrl_mean = np.array(ctrl_rep.mean(axis=0)).flatten()
        else:
            ctrl_mean = np.mean(ctrl_rep, axis=0)

        cond_disp: Dict[str, np.ndarray] = {}

        for p in perturbations:
            p_mask = cond_mask & (adata.obs[pert_key] == p)
            if p_mask.sum() < min_cells:
                continue

            p_rep = rep_data[p_mask]
            if sparse.issparse(p_rep):
                p_mean = np.array(p_rep.mean(axis=0)).flatten()
            else:
                p_mean = np.mean(p_rep, axis=0)

            disp_vec = p_mean - ctrl_mean
            cond_disp[p] = disp_vec

        df_disp = pd.DataFrame(cond_disp, index=feature_names)
        elsd_by_condition[cond] = df_disp

    return elsd_by_condition


def compute_pairwise_alignment(
    elsd_dict: Dict[str, pd.DataFrame],
    metric: str = "cosine",
) -> Dict[str, pd.DataFrame]:
    """
    Compute pairwise similarity or correlation matrices between perturbations for each condition.

    Parameters
    ----------
    elsd_dict : dict of {str: pd.DataFrame}
        Output from `compute_elsd`, mapping condition names to feature-by-perturbation DataFrames.
    metric : {'cosine', 'pearson', 'spearman', 'dot_product'}, default 'cosine'
        Alignment metric to evaluate between displacement vectors.

    Returns
    -------
    dict of {str: pd.DataFrame}
        Dictionary mapping each condition name to a symmetric (n_perts x n_perts) similarity matrix.
    """
    similarity_by_condition: Dict[str, pd.DataFrame] = {}

    for cond, df_disp in elsd_dict.items():
        perts = list(df_disp.columns)
        X = df_disp.values.T  # shape: (n_perts, n_features)

        if metric == "cosine":
            # Normalized cosine similarity: <u, v> / (||u|| * ||v||)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            X_norm = X / norms
            sim_mat = np.dot(X_norm, X_norm.T)
        elif metric == "pearson":
            sim_mat = df_disp.corr(method="pearson").values
        elif metric == "spearman":
            sim_mat = df_disp.corr(method="spearman").values
        elif metric == "dot_product":
            sim_mat = np.dot(X, X.T)
        else:
            raise ValueError(f"Unsupported metric: '{metric}'. Use 'cosine', 'pearson', 'spearman', or 'dot_product'.")

        similarity_by_condition[cond] = pd.DataFrame(sim_mat, index=perts, columns=perts)

    return similarity_by_condition


def build_pathway_incidence_matrix(
    pathway_dict: Dict[str, List[str]],
    gene_names: List[str],
    normalize_rows: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Construct a pathway-by-gene incidence matrix M in R^{K x G}.

    Parameters
    ----------
    pathway_dict : dict of {str: list of str}
        Dictionary mapping pathway names to lists of gene symbols.
    gene_names : list of str
        Ordered list of G gene symbols matching the model feature space.
    normalize_rows : bool, default True
        If True, each row is normalized by 1 / |G_k \cap genes| so that
        M @ Delta Y computes the mean gene displacement within each pathway.

    Returns
    -------
    M : np.ndarray
        Incidence matrix of shape (n_pathways, n_genes).
    pathway_names : list of str
        List of retained pathway names.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)
    
    rows = []
    retained_names = []

    for name, gset in pathway_dict.items():
        indices = [gene_to_idx[g] for g in gset if g in gene_to_idx]
        if len(indices) == 0:
            continue
        
        row = np.zeros(n_genes, dtype=np.float32)
        if normalize_rows:
            row[indices] = 1.0 / len(indices)
        else:
            row[indices] = 1.0

        rows.append(row)
        retained_names.append(name)

    if len(rows) == 0:
        raise ValueError("None of the genes in pathway_dict overlap with gene_names.")

    M = np.stack(rows, axis=0)
    return M, retained_names


def calibrate_ncg_null(
    scores_df: pd.DataFrame,
    ncg_list: List[str],
    min_std: float = 1e-6,
) -> pd.DataFrame:
    """
    Standardize feature or pathway scores using negative control genes (NCGs).

    For each feature k:
        Z_p(k) = (S_p(k) - mu_NCG(k)) / sigma_NCG(k)

    This ensures that under the null hypothesis of no biological effect:
        E[Z_NCG] = 0 and Var(Z_NCG) = 1.

    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame of shape (n_features, n_perturbations).
    ncg_list : list of str
        List of negative control gene identifiers present in scores_df.columns.
    min_std : float, default 1e-6
        Clipping floor for standard deviation to avoid division by zero.

    Returns
    -------
    pd.DataFrame
        Calibrated Z-score DataFrame of identical shape.
    """
    present_ncgs = [c for c in ncg_list if c in scores_df.columns]
    if len(present_ncgs) < 2:
        raise ValueError(
            f"Insufficient NCGs found in scores_df columns (found {len(present_ncgs)}, need at least 2)."
        )

    mu_ncg = scores_df[present_ncgs].mean(axis=1)
    std_ncg = scores_df[present_ncgs].std(axis=1).clip(lower=min_std)

    z_df = scores_df.sub(mu_ncg, axis=0).div(std_ncg, axis=0)
    return z_df


def calibrate_pool_null(
    scores_df: pd.DataFrame,
    min_std: float = 1e-6,
) -> pd.DataFrame:
    """
    Standardize feature or pathway scores using all perturbations in the pool.

    For each feature k:
        Z_p(k) = (S_p(k) - mu_pool(k)) / sigma_pool(k)

    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame of shape (n_features, n_perturbations).
    min_std : float, default 1e-6
        Clipping floor for standard deviation to avoid division by zero.

    Returns
    -------
    pd.DataFrame
        Standardized Z-score DataFrame of identical shape.
    """
    mu_pool = scores_df.mean(axis=1)
    std_pool = scores_df.std(axis=1).clip(lower=min_std)
    z_df = scores_df.sub(mu_pool, axis=0).div(std_pool, axis=0)
    return z_df


def project_to_pathways(
    gene_disp_dict: Dict[str, pd.DataFrame],
    pathway_dict: Dict[str, List[str]],
    gene_names: Optional[List[str]] = None,
    calibrate_ncg: bool = True,
    ncg_list: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Project gene-level displacement vectors to pathway-level activation scores.

        P_p(c) = M @ Delta Y_p(c)

    Parameters
    ----------
    gene_disp_dict : dict of {str: pd.DataFrame}
        Output from `compute_elsd` on gene representations (shape: n_genes x n_perts).
    pathway_dict : dict of {str: list of str}
        Dictionary mapping pathway names to member genes.
    gene_names : list of str, optional
        Gene symbols. If None, inferred from gene_disp_dict index.
    calibrate_ncg : bool, default True
        If True, standardizes scores by NCG distribution using `calibrate_ncg_null`.
    ncg_list : list of str, optional
        List of NCG perturbation names. Required if calibrate_ncg is True.

    Returns
    -------
    dict of {str: pd.DataFrame}
        Dictionary mapping condition names to pathway score DataFrames (shape: n_pathways x n_perts).
    """
    first_cond = next(iter(gene_disp_dict.keys()))
    if gene_names is None:
        gene_names = list(gene_disp_dict[first_cond].index)

    M, pathway_names = build_pathway_incidence_matrix(
        pathway_dict=pathway_dict,
        gene_names=gene_names,
        normalize_rows=True,
    )

    pathway_by_condition: Dict[str, pd.DataFrame] = {}

    for cond, df_gene in gene_disp_dict.items():
        perts = list(df_gene.columns)
        # Compute P = M @ Delta Y (shape: n_pathways x n_perts)
        P = M @ df_gene.values
        df_path = pd.DataFrame(P, index=pathway_names, columns=perts)

        if calibrate_ncg:
            if ncg_list is None:
                raise ValueError("ncg_list must be provided when calibrate_ncg=True.")
            df_path = calibrate_ncg_null(df_path, ncg_list=ncg_list)

        pathway_by_condition[cond] = df_path

    return pathway_by_condition


def build_pathway_head_operator(
    model,
    pathway_dict: Dict[str, List[str]],
    gene_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Construct the direct Decoder Pathway Operator: W_pathway = M @ W_head.

    This linear operator projects 128-dimensional hidden decoder activations directly
    into K-dimensional pathway space, bypassing intermediate 10,004-dimensional gene space:
        P_p(c) = W_pathway @ Delta h_p(c)

    Parameters
    ----------
    model : PertVI
        Trained PertVI model instance.
    pathway_dict : dict of {str: list of str}
        Dictionary mapping pathway names to gene lists.
    gene_names : list of str, optional
        Gene names in model order. If None, extracted from model.adata_manager.

    Returns
    -------
    W_pathway : np.ndarray
        Array of shape (n_pathways, 128).
    pathway_names : list of str
        Retained pathway names.
    """
    if gene_names is None:
        if hasattr(model, 'adata_manager') and model.adata_manager is not None:
            try:
                gene_names = list(model.adata_manager.get_state_registry('X_pertvi_log1p').column_names)
            except Exception:
                if hasattr(model, 'adata') and model.adata is not None:
                    gene_names = list(model.adata.var_names)
                else:
                    raise ValueError("gene_names must be provided if model has no attached adata.")
        elif hasattr(model, 'adata') and model.adata is not None:
            gene_names = list(model.adata.var_names)
        else:
            raise ValueError("gene_names must be provided if model has no attached adata.")

    M, pathway_names = build_pathway_incidence_matrix(
        pathway_dict=pathway_dict,
        gene_names=gene_names,
        normalize_rows=True,
    )

    # Extract W_head: (n_genes, 128)
    W_head = model.module.generative_net.main_head.weight.detach().cpu().numpy()
    W_pathway = M @ W_head  # (K, 128)

    return W_pathway, pathway_names


def compute_counterfactual_causal_score(
    adata,
    model,
    condition_key: str = "condition",
    pert_key: str = "perturbed_gene_name",
    control_group: Union[str, List[str]] = "NTC",
    control_name: Optional[Union[str, List[str]]] = None,
    control: Optional[Union[str, List[str]]] = None,
    latent_key: str = "X_u_state",
    conditions: Optional[List[str]] = None,
    perturbations: Optional[List[str]] = None,
    apply_layer_norm: bool = True,
    inject_film: Optional[bool] = None,
    film_mode: str = "perturbation",
    compute_pathways: bool = False,
    pathway_dict: Optional[Dict[str, List[str]]] = None,
    W_pathway: Optional[Union[np.ndarray, torch.Tensor]] = None,
    pathway_names: Optional[List[str]] = None,
    return_gene_scores: Optional[bool] = None,
    calibrate_ncg: bool = False,
    ncg_list: Optional[List[str]] = None,
    standardize: bool = True,
    batch_index: int = 0,
    ood_val: float = 0.0,
    min_cells: int = 5,
    return_hidden: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[
    Dict[str, pd.DataFrame],
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]],
    Tuple[Dict[str, pd.DataFrame], ...],
]:
    """
    Compute Counterfactual Causal Score (CCS) via Latent Generative Pushforward.

    Evaluates the counterfactual causal effect of genetic perturbations by intervening on the
    unperturbed control cell population P_NTC(c) in the latent state manifold R^d_u,
    bypassing observational selection bias, cell cycle imbalance, and technical nuisance.

    Supports both gene-level pushforward displacement Delta Y_p(c) in R^G and optional
    canonical pathway projections P_p(c) in R^K via direct linear operators.

    Mathematical Principles (Pearl SCM & Rubin Potential Outcomes):
    --------------------------------------------------------------
    1. Latent State Displacement:
           d_p(c) = E_{i in Cells(p, c)} [u_i] - E_{j in Cells(NTC, c)} [u_j] in R^30

    2. Counterfactual Intervention (Pearl's do-calculus):
           u_j^(cf) = u_j^(NTC) + d_p(c)   for u_j ~ P_NTC(c)

    3. Latent Transduction Modulation (inject_film / film_mode):
       - If film_mode == 'perturbation' or inject_film=True (Coupled-CCS / FiLM-CCS):
           z_j^(cf) = (1 + gamma_p) * u_j^(cf) + beta_p
       - If film_mode in {'control', 'state', 'pure', 'none'} or inject_film=False (State-CCS):
           z_j^(cf) = (1 + gamma_NTC) * u_j^(cf) + beta_NTC
       - Baseline control state:
           z_j^(NTC) = (1 + gamma_NTC) * u_j^(NTC) + beta_NTC

    4. Invariant Generative Pushforward:
           Delta h_p(c) = E_{u_j ~ P_NTC(c)} [ h(z_j^(cf)) - h(z_j^(NTC)) ] in R^128

    5. Gene-Level Pushforward Displacement (Delta Y_p):
           Delta Y_p(c) = W_head @ Delta h_p(c) in R^G

    6. Direct Pathway Operator Projection (P_p):
           P_p(c) = W_pathway @ Delta h_p(c) = (M @ W_head) @ Delta h_p(c) in R^K

    7. Pathway Standardization (NCG null or Pool null):
       - If calibrate_ncg=True:
           Z_p(k) = (P_p(k) - mu_NCG(k)) / sigma_NCG(k)
       - If calibrate_ncg=False:
           Z_p(k) = (P_p(k) - mu_pool(k)) / sigma_pool(k)
           (Significant cutoff is based on the distribution of all perturbations in the pool).

    Parameters
    ----------
    adata : AnnData
        Annotated single-cell dataset containing metadata and latent representation.
    model : PertVI
        Trained PertVI model instance.
    condition_key : str, default 'condition'
        Column in adata.obs identifying experimental conditions / timepoints.
    pert_key : str, default 'perturbed_gene_name'
        Column in adata.obs identifying genetic perturbations.
    control_group : str or list of str, default 'NTC'
        Identifier(s) for negative control / unperturbed cells in pert_key (e.g. 'NTC', 'control', ['NTC_1', 'NTC_2']).
    control_name : str or list of str, optional
        Alias for control_group for backwards compatibility.
    control : str or list of str, optional
        Convenience alias for control_group.
    latent_key : str, default 'X_u_state'
        Key in adata.obsm storing latent state coordinates.
    conditions : list of str, optional
        Specific conditions to analyze. Defaults to all unique conditions.
    perturbations : list of str, optional
        Specific perturbations to compute. Defaults to all perturbations except control_group.
    inject_film : bool, optional
        Convenience boolean flag for p-FiLM injection. If provided, overrides film_mode:
        - True: Coupled-CCS (applies perturbation-specific gamma_p, beta_p).
        - False: State-CCS (holds FiLM fixed at control gamma_NTC, beta_NTC).
    film_mode : {'perturbation', 'control', 'state', 'coupled', 'none'}, default 'perturbation'
        Coupling mode for FiLM modulation:
        - 'perturbation' / 'coupled': Coupled-CCS (applies perturbation-specific FiLM).
        - 'control' / 'state' / 'none': State-CCS (holds FiLM fixed at control baseline).
    compute_pathways : bool, default False
        Whether to project latent displacements to pathway space. Automatically set to True
        if pathway_dict or W_pathway is provided.
    pathway_dict : dict of {str: list of str}, optional
        Mapping of pathway names to member gene symbols. If provided, projects directly
        to K canonical pathways via W_pathway = M @ W_head.
    W_pathway : np.ndarray or torch.Tensor, optional
        Precomputed linear pathway operator of shape (K, 128). If provided, bypasses pathway_dict.
    pathway_names : list of str, optional
        Names of pathways corresponding to rows of W_pathway.
    return_gene_scores : bool, optional
        Whether to return gene-level CCS DataFrames. Defaults to True when compute_pathways is False,
        and False when compute_pathways is True. If both return_gene_scores=True and compute_pathways=True,
        returns a tuple of (gene_scores, pathway_scores).
    calibrate_ncg : bool, default False
        Whether to standardize pathway scores against Negative Control Genes (NCGs). For gene-level only
        computations, NCG calibration is not applied. For pathway-level results, if calibrate_ncg=False,
        a warning is raised and standardization / cutoffs are based on all perturbations in the pool.
    ncg_list : list of str, optional
        List of negative control gene identifiers present in perturbations. Required if calibrate_ncg=True.
    standardize : bool, default True
        Whether to standardize pathway scores (via NCG null if calibrate_ncg=True, or pool null if False).
    batch_index : int, default 0
        Target batch index passed to decoder batch embedding to hold technical context fixed.
    ood_val : float, default 0.0
        Out-of-distribution latent value passed to decoder (0.0 eliminates nuisance leakage).
    min_cells : int, default 5
        Minimum cell count required per (perturbation, condition) pair.
    return_hidden : bool, default False
        If True, also returns dictionary of 128-dimensional hidden representation deltas Delta h_p(c).
    device : str or torch.device, optional
        Computation device. Defaults to model parameter device.

    Returns
    -------
    results : dict or tuple of dicts
        - If only gene-level scores requested: dict of {condition: DataFrame(genes x perturbations)}.
        - If only pathway-level scores requested: dict of {condition: DataFrame(pathways x perturbations)}.
        - If both requested: tuple of (gene_scores_dict, pathway_scores_dict).
        - If return_hidden=True: appends hidden_deltas_dict to the returned tuple.
    """
    # 0. Resolve control group identifier(s)
    if control is not None:
        control_group = control
    elif control_name is not None:
        control_group = control_name

    if isinstance(control_group, str):
        control_list = [control_group]
    else:
        control_list = list(control_group)

    # 1. Resolve pathway calculation option
    if pathway_dict is not None or W_pathway is not None:
        compute_pathways = True

    # 2. Resolve return_gene_scores if not explicitly specified
    if return_gene_scores is None:
        if compute_pathways:
            return_gene_scores = False
        else:
            return_gene_scores = True

    if not return_gene_scores and not compute_pathways:
        raise ValueError("At least one of 'return_gene_scores' or 'compute_pathways' must be True.")

    if compute_pathways and (pathway_dict is None and W_pathway is None):
        raise ValueError(
            "compute_pathways=True requires either 'pathway_dict' or 'W_pathway' to be provided."
        )

    # 3. Check NCG option and warning for pathway analysis
    if compute_pathways:
        if calibrate_ncg:
            if ncg_list is None or len(ncg_list) < 2:
                raise ValueError("ncg_list with at least 2 negative controls must be provided when calibrate_ncg=True.")
        else:
            warnings.warn(
                "NCG calibration is disabled (calibrate_ncg=False). The significant cutoff will be based on all perturbations in the pool.",
                UserWarning,
                stacklevel=2,
            )

    # 4. Resolve FiLM modulation mode
    if inject_film is not None:
        use_pert_film = bool(inject_film)
    else:
        norm_mode = film_mode.lower()
        if norm_mode in ("perturbation", "coupled", "film", "p_film"):
            use_pert_film = True
        elif norm_mode in ("control", "state", "pure", "none", "ntc"):
            use_pert_film = False
        else:
            raise ValueError(
                f"Unsupported film_mode: '{film_mode}'. Use 'perturbation' (Coupled-CCS) or 'control' (State-CCS)."
            )

    # 5. Model device and eval state
    model.module.eval()
    if device is None:
        device = next(model.module.parameters()).device
    else:
        device = torch.device(device)

    # 6. Extract & Normalize Latent State Representation, then Compute Displacements d_p(c)
    if condition_key not in adata.obs.columns:
        raise ValueError(f"condition_key '{condition_key}' not found in adata.obs.")
    if pert_key not in adata.obs.columns:
        raise ValueError(f"pert_key '{pert_key}' not found in adata.obs.")
    if latent_key not in adata.obsm:
        raise KeyError(f"latent_key '{latent_key}' not found in adata.obsm.")

    film = model.module.inference_net.split_film_transition
    is_m7 = getattr(film, "film_mode", "legacy_hadamard") == "m7_block_tt"

    with torch.no_grad():
        u_raw_all = torch.tensor(adata.obsm[latent_key], dtype=torch.float32, device=device)
        if apply_layer_norm:
            if is_m7:
                u_state_all = film.normalize_state_eval(u_raw_all)
            else:
                u_state_all = film.norm_state(u_raw_all)
        else:
            u_state_all = u_raw_all

    all_conditions = list(adata.obs[condition_key].unique()) if conditions is None else list(conditions)
    all_obs_perts = adata.obs[pert_key].unique()
    cand_perts = [p for p in all_obs_perts if p not in control_list] if perturbations is None else list(perturbations)

    # 7. Extract FiLM parameter modulators / sample indices for all perturbations
    sample_registry = model.adata_manager.get_state_registry("sample")
    cat_mapping = sample_registry.categorical_mapping
    sample_idx_map: Dict[str, int] = {}
    films = {}

    with torch.no_grad():
        for p in cat_mapping:
            matches = np.where(cat_mapping == p)[0]
            if len(matches) == 0:
                continue
            idx = int(matches[0])
            sample_idx_map[p] = idx
            if not is_m7:
                t = torch.tensor([[idx]], dtype=torch.long, device=device)
                emb = model.module.inference_net.sample_embedding(t).squeeze(1)
                g = film.gamma_state(emb).detach()
                b = film.beta_state(emb).detach()
                films[p] = (g, b)

    ctrl_indices = [sample_idx_map[c] for c in control_list if c in sample_idx_map]
    if not ctrl_indices:
        raise KeyError(
            f"None of the control group identifiers {control_list} were found in the model categorical mapping."
        )
    if not is_m7:
        ctrl_films = [films[c] for c in control_list if c in films]
        g_ntc = torch.stack([cf[0] for cf in ctrl_films]).mean(dim=0)
        b_ntc = torch.stack([cf[1] for cf in ctrl_films]).mean(dim=0)

    # 8. Setup Gene-level Projection Operator (W_head)
    W_head_t = model.module.generative_net.main_head.weight.detach().to(device)  # (G, 128)
    if hasattr(model, "adata_manager") and model.adata_manager is not None:
        try:
            gene_names = list(model.adata_manager.get_state_registry("X_pertvi_log1p").column_names)
        except Exception:
            gene_names = list(adata.var_names)
    else:
        gene_names = list(adata.var_names)

    # 9. Setup Pathway-level Projection Operator (W_pathway) if requested
    W_pathway_t = None
    path_feature_names = None
    if compute_pathways:
        if W_pathway is not None:
            W_pathway_t = torch.as_tensor(W_pathway, dtype=torch.float32, device=device)
            path_feature_names = pathway_names if pathway_names is not None else [f"Pathway_{i}" for i in range(W_pathway_t.shape[0])]
        elif pathway_dict is not None:
            W_path_np, path_feature_names = build_pathway_head_operator(
                model=model,
                pathway_dict=pathway_dict,
                gene_names=gene_names,
            )
            W_pathway_t = torch.tensor(W_path_np, dtype=torch.float32, device=device)

    # 10. Decoder nuisance embeddings: V_ood and batch_embed
    z_state_dim = getattr(film, "u_state_dim", None) or film.gamma_state.out_features
    main_in_dim = model.module.generative_net.main_hidden[0].in_features
    batch_embed_dim = model.module.generative_net.batch_embed.embedding_dim if model.module.generative_net.use_batch else 0
    n_ood = main_in_dim - batch_embed_dim - z_state_dim
    v_ood_tensor = torch.full((1, n_ood), fill_value=ood_val, device=device)

    if model.module.generative_net.use_batch:
        b_idx_t = torch.tensor([batch_index], dtype=torch.long, device=device)
        b_embed_0 = model.module.generative_net.batch_embed(b_idx_t)
    else:
        b_embed_0 = None

    def decode_hidden(z_input: torch.Tensor) -> torch.Tensor:
        n_c = z_input.shape[0]
        ood_expanded = v_ood_tensor.expand(n_c, -1)
        if b_embed_0 is not None:
            z_full = torch.cat([z_input, ood_expanded, b_embed_0.expand(n_c, -1)], dim=-1)
        else:
            z_full = torch.cat([z_input, ood_expanded], dim=-1)
        return model.module.generative_net.main_hidden(z_full)

    # 11. Execute Counterfactual Pushforward Iteration
    gene_ccs_by_cond: Dict[str, pd.DataFrame] = {}
    pathway_ccs_by_cond: Dict[str, pd.DataFrame] = {}
    hidden_deltas_by_cond: Dict[str, pd.DataFrame] = {}

    with torch.no_grad():
        for cond in all_conditions:
            cond_mask = (adata.obs[condition_key] == cond).values
            ctrl_mask = cond_mask & adata.obs[pert_key].isin(control_list).values
            if ctrl_mask.sum() < min_cells:
                raise ValueError(
                    f"Condition '{cond}' has insufficient control cells ({ctrl_mask.sum()} < {min_cells}) "
                    f"matching control group {control_list} in column '{pert_key}'."
                )
            u_ntc_t = u_state_all[ctrl_mask]
            u_ntc_mean = u_ntc_t.mean(dim=0, keepdim=True)

            # Invariant unperturbed baseline hidden representation
            if is_m7:
                z_ntc_base = torch.stack(
                    [film.modulate_state_eval(u_ntc_t, c_idx) for c_idx in ctrl_indices], dim=0
                ).mean(dim=0)
            else:
                z_ntc_base = (1.0 + g_ntc) * u_ntc_t + b_ntc
            h_ntc_base = decode_hidden(z_ntc_base).mean(dim=0)  # (128,)

            cond_gene_scores: Dict[str, np.ndarray] = {}
            cond_path_scores: Dict[str, np.ndarray] = {}
            cond_deltas: Dict[str, np.ndarray] = {}

            for p in cand_perts:
                if p not in sample_idx_map:
                    continue
                p_mask = cond_mask & (adata.obs[pert_key] == p).values
                if p_mask.sum() < min_cells:
                    continue

                d_p_t = u_state_all[p_mask].mean(dim=0, keepdim=True) - u_ntc_mean

                # Counterfactual shift on the control manifold: u_cf = u_ntc + d_p
                u_cf = u_ntc_t + d_p_t

                # FiLM Modulation selection
                if is_m7:
                    if use_pert_film:
                        z_cf = film.modulate_state_eval(u_cf, sample_idx_map[p])
                    else:
                        z_cf = torch.stack(
                            [film.modulate_state_eval(u_cf, c_idx) for c_idx in ctrl_indices], dim=0
                        ).mean(dim=0)
                else:
                    if use_pert_film:
                        g_p, b_p = films[p]
                        z_cf = (1.0 + g_p) * u_cf + b_p
                    else:
                        z_cf = (1.0 + g_ntc) * u_cf + b_ntc

                h_cf = decode_hidden(z_cf).mean(dim=0)  # (128,)
                delta_h = h_cf - h_ntc_base             # (128,)

                if return_gene_scores:
                    gene_vec = torch.matmul(W_head_t, delta_h).cpu().numpy()
                    cond_gene_scores[p] = gene_vec

                if compute_pathways:
                    path_vec = torch.matmul(W_pathway_t, delta_h).cpu().numpy()
                    cond_path_scores[p] = path_vec

                cond_deltas[p] = delta_h.cpu().numpy()

            if return_gene_scores:
                gene_ccs_by_cond[cond] = pd.DataFrame(cond_gene_scores, index=gene_names)

            if compute_pathways:
                df_path = pd.DataFrame(cond_path_scores, index=path_feature_names)
                if standardize:
                    if calibrate_ncg:
                        df_path = calibrate_ncg_null(df_path, ncg_list=ncg_list)
                    else:
                        df_path = calibrate_pool_null(df_path)
                pathway_ccs_by_cond[cond] = df_path

            hidden_deltas_by_cond[cond] = pd.DataFrame(cond_deltas)

    # 12. Return packaging
    if return_gene_scores and compute_pathways:
        if return_hidden:
            return gene_ccs_by_cond, pathway_ccs_by_cond, hidden_deltas_by_cond
        return gene_ccs_by_cond, pathway_ccs_by_cond
    elif return_gene_scores:
        if return_hidden:
            return gene_ccs_by_cond, hidden_deltas_by_cond
        return gene_ccs_by_cond
    else:
        if return_hidden:
            return pathway_ccs_by_cond, hidden_deltas_by_cond
        return pathway_ccs_by_cond


# Formal functional aliases
compute_ccs = compute_counterfactual_causal_score
compute_css = compute_counterfactual_causal_score
compute_causal_pushforward = compute_counterfactual_causal_score


@torch.inference_mode(True)
def compute_momentum_score(
    adata,
    model,
    style: str = "linear",
    pert_key: str = "perturbed_gene_name",
    control_group: Union[str, List[str]] = "NTC",
    control_name: Optional[Union[str, List[str]]] = None,
    control: Optional[Union[str, List[str]]] = None,
    condition_key: Optional[str] = "condition",
    state_key: Optional[str] = None,
    start_state: Optional[str] = None,
    end_state: Optional[str] = None,
    dpt_key: Optional[str] = None,
    pseudotime_key: Optional[str] = None,
    latent_key: str = "X_u_state",
    perturbations: Optional[List[str]] = None,
    steps: int = 1000,
    n_bins: int = 100,
    n_waypoints: int = 50,
    n_substeps: int = 20,
    binning: str = "quantile",
    apply_layer_norm: bool = True,
    compute_pathways: bool = False,
    pathway_dict: Optional[Dict[str, List[str]]] = None,
    W_pathway: Optional[Union[np.ndarray, torch.Tensor]] = None,
    pathway_names: Optional[List[str]] = None,
    return_gene_scores: Optional[bool] = None,
    return_decomposition: bool = False,
    calibrate_ncg: bool = False,
    ncg_list: Optional[List[str]] = None,
    standardize: bool = False,
    batch_index: Optional[int] = None,
    ood_val: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Compute path-integrated PertVI perturbation momentum scores along latent transition trajectories.

    The Momentum Score quantifies the average causal transcriptional drive exerted by a genetic
    perturbation along a continuous latent state transition path P = {u_t}_{t=1}^T:

        M_p = (1 / T) * sum_{t=1}^T [ Decoder(z_p(u_t)) - Decoder(z_ctrl(u_t)) ]

    Because the PertVI generative head (W_head) is linear, the discrete path integration is
    factored into the 128-dimensional hidden representation space:

        M_p = W_head * ( (1 / T) * sum_{t=1}^T [ h(z_p(u_t)) - h(z_ctrl(u_t)) ] )

    This reduces computational FLOPs and peak VRAM requirements by over 900x while maintaining
    exact mathematical equivalence down to machine precision.

    Parameters
    ----------
    adata : AnnData
        Annotated single-cell dataset containing cell metadata and latent representations.
    model : PertVI
        Trained PertVI model instance.
    style : {'linear', 'trajectory-basic', 'trajectory-waypoint'}, default 'linear'
        Momentum trajectory construction style:
        - 'linear' (Algorithm A, default):
            Linear interpolation in normalized state space between start and end state centroids
            (u_start -> u_end) over `steps` uniform increments.
        - 'trajectory-basic' (Algorithm C):
            Empirical unperturbed control cell path ordered by continuous pseudotime (`dpt_key`),
            binned into `n_bins` quantile or equal-width centroid steps.
        - 'trajectory-waypoint' (Algorithm D):
            Anchor baseline centroid in `start_state` connected to `n_waypoints` condition-specific
            waypoints in `end_state` ordered by pseudotime, with `n_substeps` linear interpolation
            increments between adjacent nodes.
    pert_key : str, default 'perturbed_gene_name'
        Column name in adata.obs identifying genetic perturbations.
    control_group : str or list of str, default 'NTC'
        Identifier(s) for negative control / unperturbed cells in pert_key.
    control_name : str or list of str, optional
        Alias for control_group for backwards compatibility.
    control : str or list of str, optional
        Convenience alias for control_group.
    condition_key : str, optional, default 'condition'
        Column name in adata.obs identifying biological conditions or states.
    state_key : str, optional
        Alternative column name in adata.obs for state/condition identification (e.g. 'cell_state').
        Overrides condition_key if provided.
    start_state : str, optional
        Starting state identifier in adata.obs[condition_key] (e.g. 'Rest' or 'State1').
        Required for style='linear' and style='trajectory-waypoint'.
    end_state : str, optional
        End / destination state identifier in adata.obs[condition_key] (e.g. 'Stim8hr' or 'State3').
        Required for style='linear'. Optional for style='trajectory-waypoint' and style='trajectory-basic'.
    dpt_key : str, optional
        Column name in adata.obs containing continuous pseudotime coordinates (e.g. 'diff_pseudotime',
        'dpt_pseudotime'). Required for style='trajectory-basic' and style='trajectory-waypoint'.
    pseudotime_key : str, optional
        Alias for dpt_key.
    latent_key : str, default 'X_u_state'
        Key in adata.obsm storing unperturbed latent state coordinates.
    perturbations : list of str, optional
        Specific perturbations to compute. If None, evaluates all perturbations in the model
        categorical mapping excluding control identifiers.
    steps : int, default 1000
        Number of interpolation points for style='linear'.
    n_bins : int, default 100
        Number of empirical pseudotime bins for style='trajectory-basic'.
    n_waypoints : int, default 50
        Number of condition-specific waypoints for style='trajectory-waypoint'.
    n_substeps : int, default 20
        Number of linear interpolation increments between adjacent waypoints in style='trajectory-waypoint'.
    binning : {'quantile', 'equal_width'}, default 'quantile'
        Binning strategy for pseudotime trajectories. 'quantile' ensures equal cell counts per bin;
        'equal_width' matches uniform pseudotime discretization.
    apply_layer_norm : bool, default True
        If True, applies model.module.inference_net.split_film_transition.norm_state to raw u latents
        before computing centroids and waypoints.
    compute_pathways : bool, default False
        If True, projects momentum scores into pathway space via direct operator W_pathway.
    pathway_dict : dict of {str: list of str}, optional
        Dictionary mapping pathway names to member gene lists.
    W_pathway : np.ndarray or torch.Tensor, optional
        Precomputed pathway projection operator (shape: n_pathways x 128).
    pathway_names : list of str, optional
        Custom pathway names corresponding to rows of W_pathway.
    return_gene_scores : bool, optional
        If True, returns gene-level momentum scores. Defaults to True when compute_pathways=False,
        and False when compute_pathways=True unless explicitly requested.
    return_decomposition : bool, default False
        If True, returns a dictionary containing 'gene' (mean momentum), 'tiv' (trajectory interaction
        variance along the path), and 'ld' (landmark divergence between final and initial path points).
    calibrate_ncg : bool, default False
        If True, standardizes pathway scores against negative control genes (requires ncg_list).
    ncg_list : list of str, optional
        List of negative control gene identifiers for null calibration.
    standardize : bool, default False
        If True, standardizes pathway scores across perturbations.
    batch_index : int, optional
        Batch index for generative decoder batch embedding. If None, automatically infers the
        dominant batch among unperturbed control cells.
    ood_val : float, optional
        Nuisance value for the OOD latent dimension. If None, automatically uses the empirical mean
        of unperturbed control cells from adata.obsm['X_v_ood'] (or 0.0 if unavailable).
    device : str or torch.device, optional
        Computation device. Defaults to model parameter device.

    Returns
    -------
    results : pd.DataFrame, tuple of pd.DataFrame, or dict of pd.DataFrame
        - Default: pd.DataFrame of shape (n_genes, n_perturbations) containing gene-level momentum scores.
        - If compute_pathways=True and return_gene_scores=False: pd.DataFrame of shape (n_pathways, n_perturbations).
        - If both requested: tuple of (df_genes, df_pathways).
        - If return_decomposition=True: dict mapping keys ('gene', 'tiv', 'ld') to DataFrames.
    """
    # 1. Model device & eval state
    model.module.eval()
    if device is None:
        device = next(model.module.parameters()).device
    else:
        device = torch.device(device)

    # 2. Resolve Key Aliases & Columns
    cond_col = state_key if state_key is not None else condition_key
    if cond_col is None or cond_col not in adata.obs.columns:
        for cand in ("cell_state", "condition", "state", "cell_type"):
            if cand in adata.obs.columns:
                cond_col = cand
                break

    norm_style = style.lower().strip()
    if norm_style in ("linear", "trajectory-waypoint", "waypoint"):
        if cond_col is None or cond_col not in adata.obs.columns:
            raise KeyError(
                f"style='{style}' requires a valid condition/state column in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

    time_col = dpt_key if dpt_key is not None else pseudotime_key
    if time_col is None or time_col not in adata.obs.columns:
        for cand in ("diff_pseudotime", "dpt_pseudotime", "dpt_allcells", "pseudotime"):
            if cand in adata.obs.columns:
                time_col = cand
                break

    if norm_style in ("trajectory-basic", "trajectory-waypoint", "basic", "waypoint", "pseudotime"):
        if time_col is None or time_col not in adata.obs.columns:
            raise KeyError(
                f"style='{style}' requires a valid pseudotime column in adata.obs (specify via 'dpt_key'). "
                f"Available columns: {list(adata.obs.columns)}"
            )

    # Resolve Control Group Identifier(s)
    if control is not None:
        ctrl_list = [control] if isinstance(control, str) else list(control)
    elif control_name is not None:
        ctrl_list = [control_name] if isinstance(control_name, str) else list(control_name)
    else:
        ctrl_list = [control_group] if isinstance(control_group, str) else list(control_group)

    if pert_key not in adata.obs.columns:
        raise KeyError(f"pert_key '{pert_key}' not found in adata.obs.")

    ctrl_mask = adata.obs[pert_key].isin(ctrl_list).values
    if ctrl_mask.sum() == 0:
        raise ValueError(
            f"No control cells found matching control_group={ctrl_list} in adata.obs['{pert_key}']."
        )

    # 3. Latent Representation & LayerNorm Extraction
    if latent_key not in adata.obsm:
        raise KeyError(f"latent_key '{latent_key}' not found in adata.obsm.")
    u_raw_all = torch.tensor(adata.obsm[latent_key], dtype=torch.float32, device=device)

    film = getattr(model.module.inference_net, "split_film_transition", None)
    is_m7 = getattr(film, "film_mode", "legacy_hadamard") == "m7_block_tt"
    if apply_layer_norm and film is not None:
        if is_m7:
            u_norm_all = film.normalize_state_eval(u_raw_all)
        elif hasattr(film, "norm_state"):
            u_norm_all = film.norm_state(u_raw_all)
        else:
            u_norm_all = u_raw_all
    else:
        u_norm_all = u_raw_all

    # 4. Construct Latent Transition Path
    if norm_style == "linear":
        if start_state is None or end_state is None:
            raise ValueError(
                f"style='linear' requires both 'start_state' and 'end_state'. "
                f"Found start_state={start_state}, end_state={end_state}."
            )
        m_start = ctrl_mask & (adata.obs[cond_col] == start_state).values
        if m_start.sum() == 0:
            m_start = (adata.obs[cond_col] == start_state).values
        if m_start.sum() == 0:
            raise ValueError(f"No cells found for start_state='{start_state}' in adata.obs['{cond_col}'].")

        m_end = ctrl_mask & (adata.obs[cond_col] == end_state).values
        if m_end.sum() == 0:
            m_end = (adata.obs[cond_col] == end_state).values
        if m_end.sum() == 0:
            raise ValueError(f"No cells found for end_state='{end_state}' in adata.obs['{cond_col}'].")

        u_start = u_norm_all[m_start].mean(dim=0, keepdim=True)
        u_end = u_norm_all[m_end].mean(dim=0, keepdim=True)
        alphas = torch.linspace(0, 1, steps=steps, device=device).unsqueeze(1)
        path_tensor = u_start + alphas * (u_end - u_start)  # (steps, d_u)

    elif norm_style in ("trajectory-basic", "basic", "pseudotime"):
        if start_state is not None and end_state is not None and cond_col in adata.obs.columns:
            c_mask = ctrl_mask & adata.obs[cond_col].isin([start_state, end_state]).values
            if c_mask.sum() == 0:
                c_mask = ctrl_mask
        else:
            c_mask = ctrl_mask

        pts = adata.obs.loc[c_mask, time_col].values
        u_sub = u_norm_all[c_mask]

        valid_idx = ~np.isnan(pts)
        if valid_idx.sum() == 0:
            raise ValueError(f"All values in pseudotime column '{time_col}' are NaN.")
        pts = pts[valid_idx]
        u_sub = u_sub[valid_idx]

        sort_idx = np.argsort(pts)
        pts_sorted = pts[sort_idx]
        u_sorted = u_sub[sort_idx]

        if binning == "quantile":
            bin_edges = np.percentile(pts_sorted, np.linspace(0, 100, n_bins + 1))
        else:
            bin_edges = np.linspace(pts_sorted.min(), pts_sorted.max(), n_bins + 1)

        waypoints = []
        for i in range(len(bin_edges) - 1):
            if i == len(bin_edges) - 2:
                b_mask = (pts_sorted >= bin_edges[i]) & (pts_sorted <= bin_edges[i + 1])
            else:
                b_mask = (pts_sorted >= bin_edges[i]) & (pts_sorted < bin_edges[i + 1])
            if b_mask.sum() > 0:
                waypoints.append(u_sorted[b_mask].mean(dim=0))
            elif len(waypoints) > 0:
                waypoints.append(waypoints[-1])

        if len(waypoints) == 0:
            raise ValueError(f"Could not construct any valid trajectory bins along '{time_col}'.")
        path_tensor = torch.stack(waypoints, dim=0)  # (n_valid_bins, d_u)

    elif norm_style in ("trajectory-waypoint", "waypoint"):
        if start_state is None:
            raise ValueError("style='trajectory-waypoint' requires 'start_state' to define the baseline anchor.")
        m_start = ctrl_mask & (adata.obs[cond_col] == start_state).values
        if m_start.sum() == 0:
            m_start = (adata.obs[cond_col] == start_state).values
        if m_start.sum() == 0:
            raise ValueError(f"No cells found for start_state='{start_state}' in adata.obs['{cond_col}'].")
        u_anchor = u_norm_all[m_start].mean(dim=0, keepdim=True)  # (1, d_u)

        if end_state is not None:
            m_end = ctrl_mask & (adata.obs[cond_col] == end_state).values
            if m_end.sum() == 0:
                m_end = (adata.obs[cond_col] == end_state).values
        else:
            m_end = ctrl_mask & (adata.obs[cond_col] != start_state).values
            if m_end.sum() == 0:
                m_end = (adata.obs[cond_col] != start_state).values

        if m_end.sum() == 0:
            raise ValueError("Insufficient destination cells found to construct waypoints.")

        pts_end = adata.obs.loc[m_end, time_col].values
        u_end_sub = u_norm_all[m_end]

        valid_idx = ~np.isnan(pts_end)
        if valid_idx.sum() == 0:
            raise ValueError(f"All values in destination cells for '{time_col}' are NaN.")
        pts_end = pts_end[valid_idx]
        u_end_sub = u_end_sub[valid_idx]

        sort_idx = np.argsort(pts_end)
        pts_sorted = pts_end[sort_idx]
        u_sorted = u_end_sub[sort_idx]

        if binning == "quantile":
            bin_edges = np.percentile(pts_sorted, np.linspace(0, 100, n_waypoints + 1))
        else:
            bin_edges = np.linspace(pts_sorted.min(), pts_sorted.max(), n_waypoints + 1)

        nodes = [u_anchor.squeeze(0)]
        for i in range(len(bin_edges) - 1):
            if i == len(bin_edges) - 2:
                b_mask = (pts_sorted >= bin_edges[i]) & (pts_sorted <= bin_edges[i + 1])
            else:
                b_mask = (pts_sorted >= bin_edges[i]) & (pts_sorted < bin_edges[i + 1])
            if b_mask.sum() > 0:
                nodes.append(u_sorted[b_mask].mean(dim=0))
            elif len(nodes) > 1:
                nodes.append(nodes[-1])

        sub_steps_list = []
        alphas = torch.linspace(0, 1, steps=n_substeps + 1, device=device)[:-1].unsqueeze(1)
        for i in range(len(nodes) - 1):
            seg = nodes[i] + alphas * (nodes[i + 1] - nodes[i])
            sub_steps_list.append(seg)
        sub_steps_list.append(nodes[-1].unsqueeze(0))
        path_tensor = torch.cat(sub_steps_list, dim=0)  # (T, d_u)
    else:
        raise ValueError(
            f"Unsupported style: '{style}'. Choose from 'linear', 'trajectory-basic', or 'trajectory-waypoint'."
        )

    # 5. Extract FiLM Parameter Modulators / Sample Indices
    sample_reg = model.adata_manager.get_state_registry("sample")
    cat_mapping = sample_reg.categorical_mapping
    all_perts = [p for p in cat_mapping if p not in ctrl_list] if perturbations is None else perturbations

    sample_idx_map: Dict[str, int] = {}
    film_dict = {}
    for p in cat_mapping:
        matches = np.where(cat_mapping == p)[0]
        if len(matches) == 0:
            continue
        idx = int(matches[0])
        sample_idx_map[p] = idx
        if not is_m7:
            t = torch.tensor([[idx]], dtype=torch.long, device=device)
            emb = model.module.inference_net.sample_embedding(t).squeeze(1)
            g = film.gamma_state(emb).detach()
            b = film.beta_state(emb).detach()
            film_dict[p] = (g, b)

    ctrl_indices = [sample_idx_map[c] for c in ctrl_list if c in sample_idx_map]
    if not ctrl_indices:
        raise KeyError(
            f"None of the control group identifiers {ctrl_list} were found in the model categorical mapping."
        )
    if not is_m7:
        ctrl_films = [film_dict[c] for c in ctrl_list if c in film_dict]
        g_ctrl = torch.stack([cf[0] for cf in ctrl_films]).mean(dim=0)
        b_ctrl = torch.stack([cf[1] for cf in ctrl_films]).mean(dim=0)

    # 6. Setup Decoder Nuisance Embeddings: V_ood and batch_embed
    z_state_dim = getattr(film, "u_state_dim", None) or film.gamma_state.out_features
    main_in_dim = model.module.generative_net.main_hidden[0].in_features
    batch_embed_dim = (
        model.module.generative_net.batch_embed.embedding_dim if model.module.generative_net.use_batch else 0
    )
    n_ood = main_in_dim - batch_embed_dim - z_state_dim

    if ood_val is not None:
        v_ood = torch.full((1, n_ood), fill_value=float(ood_val), device=device)
    elif "X_v_ood" in adata.obsm and ctrl_mask.sum() > 0:
        v_ood_np = adata.obsm["X_v_ood"][ctrl_mask].mean(axis=0, keepdims=True)
        v_ood = torch.tensor(v_ood_np, dtype=torch.float32, device=device)
    else:
        v_ood = torch.zeros((1, n_ood), device=device)

    if model.module.generative_net.use_batch:
        if batch_index is None:
            batch_reg = model.adata_manager.get_state_registry("batch")
            batch_key = batch_reg.original_key
            batch_map = batch_reg.categorical_mapping
            if ctrl_mask.sum() > 0 and batch_key in adata.obs.columns:
                dominant_batch = adata.obs.loc[ctrl_mask, batch_key].mode()[0]
            else:
                dominant_batch = adata.obs[batch_key].mode()[0]
            batch_index = batch_map.tolist().index(dominant_batch)
        b_idx_t = torch.tensor([batch_index], dtype=torch.long, device=device)
        b_embed = model.module.generative_net.batch_embed(b_idx_t)
    else:
        b_embed = None

    def decode_hidden(z_input: torch.Tensor) -> torch.Tensor:
        n_c = z_input.shape[0]
        ood_exp = v_ood.expand(n_c, -1)
        if b_embed is not None:
            z_full = torch.cat([z_input, ood_exp, b_embed.expand(n_c, -1)], dim=-1)
        else:
            z_full = torch.cat([z_input, ood_exp], dim=-1)
        return model.module.generative_net.main_hidden(z_full)

    # 7. Evaluate Control Path Once: (T, 128)
    if is_m7:
        z_ctrl_path = torch.stack(
            [film.modulate_state_eval(path_tensor, c_idx) for c_idx in ctrl_indices], dim=0
        ).mean(dim=0)
    else:
        z_ctrl_path = (1.0 + g_ctrl) * path_tensor + b_ctrl
    h_ctrl_path = decode_hidden(z_ctrl_path)

    # 8. Factored Hidden-Space Momentum Accumulation
    delta_h_dict: Dict[str, torch.Tensor] = {}
    tiv_dict: Dict[str, torch.Tensor] = {}
    ld_dict: Dict[str, torch.Tensor] = {}

    active_perts = [p for p in all_perts if p in sample_idx_map]
    if not active_perts:
        raise ValueError("None of the requested perturbations were found in the model categorical mapping.")

    for p in active_perts:
        if is_m7:
            z_p_path = film.modulate_state_eval(path_tensor, sample_idx_map[p])
        else:
            g_p, b_p = film_dict[p]
            z_p_path = (1.0 + g_p) * path_tensor + b_p
        h_p_path = decode_hidden(z_p_path)

        diff_h = h_p_path - h_ctrl_path  # (T, 128)
        delta_h_dict[p] = diff_h.mean(dim=0)  # (128,)

        if return_decomposition:
            tiv_dict[p] = diff_h.var(dim=0)
            ld_dict[p] = diff_h[-1] - diff_h[0]

    delta_h_mat = torch.stack([delta_h_dict[p] for p in active_perts], dim=1)  # (128, n_perts)

    # 9. Output Projections
    if return_gene_scores is None:
        return_gene_scores = not compute_pathways

    if hasattr(model, "adata_manager") and model.adata_manager is not None:
        try:
            gene_names = list(model.adata_manager.get_state_registry("X_pertvi_log1p").column_names)
        except Exception:
            gene_names = list(adata.var_names)
    else:
        gene_names = list(adata.var_names)

    results: Dict[str, pd.DataFrame] = {}

    if return_gene_scores or return_decomposition:
        W_head = model.module.generative_net.main_head.weight.detach().to(device)  # (G, 128)
        gene_mom = torch.matmul(W_head, delta_h_mat).cpu().numpy()
        df_genes = pd.DataFrame(gene_mom, index=gene_names, columns=active_perts)
        results["gene"] = df_genes

        if return_decomposition:
            tiv_mat = torch.stack([tiv_dict[p] for p in active_perts], dim=1)
            ld_mat = torch.stack([ld_dict[p] for p in active_perts], dim=1)
            results["tiv"] = pd.DataFrame(
                torch.matmul(W_head.abs(), tiv_mat).cpu().numpy(), index=gene_names, columns=active_perts
            )
            results["ld"] = pd.DataFrame(
                torch.matmul(W_head, ld_mat).cpu().numpy(), index=gene_names, columns=active_perts
            )

    if compute_pathways:
        if W_pathway is not None:
            W_path_t = torch.as_tensor(W_pathway, dtype=torch.float32, device=device)
            p_names = (
                pathway_names
                if pathway_names is not None
                else [f"Pathway_{i}" for i in range(W_path_t.shape[0])]
            )
        elif pathway_dict is not None:
            W_np, p_names = build_pathway_head_operator(model, pathway_dict, gene_names=gene_names)
            W_path_t = torch.tensor(W_np, dtype=torch.float32, device=device)
        else:
            raise ValueError("compute_pathways=True requires 'pathway_dict' or 'W_pathway' to be provided.")

        path_mom = torch.matmul(W_path_t, delta_h_mat).cpu().numpy()
        df_path = pd.DataFrame(path_mom, index=p_names, columns=active_perts)

        if standardize:
            if calibrate_ncg:
                if ncg_list is None or len(ncg_list) < 2:
                    raise ValueError("ncg_list with at least 2 negative controls required when calibrate_ncg=True.")
                df_path = calibrate_ncg_null(df_path, ncg_list=ncg_list)
            else:
                df_path = calibrate_pool_null(df_path)
        results["pathway"] = df_path

    # 10. Packaging Returns
    if return_decomposition:
        return results
    if return_gene_scores and compute_pathways:
        return results["gene"], results["pathway"]
    return results["gene"] if return_gene_scores else results["pathway"]


# Momentum functional aliases
compute_momentum = compute_momentum_score


# ==============================================================================
# Out-of-Distribution (OOD) Analysis & Clean Signature Isolation
# ==============================================================================

def compute_ood_distance(
    adata,
    pert_key: str = "perturbed_gene_name",
    control_group: Union[str, List[str]] = "NTC",
    control_name: Optional[Union[str, List[str]]] = None,
    control: Optional[Union[str, List[str]]] = None,
    ood_key: str = "X_v_ood",
    perturbations: Optional[List[str]] = None,
    min_cells: int = 5,
    operational_threshold: float = 0.30,
    fit_gmm: bool = True,
    n_bootstraps_null: int = 1000,
) -> pd.DataFrame:
    """
    Compute Out-of-Distribution (OOD) latent displacement metrics for perturbations.

    The OOD latent distance measures the Euclidean displacement of the perturbed cell
    population from the unperturbed control centroid in PertVI's OOD latent space:

        c_p = E_{i in Cells(p)} [v_{ood, i}] in R^5
        c_ctrl = E_{j in Cells(control)} [v_{ood, j}] in R^5
        d_p = ||c_p - c_ctrl||_2

    This metric quantifies whether a perturbation induces an unconstrained cellular stress,
    toxicity, or chromatin crisis that escapes the physiological cell state manifold.

    Parameters
    ----------
    adata : AnnData
        Annotated single-cell dataset containing metadata and OOD latent representation.
    pert_key : str, default 'perturbed_gene_name'
        Column name in adata.obs identifying perturbations (e.g. 'perturbed_gene_name', 'compound_name').
    control_group : str or list of str, default 'NTC'
        Identifier(s) for negative control / unperturbed cells (e.g. 'NTC', 'DMSO').
    control_name : str or list of str, optional
        Alias for control_group.
    control : str or list of str, optional
        Convenience alias for control_group.
    ood_key : str, default 'X_v_ood'
        Key in adata.obsm storing OOD latent representations (R^d_ood).
    perturbations : list of str, optional
        Specific perturbations to compute. If None, computes all perturbations except control.
    min_cells : int, default 5
        Minimum number of cells required to compute distance for a perturbation.
    operational_threshold : float, default 0.30
        Operational distance cutoff defining off-manifold perturbations.
    fit_gmm : bool, default True
        If True, fits a 2-component Gaussian Mixture Model (GMM) to estimate posterior
        probability P(Off | d) of belonging to the off-manifold regime.
    n_bootstraps_null : int, default 1000
        Number of bootstrap iterations on control cells to calculate technical noise floor
        (99.9th percentile null displacement). If 0, skips null model calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by perturbation containing:
        - 'ood_distance': Euclidean distance ||c_p - c_ctrl||_2
        - 'rank': Rank ordered by distance (descending)
        - 'n_cells': Number of cells per perturbation
        - 'is_off_manifold': Boolean flag (ood_distance >= operational_threshold)
        - 'is_tukey_outlier': Boolean flag (ood_distance >= Q3 + 1.5 * IQR)
        - 'gmm_prob_off': Posterior probability of off-manifold regime (if fit_gmm=True)
        - 'is_above_null': Boolean flag exceeding 99.9th percentile null (if n_bootstraps_null > 0)
    """
    # Auto-resolve pert_key fallback
    if pert_key not in adata.obs.columns:
        if "compound_name" in adata.obs.columns:
            pert_key = "compound_name"
        else:
            raise KeyError(f"pert_key '{pert_key}' not found in adata.obs.")

    if ood_key not in adata.obsm:
        raise KeyError(f"ood_key '{ood_key}' not found in adata.obsm.")

    # Resolve control group identifier(s)
    if control is not None:
        control_group = control
    elif control_name is not None:
        control_group = control_name

    if isinstance(control_group, str):
        control_list = [control_group]
    else:
        control_list = list(control_group)

    # Auto-fallback for control_group (NTC vs DMSO)
    all_perts_in_obs = set(adata.obs[pert_key].unique())
    if not any(c in all_perts_in_obs for c in control_list):
        if "DMSO" in all_perts_in_obs:
            control_list = ["DMSO"]
        elif "NTC" in all_perts_in_obs:
            control_list = ["NTC"]
        else:
            raise ValueError(f"Control identifier(s) {control_list} not found in adata.obs['{pert_key}'].")

    ctrl_mask = adata.obs[pert_key].isin(control_list)
    if ctrl_mask.sum() == 0:
        raise ValueError(f"No control cells found matching {control_list} in adata.obs['{pert_key}'].")

    v_ood_all = adata.obsm[ood_key]
    v_ctrl = v_ood_all[ctrl_mask]
    c_ctrl = np.mean(v_ctrl, axis=0)

    # Resolve perturbations list
    if perturbations is None:
        candidate_perts = [p for p in adata.obs[pert_key].unique() if p not in control_list]
    else:
        candidate_perts = [p for p in perturbations if p not in control_list]

    records = []
    for p in candidate_perts:
        p_mask = adata.obs[pert_key] == p
        n_c = int(p_mask.sum())
        if n_c < min_cells:
            continue
        c_p = np.mean(v_ood_all[p_mask], axis=0)
        d_p = float(np.linalg.norm(c_p - c_ctrl))
        records.append({
            "perturbation": p,
            "ood_distance": d_p,
            "n_cells": n_c
        })

    if not records:
        raise ValueError("No perturbations had sufficient cells to compute OOD distance.")

    df = pd.DataFrame(records).sort_values("ood_distance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # Statistical fences
    q1 = df["ood_distance"].quantile(0.25)
    q3 = df["ood_distance"].quantile(0.75)
    iqr = q3 - q1
    tukey_fence = float(q3 + 1.5 * iqr)

    df["is_off_manifold"] = df["ood_distance"] >= operational_threshold
    df["is_tukey_outlier"] = df["ood_distance"] >= tukey_fence

    # Gaussian Mixture Model (GMM) decomposition
    if fit_gmm and len(df) >= 4:
        try:
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=2, random_state=42)
            X_d = df[["ood_distance"]].values
            gmm.fit(X_d)
            off_comp = int(np.argmax(gmm.means_.ravel()))
            probs = gmm.predict_proba(X_d)
            df["gmm_prob_off"] = probs[:, off_comp]
            df.attrs["gmm_means"] = gmm.means_.ravel()
            df.attrs["gmm_weights"] = gmm.weights_.ravel()
        except Exception as e:
            warnings.warn(f"GaussianMixture fitting skipped: {e}")

    # Bootstrap null model for technical noise floor
    if n_bootstraps_null > 0 and len(v_ctrl) >= 50:
        null_dists = []
        n_sub = min(500, len(v_ctrl) // 2)
        rng = np.random.default_rng(42)
        for _ in range(n_bootstraps_null):
            idx = rng.choice(len(v_ctrl), size=n_sub, replace=False)
            sub_c = np.mean(v_ctrl[idx], axis=0)
            null_dists.append(float(np.linalg.norm(sub_c - c_ctrl)))
        null_999 = float(np.percentile(null_dists, 99.9))
        df.attrs["null_noise_floor_999"] = null_999
        df["is_above_null"] = df["ood_distance"] > null_999

    df.attrs["operational_threshold"] = operational_threshold
    df.attrs["tukey_fence"] = tukey_fence
    df.set_index("perturbation", inplace=True)
    return df


def compute_ood_signature(
    adata,
    model,
    pert_key: str = "perturbed_gene_name",
    control_group: Union[str, List[str]] = "NTC",
    control_name: Optional[Union[str, List[str]]] = None,
    control: Optional[Union[str, List[str]]] = None,
    ood_key: str = "X_v_ood",
    state_key: str = "X_z_state",
    perturbations: Optional[List[str]] = None,
    method: str = "mco",
    batch_index: Optional[int] = None,
    chunk_size: int = 2048,
    return_tangent_operator: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Compute clean, leakage-free gene-level Out-of-Distribution (OOD) signatures.

    Derives the isolated gene expression modulation governed strictly by the OOD latent
    space (v_ood), provably free of variance leakage from cell cycle (z_cycle) or cell
    state (z_state) transitions.

    Mathematical Principles:
    ------------------------
    1. Architectural Cycle Invariance:
       In PertVI's BifurcatedDecoder, z_cycle is routed strictly to a masked linear head
       restricted to canonical cycle genes. Main head logits h_main satisfy:
           dh_main / dz_cycle == 0  (algebraically zero cycle leakage).

    2. State Variance Leakage in Naive Evaluation:
       Evaluating h_main at treated cell states z_state(c) modulates non-linear ReLU gating:
           D(a) = diag(I(a > 0)) in R^{128 x 128}
       This introduces 12% - 34% state variance leakage from the drug's phenotypic shift.

    3. Marginalized Counterfactual Operator (MCO, method='mco'):
       By defining expectation strictly over the invariant control reference manifold P_0:
           s_OOD*(p) = E_{(z, b) ~ P_0} [ h_main(z, v_ood(p), b) - h_main(z, v_ood(ctrl), b) ] in R^G
       state variance leakage is provably eliminated:
           d s_OOD*(p) / d(Delta z_state) == 0.

    4. Canonical Linear Tangent Operator (CLTO, method='clto'):
       First-order Taylor expansion around control manifold yields a linear tangent basis:
           M_OOD = W_head @ diag(p_active) @ W_ood_eff in R^{G x d_ood}
           s_CLTO(p) = M_OOD @ (v_ood(p) - v_ood(ctrl))
       where p_active = E_{P_0} [ I(a > 0) ] is the empirical activation frequency (r ~ 0.99 with MCO).

    Parameters
    ----------
    adata : AnnData
        Annotated single-cell dataset containing metadata and latent representations.
    model : PertVI
        Trained PertVI model instance.
    pert_key : str, default 'perturbed_gene_name'
        Column in adata.obs identifying perturbations (e.g. 'perturbed_gene_name', 'compound_name').
    control_group : str or list of str, default 'NTC'
        Identifier(s) for negative control / unperturbed cells (e.g. 'NTC', 'DMSO').
    control_name : str or list of str, optional
        Alias for control_group.
    control : str or list of str, optional
        Convenience alias for control_group.
    ood_key : str, default 'X_v_ood'
        Key in adata.obsm storing OOD latent representations.
    state_key : str, default 'X_z_state'
        Key in adata.obsm storing cell state latent representations.
    perturbations : list of str, optional
        List of perturbations to compute. If None, computes all perturbations except control.
    method : {'mco', 'clto', 'both'}, default 'mco'
        Decoding method:
        - 'mco': Marginalized Counterfactual Operator (exact, non-linear expectation).
        - 'clto': Canonical Linear Tangent Operator (linear pushforward).
        - 'both': Returns tuple of (df_mco, df_clto).
    batch_index : int, optional
        Specific batch index for batch embedding. Defaults to control cell batch distribution.
    chunk_size : int, default 2048
        Chunk size for counterfactual forward pass to prevent GPU out-of-memory.
    return_tangent_operator : bool, default False
        If True, also returns the tangent projection matrix M_OOD in R^{G x d_ood}.
    device : str or torch.device, optional
        Computation device. Defaults to model device.

    Returns
    -------
    pd.DataFrame or tuple of (pd.DataFrame, pd.DataFrame)
        DataFrame of shape (n_genes, n_perturbations) containing clean OOD log-fold changes.
        If method='both', returns (df_mco, df_clto).
        If return_tangent_operator=True, returns (df_signatures, M_ood_df).
    """
    if device is None:
        device = model.device

    gen = model.module.generative_net
    gen.eval()

    # Auto-resolve pert_key fallback
    if pert_key not in adata.obs.columns:
        if "compound_name" in adata.obs.columns:
            pert_key = "compound_name"
        else:
            raise KeyError(f"pert_key '{pert_key}' not found in adata.obs.")

    if ood_key not in adata.obsm:
        raise KeyError(f"ood_key '{ood_key}' not found in adata.obsm.")
    if state_key not in adata.obsm:
        raise KeyError(f"state_key '{state_key}' not found in adata.obsm.")

    # Resolve control group identifier(s)
    if control is not None:
        control_group = control
    elif control_name is not None:
        control_group = control_name

    if isinstance(control_group, str):
        control_list = [control_group]
    else:
        control_list = list(control_group)

    # Auto-fallback for control_group
    all_perts_in_obs = set(adata.obs[pert_key].unique())
    if not any(c in all_perts_in_obs for c in control_list):
        if "DMSO" in all_perts_in_obs:
            control_list = ["DMSO"]
        elif "NTC" in all_perts_in_obs:
            control_list = ["NTC"]
        else:
            raise ValueError(f"Control identifier(s) {control_list} not found in adata.obs['{pert_key}'].")

    ctrl_mask = adata.obs[pert_key].isin(control_list)
    if ctrl_mask.sum() == 0:
        raise ValueError(f"No control cells found matching {control_list} in adata.obs['{pert_key}'].")

    z_ctrl_all = torch.tensor(adata.obsm[state_key][ctrl_mask], device=device, dtype=torch.float32)
    v_ctrl_mean = torch.tensor(adata.obsm[ood_key][ctrl_mask], device=device, dtype=torch.float32).mean(dim=0, keepdim=True)
    n_ctrl = len(z_ctrl_all)

    # Resolve batch embeddings
    if hasattr(gen, "batch_embed") and gen.batch_embed is not None:
        if batch_index is not None:
            b_idx_t = torch.tensor([batch_index], dtype=torch.long, device=device)
            b_embed_ctrl = gen.batch_embed(b_idx_t).expand(n_ctrl, -1)
        elif "_scvi_batch" in adata.obs:
            b_indices = torch.tensor(adata.obs["_scvi_batch"][ctrl_mask].values, dtype=torch.long, device=device)
            b_embed_ctrl = gen.batch_embed(b_indices)
        else:
            b_embed_ctrl = gen.batch_embed(torch.zeros(n_ctrl, dtype=torch.long, device=device))
    else:
        b_embed_ctrl = None

    def forward_main(z_st: torch.Tensor, v_ood: torch.Tensor, b_emb: Optional[torch.Tensor]) -> torch.Tensor:
        if b_emb is not None:
            full_x = torch.cat([z_st, v_ood, b_emb], dim=-1)
        else:
            full_x = torch.cat([z_st, v_ood], dim=-1)
        return gen.main_head(gen.main_hidden(full_x))

    # Baseline control forward pass (chunked for memory safety)
    h_ctrl_list = []
    with torch.no_grad():
        for i in range(0, n_ctrl, chunk_size):
            end_i = min(i + chunk_size, n_ctrl)
            z_chunk = z_ctrl_all[i:end_i]
            v_chunk = v_ctrl_mean.expand(end_i - i, -1)
            b_chunk = b_embed_ctrl[i:end_i] if b_embed_ctrl is not None else None
            h_ctrl_list.append(forward_main(z_chunk, v_chunk, b_chunk))
        h_ctrl = torch.cat(h_ctrl_list, dim=0)

    # Gene names
    if hasattr(model, "adata_manager") and model.adata_manager is not None:
        try:
            gene_names = list(model.adata_manager.get_state_registry("X_pertvi_log1p").column_names)
        except Exception:
            gene_names = list(adata.var_names)
    else:
        gene_names = list(adata.var_names)

    # Resolve perturbations list
    if perturbations is None:
        candidate_perts = [p for p in adata.obs[pert_key].unique() if p not in control_list]
    else:
        candidate_perts = [p for p in perturbations if p not in control_list]

    mco_sigs = {}
    for p in candidate_perts:
        p_mask = adata.obs[pert_key] == p
        if p_mask.sum() == 0:
            continue
        v_p_mean = torch.tensor(adata.obsm[ood_key][p_mask], device=device, dtype=torch.float32).mean(dim=0, keepdim=True)

        h_p_list = []
        with torch.no_grad():
            for i in range(0, n_ctrl, chunk_size):
                end_i = min(i + chunk_size, n_ctrl)
                z_chunk = z_ctrl_all[i:end_i]
                v_chunk = v_p_mean.expand(end_i - i, -1)
                b_chunk = b_embed_ctrl[i:end_i] if b_embed_ctrl is not None else None
                h_p_list.append(forward_main(z_chunk, v_chunk, b_chunk))
            h_p = torch.cat(h_p_list, dim=0)
            delta_mco = (h_p - h_ctrl).mean(dim=0).cpu().numpy()

        mco_sigs[p] = delta_mco

    df_mco = pd.DataFrame(mco_sigs, index=gene_names)

    # Canonical Linear Tangent Operator (CLTO)
    M_ood_df = None
    if method in ["clto", "both"] or return_tangent_operator:
        with torch.no_grad():
            n_sub = min(n_ctrl, 2048)
            z_sub = z_ctrl_all[:n_sub]
            v_sub = v_ctrl_mean.expand(n_sub, -1)
            b_sub = b_embed_ctrl[:n_sub] if b_embed_ctrl is not None else None
            if b_sub is not None:
                full_sub = torch.cat([z_sub, v_sub, b_sub], dim=-1)
            else:
                full_sub = torch.cat([z_sub, v_sub], dim=-1)

            a_ctrl = gen.main_hidden[1](gen.main_hidden[0](full_sub))
            p_active = (a_ctrl > 0).float().mean(dim=0)  # (128,)

            w1 = gen.main_hidden[0].weight.detach()
            bn = gen.main_hidden[1]
            scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
            w1_eff = w1 * scale.unsqueeze(1)
            z_dim = z_ctrl_all.shape[1]
            v_dim = v_ctrl_mean.shape[1]
            w_ood = w1_eff[:, z_dim : z_dim + v_dim]
            w_head = gen.main_head.weight.detach()
            M_ood = torch.matmul(w_head * p_active.unsqueeze(0), w_ood)  # (G, v_dim)
            M_ood_df = pd.DataFrame(M_ood.cpu().numpy(), index=gene_names, columns=[f"v_ood_{i}" for i in range(v_dim)])

        clto_sigs = {}
        for p in candidate_perts:
            p_mask = adata.obs[pert_key] == p
            if p_mask.sum() == 0:
                continue
            v_p_mean = torch.tensor(adata.obsm[ood_key][p_mask], device=device, dtype=torch.float32).mean(dim=0, keepdim=True)
            delta_v = (v_p_mean - v_ctrl_mean).squeeze()
            delta_clto = torch.matmul(M_ood, delta_v).cpu().numpy()
            clto_sigs[p] = delta_clto

        df_clto = pd.DataFrame(clto_sigs, index=gene_names)

        if method == "clto":
            return (df_clto, M_ood_df) if return_tangent_operator else df_clto
        elif method == "both":
            return (df_mco, df_clto, M_ood_df) if return_tangent_operator else (df_mco, df_clto)

    if return_tangent_operator and M_ood_df is not None:
        return df_mco, M_ood_df

    return df_mco


# OOD Functional Aliases
compute_mco_ood_signature = compute_ood_signature
compute_clean_ood_signature = compute_ood_signature



