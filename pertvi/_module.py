from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.distributions import NegativeBinomial
from ._components import PertVI_Inference, BifurcatedDecoder


def _linear_cov_penalty(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape[0] <= 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    x_c = x - x.mean(dim=0, keepdim=True)
    y_c = y - y.mean(dim=0, keepdim=True)
    cov_mat = torch.matmul(x_c.t(), y_c) / (x.shape[0] - 1.0 + 1e-8)
    return torch.sum(cov_mat ** 2)


def _multiscale_rbf_kernel(x: torch.Tensor, scales: tuple = (0.5, 1.0, 2.0)) -> torch.Tensor:
    sq_norms = (x ** 2).sum(dim=-1, keepdim=True)
    dist_sq = (sq_norms + sq_norms.t() - 2.0 * torch.matmul(x, x.t())).clamp(min=0.0)
    with torch.no_grad():
        n = x.shape[0]
        triu_mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        off_diag = dist_sq.detach()[triu_mask]
        med_sq = torch.median(off_diag) if off_diag.numel() > 0 else torch.tensor(1.0, device=x.device)
        sigma_sq = torch.clamp(med_sq, min=1e-4)
    k_mat = sum(torch.exp(-dist_sq / (2.0 * (s ** 2) * sigma_sq)) for s in scales) / float(len(scales))
    return k_mat - torch.diag_embed(torch.diagonal(k_mat))


def _unbiased_hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    if n < 4:
        return _linear_cov_penalty(x, y)
    k_tilde = _multiscale_rbf_kernel(x)
    l_tilde = _multiscale_rbf_kernel(y)

    term1 = (k_tilde * l_tilde).sum()
    k_row = k_tilde.sum(dim=1)
    l_row = l_tilde.sum(dim=1)
    term2 = (k_row.sum() * l_row.sum()) / ((n - 1.0) * (n - 2.0))
    term3 = (2.0 / (n - 2.0)) * (k_row * l_row).sum()

    hsic_u = (term1 + term2 - term3) / (n * (n - 3.0))
    return hsic_u.clamp(min=0.0)


class PertVIModule(BaseModuleClass):
    def __init__(
        self,
        n_input: int,
        cycle_gene_mask: torch.Tensor,
        n_samples: int = 0,
        n_batch: int = 0,
        n_labels: int = 0,
        n_latent_cycle: int = 3,
        n_latent_state: int = 20,
        n_latent_ood: int = 5,  # Reduced capacity for OOD bottleneck
        drug_embed_dim: int = 16,  # Added embedding dimension
        n_obs_total: int = 1,
        sample_counts: Optional[torch.Tensor] = None,
        is_control_sample: Optional[torch.Tensor] = None,
        film_mode: str = "m7_block_tt",
        rank_gamma: int = 16,
        tau_snr: float = 1.5,
        m0_shrink: float = 10.0,
        eta0: float = 0.005,
        eps_iso: float = 0.05,
        lambda_film_gamma: float = 1.5,
        lambda_film_alpha: float = 1.0,
        basal_gene_dropout_rate: Union[float, str] = 0.15,
        ood_dropout_rate: float = 0.3,
        max_col_norm: Union[float, str, None] = "auto",
        ortho_mode: str = "linear_cov",
        **kwargs,
    ):
        super().__init__()
        if ortho_mode not in ("linear_cov", "hsic"):
            raise ValueError(f"Unsupported ortho_mode='{ortho_mode}'. Choose 'linear_cov' or 'hsic'.")

        self.n_latent_total = n_latent_cycle + n_latent_state + n_latent_ood
        self.film_mode = film_mode
        self.ortho_mode = ortho_mode
        self.lambda_film_gamma = float(lambda_film_gamma)
        self.lambda_film_alpha = float(lambda_film_alpha)

        if basal_gene_dropout_rate == "auto":
            resolved_basal_drop = 0.15 if film_mode == "m7_block_tt" else 0.0
        else:
            resolved_basal_drop = float(basal_gene_dropout_rate)

        if max_col_norm == "auto":
            resolved_max_col_norm = 2.0 if film_mode == "m7_block_tt" else None
        else:
            resolved_max_col_norm = max_col_norm

        self.inference_net = PertVI_Inference(
            n_input=n_input,
            n_samples=n_samples,
            n_latent_cycle=n_latent_cycle,
            n_latent_state=n_latent_state,
            n_latent_ood=n_latent_ood,
            ood_dropout_rate=ood_dropout_rate,
            basal_gene_dropout_rate=resolved_basal_drop,
            drug_embed_dim=drug_embed_dim,
            film_mode=film_mode,
            rank_gamma=rank_gamma,
            tau_snr=tau_snr,
            m0_shrink=m0_shrink,
            eta0=eta0,
            eps_iso=eps_iso,
            n_obs_total=n_obs_total,
            sample_counts=sample_counts,
            is_control_sample=is_control_sample,
        )

        # use custom bifurcated decoder
        self.generative_net = BifurcatedDecoder(
            n_input=self.n_latent_total,
            n_output=n_input,
            n_latent_cycle=n_latent_cycle,
            cycle_gene_mask=cycle_gene_mask,
            n_cat_list=[n_batch] if n_batch > 0 else None,
            n_layers=1,
            n_hidden=128,
            max_col_norm=resolved_max_col_norm,
        )

        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        self.current_lambda_adv = 0.0
        self.current_lambda_ood = 0.0

    # --- DATA ROUTERS ---
    def _get_inference_input(self, tensors, **kwargs):
        return {"tensors": tensors}

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        return {"tensors": tensors, "inference_outputs": inference_outputs}

    def _get_loss_input(self, tensors, inference_outputs, generative_outputs, **kwargs):
        return {
            "tensors": tensors,
            "inference_outputs": inference_outputs,
            "generative_outputs": generative_outputs,
        }

    # --- FORWARD PASSES ---
    @auto_move_data
    def inference(self, tensors):
        x = tensors["X"]
        x_safe = torch.clamp(x, min=0.0)
        x_log = torch.log(1 + x_safe)
        sample_index = tensors.get("sample")
        is_treated = tensors.get("is_treated")

        outputs = self.inference_net(x_log, sample_index, is_treated=is_treated)
        return outputs

    @auto_move_data
    def generative(self, tensors, inference_outputs):
        z = inference_outputs["z_final"]
        batch_index = tensors.get("batch")

        library_log = inference_outputs["library_log"].view(-1, 1)

        if batch_index is not None:
            px_scale, px_r, px_rate, px_dropout = self.generative_net("gene", z, library_log, batch_index)
        else:
            px_scale, px_r, px_rate, px_dropout = self.generative_net("gene", z, library_log)

        px_rate = torch.clamp(px_rate, min=1e-5)

        return {
            "px_scale": px_scale,
            "px_r": px_r,
            "px_rate": px_rate,
            "px_dropout": px_dropout,
        }

    # --- LOSS COMPUTATION ---
    def loss(self, tensors, inference_outputs, generative_outputs):
        # --- Unpack tensors & clamp ---
        x = tensors["X"]
        x_safe = torch.clamp(x, min=0.0)
        is_treated = tensors.get("is_treated", torch.ones(x.shape[0], 1, device=x.device))

        # Extract 1D masks for v_ood KL calculations (strictly preserving cell-level is_treated)
        treated_mask_1d = is_treated.view(-1)
        control_mask_1d = (1.0 - is_treated).view(-1)

        # --- Unpack inference outputs ---
        q_m_cycle, q_v_cycle = inference_outputs["q_m_cycle"], inference_outputs["q_v_cycle"]
        q_m_state, q_v_state = inference_outputs["q_m_state"], inference_outputs["q_v_state"]
        q_m_ood, q_v_ood = inference_outputs["q_m_ood"], inference_outputs["q_v_ood"]
        q_m_lib, q_v_lib = inference_outputs["q_m_lib"], inference_outputs["q_v_lib"]

        px_rate = generative_outputs["px_rate"]

        # --- Base VAE losses ---
        px_r_safe = torch.clamp(self.px_r, min=-10.0, max=15.0)
        px_r = torch.exp(px_r_safe) + 1e-4

        reconst_loss = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x_safe).sum(dim=-1)

        prior = Normal(0, 1)
        kl_cycle = kl(Normal(q_m_cycle, torch.sqrt(q_v_cycle)), prior).sum(dim=-1)
        kl_state = kl(Normal(q_m_state, torch.sqrt(q_v_state)), prior).sum(dim=-1)

        # Standard KL Beta for OOD
        # Control cells anchored to 0
        kl_ood_control = kl(Normal(torch.zeros_like(q_m_ood), torch.sqrt(q_v_ood)), prior).sum(dim=-1)
        kl_ood_treated = kl(Normal(q_m_ood, torch.sqrt(q_v_ood)), prior).sum(dim=-1)
        kl_ood_final = (kl_ood_control * control_mask_1d) + (kl_ood_treated * treated_mask_1d)

        # KL for library size
        empirical_counts = torch.clamp(x_safe.sum(dim=-1, keepdim=True), min=1.0)
        empirical_library_log = torch.log(empirical_counts).detach()
        prior_lib = Normal(empirical_library_log, torch.ones_like(empirical_library_log))
        kl_library = kl(Normal(q_m_lib, torch.sqrt(q_v_lib)), prior_lib).sum(dim=-1)

        # Topological relaxation
        beta_cyc = 0.5
        kl_local_unaggregated = (beta_cyc * kl_cycle) + kl_state + kl_ood_final + kl_library

        # --- Orthogonality / Independence constraints ---
        ortho_fn = _unbiased_hsic if self.ortho_mode == "hsic" else _linear_cov_penalty
        disentangle_state_cycle = ortho_fn(q_m_state, q_m_cycle)

        # OOD shielded constraints (Only on treated cells)
        treated_mask = treated_mask_1d > 0.5
        n_treat_mb = int(treated_mask.sum().item())
        min_treat = 4 if self.ortho_mode == "hsic" else 2
        if n_treat_mb >= min_treat:
            q_m_cycle_treated = q_m_cycle[treated_mask]
            q_m_state_treated = q_m_state[treated_mask]
            q_m_ood_treated = q_m_ood[treated_mask]
            disentangle_ood_cycle = ortho_fn(q_m_ood_treated, q_m_cycle_treated)
            disentangle_ood_state = ortho_fn(q_m_ood_treated, q_m_state_treated)
        elif n_treat_mb > 1:
            q_m_cycle_treated = q_m_cycle[treated_mask]
            q_m_state_treated = q_m_state[treated_mask]
            q_m_ood_treated = q_m_ood[treated_mask]
            disentangle_ood_cycle = _linear_cov_penalty(q_m_ood_treated, q_m_cycle_treated)
            disentangle_ood_state = _linear_cov_penalty(q_m_ood_treated, q_m_state_treated)
        else:
            disentangle_ood_cycle = torch.tensor(0.0, device=x.device)
            disentangle_ood_state = torch.tensor(0.0, device=x.device)

        # --- Single-Tiered L1 Sparsity ---
        control_mask_2d = 1.0 - is_treated.view(-1, 1)

        # Crush controls to absolute zero
        ood_control_l1 = torch.sum(torch.abs(q_m_ood * control_mask_2d), dim=-1)

        # --- Soft L1 penalty mask on main head ---
        cycle_mask_float = self.generative_net.cycle_gene_mask.float().unsqueeze(1)
        masked_weights = self.generative_net.main_head.weight * cycle_mask_float
        soft_l1_loss = torch.sum(torch.abs(masked_weights))

        # --- Closed-Form Functional Prior on Active FiLM Operators ---
        film_module = self.inference_net.split_film_transition
        if getattr(film_module, "film_mode", "legacy_hadamard") == "m7_block_tt":
            n_total = max(1.0, float(film_module.n_obs_total.item()))
            film_prior_loss = (
                film_module.compute_film_prior_loss(
                    lambda_gamma=self.lambda_film_gamma,
                    lambda_alpha=self.lambda_film_alpha,
                )
                / n_total
            )
        else:
            film_prior_loss = torch.tensor(0.0, device=x.device)

        # --- Dynamic Weighting & Aggregation ---
        lambda_kl = 1.0
        lambda_adv_state = getattr(self, "current_lambda_adv_state", 100.0)
        lambda_adv_ood = getattr(self, "current_lambda_adv_ood", 5.0)
        lambda_adv_state_ood = 5.0
        lambda_ood_anchor = getattr(self, "current_lambda_ood", 50.0)
        lambda_soft = 50.0

        # --- Base loss ---
        base_loss_unaggregated = (
            reconst_loss
            + (lambda_kl * kl_local_unaggregated)
            + (lambda_ood_anchor * ood_control_l1)
        )

        # --- Total loss ---
        total_loss = (
            torch.mean(base_loss_unaggregated)
            + (lambda_adv_state * disentangle_state_cycle)
            + (lambda_adv_ood * disentangle_ood_cycle)
            + (lambda_adv_state_ood * disentangle_ood_state)
            + (lambda_soft * soft_l1_loss)
            + film_prior_loss
        )

        return LossOutput(
            loss=total_loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local_unaggregated,
            n_obs_minibatch=x.shape[0],
            extra_metrics={
                "covariance_penalty_state": disentangle_state_cycle,
                "covariance_penalty_ood": disentangle_ood_cycle,
                "covariance_penalty_state_ood": disentangle_ood_state,
                "soft_l1_penalty": soft_l1_loss,
                "film_prior_loss": film_prior_loss,
                "v_ood_active_fraction": torch.mean((torch.abs(q_m_ood) > 0.5).float()),
            },
        )
