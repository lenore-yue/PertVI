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


def _within_perturbation_eb_center(
    q_m: torch.Tensor,
    sample_idx: Optional[torch.Tensor],
    kappa0: float = 5.0,
) -> torch.Tensor:
    """Computes empirical-Bayes shrunk within-perturbation residuals:
       delta_q_{m,i} = q_{m,i} - ((n_s / (n_s + kappa0)) * mu_s + (kappa0 / (n_s + kappa0)) * mu_batch).
    """
    if sample_idx is None or q_m.shape[0] <= 2:
        return q_m - q_m.mean(dim=0, keepdim=True)
    idx = sample_idx.reshape(-1).long()
    mu_batch = q_m.mean(dim=0, keepdim=True)
    mu_shrunk = mu_batch.expand_as(q_m).clone()
    for s_val in torch.unique(idx):
        mask_s = idx == s_val
        n_s = float(mask_s.sum().item())
        if n_s >= 1.0:
            w_s = n_s / (n_s + float(kappa0))
            mu_s = q_m[mask_s].mean(dim=0, keepdim=True)
            mu_shrunk[mask_s] = w_s * mu_s + (1.0 - w_s) * mu_batch
    return q_m - mu_shrunk


def _snr_whitened_harmonic_hsic_penalty(
    q_m_state: torch.Tensor,
    q_v_state: torch.Tensor,
    q_m_cycle_detached: torch.Tensor,
) -> torch.Tensor:
    """Scale-Invariant SNR-Whitened Spherical/Cylindrical Harmonic + RBF-HSIC Orthogonality Penalty."""
    n = q_m_state.shape[0]
    if n < 4:
        return _linear_cov_penalty(q_m_state, q_m_cycle_detached)

    xc = q_m_state - q_m_state.mean(dim=0, keepdim=True)
    var_qm = xc.pow(2).mean(dim=0, keepdim=True)
    mean_qv = q_v_state.detach().mean(dim=0, keepdim=True)
    x_tilde = xc / torch.sqrt(var_qm + mean_qv + 1e-5)

    y_raw = q_m_cycle_detached - q_m_cycle_detached.mean(dim=0, keepdim=True)
    if y_raw.shape[1] < 3:
        y1 = F.pad(y_raw, (0, 3 - y_raw.shape[1]))
    else:
        y1 = y_raw[:, :3]

    rad = torch.sqrt(y_raw.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
    y2 = torch.cat(
        [
            y_raw.pow(2),
            y1[:, 0:1] * y1[:, 1:2],
            y1[:, 0:1] * y1[:, 2:3],
            y1[:, 1:2] * y1[:, 2:3],
        ],
        dim=-1,
    )
    y3 = torch.cat(
        [
            y1.pow(3),
            (y1[:, 0:1].pow(2)) * y1[:, 1:2],
            (y1[:, 1:2].pow(2)) * y1[:, 0:1],
            y1[:, 0:1] * y1[:, 1:2] * y1[:, 2:3],
        ],
        dim=-1,
    )
    phi = torch.cat([y_raw, rad, y2, y3], dim=-1)
    phi_c = phi - phi.mean(dim=0, keepdim=True)
    phi_hat = phi_c / torch.sqrt(phi_c.pow(2).mean(dim=0, keepdim=True) + 1e-6)

    corr_mat = torch.matmul(x_tilde.t(), phi_hat) / (n - 1.0 + 1e-8)
    harm_pen = torch.sum(corr_mat ** 2) / 3.0

    y_hat = y_raw / torch.sqrt(y_raw.pow(2).mean(dim=0, keepdim=True) + 1e-6)
    hsic_pen = 10.0 * _unbiased_hsic(x_tilde, y_hat)
    return harm_pen + hsic_pen


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
        n_latent_ood: int = 5,
        drug_embed_dim: int = 16,
        n_obs_total: int = 1,
        sample_counts: Optional[torch.Tensor] = None,
        is_control_sample: Optional[torch.Tensor] = None,
        state_exclude_mask: Optional[torch.Tensor] = None,
        ood_quarantine_mask: Optional[torch.Tensor] = None,
        pathway_mask: Optional[torch.Tensor] = None,
        cycle_encoder_mode: str = "standard",
        cycle_decoder_mode: str = "linear",
        split_ood_decoder: bool = False,
        direct_pathway_latent: bool = False,
        n_hidden_per_pathway: int = 16,
        film_mode: str = "m7_block_tt",
        transition_mode: Optional[str] = None,
        cycle_norm_mode: str = "matched_ln",
        cycle_prior_mode: str = "standard_normal",
        sigma_cycle_xy: float = 1.0,
        sigma_cycle_z: float = 0.05,
        r0_ring: float = 1.02,
        tau_hole: float = 0.0,
        lambda_circ: float = 0.0,
        lambda_drug_cycle_align: float = 0.0,
        lambda_go_pathway: float = 5.0,
        deleak_radial_cycle: bool = False,
        selective_cycle_input: bool = False,
        ood_all_cells_hsic: bool = False,
        within_pert_hsic_weight: float = 0.0,
        rank_gamma: int = 16,
        tau_snr: float = 1.5,
        m0_shrink: float = 10.0,
        eta0: float = 0.005,
        eps_iso: float = 0.05,
        lambda_film_gamma: float = 1.5,
        lambda_film_alpha: float = 1.0,
        lambda_film_gamma_cycle: Optional[float] = None,
        lambda_film_alpha_cycle: Optional[float] = None,
        basal_gene_dropout_rate: Union[float, str] = 0.15,
        ood_sub_dropout_rate: float = 0.0,
        ood_dropout_rate: float = 0.3,
        dropout_rate: float = 0.1,
        max_col_norm: Union[float, str, None] = "auto",
        ortho_mode: str = "linear_cov",
        ood_anchor_mode: str = "l1",
        max_lambda_adv_state: float = 100.0,
        max_lambda_adv_ood: float = 5.0,
        lambda_soft: float = 50.0,
        floor_lambda_ood: float = 5.0,
        max_lambda_ood: float = 50.0,
        **kwargs,
    ):
        super().__init__()
        if transition_mode is not None:
            film_mode = transition_mode
        if ortho_mode == "snr_whitened_harmonic_hsic":
            ortho_mode = "snr_harmonic_hsic"
        if ortho_mode not in ("snr_harmonic_hsic", "linear_cov", "hsic"):
            raise ValueError(
                f"Unsupported ortho_mode='{ortho_mode}'. Choose 'snr_harmonic_hsic', 'linear_cov', or 'hsic'."
            )
        if ood_anchor_mode not in ("l2", "l1"):
            raise ValueError(
                f"Unsupported ood_anchor_mode='{ood_anchor_mode}'. Choose 'l2' or 'l1'."
            )

        self.n_latent_cycle = int(n_latent_cycle)
        self.n_latent_state = int(n_latent_state)
        self.n_latent_ood = int(n_latent_ood)
        self.n_latent_total = self.n_latent_cycle + self.n_latent_state + self.n_latent_ood
        self.film_mode = film_mode
        self.cycle_encoder_mode = str(cycle_encoder_mode)
        self.cycle_decoder_mode = str(cycle_decoder_mode)
        self.split_ood_decoder = bool(split_ood_decoder)
        self.cycle_norm_mode = str(cycle_norm_mode)
        self.cycle_prior_mode = str(cycle_prior_mode)
        self.sigma_cycle_xy = float(sigma_cycle_xy)
        self.sigma_cycle_z = float(sigma_cycle_z)
        self.r0_ring = float(r0_ring)
        self.tau_hole = float(tau_hole)
        self.lambda_circ = float(lambda_circ)
        self.lambda_drug_cycle_align = float(lambda_drug_cycle_align)
        self.lambda_go_pathway = float(lambda_go_pathway)
        self.r_star_cycle = (
            0.5 * (self.r0_ring + ((self.r0_ring ** 2 + 4.0 * self.tau_hole * (self.sigma_cycle_xy ** 2)) ** 0.5))
            if (self.cycle_prior_mode == "annular_ring" and self.tau_hole > 0.0)
            else 1.0
        )
        self.selective_cycle_input = bool(selective_cycle_input)
        self.ood_all_cells_hsic = bool(ood_all_cells_hsic)
        self.within_pert_hsic_weight = float(within_pert_hsic_weight)
        self.ortho_mode = ortho_mode
        self.ood_anchor_mode = ood_anchor_mode
        self.lambda_film_gamma = float(lambda_film_gamma)
        self.lambda_film_alpha = float(lambda_film_alpha)
        self.max_lambda_adv_state = float(max_lambda_adv_state)
        self.max_lambda_adv_ood = float(max_lambda_adv_ood)
        self.lambda_soft = float(lambda_soft)
        self.floor_lambda_ood = float(floor_lambda_ood)
        self.max_lambda_ood = float(max_lambda_ood)

        if basal_gene_dropout_rate == "auto":
            resolved_basal_drop = 0.05 if self.selective_cycle_input else (0.15 if film_mode == "m7_block_tt" else 0.0)
        else:
            resolved_basal_drop = float(basal_gene_dropout_rate)

        resolved_ood_sub_drop = float(ood_sub_dropout_rate) if film_mode == "m7_block_tt" else 0.0
        resolved_selective_cyc = bool(selective_cycle_input) if film_mode == "m7_block_tt" else False

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
            ood_sub_dropout_rate=resolved_ood_sub_drop,
            selective_cycle_input=resolved_selective_cyc,
            cycle_encoder_mode=self.cycle_encoder_mode,
            cycle_norm_mode=self.cycle_norm_mode,
            r_star_cycle=self.r_star_cycle,
            deleak_radial_cycle=deleak_radial_cycle,
            direct_pathway_latent=direct_pathway_latent,
            n_hidden_per_pathway=n_hidden_per_pathway,
            dropout_rate=dropout_rate,
            cycle_gene_mask=cycle_gene_mask,
            state_exclude_mask=state_exclude_mask,
            ood_quarantine_mask=ood_quarantine_mask,
            pathway_mask=pathway_mask,
            drug_embed_dim=drug_embed_dim,
            film_mode=film_mode,
            lambda_gamma_state=lambda_film_gamma,
            lambda_alpha_state=lambda_film_alpha,
            lambda_gamma_cycle=lambda_film_gamma_cycle,
            lambda_alpha_cycle=lambda_film_alpha_cycle,
            rank_gamma=rank_gamma,
            tau_snr=tau_snr,
            m0_shrink=m0_shrink,
            eta0=eta0,
            eps_iso=eps_iso,
            n_obs_total=n_obs_total,
            sample_counts=sample_counts,
            is_control_sample=is_control_sample,
        )

        self.generative_net = BifurcatedDecoder(
            n_input=self.n_latent_total,
            n_output=n_input,
            n_latent_cycle=n_latent_cycle,
            n_latent_state=n_latent_state,
            n_latent_ood=n_latent_ood,
            cycle_gene_mask=cycle_gene_mask,
            pathway_mask=pathway_mask,
            state_exclude_mask=state_exclude_mask,
            ood_quarantine_mask=ood_quarantine_mask,
            cycle_decoder_mode=self.cycle_decoder_mode,
            split_ood_decoder=self.split_ood_decoder,
            n_cat_list=[n_batch] if n_batch > 0 else None,
            n_layers=1,
            n_hidden=128,
            max_col_norm=resolved_max_col_norm,
        )

        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        self.current_lambda_adv = 0.0
        self.current_lambda_ood = 0.0

    @property
    def z_encoder(self) -> PertVI_Inference:
        return self.inference_net

    def _snr_whitened_harmonic_hsic_penalty(
        self,
        q_m_state: torch.Tensor,
        q_v_state: torch.Tensor,
        q_m_cycle_detached: torch.Tensor,
    ) -> torch.Tensor:
        return _snr_whitened_harmonic_hsic_penalty(q_m_state, q_v_state, q_m_cycle_detached)

    def _compute_disentangle_penalty(
        self,
        q_m: torch.Tensor,
        q_v: torch.Tensor,
        q_m_cycle_detached: torch.Tensor,
        sample_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.ortho_mode == "snr_harmonic_hsic":
            marg_pen = _snr_whitened_harmonic_hsic_penalty(q_m, q_v, q_m_cycle_detached)
            if self.within_pert_hsic_weight > 0.0 and sample_idx is not None:
                dq_m = _within_perturbation_eb_center(q_m, sample_idx)
                dq_cyc = _within_perturbation_eb_center(q_m_cycle_detached, sample_idx).detach()
                cond_pen = _snr_whitened_harmonic_hsic_penalty(dq_m, q_v, dq_cyc)
                w = float(self.within_pert_hsic_weight)
                return (1.0 - w) * marg_pen + w * cond_pen
            return marg_pen
        elif self.ortho_mode == "hsic":
            return _unbiased_hsic(q_m, q_m_cycle_detached)
        else:
            return _linear_cov_penalty(q_m, q_m_cycle_detached)

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
        if self.cycle_decoder_mode in ("go_cylindrical", "cylindrical_harmonic"):
            outputs["u_cycle_go"] = self.generative_net.compute_go_pathway_coordinates(outputs["u_cycle_src"])
            outputs["z_cycle_go"] = self.generative_net.compute_go_pathway_coordinates(outputs["z_cycle"])
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
        x = tensors["X"]
        x_safe = torch.clamp(x, min=0.0)
        is_treated = tensors.get("is_treated", torch.ones(x.shape[0], 1, device=x.device))
        sample_idx = tensors.get("sample")
        if sample_idx is not None:
            sample_idx = sample_idx.view(-1).long()

        treated_mask_1d = is_treated.view(-1)
        control_mask_1d = (1.0 - is_treated).view(-1)

        q_m_cycle, q_v_cycle = inference_outputs["q_m_cycle"], inference_outputs["q_v_cycle"]
        q_m_state, q_v_state = inference_outputs["q_m_state"], inference_outputs["q_v_state"]
        q_m_ood, q_v_ood = inference_outputs["q_m_ood"], inference_outputs["q_v_ood"]
        q_m_lib, q_v_lib = inference_outputs["q_m_lib"], inference_outputs["q_v_lib"]

        px_rate = generative_outputs["px_rate"]

        px_r_safe = torch.clamp(self.px_r, min=-10.0, max=15.0)
        px_r = torch.exp(px_r_safe) + 1e-4

        reconst_loss = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x_safe).sum(dim=-1)

        prior = Normal(0, 1)
        kl_state = kl(Normal(q_m_state, torch.sqrt(q_v_state)), prior).sum(dim=-1)

        kl_ood_control = kl(Normal(torch.zeros_like(q_m_ood), torch.sqrt(q_v_ood)), prior).sum(dim=-1)
        kl_ood_treated = kl(Normal(q_m_ood, torch.sqrt(q_v_ood)), prior).sum(dim=-1)
        kl_ood_final = (kl_ood_control * control_mask_1d) + (kl_ood_treated * treated_mask_1d)

        empirical_counts = torch.clamp(x_safe.sum(dim=-1, keepdim=True), min=1.0)
        empirical_library_log = torch.log(empirical_counts).detach()
        prior_lib = Normal(empirical_library_log, torch.ones_like(empirical_library_log))
        kl_library = kl(Normal(q_m_lib, torch.sqrt(q_v_lib)), prior_lib).sum(dim=-1)

        # --- Cycle KL Prior & Optional Topological Terms ---
        l_circ = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        l_drug_cycle = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.cycle_prior_mode == "anisotropic_normal" and q_m_cycle.shape[1] == 3:
            prior_xy = Normal(0.0, self.sigma_cycle_xy)
            kl_xy = kl(Normal(q_m_cycle[:, :2], torch.sqrt(q_v_cycle[:, :2])), prior_xy).sum(dim=-1)
            sz2 = self.sigma_cycle_z ** 2
            kl_z2 = (q_m_cycle[:, 2].pow(2)) / (2.0 * sz2) + 0.5 * (
                (q_v_cycle[:, 2] / sz2) - torch.log(q_v_cycle[:, 2] / sz2) - 1.0
            )
            kl_cycle_weighted = kl_xy + kl_z2
        elif self.cycle_prior_mode == "annular_ring" and q_m_cycle.shape[1] == 3:
            r12 = torch.sqrt(q_m_cycle[:, 0].pow(2) + q_m_cycle[:, 1].pow(2) + 1e-6)
            l_rad = ((r12 - self.r0_ring).pow(2)) / (2.0 * (self.sigma_cycle_xy ** 2)) - (
                self.tau_hole * torch.log(r12 + 1e-3)
            )
            l_z2 = (q_m_cycle[:, 2].pow(2)) / (2.0 * (self.sigma_cycle_z ** 2))
            var_target = torch.tensor(
                [self.sigma_cycle_xy ** 2, self.sigma_cycle_xy ** 2, self.sigma_cycle_z ** 2],
                device=x.device,
                dtype=x.dtype,
            ).unsqueeze(0)
            l_qv = 0.5 * torch.sum((q_v_cycle / var_target) - torch.log(q_v_cycle / var_target) - 1.0, dim=-1)
            kl_cycle_weighted = l_rad + l_z2 + l_qv
            if self.lambda_circ > 0.0:
                c1 = q_m_cycle[:, 0] / r12
                s1 = q_m_cycle[:, 1] / r12
                c2 = c1.pow(2) - s1.pow(2)
                s2 = 2.0 * s1 * c1
                l_circ = self.lambda_circ * (
                    c1.mean().pow(2) + s1.mean().pow(2) + 0.5 * (c2.mean().pow(2) + s2.mean().pow(2))
                )
        else:
            beta_cyc = 0.5
            kl_cycle = kl(Normal(q_m_cycle, torch.sqrt(q_v_cycle)), prior).sum(dim=-1)
            kl_cycle_weighted = beta_cyc * kl_cycle

        if self.lambda_drug_cycle_align > 0.0 and sample_idx is not None:
            r12_all = (
                torch.sqrt(q_m_cycle[:, 0].pow(2) + q_m_cycle[:, 1].pow(2) + 1e-6)
                if q_m_cycle.shape[1] >= 2
                else torch.sqrt(q_m_cycle.pow(2).sum(dim=-1) + 1e-6)
            )
            r_global = r12_all.mean().detach()
            n_groups = 0
            for s_val in torch.unique(sample_idx):
                m_s = sample_idx == s_val
                if int(m_s.sum().item()) >= 3:
                    r_mean_s = r12_all[m_s].mean()
                    u_mean_s = q_m_cycle[m_s].mean(dim=0)
                    l_drug_cycle = l_drug_cycle + ((r_mean_s - r_global).pow(2)) + 0.5 * u_mean_s.pow(2).sum()
                    n_groups += 1
            if n_groups > 0:
                l_drug_cycle = self.lambda_drug_cycle_align * (l_drug_cycle / float(n_groups))

        # --- GO Pathway Coordinate Alignment Loss (v0.4.2) ---
        l_go_pathway = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if (
            self.lambda_go_pathway > 0.0
            and "emp_go_scores" in inference_outputs
            and inference_outputs["emp_go_scores"] is not None
        ):
            emp_go = inference_outputs["emp_go_scores"].detach()
            emp_go_std = emp_go / (emp_go.std(dim=0, keepdim=True) + 1e-5)
            if "u_cycle_go" in inference_outputs and inference_outputs["u_cycle_go"] is not None:
                u_go = inference_outputs["u_cycle_go"]
                l_go_pathway = l_go_pathway + F.mse_loss(u_go, emp_go_std)
            if "u_cycle_go_enc" in inference_outputs and inference_outputs["u_cycle_go_enc"] is not None:
                u_go_enc = inference_outputs["u_cycle_go_enc"]
                l_go_pathway = l_go_pathway + 0.5 * F.mse_loss(u_go_enc, emp_go_std)
            l_go_pathway = self.lambda_go_pathway * l_go_pathway

        kl_local_unaggregated = kl_cycle_weighted + kl_state + kl_ood_final + kl_library

        # --- Orthogonality / Independence constraints (always detach q_m_cycle) ---
        q_m_cycle_detached = q_m_cycle.detach()
        disentangle_state_cycle = self._compute_disentangle_penalty(
            q_m_state, q_v_state, q_m_cycle_detached, sample_idx=sample_idx
        )

        treated_mask = treated_mask_1d > 0.5
        n_treat_mb = int(treated_mask.sum().item())

        if self.ood_all_cells_hsic and self.ortho_mode == "snr_harmonic_hsic":
            disentangle_ood_cycle = self._compute_disentangle_penalty(
                q_m_ood, q_v_ood, q_m_cycle_detached, sample_idx=sample_idx
            )
            if n_treat_mb >= 2:
                disentangle_ood_state = _linear_cov_penalty(q_m_ood[treated_mask], q_m_state[treated_mask])
            else:
                disentangle_ood_state = torch.tensor(0.0, device=x.device)
        else:
            ood_ortho_fn = _unbiased_hsic if self.ortho_mode == "hsic" else _linear_cov_penalty
            min_treat = 4 if self.ortho_mode == "hsic" else 2
            if n_treat_mb >= min_treat:
                q_m_cycle_treated = q_m_cycle_detached[treated_mask]
                q_m_state_treated = q_m_state[treated_mask]
                q_m_ood_treated = q_m_ood[treated_mask]
                disentangle_ood_cycle = ood_ortho_fn(q_m_ood_treated, q_m_cycle_treated)
                disentangle_ood_state = ood_ortho_fn(q_m_ood_treated, q_m_state_treated)
            elif n_treat_mb > 1:
                q_m_cycle_treated = q_m_cycle_detached[treated_mask]
                q_m_state_treated = q_m_state[treated_mask]
                q_m_ood_treated = q_m_ood[treated_mask]
                disentangle_ood_cycle = _linear_cov_penalty(q_m_ood_treated, q_m_cycle_treated)
                disentangle_ood_state = _linear_cov_penalty(q_m_ood_treated, q_m_state_treated)
            else:
                disentangle_ood_cycle = torch.tensor(0.0, device=x.device)
                disentangle_ood_state = torch.tensor(0.0, device=x.device)

        # --- Control Anchor on v_ood (Smooth L2 or L1) ---
        if self.ood_anchor_mode == "l2":
            ood_ctrl_penalty_per_cell = 0.5 * torch.sum(q_m_ood ** 2, dim=-1)
        else:
            ood_ctrl_penalty_per_cell = torch.sum(torch.abs(q_m_ood), dim=-1)
        ood_control_pen = ood_ctrl_penalty_per_cell * control_mask_1d

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
        lambda_adv_state = getattr(self, "current_lambda_adv_state", self.max_lambda_adv_state)
        lambda_adv_ood = getattr(self, "current_lambda_adv_ood", self.max_lambda_adv_ood)
        lambda_adv_state_ood = 5.0
        lambda_ood_anchor = getattr(self, "current_lambda_ood", self.max_lambda_ood)
        lambda_soft = self.lambda_soft

        base_loss_unaggregated = (
            reconst_loss
            + (lambda_kl * kl_local_unaggregated)
            + (lambda_ood_anchor * ood_control_pen)
        )

        total_loss = (
            torch.mean(base_loss_unaggregated)
            + l_circ
            + l_drug_cycle
            + l_go_pathway
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
                "go_pathway_loss": l_go_pathway,
                "soft_l1_penalty": soft_l1_loss,
                "film_prior_loss": film_prior_loss,
                "v_ood_active_fraction": torch.mean((torch.abs(q_m_ood) > 0.5).float()),
            },
        )
