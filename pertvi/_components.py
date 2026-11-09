from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
from scvi.nn import Encoder


# ==========================================
# Decoder Column Max-Norm Parametrization
# ==========================================
class MaxColNorm(nn.Module):
    """Bounds each column norm of a Linear weight matrix: ||W[:, d]||_2 <= max_col_norm."""

    def __init__(self, max_col_norm: float = 2.0):
        super().__init__()
        self.max_col_norm = float(max_col_norm)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        col_norms = torch.sqrt(W.pow(2).sum(dim=0, keepdim=True) + 1e-12)
        scale = (col_norms / self.max_col_norm).clamp(min=1.0)
        return W / scale


# ==========================================
# FiLM transition block (v0.4.0 M7-InvDrop + legacy Hadamard)
# ==========================================
class SplitNaturalShiftFiLM(nn.Module):
    """Disentangled state and cell-cycle FiLM transition module.

    Supports:
    - ``film_mode="m7_block_tt"`` (default in v0.4.0): Stoch-SNR Block-TT WP-FiLM with
      variance-matched bias-free LayerNorm (``MatchedLN``), sample-level control identity
      clamping, count-weighted EMA sufficient statistics, Pre-LN Hill-2 KL-SNR gates,
      and exact Schur-product functional L2 prior over pooled + control reference covariance.
    - ``film_mode="legacy_hadamard"``: backward-compatible v0.3.0 diagonal affine FiLM.
    """

    def __init__(
        self,
        u_cycle_dim: int = 10,
        u_state_dim: int = 20,
        drug_embed_dim: int = 16,
        n_samples: int = 1,
        film_mode: str = "m7_block_tt",
        rank_gamma: int = 16,
        tau_snr: float = 1.5,
        m0_shrink: float = 10.0,
        eta0: float = 0.005,
        eps_iso: float = 0.05,
        ln_eps: float = 1e-5,
        n_obs_total: int = 1,
        sample_counts: Optional[torch.Tensor] = None,
        is_control_sample: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if film_mode not in ("m7_block_tt", "legacy_hadamard"):
            raise ValueError(
                f"Unsupported film_mode='{film_mode}'. Choose 'm7_block_tt' or 'legacy_hadamard'."
            )

        self.u_cycle_dim = int(u_cycle_dim)
        self.u_state_dim = int(u_state_dim)
        self.drug_embed_dim = int(drug_embed_dim)
        self.film_mode = film_mode

        if film_mode == "legacy_hadamard":
            # Layer-Norm for u_cycle and u_state
            self.norm_cycle = nn.LayerNorm(u_cycle_dim)
            self.norm_state = nn.LayerNorm(u_state_dim)

            # Cycle-Specific Modulators
            self.gamma_cycle = nn.Linear(drug_embed_dim, u_cycle_dim)
            self.beta_cycle = nn.Linear(drug_embed_dim, u_cycle_dim)

            # State-Specific Modulators
            self.gamma_state = nn.Linear(drug_embed_dim, u_state_dim)
            self.beta_state = nn.Linear(drug_embed_dim, u_state_dim)

            # Residual Initializations (1 + gamma) to prevent manifold erasure
            nn.init.zeros_(self.gamma_cycle.weight)
            nn.init.zeros_(self.gamma_cycle.bias)
            nn.init.zeros_(self.gamma_state.weight)
            nn.init.zeros_(self.gamma_state.bias)

            nn.init.zeros_(self.beta_cycle.weight)
            nn.init.zeros_(self.beta_cycle.bias)
            nn.init.zeros_(self.beta_state.weight)
            nn.init.zeros_(self.beta_state.bias)
            return

        # --- M7-InvDrop: Stoch-SNR Block-TT WP-FiLM ---
        S = max(1, int(n_samples))
        D_cyc = self.u_cycle_dim
        D_st = self.u_state_dim
        R_cyc = D_cyc
        R_st = min(int(rank_gamma), D_st)

        self.n_samples = S
        self.rank_cycle = R_cyc
        self.rank_state = R_st
        self.tau_snr = float(tau_snr)
        self.m0_shrink = float(m0_shrink)
        self.eta0 = float(eta0)
        self.eps_iso = float(eps_iso)
        self.ln_eps = float(ln_eps)

        # 1. Variance-Matched Bias-Free LayerNorm (MatchedLN) scale weights (b_LN === 0)
        self.norm_weight_cycle = nn.Parameter(torch.ones(D_cyc))
        self.norm_weight_state = nn.Parameter(torch.ones(D_st))

        # 2. Sample-level metadata & EMA buffers
        self.register_buffer("n_obs_total", torch.tensor(float(max(1, int(n_obs_total))), dtype=torch.float32))

        if sample_counts is not None:
            sc_t = torch.as_tensor(sample_counts, dtype=torch.float32).reshape(S).clone().clamp(min=1.0)
        else:
            sc_t = torch.ones(S, dtype=torch.float32)
        self.register_buffer("sample_counts", sc_t)

        if is_control_sample is not None:
            ic_t = torch.as_tensor(is_control_sample, dtype=torch.float32).reshape(S).clone()
        else:
            ic_t = torch.zeros(S, dtype=torch.float32)
        self.register_buffer("is_control_sample", ic_t)

        self.register_buffer("ctrl_stats_init", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("pool_stats_init", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("pert_means_init", torch.zeros(S, dtype=torch.float32))

        for ch, D_c in (("cycle", D_cyc), ("state", D_st)):
            self.register_buffer(f"ctrl_qm_var_{ch}", torch.ones(D_c, dtype=torch.float32))
            self.register_buffer(f"ctrl_qv_mean_{ch}", torch.ones(D_c, dtype=torch.float32))
            self.register_buffer(f"ctrl_mean_{ch}", torch.zeros(D_c, dtype=torch.float32))
            self.register_buffer(f"ctrl_cov_{ch}", torch.eye(D_c, dtype=torch.float32))
            self.register_buffer(f"pool_cov_{ch}", torch.eye(D_c, dtype=torch.float32))
            self.register_buffer(f"pert_means_{ch}", torch.zeros(S, D_c, dtype=torch.float32))

        # 3. Decoupled perturbation embeddings & Block-TT WP-FiLM heads
        self.emb_gamma = nn.Embedding(S, self.drug_embed_dim)
        self.emb_alpha = nn.Embedding(S, self.drug_embed_dim)

        self.gamma_par_cycle = nn.Linear(self.drug_embed_dim, D_cyc, bias=False)
        self.gamma_perp_cycle = nn.Linear(self.drug_embed_dim, R_cyc, bias=False)
        self.alpha_cycle = nn.Linear(self.drug_embed_dim, D_cyc, bias=False)
        self.V_read_cycle = nn.Linear(D_cyc, R_cyc, bias=False)
        self.U_write_cycle = nn.Linear(R_cyc, D_cyc, bias=False)

        self.gamma_par_state = nn.Linear(self.drug_embed_dim, D_st, bias=False)
        self.gamma_perp_state = nn.Linear(self.drug_embed_dim, R_st, bias=False)
        self.alpha_state = nn.Linear(self.drug_embed_dim, D_st, bias=False)
        self.V_read_state = nn.Linear(D_st, R_st, bias=False)
        self.U_write_state = nn.Linear(R_st, D_st, bias=False)

        for head in (
            self.gamma_par_cycle,
            self.gamma_perp_cycle,
            self.alpha_cycle,
            self.gamma_par_state,
            self.gamma_perp_state,
            self.alpha_state,
        ):
            nn.init.zeros_(head.weight)

        for basis in (self.V_read_cycle, self.U_write_cycle, self.V_read_state, self.U_write_state):
            nn.init.orthogonal_(basis.weight)

    # --- MatchedLN Helpers ---
    def _normalize_stochastic(self, u: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        mean = u.mean(dim=-1, keepdim=True)
        var = u.var(dim=-1, unbiased=False, keepdim=True)
        w = weight.to(device=u.device, dtype=u.dtype)
        return ((u - mean) / torch.sqrt(var + self.ln_eps)) * w

    def _normalize_matched(
        self,
        channel: str,
        q_m: torch.Tensor,
        q_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        D_c = q_m.shape[-1]
        weight = getattr(self, f"norm_weight_{channel}").to(device=q_m.device, dtype=q_m.dtype)
        mean = q_m.mean(dim=-1, keepdim=True)
        var = q_m.var(dim=-1, unbiased=False, keepdim=True)
        if q_v is not None:
            qv_bar = q_v.to(device=q_m.device, dtype=q_m.dtype).mean(dim=-1, keepdim=True)
        else:
            qv_bar = getattr(self, f"ctrl_qv_mean_{channel}").to(device=q_m.device, dtype=q_m.dtype).mean()
        matched_var = var + ((D_c - 1.0) / max(1, D_c)) * qv_bar
        return ((q_m - mean) / torch.sqrt(matched_var + self.ln_eps)) * weight

    def normalize_cycle_eval(self, q_m: torch.Tensor, q_v: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.film_mode == "legacy_hadamard":
            return self.norm_cycle(q_m)
        return self._normalize_matched("cycle", q_m, q_v=q_v)

    def normalize_state_eval(self, q_m: torch.Tensor, q_v: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.film_mode == "legacy_hadamard":
            return self.norm_state(q_m)
        return self._normalize_matched("state", q_m, q_v=q_v)

    # --- Pre-LN Hill-2 KL-SNR Gates, Centroids, & Orthogonal Bases ---
    def get_snr(self, channel: str) -> torch.Tensor:
        qm_var = getattr(self, f"ctrl_qm_var_{channel}")
        qv_mean = getattr(self, f"ctrl_qv_mean_{channel}")
        return qm_var / (qv_mean + 1e-5)

    def get_snr_gates(self, channel: str) -> Tuple[torch.Tensor, torch.Tensor]:
        qm_var = getattr(self, f"ctrl_qm_var_{channel}")
        if self.ctrl_stats_init.item() == 0.0:
            g_act = torch.full_like(qm_var, 0.5)
        else:
            snr = self.get_snr(channel)
            snr_sq = snr.pow(2)
            g_act = snr_sq / (snr_sq + (self.tau_snr ** 2))
        g_perp = 1.0 - g_act
        return g_act, g_perp

    def get_shrunk_centroids(self, channel: str) -> torch.Tensor:
        pert_means = getattr(self, f"pert_means_{channel}")
        ctrl_mean = getattr(self, f"ctrl_mean_{channel}")
        w = (self.sample_counts / (self.sample_counts + self.m0_shrink)) * self.pert_means_init
        w = w.unsqueeze(-1)
        return w * pert_means + (1.0 - w) * ctrl_mean.unsqueeze(0)

    def get_effective_bases(
        self,
        channel: str,
        g_act: torch.Tensor,
        g_perp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        V_read = getattr(self, f"V_read_{channel}")
        U_write = getattr(self, f"U_write_{channel}")
        V_hat = F.normalize(V_read.weight, dim=1) * g_act.unsqueeze(0)
        U_tilde = g_perp.unsqueeze(1) * U_write.weight
        col_norms = torch.sqrt(U_tilde.pow(2).sum(dim=0, keepdim=True) + 1e-12).clamp(min=1.0)
        U_hat = U_tilde / col_norms
        return V_hat, U_hat

    # --- Count-Weighted Sufficient-Statistic EMA Update ---
    @torch.no_grad()
    def _update_ema_stats(
        self,
        q_m_cycle: torch.Tensor,
        q_v_cycle: torch.Tensor,
        u_cycle_match: torch.Tensor,
        q_m_state: torch.Tensor,
        q_v_state: torch.Tensor,
        u_state_match: torch.Tensor,
        sample_idx: torch.Tensor,
        ctrl_mask: torch.Tensor,
    ) -> None:
        n0_b = int(ctrl_mask.sum().item())
        n_b = int(sample_idx.shape[0])

        channel_inputs = (
            ("cycle", q_m_cycle.detach(), q_v_cycle.detach(), u_cycle_match.detach()),
            ("state", q_m_state.detach(), q_v_state.detach(), u_state_match.detach()),
        )

        # 1. Control-sample Pre-LN & MatchedLN statistics
        if n0_b >= 2:
            rho0 = 1.0 - ((1.0 - self.eta0) ** n0_b)
            first_ctrl_init = self.ctrl_stats_init.item() == 0.0
            for ch, qm_d, qv_d, um_d in channel_inputs:
                b_qm_var = qm_d[ctrl_mask].var(dim=0, unbiased=False)
                b_qv_mean = qv_d[ctrl_mask].mean(dim=0)
                b_mean = um_d[ctrl_mask].mean(dim=0)

                ctrl_qm_var = getattr(self, f"ctrl_qm_var_{ch}")
                ctrl_qv_mean = getattr(self, f"ctrl_qv_mean_{ch}")
                ctrl_mean = getattr(self, f"ctrl_mean_{ch}")
                ctrl_cov = getattr(self, f"ctrl_cov_{ch}")

                if first_ctrl_init:
                    ctrl_qm_var.copy_(b_qm_var)
                    ctrl_qv_mean.copy_(b_qv_mean)
                    ctrl_mean.copy_(b_mean)
                    d0 = um_d[ctrl_mask] - b_mean.unsqueeze(0)
                    ctrl_cov.copy_((d0.T @ d0) / n0_b)
                else:
                    ctrl_qm_var.mul_(1.0 - rho0).add_(rho0 * b_qm_var)
                    ctrl_qv_mean.mul_(1.0 - rho0).add_(rho0 * b_qv_mean)
                    ctrl_mean.mul_(1.0 - rho0).add_(rho0 * b_mean)
                    d0 = um_d[ctrl_mask] - ctrl_mean.unsqueeze(0)
                    ctrl_cov.mul_(1.0 - rho0).add_(rho0 * ((d0.T @ d0) / n0_b))

            if first_ctrl_init:
                self.ctrl_stats_init.fill_(1.0)
        elif n0_b == 1:
            rho0 = self.eta0
            for ch, _, qv_d, um_d in channel_inputs:
                ctrl_qv_mean = getattr(self, f"ctrl_qv_mean_{ch}")
                ctrl_mean = getattr(self, f"ctrl_mean_{ch}")
                ctrl_qv_mean.mul_(1.0 - rho0).add_(rho0 * qv_d[ctrl_mask].mean(dim=0))
                ctrl_mean.mul_(1.0 - rho0).add_(rho0 * um_d[ctrl_mask].mean(dim=0))

        # 2. Pooled covariance over entire minibatch
        if n_b >= 2:
            rho_pool = 1.0 - ((1.0 - self.eta0) ** n_b)
            first_pool_init = self.pool_stats_init.item() == 0.0
            for ch, _, _, um_d in channel_inputs:
                dp = um_d - um_d.mean(dim=0, keepdim=True)
                b_pool_cov = (dp.T @ dp) / n_b
                pool_cov = getattr(self, f"pool_cov_{ch}")
                if first_pool_init:
                    pool_cov.copy_(b_pool_cov)
                else:
                    pool_cov.mul_(1.0 - rho_pool).add_(rho_pool * b_pool_cov)
            if first_pool_init:
                self.pool_stats_init.fill_(1.0)

        # 3. Per-perturbation MatchedLN centroids
        for p_val in torch.unique(sample_idx).tolist():
            pm = sample_idx == p_val
            np_b = int(pm.sum().item())
            if np_b < 1:
                continue
            rho_p = 1.0 - ((1.0 - self.eta0) ** np_b)
            first_p = self.pert_means_init[p_val].item() == 0.0
            for ch, _, _, um_d in channel_inputs:
                mb = um_d[pm].mean(dim=0)
                pert_means = getattr(self, f"pert_means_{ch}")
                if first_p:
                    pert_means[p_val].copy_(mb)
                else:
                    pert_means[p_val].mul_(1.0 - rho_p).add_(rho_p * mb)
            if first_p:
                self.pert_means_init[p_val] = 1.0

    # --- Block-TT WP-FiLM Modulation ---
    def _modulate_channel(
        self,
        channel: str,
        u_src: torch.Tensor,
        sample_idx: torch.Tensor,
        c_film: torch.Tensor,
    ) -> torch.Tensor:
        mu_hat_all = self.get_shrunk_centroids(channel)
        mu_hat = mu_hat_all[sample_idx]
        u_check = u_src - mu_hat

        e_gamma = self.emb_gamma(sample_idx)
        e_alpha = self.emb_alpha(sample_idx)

        gamma_par = getattr(self, f"gamma_par_{channel}")(e_gamma)
        gamma_perp = getattr(self, f"gamma_perp_{channel}")(e_gamma)
        alpha = getattr(self, f"alpha_{channel}")(e_alpha)

        g_act, g_perp = self.get_snr_gates(channel)
        V_hat, U_hat = self.get_effective_bases(channel, g_act, g_perp)

        delta_tan = (g_act.unsqueeze(0) * gamma_par) * u_check
        delta_perp = F.linear(gamma_perp * F.linear(u_check, V_hat), U_hat)
        delta_z = c_film * (delta_tan + delta_perp + alpha)
        return u_src + delta_z

    def _modulate_eval_helper(
        self,
        channel: str,
        u_norm: torch.Tensor,
        sample_idx: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        dev = self.emb_gamma.weight.device
        n_pts = u_norm.shape[0]
        if isinstance(sample_idx, (int, np.integer)):
            idx = torch.full((n_pts,), int(sample_idx), dtype=torch.long, device=dev)
        else:
            idx = torch.as_tensor(sample_idx, dtype=torch.long, device=dev).reshape(-1)
            if idx.numel() == 1 and n_pts > 1:
                idx = idx.expand(n_pts)

        if self.is_control_sample.sum() > 0:
            c_film = (1.0 - self.is_control_sample[idx]).unsqueeze(-1)
        else:
            c_film = torch.ones(n_pts, 1, device=dev, dtype=torch.float32)

        if bool((c_film == 0.0).all()):
            return u_norm

        u_in = u_norm.to(device=dev, dtype=torch.float32)
        z_out = self._modulate_channel(channel, u_in, idx, c_film)
        return z_out.to(device=u_norm.device, dtype=u_norm.dtype)

    def modulate_cycle_eval(
        self,
        u_norm: torch.Tensor,
        sample_idx: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        if self.film_mode != "m7_block_tt":
            raise RuntimeError("modulate_cycle_eval requires film_mode='m7_block_tt'.")
        return self._modulate_eval_helper("cycle", u_norm, sample_idx)

    def modulate_state_eval(
        self,
        u_norm: torch.Tensor,
        sample_idx: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        if self.film_mode != "m7_block_tt":
            raise RuntimeError("modulate_state_eval requires film_mode='m7_block_tt'.")
        return self._modulate_eval_helper("state", u_norm, sample_idx)

    def forward(
        self,
        u_cycle: torch.Tensor,
        u_state: torch.Tensor,
        sample_or_embed: torch.Tensor,
        q_m_cycle: Optional[torch.Tensor] = None,
        q_v_cycle: Optional[torch.Tensor] = None,
        q_m_state: Optional[torch.Tensor] = None,
        q_v_state: Optional[torch.Tensor] = None,
        is_treated: Optional[torch.Tensor] = None,
        return_src: bool = False,
    ):
        if self.film_mode == "legacy_hadamard":
            u_cycle_norm = self.norm_cycle(u_cycle)
            u_state_norm = self.norm_state(u_state)

            g_cyc, b_cyc = self.gamma_cycle(sample_or_embed), self.beta_cycle(sample_or_embed)
            g_st, b_st = self.gamma_state(sample_or_embed), self.beta_state(sample_or_embed)

            z_cycle = ((1.0 + g_cyc) * u_cycle_norm) + b_cyc
            z_state = ((1.0 + g_st) * u_state_norm) + b_st
            if return_src:
                return z_cycle, z_state, u_cycle_norm, u_state_norm
            return z_cycle, z_state

        # --- M7-InvDrop Forward ---
        sample_idx = sample_or_embed.reshape(-1).long()
        if q_m_cycle is None:
            q_m_cycle = u_cycle
        if q_m_state is None:
            q_m_state = u_state

        # Resolve sample-level control identity clamping & EMA control mask
        if self.is_control_sample.sum() > 0:
            ctrl_mask = self.is_control_sample[sample_idx] > 0.5
            c_film = (1.0 - self.is_control_sample[sample_idx]).unsqueeze(-1).to(dtype=u_cycle.dtype)
        elif is_treated is not None:
            is_treat_1d = is_treated.reshape(-1)
            if bool((is_treat_1d < 0.5).any()):
                ctrl_mask = is_treat_1d < 0.5
            else:
                ctrl_mask = torch.ones_like(sample_idx, dtype=torch.bool)
            c_film = is_treat_1d.unsqueeze(-1).to(dtype=u_cycle.dtype)
        else:
            ctrl_mask = torch.ones_like(sample_idx, dtype=torch.bool)
            c_film = torch.ones(sample_idx.shape[0], 1, device=u_cycle.device, dtype=u_cycle.dtype)

        if self.training:
            with torch.no_grad():
                qv_cyc_d = q_v_cycle.detach() if q_v_cycle is not None else None
                qv_st_d = q_v_state.detach() if q_v_state is not None else None
                u_cyc_match_d = self._normalize_matched("cycle", q_m_cycle.detach(), q_v=qv_cyc_d)
                u_st_match_d = self._normalize_matched("state", q_m_state.detach(), q_v=qv_st_d)
                if q_v_cycle is not None and q_v_state is not None:
                    self._update_ema_stats(
                        q_m_cycle=q_m_cycle,
                        q_v_cycle=q_v_cycle,
                        u_cycle_match=u_cyc_match_d,
                        q_m_state=q_m_state,
                        q_v_state=q_v_state,
                        u_state_match=u_st_match_d,
                        sample_idx=sample_idx,
                        ctrl_mask=ctrl_mask,
                    )
            u_cycle_src = self._normalize_stochastic(u_cycle, self.norm_weight_cycle)
            u_state_src = self._normalize_stochastic(u_state, self.norm_weight_state)
        else:
            u_cycle_src = self._normalize_matched("cycle", q_m_cycle, q_v=q_v_cycle)
            u_state_src = self._normalize_matched("state", q_m_state, q_v=q_v_state)

        z_cycle = self._modulate_channel("cycle", u_cycle_src, sample_idx, c_film)
        z_state = self._modulate_channel("state", u_state_src, sample_idx, c_film)

        if return_src:
            return z_cycle, z_state, u_cycle_src, u_state_src
        return z_cycle, z_state

    # --- Exact Closed-Form Schur-Product Functional Prior ---
    def compute_film_prior_loss(
        self,
        lambda_gamma: float = 1.5,
        lambda_alpha: float = 1.0,
    ) -> torch.Tensor:
        if self.film_mode != "m7_block_tt":
            return torch.tensor(0.0, device=self.gamma_state.weight.device)

        dev = self.emb_gamma.weight.device
        if self.is_control_sample.sum() > 0:
            p_treat = torch.where(self.is_control_sample < 0.5)[0]
        else:
            p_treat = torch.arange(self.n_samples, device=dev, dtype=torch.long)

        if p_treat.numel() == 0:
            return torch.tensor(0.0, device=dev, dtype=torch.float32)

        e_gamma = self.emb_gamma(p_treat)
        e_alpha = self.emb_alpha(p_treat)

        total_raw = torch.tensor(0.0, device=dev, dtype=torch.float32)
        for ch, D_c in (("state", self.u_state_dim), ("cycle", self.u_cycle_dim)):
            ctrl_cov = getattr(self, f"ctrl_cov_{ch}")
            pool_cov = getattr(self, f"pool_cov_{ch}")
            eye_c = torch.eye(D_c, device=dev, dtype=ctrl_cov.dtype)
            Sigma_ref = 0.5 * (ctrl_cov + pool_cov) + self.eps_iso * eye_c

            g_act, g_perp = self.get_snr_gates(ch)
            V_hat, U_hat = self.get_effective_bases(ch, g_act, g_perp)

            C_v = V_hat @ Sigma_ref @ V_hat.T
            G_U = U_hat.T @ U_hat

            gamma_par = getattr(self, f"gamma_par_{ch}")(e_gamma)
            gamma_perp = getattr(self, f"gamma_perp_{ch}")(e_gamma)
            alpha = getattr(self, f"alpha_{ch}")(e_alpha)

            diag_weights = (g_act.pow(2)) * torch.diagonal(Sigma_ref)
            tan_energy = ((gamma_par.pow(2)) * diag_weights.unsqueeze(0)).sum()
            perp_energy = ((gamma_perp @ (G_U * C_v)) * gamma_perp).sum()
            alpha_energy = (alpha.pow(2)).sum()

            total_raw = total_raw + 0.5 * (
                float(lambda_gamma) * (tan_energy + perp_energy) + float(lambda_alpha) * alpha_energy
            )

        return total_raw


# ==========================================
# Bifurcated decoder: cell cycle head + main head
# ==========================================
class BifurcatedDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_latent_cycle: int,
        cycle_gene_mask: torch.Tensor,
        n_cat_list: list = None,
        n_hidden: int = 128,
        n_layers: int = 1,
        max_col_norm: Optional[float] = 2.0,
    ):
        super().__init__()
        self.n_latent_cycle = n_latent_cycle
        self.n_latent_main = n_input - n_latent_cycle
        self.n_genes = n_output
        self.max_col_norm = max_col_norm
        self.register_buffer("cycle_gene_mask", cycle_gene_mask.float())

        # Pathway A: Strictly Linear Cycle Head
        self.cycle_head = nn.Linear(self.n_latent_cycle, self.n_genes)

        # Pathway B: Dense Main Head & Batch Bypass
        self.use_batch = n_cat_list is not None and len(n_cat_list) > 0
        if self.use_batch:
            self.batch_embed = nn.Embedding(n_cat_list[0], 16)
            main_in_dim = self.n_latent_main + 16
            self.batch_gene_bias = nn.Linear(16, self.n_genes)  # Batch bypass for cycle genes
        else:
            main_in_dim = self.n_latent_main

        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(main_in_dim, n_hidden))
            layers.append(nn.BatchNorm1d(n_hidden))
            layers.append(nn.ReLU())
            main_in_dim = n_hidden

        self.main_hidden = nn.Sequential(*layers)
        if max_col_norm is not None and float(max_col_norm) > 0.0:
            register_parametrization(
                self.main_hidden[0],
                "weight",
                MaxColNorm(max_col_norm=float(max_col_norm)),
            )

        self.main_head = nn.Linear(n_hidden, self.n_genes)
        self.px_dropout_decoder = nn.Linear(n_hidden, self.n_genes)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, batch_index: torch.Tensor = None):
        z_cycle = z[:, :self.n_latent_cycle]
        z_main = z[:, self.n_latent_cycle:]

        if self.use_batch and batch_index is not None:
            b_embed = self.batch_embed(batch_index.squeeze(-1))
            z_main = torch.cat([z_main, b_embed], dim=-1)
            h_batch = self.batch_gene_bias(b_embed)
        else:
            h_batch = 0.0

        # Hard Mask implementation
        masked_cycle_weight = self.cycle_head.weight * self.cycle_gene_mask.unsqueeze(1)
        masked_cycle_bias = self.cycle_head.bias * self.cycle_gene_mask
        h_cycle = F.linear(z_cycle, masked_cycle_weight, masked_cycle_bias)

        h_hidden = self.main_hidden(z_main)
        h_main = self.main_head(h_hidden)

        h_total = h_cycle + h_main + h_batch

        px_scale = F.softmax(h_total, dim=-1)
        px_rate = torch.exp(library) * px_scale
        px_dropout = self.px_dropout_decoder(h_hidden)

        return px_scale, None, px_rate, px_dropout


# ==========================================
# Perturbation-specialized Multi-resolution Variational Inference Architecture
# ==========================================
class PertVI_Inference(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_samples: int,
        n_latent_cycle: int = 3,
        n_latent_state: int = 20,
        n_latent_ood: int = 10,
        ood_dropout_rate: float = 0.3,
        basal_gene_dropout_rate: float = 0.15,
        drug_embed_dim: int = 16,
        film_mode: str = "m7_block_tt",
        rank_gamma: int = 16,
        tau_snr: float = 1.5,
        m0_shrink: float = 10.0,
        eta0: float = 0.005,
        eps_iso: float = 0.05,
        n_obs_total: int = 1,
        sample_counts: Optional[torch.Tensor] = None,
        is_control_sample: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.film_mode = film_mode

        # --- Basal Input Gene Masking (applied exclusively to encoder_cycle & encoder_state) ---
        self.basal_gene_dropout = nn.Dropout(float(basal_gene_dropout_rate))

        # --- Dual Encoders for Sample-Unaware (u) Spaces ---
        self.encoder_cycle = Encoder(
            n_input=n_input,
            n_output=n_latent_cycle,
            n_hidden=128,
            n_layers=2,
        )
        self.encoder_state = Encoder(
            n_input=n_input,
            n_output=n_latent_state,
            n_hidden=128,
            n_layers=2,
        )

        if film_mode == "legacy_hadamard":
            self.sample_embedding = nn.Embedding(n_samples, drug_embed_dim)

        # --- The OOD Escape Hatch (v_ood) ---
        self.encoder_ood = Encoder(
            n_input=n_input,
            n_output=n_latent_ood,
            n_hidden=128,
            n_layers=1,
        )

        # --- Encoders to learn library size ---
        self.encoder_library = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2),
        )

        # --- The FiLM Transition ---
        self.split_film_transition = SplitNaturalShiftFiLM(
            u_cycle_dim=n_latent_cycle,
            u_state_dim=n_latent_state,
            drug_embed_dim=drug_embed_dim,
            n_samples=n_samples,
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

        # --- The OOD dropout ---
        self.ood_dropout = nn.Dropout(ood_dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        sample_index: torch.Tensor,
        is_treated: Optional[torch.Tensor] = None,
    ):
        """
        x: Raw log-normalized counts tensor
        """
        # Basal encoders receive expectation-preserving gene-masked x_basal; OOD & library receive unmasked x
        x_basal = self.basal_gene_dropout(x)

        # Extract Latents
        q_m_cycle, q_v_cycle, u_cycle = self.encoder_cycle(x_basal)
        q_m_state, q_v_state, u_state = self.encoder_state(x_basal)
        q_m_ood, q_v_ood, v_ood = self.encoder_ood(x)

        # FiLM Transition
        u_base = torch.cat([u_cycle, u_state], dim=-1)
        if self.film_mode == "legacy_hadamard":
            sample_embed = self.sample_embedding(sample_index.squeeze(-1).long())
            z_cycle, z_state, u_cycle_src, u_state_src = self.split_film_transition(
                u_cycle, u_state, sample_embed, return_src=True
            )
            z_base = torch.cat([z_cycle, z_state], dim=-1)
            z_shift = z_base - u_base
        else:
            z_cycle, z_state, u_cycle_src, u_state_src = self.split_film_transition(
                u_cycle,
                u_state,
                sample_index,
                q_m_cycle=q_m_cycle,
                q_v_cycle=q_v_cycle,
                q_m_state=q_m_state,
                q_v_state=q_v_state,
                is_treated=is_treated,
                return_src=True,
            )
            z_base = torch.cat([z_cycle, z_state], dim=-1)
            # Exact learned FiLM displacement in MatchedLN space
            z_shift = z_base - torch.cat([u_cycle_src, u_state_src], dim=-1)

        # Composition with Information Dropout on OOD
        v_ood_dropped = self.ood_dropout(v_ood)

        # Final composition
        z_final = torch.cat([z_base, v_ood_dropped], dim=-1)

        # Library Size
        empirical_counts = torch.clamp(x.sum(dim=-1, keepdim=True), min=1.0)  # Calculate the reality (empirical sum)
        empirical_lib_log = torch.log(empirical_counts)
        q_lib_params = self.encoder_library(x)  # Let the network guess a small TWEAK, not the absolute number
        q_m_lib = empirical_lib_log + q_lib_params[:, 0:1]
        q_v_lib_raw = torch.clamp(q_lib_params[:, 1:2], min=-5.0, max=2.0)  # Tame the variance
        q_v_lib = torch.exp(q_v_lib_raw) + 1e-4
        library_log = q_m_lib + torch.sqrt(q_v_lib) * torch.randn_like(q_m_lib)  # Sample the final latent library size

        return {
            "u_cycle": u_cycle,
            "u_state": u_state,
            "u_cycle_src": u_cycle_src,
            "u_state_src": u_state_src,
            "v_ood": v_ood,
            "q_m_cycle": q_m_cycle,
            "q_v_cycle": q_v_cycle,
            "q_m_state": q_m_state,
            "q_v_state": q_v_state,
            "q_m_ood": q_m_ood,
            "q_v_ood": q_v_ood,
            "u_base": u_base,
            "z_shift": z_shift,
            "z_base": z_base,
            "z_cycle": z_cycle,
            "z_state": z_state,
            "z_final": z_final,
            "q_m_lib": q_m_lib,
            "q_v_lib": q_v_lib,
            "library_log": library_log,
        }
