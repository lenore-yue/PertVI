import math
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
# Cylindrical-Harmonic Feature Expansion (v0.4.2)
# ==========================================
def expand_cylindrical_harmonics(
    z_cycle: torch.Tensor,
    harmonic_2theta_scale: float = 0.75,
    z2_scale: float = 0.05,
    r0_smooth: float = 0.45,
    radial_amp_scale: float = 1.0,
    harmonic_basis_mode: str = "soft_unit_ribbon",
) -> torch.Tensor:
    """Expands 3D cylindrical cycle coordinates (z_0, z_1, z_2) into a 6D harmonic basis.

    Modes:
      - ``soft_unit_ribbon`` (v0.4.2 default): C^infty origin-vanishing soft-unit angular harmonics
        with Riemannian tangential gradient projection:
            phi_rib(z) = [z_tan,0 / sqrt(r_sg^2 + r_0^2), z_tan,1 / sqrt(r_sg^2 + r_0^2),
                          z2_scale * z_2, s_r * r_sg^2 / (r_sg^2 + r_0^2),
                          s_2 * (z_tan,0^2 - z_tan,1^2) / (r_sg^2 + r_0^2),
                          s_2 * (2 z_tan,0 z_tan,1) / (r_sg^2 + r_0^2)],
        where z_tan = z_xy - (<z_xy - z_sg, z_sg> / (r_sg^2 + 1e-6)) z_sg equals z_xy identically
        in forward evaluation while projecting decoder gradients strictly onto the angular tangent
        direction e_theta. On the active ribbon (r ~ 1), phi_rib spans the complete 5D trigonometric
        polynomial space {1, cos(theta), sin(theta), cos(2*theta), sin(2*theta)} on S^1 while
        vanishing identically at z = 0 on non-cycling/null cohorts.
      - ``smooth_polynomial``: Unbounded C^infty origin-vanishing polynomial harmonics.
      - ``legacy_cylindrical``: Earlier v0.4.2 basis [z_0, z_1, z2_scale * z_2, r_12 - 1, c_2, s_2].
    """
    if z_cycle.shape[-1] != 3:
        return z_cycle
    if harmonic_basis_mode == "legacy_cylindrical":
        r_z = torch.sqrt(z_cycle[:, 0:1].pow(2) + z_cycle[:, 1:2].pow(2) + 1e-6)
        c1 = z_cycle[:, 0:1] / r_z
        s1 = z_cycle[:, 1:2] / r_z
        c2 = float(harmonic_2theta_scale) * r_z * (c1.pow(2) - s1.pow(2))
        s2 = float(harmonic_2theta_scale) * r_z * (2.0 * s1 * c1)
        return torch.cat(
            [
                z_cycle[:, 0:2],
                float(z2_scale) * z_cycle[:, 2:3],
                r_z - 1.0,
                c2,
                s2,
            ],
            dim=-1,
        )

    z0 = z_cycle[:, 0:1]
    z1 = z_cycle[:, 1:2]
    r2 = z0.pow(2) + z1.pow(2)

    if harmonic_basis_mode == "soft_unit_ribbon":
        r0_sq = max(float(r0_smooth) ** 2, 1e-4)
        z_xy = z_cycle[:, 0:2]
        z_sg = z_xy.detach()
        r2_sg = z_sg.pow(2).sum(dim=-1, keepdim=True)
        w_tan = r2_sg / (r2_sg + 0.04)
        rad_proj = w_tan * (((z_xy - z_sg) * z_sg).sum(dim=-1, keepdim=True) / (r2_sg + 1e-6))
        z_tan = z_xy - rad_proj * z_sg
        zt0 = z_tan[:, 0:1]
        zt1 = z_tan[:, 1:2]
        denom1 = torch.sqrt(r2_sg + r0_sq)
        denom2 = r2_sg + r0_sq
        c1_sat = zt0 / denom1
        s1_sat = zt1 / denom1
        r_sat = float(radial_amp_scale) * (r2_sg / denom2)
        c2_sat = float(harmonic_2theta_scale) * (zt0.pow(2) - zt1.pow(2)) / denom2
        s2_sat = float(harmonic_2theta_scale) * (2.0 * zt0 * zt1) / denom2
        return torch.cat(
            [
                c1_sat,
                s1_sat,
                float(z2_scale) * z_cycle[:, 2:3],
                r_sat,
                c2_sat,
                s2_sat,
            ],
            dim=-1,
        )

    if harmonic_basis_mode == "saturating_rational":
        r0_sq = max(float(r0_smooth) ** 2, 1e-4)
        denom1 = torch.sqrt(r2 + r0_sq)
        denom2 = r2 + r0_sq
        c1_sat = z0 / denom1
        s1_sat = z1 / denom1
        r_sat = float(radial_amp_scale) * (r2 / denom2)
        c2_sat = float(harmonic_2theta_scale) * (z0.pow(2) - z1.pow(2)) / denom2
        s2_sat = float(harmonic_2theta_scale) * (2.0 * z0 * z1) / denom2
        return torch.cat(
            [
                c1_sat,
                s1_sat,
                float(z2_scale) * z_cycle[:, 2:3],
                r_sat,
                c2_sat,
                s2_sat,
            ],
            dim=-1,
        )

    # Default: smooth_polynomial (unbounded C^infty origin-vanishing harmonic basis)
    r_quad = float(radial_amp_scale) * r2
    c2_poly = float(harmonic_2theta_scale) * (z0.pow(2) - z1.pow(2))
    s2_poly = float(harmonic_2theta_scale) * (2.0 * z0 * z1)
    return torch.cat(
        [
            z0,
            z1,
            float(z2_scale) * z_cycle[:, 2:3],
            r_quad,
            c2_poly,
            s2_poly,
        ],
        dim=-1,
    )


# ==========================================
# GO Pathway-Masked Cell-Cycle Encoder (v0.4.2)
# ==========================================
class PathwayCycleEncoder(nn.Module):
    """GO/Reactome Pathway-Masked Cell-Cycle Encoder (v0.4.2).

    Architecture:
      1. Hard-masks the input weight matrix W_in in R^{(K * H_p) x G} by the binary pathway
         membership matrix M_path in {0, 1}^{K x G} (scaled by sqrt(G_cyc / |P_k|) per pathway),
         so sub-block k exclusively reads genes belonging to GO cell-cycle module P_k.
      2. Extracts K=6 explicit basal GO pathway activations u_cycle_go_enc in R^{N x K}.
      3. In ``hierarchical_3d`` mode (default, n_output=3), projects the concatenated K*H_p
         pathway representations onto the 3D unforced hysteresis-loop posterior
         (q_m_cycle, q_v_cycle, u_cycle) in R^3 (capturing 97.59% of 6-module variance while
         filtering out the 2.41% collinear noise subspace).
      4. In ``direct_6d`` mode (n_output=K), maps each pathway sub-block k directly to its own
         latent coordinate (q_m_cycle[:, k], q_v_cycle[:, k]) in R^K.
    """

    def __init__(
        self,
        n_input: int,
        pathway_mask: torch.Tensor,
        n_output: int = 3,
        n_hidden_per_pathway: int = 16,
        n_hidden: int = 128,
        dropout_rate: float = 0.0,
        direct_pathway_latent: bool = False,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        M_path = torch.as_tensor(pathway_mask, dtype=torch.float32)
        if M_path.ndim != 2 or M_path.shape[1] != n_input:
            raise ValueError(
                f"pathway_mask must have shape (K, {n_input}), got {tuple(M_path.shape)}."
            )
        self.n_input = int(n_input)
        self.n_pathways = int(M_path.shape[0])
        self.n_output = int(n_output)
        self.n_hidden_per_pathway = int(n_hidden_per_pathway)
        self.n_hidden = int(n_hidden)
        self.direct_pathway_latent = bool(direct_pathway_latent) and (self.n_output == self.n_pathways)
        self.var_eps = float(var_eps)

        # Normalize per-pathway input mask by sqrt(G_cyc / |P_k|) so smaller GO modules have balanced gradient norm
        cyc_union = (M_path.sum(dim=0) > 0).float()
        g_cyc_total = cyc_union.sum().clamp(min=1.0)
        p_counts = M_path.sum(dim=1, keepdim=True).clamp(min=1.0)
        scaled_M_path = M_path * torch.sqrt(g_cyc_total / p_counts)

        # Expand (K, G) -> (K * H_p, G)
        expanded_mask = scaled_M_path.repeat_interleave(self.n_hidden_per_pathway, dim=0)
        self.register_buffer("pathway_mask", M_path)
        self.register_buffer("expanded_input_mask", expanded_mask)

        M_half_a, M_half_b = self._make_disjoint_split_halves(M_path)
        self.register_buffer("pathway_mask_half_a", M_half_a, persistent=False)
        self.register_buffer("pathway_mask_half_b", M_half_b, persistent=False)

        total_path_hid = self.n_pathways * self.n_hidden_per_pathway
        self.pathway_linear = nn.Linear(self.n_input, total_path_hid, bias=False)
        with torch.no_grad():
            self.pathway_linear.weight.mul_(expanded_mask > 0)

        self.pathway_norm = nn.LayerNorm(self.n_hidden_per_pathway, elementwise_affine=False)
        self.pathway_act = nn.GELU()
        self.pathway_dropout = nn.Dropout(float(dropout_rate)) if float(dropout_rate) > 0.0 else nn.Identity()

        # Explicit per-pathway scalar readout: (N, K, H_p) -> (N, K)
        self.pathway_score_weight = nn.Parameter(
            torch.randn(self.n_pathways, self.n_hidden_per_pathway) / math.sqrt(self.n_hidden_per_pathway)
        )
        self.pathway_score_bias = nn.Parameter(torch.zeros(self.n_pathways))

        if self.direct_pathway_latent:
            self.direct_mean_weight = nn.Parameter(
                torch.randn(self.n_pathways, self.n_hidden_per_pathway) / math.sqrt(self.n_hidden_per_pathway)
            )
            self.direct_mean_bias = nn.Parameter(torch.zeros(self.n_pathways))
            self.direct_var_weight = nn.Parameter(
                torch.zeros(self.n_pathways, self.n_hidden_per_pathway)
            )
            self.direct_var_bias = nn.Parameter(torch.full((self.n_pathways,), -2.0))
        else:
            self.loop_fc = nn.Sequential(
                nn.Linear(total_path_hid, self.n_hidden),
                nn.BatchNorm1d(self.n_hidden),
                nn.ReLU(),
                self.pathway_dropout,
            )
            self.mean_encoder = nn.Linear(self.n_hidden, self.n_output)
            self.var_encoder = nn.Linear(self.n_hidden, self.n_output)
            nn.init.constant_(self.var_encoder.bias, -2.0)

        self.last_u_cycle_go_enc: Optional[torch.Tensor] = None
        self.split_r_gate_tau: float = 0.35
        self.sig_res_gate_tau: float = 0.42
        self.register_buffer("ema_r_split", torch.tensor(0.85, dtype=torch.float32), persistent=False)
        self.register_buffer("ema_sig_res", torch.tensor(0.75, dtype=torch.float32), persistent=False)
        self.register_buffer("ema_q05", torch.zeros(self.n_pathways, dtype=torch.float32), persistent=False)
        self.register_buffer("ema_q95", torch.ones(self.n_pathways, dtype=torch.float32), persistent=False)
        self.register_buffer("ema_go_initialized", torch.tensor(0, dtype=torch.long), persistent=False)

    @staticmethod
    def _make_disjoint_split_halves(M_path: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        union_idx = torch.where(M_path.sum(dim=0) > 0)[0]
        M_half_a = torch.zeros_like(M_path)
        M_half_b = torch.zeros_like(M_path)
        if union_idx.numel() >= 2:
            M_half_a[:, union_idx[0::2]] = M_path[:, union_idx[0::2]]
            M_half_b[:, union_idx[1::2]] = M_path[:, union_idx[1::2]]
        else:
            M_half_a.copy_(M_path)
            M_half_b.copy_(M_path)
        return M_half_a, M_half_b

    @torch.no_grad()
    def reset_pathway_masks(self, pathway_mask: torch.Tensor) -> None:
        M_path = torch.as_tensor(
            pathway_mask, device=self.pathway_mask.device, dtype=self.pathway_mask.dtype
        )
        cyc_union = (M_path.sum(dim=0) > 0).float()
        g_cyc_total = cyc_union.sum().clamp(min=1.0)
        p_counts = M_path.sum(dim=1, keepdim=True).clamp(min=1.0)
        scaled_M_path = M_path * torch.sqrt(g_cyc_total / p_counts)
        expanded_mask = scaled_M_path.repeat_interleave(self.n_hidden_per_pathway, dim=0)
        self.pathway_mask.copy_(M_path)
        self.expanded_input_mask.copy_(expanded_mask)
        M_half_a, M_half_b = self._make_disjoint_split_halves(M_path)
        self.pathway_mask_half_a.copy_(M_half_a)
        self.pathway_mask_half_b.copy_(M_half_b)
        self.pathway_linear.weight.mul_(expanded_mask > 0)
        self.ema_go_initialized.fill_(0)

    def get_reliability_gate(self) -> float:
        r_sp = float(self.ema_r_split.item())
        s_rs = float(self.ema_sig_res.item())
        g_r = (r_sp ** 4) / ((r_sp ** 4) + (self.split_r_gate_tau ** 4))
        g_s = (s_rs ** 4) / ((s_rs ** 4) + (self.sig_res_gate_tau ** 4))
        raw_g = float(g_r * g_s)
        return float(max(0.0, min(1.0, (raw_g - 0.02) / 0.98)))

    def compute_empirical_go_targets(
        self, x_full_log: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes Split-Half Reliability-Gated GO Pathway & Forward Soft-Ribbon Phase Targets.

        Mathematical Guarantees:
          1. Library-size normalizes x_full_log = log(1 + counts) to log(1 + CP10k) so low-UMI
             droplets are not attenuated into the origin hole.
          2. Evaluates genuine multi-gene cell-cycle coherence via strictly disjoint even/odd
             cycle-gene split-half correlation r_split and 3-pole dispersion sig_res AFTER
             quadratic library-size residualization [1, log_lib, log_lib^2], giving gamma_cycle -> 0
             identically on column-shuffled noise or pure quiescent G0 cohorts.
          3. Normalizes each GO module k by span_k = max(q95_k - q05_k, 0.85 * max_j(q95_j - q05_j, 0.75))
             so active cycling phases map to [0, 1] across datasets while absent phases (e.g., M-depleted
             open C-arcs where G1 and M do not connect) are NEVER artificially stretched to close a loop.
          4. Applies canonical de-shearing around the biochemically forbidden hysteresis interior
             (c_S=0.44, c_M=0.42) and smooth rational forward ribbon mapping with adaptive core
             r0_eff^2 = 0.40^2 + 2.5 * (1 - gamma_cycle)^2 (hollow ribbon when gamma_cycle ~ 1,
             linear unimodal contraction when gamma_cycle -> 0).
        """
        M = self.pathway_mask.to(device=x_full_log.device, dtype=x_full_log.dtype)
        denom = M.sum(dim=1, keepdim=True).clamp(min=1.0).t()  # (1, K)
        counts_est = torch.clamp(torch.expm1(x_full_log), min=0.0)
        tot_counts = counts_est.sum(dim=-1, keepdim=True).clamp(min=1.0)
        x_cp10k = torch.log1p(counts_est * (1e4 / tot_counts))
        s_full = torch.matmul(x_cp10k, M.t()) / denom  # (N, K)

        if (self.training or int(self.ema_go_initialized.item()) == 0) and s_full.shape[0] >= 64:
            with torch.no_grad():
                x_cp_f = x_cp10k.float()
                s_f = s_full.float()
                M_a = self.pathway_mask_half_a.to(device=x_cp_f.device, dtype=torch.float32)
                M_b = self.pathway_mask_half_b.to(device=x_cp_f.device, dtype=torch.float32)
                s_a = torch.matmul(x_cp_f, M_a.t()) / M_a.sum(dim=1, keepdim=True).clamp(min=1.0).t()
                s_b = torch.matmul(x_cp_f, M_b.t()) / M_b.sum(dim=1, keepdim=True).clamp(min=1.0).t()

                ll = torch.log(tot_counts.float())
                ll_z = (ll - ll.mean()) / (ll.std() + 1e-6)
                B_lib = torch.cat([torch.ones_like(ll_z), ll_z, ll_z.pow(2)], dim=1)  # (N, 3)
                gram = torch.matmul(B_lib.t(), B_lib) + 1e-4 * torch.eye(3, device=B_lib.device, dtype=torch.float32)

                def _res_quad(Y: torch.Tensor) -> torch.Tensor:
                    coef = torch.linalg.solve(gram, torch.matmul(B_lib.t(), Y))
                    return Y - torch.matmul(B_lib, coef)

                r_a = _res_quad(s_a)
                r_b = _res_quad(s_b)
                r_f = _res_quad(s_f)

                r_ac = r_a - r_a.mean(dim=0, keepdim=True)
                r_bc = r_b - r_b.mean(dim=0, keepdim=True)
                valid_ab = (M_a.sum(dim=1) >= 1.0) & (M_b.sum(dim=1) >= 1.0)
                corr_num = (r_ac * r_bc).mean(dim=0)
                corr_den = torch.sqrt(r_ac.pow(2).mean(dim=0) * r_bc.pow(2).mean(dim=0) + 1e-10)
                corrs = torch.clamp(corr_num / corr_den, min=0.0, max=1.0)
                if bool(valid_ab.any()):
                    r_split_b = torch.sqrt(corrs[valid_ab].pow(2).mean() + 1e-10)
                else:
                    r_split_b = torch.tensor(0.0, device=s_f.device, dtype=torch.float32)

                k_half = max(self.n_pathways // 2, 1)
                load_r = r_f.mean(dim=-1)
                sd_r = r_f[:, :k_half].mean(dim=-1) - r_f[:, k_half:].mean(dim=-1)
                m_g1 = load_r <= torch.quantile(load_r, 0.15)
                m_s = sd_r >= torch.quantile(sd_r, 0.85)
                m_m = (-sd_r) >= torch.quantile(-sd_r, 0.85)
                p_g1 = r_f[m_g1].mean(dim=0) if bool(m_g1.any()) else r_f.mean(dim=0)
                p_s = r_f[m_s].mean(dim=0) if bool(m_s.any()) else r_f.mean(dim=0)
                p_m = r_f[m_m].mean(dim=0) if bool(m_m.any()) else r_f.mean(dim=0)
                mu_r = (p_g1 + p_s + p_m) / 3.0
                sig_res_b = torch.sqrt(
                    (((p_g1 - mu_r).pow(2) + (p_s - mu_r).pow(2) + (p_m - mu_r).pow(2)) / 3.0).sum() + 1e-10
                )

                q05_b = torch.quantile(s_f, 0.05, dim=0)
                q95_b = torch.quantile(s_f, 0.95, dim=0)

                if int(self.ema_go_initialized.item()) == 0:
                    self.ema_r_split.copy_(r_split_b)
                    self.ema_sig_res.copy_(sig_res_b)
                    self.ema_q05.copy_(q05_b)
                    self.ema_q95.copy_(q95_b)
                    self.ema_go_initialized.fill_(1)
                elif self.training:
                    self.ema_r_split.mul_(0.95).add_(0.05 * r_split_b)
                    self.ema_sig_res.mul_(0.95).add_(0.05 * sig_res_b)
                    self.ema_q05.mul_(0.95).add_(0.05 * q05_b)
                    self.ema_q95.mul_(0.95).add_(0.05 * q95_b)

        r_sp = self.ema_r_split.to(device=s_full.device, dtype=s_full.dtype)
        s_rs = self.ema_sig_res.to(device=s_full.device, dtype=s_full.dtype)
        raw_gamma = (r_sp.pow(4) / (r_sp.pow(4) + (self.split_r_gate_tau ** 4))) * (
            s_rs.pow(4) / (s_rs.pow(4) + (self.sig_res_gate_tau ** 4))
        )
        gamma_gate = torch.clamp((raw_gamma - 0.02) / 0.98, min=0.0, max=1.0)

        q05 = self.ema_q05.to(device=s_full.device, dtype=s_full.dtype).unsqueeze(0)
        q95 = self.ema_q95.to(device=s_full.device, dtype=s_full.dtype).unsqueeze(0)
        raw_span = torch.clamp(q95 - q05, min=0.05)
        d_max = torch.clamp(raw_span.max(), min=0.75)
        span = torch.clamp(raw_span, min=0.85 * d_max)
        a = (s_full - q05) / span

        k_half = max(self.n_pathways // 2, 1)
        a_s = a[:, :k_half].mean(dim=-1, keepdim=True)
        a_m = a[:, k_half:].mean(dim=-1, keepdim=True)
        u_x = (a_s + a_m - 0.86) / 0.75
        u_y = (a_m - a_s + 0.02) / 0.28
        p2 = torch.cat([u_x, u_y], dim=-1)
        r0_eff_sq = (0.40 ** 2) + 2.5 * ((1.0 - gamma_gate).pow(2))
        u_ribbon_2d = (gamma_gate * 1.20) * p2 / torch.sqrt(p2.pow(2).sum(dim=-1, keepdim=True) + r0_eff_sq)
        emp_go_6d = gamma_gate * (a - 0.43)
        return emp_go_6d, u_ribbon_2d, gamma_gate

    def compute_empirical_go_scores(self, x_full_log: torch.Tensor) -> torch.Tensor:
        emp_go_6d, _, _ = self.compute_empirical_go_targets(x_full_log)
        return emp_go_6d

    def forward(
        self,
        x: torch.Tensor,
        return_go_scores: bool = False,
    ):
        mask_exp = self.expanded_input_mask.to(device=x.device, dtype=x.dtype)
        masked_w = self.pathway_linear.weight * mask_exp
        h_flat = F.linear(x, masked_w, self.pathway_linear.bias)
        h_path = h_flat.view(x.shape[0], self.n_pathways, self.n_hidden_per_pathway)
        h_path = self.pathway_dropout(self.pathway_act(self.pathway_norm(h_path)))

        u_cycle_go_enc = (h_path * self.pathway_score_weight.unsqueeze(0)).sum(dim=-1) + self.pathway_score_bias.unsqueeze(0)
        self.last_u_cycle_go_enc = u_cycle_go_enc

        if self.direct_pathway_latent:
            q_m = (h_path * self.direct_mean_weight.unsqueeze(0)).sum(dim=-1) + self.direct_mean_bias.unsqueeze(0)
            lv = (h_path * self.direct_var_weight.unsqueeze(0)).sum(dim=-1) + self.direct_var_bias.unsqueeze(0)
            q_v = torch.exp(torch.clamp(lv, min=-8.0, max=3.0)) + self.var_eps
        else:
            h_loop = self.loop_fc(h_path.reshape(x.shape[0], -1))
            q_m = self.mean_encoder(h_loop)
            lv = self.var_encoder(h_loop)
            q_v = torch.exp(torch.clamp(lv, min=-8.0, max=3.0)) + self.var_eps

        if self.training:
            latent = q_m + torch.sqrt(q_v) * torch.randn_like(q_m)
        else:
            latent = q_m

        if return_go_scores:
            return q_m, q_v, latent, u_cycle_go_enc
        return q_m, q_v, latent


# ==========================================
# FiLM transition block (v0.4.2 Natural-Identity + v0.4.1 RMSNorm + v0.4.0 MatchedLN + legacy Hadamard)
# ==========================================
class SplitNaturalShiftFiLM(nn.Module):
    """Disentangled state and cell-cycle FiLM transition module.

    Supports:
    - ``film_mode="m7_block_tt"`` (default): Stoch-SNR Block-TT WP-FiLM with:
      - ``cycle_norm_mode="natural_identity"`` (v0.4.2 GO-pathway default): preserves natural
        radial amplitude and origin-anchored quiescence (u_cycle / R_star) with planar
        origin-centered cycle FiLM and per-perturbation radial de-leaking.
      - ``cycle_norm_mode="rms_norm"`` (v0.4.1 J2): variance-matched bias-free RMSNorm on u_cycle.
      - ``cycle_norm_mode="matched_ln"`` (v0.4.0 M7): variance-matched bias-free LayerNorm.
    - ``film_mode="legacy_hadamard"``: backward-compatible v0.3.0 diagonal affine FiLM.
    """

    def __init__(
        self,
        u_cycle_dim: int = 10,
        u_state_dim: int = 20,
        drug_embed_dim: int = 16,
        n_samples: int = 1,
        film_mode: str = "m7_block_tt",
        transition_mode: Optional[str] = None,
        cycle_norm_mode: str = "matched_ln",
        r_star_cycle: float = 1.0,
        deleak_radial_cycle: bool = False,
        lambda_gamma_state: float = 1.5,
        lambda_alpha_state: float = 1.0,
        lambda_gamma_cycle: Optional[float] = None,
        lambda_alpha_cycle: Optional[float] = None,
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
        if transition_mode is not None:
            film_mode = transition_mode
        if film_mode not in ("m7_block_tt", "legacy_hadamard"):
            raise ValueError(
                f"Unsupported film_mode='{film_mode}'. Choose 'm7_block_tt' or 'legacy_hadamard'."
            )
        if cycle_norm_mode not in ("natural_identity", "rms_norm", "matched_ln"):
            raise ValueError(
                f"Unsupported cycle_norm_mode='{cycle_norm_mode}'. "
                "Choose 'natural_identity', 'rms_norm', or 'matched_ln'."
            )

        self.u_cycle_dim = int(u_cycle_dim)
        self.u_state_dim = int(u_state_dim)
        self.drug_embed_dim = int(drug_embed_dim)
        self.film_mode = film_mode
        self.transition_mode = film_mode
        self.cycle_norm_mode = cycle_norm_mode
        self.r_star_cycle = float(r_star_cycle)
        self.deleak_radial_cycle = bool(deleak_radial_cycle)
        self.lambda_gamma_state = float(lambda_gamma_state)
        self.lambda_alpha_state = float(lambda_alpha_state)
        self.lambda_gamma_cycle = (
            float(lambda_gamma_cycle)
            if lambda_gamma_cycle is not None
            else (0.02 if cycle_norm_mode == "natural_identity" else float(lambda_gamma_state))
        )
        self.lambda_alpha_cycle = (
            float(lambda_alpha_cycle)
            if lambda_alpha_cycle is not None
            else (0.008 if cycle_norm_mode == "natural_identity" else float(lambda_alpha_state))
        )

        if film_mode == "legacy_hadamard":
            self.norm_cycle = nn.LayerNorm(u_cycle_dim)
            self.norm_state = nn.LayerNorm(u_state_dim)

            self.gamma_cycle = nn.Linear(drug_embed_dim, u_cycle_dim)
            self.beta_cycle = nn.Linear(drug_embed_dim, u_cycle_dim)
            self.gamma_state = nn.Linear(drug_embed_dim, u_state_dim)
            self.beta_state = nn.Linear(drug_embed_dim, u_state_dim)

            nn.init.zeros_(self.gamma_cycle.weight)
            nn.init.zeros_(self.gamma_cycle.bias)
            nn.init.zeros_(self.gamma_state.weight)
            nn.init.zeros_(self.gamma_state.bias)
            nn.init.zeros_(self.beta_cycle.weight)
            nn.init.zeros_(self.beta_cycle.bias)
            nn.init.zeros_(self.beta_state.weight)
            nn.init.zeros_(self.beta_state.bias)
            return

        # --- M7-InvDrop / v0.4.2 Block-TT WP-FiLM ---
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

        self.norm_weight_cycle = nn.Parameter(torch.ones(D_cyc))
        self.norm_weight_state = nn.Parameter(torch.ones(D_st))

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
        self.register_buffer("pert_r_mean_cycle", torch.ones(S, dtype=torch.float32))
        self.register_buffer("ctrl_r_mean_cycle", torch.tensor(1.0, dtype=torch.float32))

        for ch, D_c in (("cycle", D_cyc), ("state", D_st)):
            self.register_buffer(f"ctrl_qm_var_{ch}", torch.ones(D_c, dtype=torch.float32))
            self.register_buffer(f"ctrl_qv_mean_{ch}", torch.ones(D_c, dtype=torch.float32))
            self.register_buffer(f"ctrl_mean_{ch}", torch.zeros(D_c, dtype=torch.float32))
            self.register_buffer(f"ctrl_cov_{ch}", torch.eye(D_c, dtype=torch.float32))
            self.register_buffer(f"pool_cov_{ch}", torch.eye(D_c, dtype=torch.float32))
            self.register_buffer(f"pert_means_{ch}", torch.zeros(S, D_c, dtype=torch.float32))

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

    @property
    def ln_cycle_weight(self) -> torch.Tensor:
        return self.norm_weight_cycle

    @property
    def ln_state_weight(self) -> torch.Tensor:
        return self.norm_weight_state

    # --- Normalization Helpers ---
    def _normalize_stochastic(
        self,
        subspace_or_u: Union[str, torch.Tensor],
        u_or_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(subspace_or_u, str):
            subspace = subspace_or_u
            u = u_or_weight
            weight = getattr(self, f"norm_weight_{subspace}")
        else:
            u = subspace_or_u
            weight = u_or_weight if u_or_weight is not None else self.norm_weight_state
            subspace = "cycle" if weight is self.norm_weight_cycle else "state"

        w = weight.to(device=u.device, dtype=u.dtype)
        if subspace == "cycle":
            if self.cycle_norm_mode == "natural_identity":
                if self.deleak_radial_cycle:
                    return u / self.r_star_cycle
                return (u / self.r_star_cycle) * w
            if self.cycle_norm_mode == "rms_norm":
                rms = torch.sqrt(u.pow(2).mean(dim=-1, keepdim=True) + self.ln_eps)
                return (u / rms) * w

        mean = u.mean(dim=-1, keepdim=True)
        var = u.var(dim=-1, unbiased=False, keepdim=True)
        return ((u - mean) / torch.sqrt(var + self.ln_eps)) * w

    def _normalize_matched(
        self,
        channel: str,
        q_m: torch.Tensor,
        q_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight = getattr(self, f"norm_weight_{channel}").to(device=q_m.device, dtype=q_m.dtype)
        if channel == "cycle":
            if self.cycle_norm_mode == "natural_identity":
                if self.deleak_radial_cycle:
                    return q_m / self.r_star_cycle
                return (q_m / self.r_star_cycle) * weight

        if q_v is not None:
            qv_bar = q_v.to(device=q_m.device, dtype=q_m.dtype).mean(dim=-1, keepdim=True)
        else:
            qv_bar = getattr(self, f"ctrl_qv_mean_{channel}").to(device=q_m.device, dtype=q_m.dtype).mean()

        if channel == "cycle" and self.cycle_norm_mode == "rms_norm":
            denom = torch.sqrt(q_m.pow(2).mean(dim=-1, keepdim=True) + qv_bar + self.ln_eps)
            return (q_m / denom) * weight

        D_c = q_m.shape[-1]
        mean = q_m.mean(dim=-1, keepdim=True)
        var = q_m.var(dim=-1, unbiased=False, keepdim=True)
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
        if channel == "cycle" and self.cycle_norm_mode == "natural_identity" and self.deleak_radial_cycle and self.training:
            with torch.no_grad():
                if u_src.shape[-1] >= 2:
                    r_b = torch.sqrt(u_src[:, 0].pow(2) + u_src[:, 1].pow(2) + 1e-6)
                else:
                    r_b = torch.sqrt(u_src.pow(2).sum(dim=-1) + 1e-6)
                ctrl_m = c_film.view(-1) < 0.5
                if int(ctrl_m.sum().item()) >= 2:
                    self.ctrl_r_mean_cycle.mul_(0.95).add_(0.05 * r_b[ctrl_m].mean())
                for p_val in torch.unique(sample_idx).tolist():
                    pm = sample_idx == p_val
                    if int(pm.sum().item()) >= 2:
                        self.pert_r_mean_cycle[p_val].mul_(0.90).add_(0.10 * r_b[pm].mean())

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
        alpha_eff = (g_act[:2].mean() * alpha) if (channel == "cycle" and g_act.numel() >= 2) else alpha

        if channel == "cycle" and self.cycle_norm_mode == "natural_identity" and self.deleak_radial_cycle:
            if u_src.shape[-1] == 3:
                plane_scale = torch.tensor([[1.0, 1.0, 0.05]], device=u_src.device, dtype=u_src.dtype)
                delta_tan_ring = (g_act.unsqueeze(0) * gamma_par) * u_src
                delta_perp_ring = F.linear(gamma_perp * F.linear(u_src, V_hat), U_hat)
                delta_z = c_film * (delta_tan_ring + delta_perp_ring + alpha_eff) * plane_scale
                r2_cur = u_src[:, 0:1].pow(2) + u_src[:, 1:2].pow(2)
                r0_rad_sq = 0.25 ** 2
                r_cur = torch.sqrt(r2_cur + 1e-8)
                e_rad_smooth = torch.cat(
                    [u_src[:, 0:2] * (r_cur / (r2_cur + r0_rad_sq)), torch.zeros_like(r_cur)],
                    dim=-1,
                )
                dr_leak = (self.pert_r_mean_cycle[sample_idx] - self.ctrl_r_mean_cycle).unsqueeze(-1)
                u_base_clean = u_src - c_film * (dr_leak * e_rad_smooth).detach()
                u_xy = u_base_clean[:, 0:2]
                r_u = torch.sqrt(u_xy.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
                z_raw_xy = u_xy + delta_z[:, 0:2]
                z_retr_xy = r_u * (z_raw_xy / torch.sqrt(z_raw_xy.pow(2).sum(dim=-1, keepdim=True) + 1e-6))
                z_xy = torch.where(c_film > 0.5, z_retr_xy, u_src[:, 0:2])
                z_z2 = u_src[:, 2:3]
                return torch.cat([z_xy, z_z2], dim=-1)
            else:
                delta_tan_ring = (g_act.unsqueeze(0) * gamma_par) * u_src
                delta_perp_ring = F.linear(gamma_perp * F.linear(u_src, V_hat), U_hat)
                delta_z = c_film * (delta_tan_ring + delta_perp_ring + alpha_eff)
                return u_src + delta_z

        delta_tan = (g_act.unsqueeze(0) * gamma_par) * u_check
        delta_perp = F.linear(gamma_perp * F.linear(u_check, V_hat), U_hat)
        delta_z = c_film * (delta_tan + delta_perp + alpha_eff)
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

        sample_idx = sample_or_embed.reshape(-1).long()
        if q_m_cycle is None:
            q_m_cycle = u_cycle
        if q_m_state is None:
            q_m_state = u_state

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
            u_cycle_src = self._normalize_stochastic("cycle", u_cycle)
            u_state_src = self._normalize_stochastic("state", u_state)
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
        lambda_gamma: Optional[float] = None,
        lambda_alpha: Optional[float] = None,
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

        lg_st = float(lambda_gamma) if lambda_gamma is not None else self.lambda_gamma_state
        la_st = float(lambda_alpha) if lambda_alpha is not None else self.lambda_alpha_state
        lg_cyc = (
            self.lambda_gamma_cycle
            if self.cycle_norm_mode == "natural_identity"
            else (float(lambda_gamma) if lambda_gamma is not None else self.lambda_gamma_cycle)
        )
        la_cyc = (
            self.lambda_alpha_cycle
            if self.cycle_norm_mode == "natural_identity"
            else (float(lambda_alpha) if lambda_alpha is not None else self.lambda_alpha_cycle)
        )

        total_raw = torch.tensor(0.0, device=dev, dtype=torch.float32)
        for ch, D_c, lg_c, la_c in (
            ("state", self.u_state_dim, lg_st, la_st),
            ("cycle", self.u_cycle_dim, lg_cyc, la_cyc),
        ):
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
                float(lg_c) * (tan_energy + perp_energy) + float(la_c) * alpha_energy
            )

        return total_raw


# Alias for SplitWPFiLMTransition
SplitWPFiLMTransition = SplitNaturalShiftFiLM


# ==========================================
# Bifurcated / Trifurcated Decoder (v0.4.2 GO-Cylindrical + Gauge-Fixed State + Quarantined OOD)
# ==========================================
class BifurcatedDecoder(nn.Module):
    """Multi-head generative decoder for PertVI.

    Supports:
      - ``cycle_decoder_mode="linear"`` (v0.3.0 / v0.4.0 default): linear projection of z_cycle
        masked by ``cycle_gene_mask``.
      - ``cycle_decoder_mode="cylindrical_harmonic"`` (R3 / R4): 6D cylindrical-harmonic basis
        phi_cyl(z_cycle) masked by ``cycle_gene_mask`` + Gauge-Fixed Zero-Mean Non-Cycle Logit
        Centering on ``main_head``.
      - ``cycle_decoder_mode="go_cylindrical"`` (v0.4.2 default when ``pathway_mask`` is provided):
        expands z_cycle into 6D cylindrical harmonics phi_cyl(z_cycle), maps to K=6 explicit
        GO pathway activations z_cycle_go = cyl_to_go(phi_cyl(z_cycle)) in R^6, and decodes onto
        cycle genes via non-negative pathway-masked weights Softplus(W_go) * M_path^T plus a
        norm-bounded harmonic residual W_harm * m_cyc.
      - ``split_ood_decoder=True`` (v0.4.2): separates ``state_head(z_state)`` (masked by
        ``1 - state_exclude_mask`` and zero-mean centered across non-cycle genes) from
        ``ood_head(v_ood)`` (bias-free so h_ood(0) === 0 on controls, and hard-masked by
        ``1 - ood_quarantine_mask`` so d(h_ood,g)/d(v_ood) === 0 on all mitotic genes).
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_latent_cycle: int,
        cycle_gene_mask: torch.Tensor,
        n_latent_state: Optional[int] = None,
        n_latent_ood: Optional[int] = None,
        pathway_mask: Optional[torch.Tensor] = None,
        state_exclude_mask: Optional[torch.Tensor] = None,
        ood_quarantine_mask: Optional[torch.Tensor] = None,
        cycle_decoder_mode: str = "linear",
        split_ood_decoder: bool = False,
        gauge_fix_state: bool = False,
        harmonic_2theta_scale: float = 0.75,
        r0_smooth: float = 0.45,
        radial_amp_scale: float = 1.0,
        harmonic_basis_mode: str = "soft_unit_ribbon",
        cycle_max_col_norm: Optional[float] = 10.0,
        ood_max_col_norm: Optional[float] = 1.85,
        decouple_batch_state: bool = True,
        n_cat_list: list = None,
        n_hidden: int = 128,
        n_hidden_ood: int = 64,
        n_layers: int = 1,
        max_col_norm: Optional[float] = 2.0,
    ):
        super().__init__()
        self.n_latent_cycle = int(n_latent_cycle)
        self.n_latent_main = int(n_input - n_latent_cycle)
        self.n_genes = int(n_output)
        self.max_col_norm = max_col_norm
        self.ood_max_col_norm = ood_max_col_norm
        self.decouple_batch_state = bool(decouple_batch_state)
        self.cycle_decoder_mode = str(cycle_decoder_mode)
        self.split_ood_decoder = bool(split_ood_decoder)
        self.gauge_fix_state = bool(gauge_fix_state) or (
            self.cycle_decoder_mode in ("cylindrical_harmonic", "go_cylindrical")
        )
        self.harmonic_2theta_scale = float(harmonic_2theta_scale)
        self.r0_smooth = float(r0_smooth)
        self.radial_amp_scale = float(radial_amp_scale)
        self.harmonic_basis_mode = str(harmonic_basis_mode)

        if n_latent_state is not None and n_latent_ood is not None:
            self.n_latent_state = int(n_latent_state)
            self.n_latent_ood = int(n_latent_ood)
        else:
            self.n_latent_state = self.n_latent_main
            self.n_latent_ood = 0
            self.split_ood_decoder = False

        cyc_mask_f = torch.as_tensor(cycle_gene_mask, dtype=torch.float32).reshape(-1).clone()
        self.register_buffer("cycle_gene_mask", cyc_mask_f)

        if state_exclude_mask is not None:
            st_excl_f = torch.as_tensor(state_exclude_mask, dtype=torch.float32).reshape(-1).clone()
        else:
            st_excl_f = cyc_mask_f.clone()
        self.register_buffer("state_exclude_mask", st_excl_f)

        if ood_quarantine_mask is not None:
            ood_quar_f = torch.as_tensor(ood_quarantine_mask, dtype=torch.float32).reshape(-1).clone()
        else:
            ood_quar_f = cyc_mask_f.clone()
        self.register_buffer("ood_quarantine_mask", ood_quar_f)

        # --- Pathway A: Cell-Cycle Head ---
        self.has_pathway_mask = pathway_mask is not None
        if self.has_pathway_mask:
            M_path = torch.as_tensor(pathway_mask, dtype=torch.float32).clone()
            self.n_pathways = int(M_path.shape[0])
            self.register_buffer("pathway_mask", M_path)
        else:
            self.n_pathways = 0
            self.register_buffer("pathway_mask", None)

        cyl_in_dim = 6 if self.n_latent_cycle == 3 else self.n_latent_cycle
        if self.cycle_decoder_mode == "go_cylindrical" and self.has_pathway_mask:
            use_go_bias = self.harmonic_basis_mode == "legacy_cylindrical"
            self.cyl_to_go = nn.Linear(cyl_in_dim, self.n_pathways, bias=use_go_bias)
            with torch.no_grad():
                if self.harmonic_basis_mode == "legacy_cylindrical" and cyl_in_dim == self.n_pathways:
                    nn.init.eye_(self.cyl_to_go.weight)
                else:
                    nn.init.normal_(self.cyl_to_go.weight, mean=0.0, std=0.1)
                if self.cyl_to_go.bias is not None:
                    nn.init.zeros_(self.cyl_to_go.bias)

            # Non-negative pathway-to-gene weights: Softplus(W_go_raw) * M_path^T
            self.go_gene_weight_raw = nn.Parameter(
                torch.full((self.n_genes, self.n_pathways), -1.0)
            )
            # Fine-grained cylindrical-harmonic residual head on cycle genes
            self.cycle_head = nn.Linear(cyl_in_dim, self.n_genes)
            with torch.no_grad():
                self.cycle_head.weight.mul_(cyc_mask_f.unsqueeze(1))
            if cycle_max_col_norm is not None and float(cycle_max_col_norm) > 0.0:
                register_parametrization(
                    self.cycle_head,
                    "weight",
                    MaxColNorm(max_col_norm=float(cycle_max_col_norm)),
                )
        elif self.cycle_decoder_mode == "cylindrical_harmonic":
            self.cycle_head = nn.Linear(cyl_in_dim, self.n_genes)
            with torch.no_grad():
                self.cycle_head.weight.mul_(cyc_mask_f.unsqueeze(1))
            if cycle_max_col_norm is not None and float(cycle_max_col_norm) > 0.0:
                register_parametrization(
                    self.cycle_head,
                    "weight",
                    MaxColNorm(max_col_norm=float(cycle_max_col_norm)),
                )
        else:
            self.cycle_head = nn.Linear(self.n_latent_cycle, self.n_genes)

        # --- Pathway B: State / Main Head & Batch Bypass ---
        self.use_batch = n_cat_list is not None and len(n_cat_list) > 0
        state_core_dim = self.n_latent_state if self.split_ood_decoder else self.n_latent_main
        if self.use_batch:
            self.batch_embed = nn.Embedding(n_cat_list[0], 16)
            main_in_dim = (
                state_core_dim
                if (self.split_ood_decoder and self.decouple_batch_state)
                else (state_core_dim + 16)
            )
            self.batch_gene_bias = nn.Linear(16, self.n_genes)
        else:
            main_in_dim = state_core_dim

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

        # --- Pathway C: Dedicated Quarantined OOD Head (when split_ood_decoder=True) ---
        if self.split_ood_decoder and self.n_latent_ood > 0:
            eff_ood_hid_col_norm = (
                float(max_col_norm)
                if (max_col_norm is not None and float(max_col_norm) > 0.0)
                else (float(self.ood_max_col_norm) if (self.ood_max_col_norm is not None and float(self.ood_max_col_norm) > 0.0) else None)
            )
            self.ood_hidden = nn.Sequential(
                nn.Linear(self.n_latent_ood, int(n_hidden_ood), bias=False),
                nn.ReLU(),
            )
            if eff_ood_hid_col_norm is not None:
                register_parametrization(
                    self.ood_hidden[0],
                    "weight",
                    MaxColNorm(max_col_norm=eff_ood_hid_col_norm),
                )
            self.ood_head = nn.Linear(int(n_hidden_ood), self.n_genes, bias=False)
            with torch.no_grad():
                self.ood_head.weight.mul_((1.0 - ood_quar_f).unsqueeze(1))
            if self.ood_max_col_norm is not None and float(self.ood_max_col_norm) > 0.0:
                eff_ood_head_col_norm = float(self.ood_max_col_norm) * max(1.0, (float(self.n_genes) / 1000.0) ** 0.5)
                register_parametrization(
                    self.ood_head,
                    "weight",
                    MaxColNorm(max_col_norm=eff_ood_head_col_norm),
                )

    def compute_go_pathway_coordinates(self, cycle_latent: torch.Tensor) -> torch.Tensor:
        """Maps a 3D cycle latent (u_cycle_src or z_cycle) to K=6 explicit GO pathway coordinates."""
        phi = expand_cylindrical_harmonics(
            cycle_latent,
            harmonic_2theta_scale=self.harmonic_2theta_scale,
            r0_smooth=self.r0_smooth,
            radial_amp_scale=self.radial_amp_scale,
            harmonic_basis_mode=self.harmonic_basis_mode,
        )
        if self.cycle_decoder_mode == "go_cylindrical" and self.has_pathway_mask:
            return self.cyl_to_go(phi)
        return phi

    def decode_cycle_logits(self, z_cycle: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes h_cycle and (if go_cylindrical) the K=6 GO pathway activations z_cycle_go."""
        cyc_mask_1d = self.cycle_gene_mask.float()
        if self.cycle_decoder_mode == "go_cylindrical" and self.has_pathway_mask:
            phi_cyl = expand_cylindrical_harmonics(
                z_cycle,
                harmonic_2theta_scale=self.harmonic_2theta_scale,
                r0_smooth=self.r0_smooth,
                radial_amp_scale=self.radial_amp_scale,
                harmonic_basis_mode=self.harmonic_basis_mode,
            )
            z_cycle_go = self.cyl_to_go(phi_cyl)
            M_path_t = self.pathway_mask.t().to(device=z_cycle.device, dtype=z_cycle.dtype)  # (G, K)
            p_counts = M_path_t.sum(dim=1, keepdim=True).clamp(min=1.0)
            pos_go_weight = (F.softplus(self.go_gene_weight_raw) * M_path_t) / p_counts
            h_go = F.linear(z_cycle_go, pos_go_weight)

            masked_res_w = self.cycle_head.weight * cyc_mask_1d.unsqueeze(1)
            masked_res_b = self.cycle_head.bias * cyc_mask_1d
            h_res = F.linear(phi_cyl, masked_res_w, masked_res_b)
            return h_go + h_res, z_cycle_go
        elif self.cycle_decoder_mode == "cylindrical_harmonic":
            phi_cyl = expand_cylindrical_harmonics(
                z_cycle,
                harmonic_2theta_scale=self.harmonic_2theta_scale,
                r0_smooth=self.r0_smooth,
                radial_amp_scale=self.radial_amp_scale,
                harmonic_basis_mode=self.harmonic_basis_mode,
            )
            masked_cycle_weight = self.cycle_head.weight * cyc_mask_1d.unsqueeze(1)
            masked_cycle_bias = self.cycle_head.bias * cyc_mask_1d
            h_cycle = F.linear(phi_cyl, masked_cycle_weight, masked_cycle_bias)
            return h_cycle, None
        else:
            masked_cycle_weight = self.cycle_head.weight * cyc_mask_1d.unsqueeze(1)
            masked_cycle_bias = self.cycle_head.bias * cyc_mask_1d
            h_cycle = F.linear(z_cycle, masked_cycle_weight, masked_cycle_bias)
            return h_cycle, None

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
    ):
        z_cycle = z[:, : self.n_latent_cycle]
        h_cycle, _ = self.decode_cycle_logits(z_cycle)

        if self.split_ood_decoder and self.n_latent_ood > 0:
            z_state = z[:, self.n_latent_cycle : self.n_latent_cycle + self.n_latent_state]
            v_ood = z[:, self.n_latent_cycle + self.n_latent_state :]
            if self.use_batch and batch_index is not None:
                b_embed = self.batch_embed(batch_index.squeeze(-1))
                z_state_in = z_state if self.decouple_batch_state else torch.cat([z_state, b_embed], dim=-1)
                h_batch = self.batch_gene_bias(b_embed)
            else:
                z_state_in = z_state
                h_batch = 0.0

            h_hidden = self.main_hidden(z_state_in)
            h_state_raw = self.main_head(h_hidden)

            if self.gauge_fix_state:
                non_st_mask = (1.0 - self.state_exclude_mask.float()).unsqueeze(0)
                n_non_st = non_st_mask.sum().clamp(min=1.0)
                h_state_mean = (h_state_raw * non_st_mask).sum(dim=-1, keepdim=True) / n_non_st
                h_state = (h_state_raw - h_state_mean) * non_st_mask
            else:
                h_state = h_state_raw

            non_quar_mask = (1.0 - self.ood_quarantine_mask.float()).unsqueeze(1)
            masked_ood_w = self.ood_head.weight * non_quar_mask
            h_ood_hid = self.ood_hidden(v_ood)
            h_ood_raw = F.linear(h_ood_hid, masked_ood_w)
            if self.gauge_fix_state:
                non_quar_1d = (1.0 - self.ood_quarantine_mask.float()).unsqueeze(0)
                n_non_quar = non_quar_1d.sum().clamp(min=1.0)
                h_ood_mean = (h_ood_raw * non_quar_1d).sum(dim=-1, keepdim=True) / n_non_quar
                h_ood = (h_ood_raw - h_ood_mean) * non_quar_1d
            else:
                h_ood = h_ood_raw

            h_total = h_cycle + h_state + h_ood + h_batch
        else:
            z_main = z[:, self.n_latent_cycle :]
            if self.use_batch and batch_index is not None:
                b_embed = self.batch_embed(batch_index.squeeze(-1))
                z_main = torch.cat([z_main, b_embed], dim=-1)
                h_batch = self.batch_gene_bias(b_embed)
            else:
                h_batch = 0.0

            h_hidden = self.main_hidden(z_main)
            h_main_raw = self.main_head(h_hidden)

            if self.gauge_fix_state:
                non_st_mask = (1.0 - self.state_exclude_mask.float()).unsqueeze(0)
                n_non_st = non_st_mask.sum().clamp(min=1.0)
                h_main_mean = (h_main_raw * non_st_mask).sum(dim=-1, keepdim=True) / n_non_st
                h_main = (h_main_raw - h_main_mean) * non_st_mask
            else:
                h_main = h_main_raw

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
        ood_dropout_rate: float = 0.0,
        basal_gene_dropout_rate: float = 0.05,
        ood_sub_dropout_rate: float = 0.05,
        selective_cycle_input: bool = True,
        cycle_encoder_mode: str = "standard",
        cycle_norm_mode: str = "matched_ln",
        r_star_cycle: float = 1.0,
        deleak_radial_cycle: bool = False,
        direct_pathway_latent: bool = False,
        n_hidden_per_pathway: int = 16,
        dropout_rate: float = 0.1,
        cycle_gene_mask: Optional[torch.Tensor] = None,
        state_exclude_mask: Optional[torch.Tensor] = None,
        ood_quarantine_mask: Optional[torch.Tensor] = None,
        pathway_mask: Optional[torch.Tensor] = None,
        drug_embed_dim: int = 16,
        film_mode: str = "m7_block_tt",
        transition_mode: Optional[str] = None,
        lambda_gamma_state: float = 1.5,
        lambda_alpha_state: float = 1.0,
        lambda_gamma_cycle: Optional[float] = None,
        lambda_alpha_cycle: Optional[float] = None,
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
        if transition_mode is not None:
            film_mode = transition_mode
        self.film_mode = film_mode
        self.cycle_encoder_mode = str(cycle_encoder_mode)
        self.selective_cycle_input = bool(selective_cycle_input)
        self.basal_gene_dropout_rate = float(basal_gene_dropout_rate)
        self.ood_sub_dropout_rate = float(ood_sub_dropout_rate)
        self.ood_dropout_rate = float(ood_dropout_rate)
        self.dropout_rate = float(dropout_rate)

        if cycle_gene_mask is not None:
            cyc_t = torch.as_tensor(cycle_gene_mask, dtype=torch.float32).reshape(-1).clone()
            self.register_buffer("cycle_gene_mask", cyc_t, persistent=False)
        else:
            self.register_buffer("cycle_gene_mask", None, persistent=False)

        if state_exclude_mask is not None:
            st_excl_t = torch.as_tensor(state_exclude_mask, dtype=torch.float32).reshape(-1).clone()
            self.register_buffer("state_exclude_mask", st_excl_t, persistent=False)
        else:
            self.register_buffer(
                "state_exclude_mask",
                self.cycle_gene_mask.clone() if self.cycle_gene_mask is not None else None,
                persistent=False,
            )

        if ood_quarantine_mask is not None:
            ood_quar_t = torch.as_tensor(ood_quarantine_mask, dtype=torch.float32).reshape(-1).clone()
            self.register_buffer("ood_quarantine_mask", ood_quar_t, persistent=False)
        else:
            self.register_buffer(
                "ood_quarantine_mask",
                self.cycle_gene_mask.clone() if self.cycle_gene_mask is not None else None,
                persistent=False,
            )

        # --- Selective Basal Input Gene Masking & Nested OOD Sub-Mask ---
        self.basal_gene_dropout = (
            nn.Dropout(p=self.basal_gene_dropout_rate)
            if self.basal_gene_dropout_rate > 0.0
            else nn.Identity()
        )
        self.ood_sub_dropout = (
            nn.Dropout(p=self.ood_sub_dropout_rate)
            if self.ood_sub_dropout_rate > 0.0
            else nn.Identity()
        )

        p_hid_cycle = (
            0.0
            if (self.selective_cycle_input and self.cycle_gene_mask is not None)
            else self.dropout_rate
        )
        self.p_hid_cycle = p_hid_cycle
        self.p_hid_state = self.dropout_rate

        # --- Dual Encoders for Sample-Unaware (u) Spaces ---
        if self.cycle_encoder_mode == "go_pathway" and pathway_mask is not None:
            self.encoder_cycle = PathwayCycleEncoder(
                n_input=n_input,
                pathway_mask=pathway_mask,
                n_output=n_latent_cycle,
                n_hidden_per_pathway=n_hidden_per_pathway,
                n_hidden=128,
                dropout_rate=p_hid_cycle,
                direct_pathway_latent=direct_pathway_latent,
            )
        else:
            self.encoder_cycle = Encoder(
                n_input=n_input,
                n_output=n_latent_cycle,
                n_hidden=128,
                n_layers=2,
                dropout_rate=p_hid_cycle,
            )

        self.encoder_state = Encoder(
            n_input=n_input,
            n_output=n_latent_state,
            n_hidden=128,
            n_layers=2,
            dropout_rate=self.dropout_rate,
        )

        if film_mode == "legacy_hadamard":
            self.sample_embedding = nn.Embedding(n_samples, drug_embed_dim)

        # --- The OOD Escape Hatch (v_ood) ---
        self.encoder_ood = Encoder(
            n_input=n_input,
            n_output=n_latent_ood,
            n_hidden=128,
            n_layers=1,
            dropout_rate=self.dropout_rate,
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
            cycle_norm_mode=cycle_norm_mode,
            r_star_cycle=r_star_cycle,
            deleak_radial_cycle=deleak_radial_cycle,
            lambda_gamma_state=lambda_gamma_state,
            lambda_alpha_state=lambda_alpha_state,
            lambda_gamma_cycle=lambda_gamma_cycle,
            lambda_alpha_cycle=lambda_alpha_cycle,
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
        self.ood_dropout = nn.Dropout(self.ood_dropout_rate)

    @property
    def transition(self) -> SplitNaturalShiftFiLM:
        return self.split_film_transition

    def forward(
        self,
        x: torch.Tensor,
        sample_index: torch.Tensor,
        is_treated: Optional[torch.Tensor] = None,
    ):
        """
        x: Raw log-normalized counts tensor
        """
        if self.cycle_gene_mask is not None and self.selective_cycle_input:
            cyc_mask = self.cycle_gene_mask.to(x.device, dtype=x.dtype).unsqueeze(0)
            st_excl = (
                self.state_exclude_mask.to(x.device, dtype=x.dtype).unsqueeze(0)
                if self.state_exclude_mask is not None
                else cyc_mask
            )
            ood_quar = (
                self.ood_quarantine_mask.to(x.device, dtype=x.dtype).unsqueeze(0)
                if self.ood_quarantine_mask is not None
                else cyc_mask
            )
            G_total = float(x.shape[-1])
            G_cyc = float(cyc_mask.sum().clamp(min=1.0).item())
            scale_cyc = math.sqrt(G_total / G_cyc)
            if isinstance(self.encoder_cycle, PathwayCycleEncoder):
                counts_est = torch.clamp(torch.expm1(x), min=0.0)
                tot_counts = counts_est.sum(dim=-1, keepdim=True).clamp(min=1.0).detach()
                x_cp10k = torch.log1p(counts_est * (1e4 / tot_counts))
                x_cycle = (x_cp10k * cyc_mask) * scale_cyc
            else:
                x_cycle = (x * cyc_mask) * scale_cyc
            x_non_cycle = x * (1.0 - st_excl)
            x_state = self.basal_gene_dropout(x_non_cycle)
            x_ood = self.ood_sub_dropout(x_state * (1.0 - ood_quar))
        else:
            x_cycle = x
            x_state = self.basal_gene_dropout(x)
            x_ood = self.ood_sub_dropout(x_state)

        # Extract Latents
        if isinstance(self.encoder_cycle, PathwayCycleEncoder):
            q_m_cycle, q_v_cycle, u_cycle, u_cycle_go_enc = self.encoder_cycle(
                x_cycle, return_go_scores=True
            )
            emp_go_scores, emp_ribbon_2d, cycle_reliability_gate = (
                self.encoder_cycle.compute_empirical_go_targets(x)
            )
        else:
            q_m_cycle, q_v_cycle, u_cycle = self.encoder_cycle(x_cycle)
            u_cycle_go_enc = None
            emp_go_scores = None
            emp_ribbon_2d = None
            cycle_reliability_gate = None

        q_m_state, q_v_state, u_state = self.encoder_state(x_state)
        q_m_ood, q_v_ood, v_ood = self.encoder_ood(x_ood)

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
            z_shift = z_base - torch.cat([u_cycle_src, u_state_src], dim=-1)

        # Composition with Information Dropout on OOD (deterministic q_m_ood in eval mode)
        v_ood_dropped = self.ood_dropout(v_ood)
        v_ood_for_z = v_ood_dropped if self.training else q_m_ood

        # Final composition
        z_final = torch.cat([z_base, v_ood_for_z], dim=-1)

        # Library Size
        empirical_counts = torch.clamp(x.sum(dim=-1, keepdim=True), min=1.0)
        empirical_lib_log = torch.log(empirical_counts)
        q_lib_params = self.encoder_library(x)
        q_m_lib = empirical_lib_log + q_lib_params[:, 0:1]
        q_v_lib_raw = torch.clamp(q_lib_params[:, 1:2], min=-5.0, max=2.0)
        q_v_lib = torch.exp(q_v_lib_raw) + 1e-4
        library_log = q_m_lib + torch.sqrt(q_v_lib) * torch.randn_like(q_m_lib)

        out = {
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
        if u_cycle_go_enc is not None:
            out["u_cycle_go_enc"] = u_cycle_go_enc
        if emp_go_scores is not None:
            out["emp_go_scores"] = emp_go_scores
        if emp_ribbon_2d is not None:
            out["emp_ribbon_2d"] = emp_ribbon_2d
        if cycle_reliability_gate is not None:
            out["cycle_reliability_gate"] = cycle_reliability_gate
        return out
