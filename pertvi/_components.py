import torch
import torch.nn as nn
from scvi.nn import Encoder

# ==========================================
# FiLM transition block
# ==========================================
class SplitNaturalShiftFiLM(nn.Module):
    def __init__(self, u_cycle_dim: int = 10, u_state_dim: int = 20, drug_embed_dim: int = 16):
        super().__init__()
        
        # Cycle-Specific Modulators
        self.gamma_cycle = nn.Linear(drug_embed_dim, u_cycle_dim)
        self.beta_cycle = nn.Linear(drug_embed_dim, u_cycle_dim)
        
        # State-Specific Modulators
        self.gamma_state = nn.Linear(drug_embed_dim, u_state_dim)
        self.beta_state = nn.Linear(drug_embed_dim, u_state_dim)
        
        # Residual Initializations (1 + gamma) to prevent manifold erasure
        nn.init.zeros_(self.gamma_cycle.weight); nn.init.zeros_(self.gamma_cycle.bias)
        nn.init.zeros_(self.gamma_state.weight); nn.init.zeros_(self.gamma_state.bias)
        
        nn.init.zeros_(self.beta_cycle.weight); nn.init.zeros_(self.beta_cycle.bias)
        nn.init.zeros_(self.beta_state.weight); nn.init.zeros_(self.beta_state.bias)
        
        self.norm_cycle = nn.LayerNorm(u_cycle_dim)
        self.norm_state = nn.LayerNorm(u_state_dim)

    def forward(self, u_cycle: torch.Tensor, u_state: torch.Tensor, drug_embedding: torch.Tensor):
        # 1. Independent Cycle Shift
        g_cyc, b_cyc = self.gamma_cycle(drug_embedding), self.beta_cycle(drug_embedding)
        z_cycle = self.norm_cycle(((1.0 + g_cyc) * u_cycle) + b_cyc)
        
        # 2. Independent State Shift
        g_st, b_st = self.gamma_state(drug_embedding), self.beta_state(drug_embedding)
        z_state = self.norm_state(((1.0 + g_st) * u_state) + b_st)
        
        return z_cycle, z_state
        
# ==========================================
# Perturbation-specialized Multi-resolution Varitional Inference Architecture
# ==========================================

class PertVI_Inference(nn.Module):
    def __init__(
        self, 
        n_input, 
        n_samples,
        n_latent_cycle=10, 
        n_latent_state=20, 
        n_latent_ood=10, # Decreased from 20 to cap the OOD Black Hole
        n_cycle_targets=2 # Will be removed in Phase 2
    ):
        super().__init__()
        
        # --- A. Dual Encoders for Sample-Unaware (u) Spaces ---
        self.encoder_cycle = Encoder(
            n_input=n_input, 
            n_output=n_latent_cycle, 
            n_hidden=128, 
            n_layers=2
        )
        self.encoder_state = Encoder(
            n_input=n_input, 
            n_output=n_latent_state, 
            n_hidden=128, 
            n_layers=2
        )
        self.sample_embedding = nn.Embedding(n_samples, 16)
        
        # --- B. The OOD Escape Hatch (v_ood) ---
        self.encoder_ood = Encoder(
            n_input=n_input,
            n_output=n_latent_ood,
            n_hidden=128,
            n_layers=1
        )

        # --- C. Encoders to learn library size ---
        self.encoder_library = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2) 
        )

        # --- D. The Adversarial cell cycle classifier ---
        self.cycle_classifier = nn.Sequential(
            nn.Linear(n_latent_cycle, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Linear(64, n_cycle_targets)
        )
        
        # --- E. The FiLM Transition ---
        self.split_film_transition = SplitNaturalShiftFiLM(
            u_cycle_dim=n_latent_cycle, 
            u_state_dim=n_latent_state, 
            drug_embed_dim=16
        )

    def forward(self, x, sample_index):
        """
        x: Raw log-normalized counts tensor
        """
        
        # Extract Latents
        q_m_cycle, q_v_cycle, u_cycle = self.encoder_cycle(x)
        q_m_state, q_v_state, u_state = self.encoder_state(x)
        q_m_ood, q_v_ood, v_ood = self.encoder_ood(x)

        # Cell cycle prediction
        pred_cycle = self.cycle_classifier(u_cycle)
        
        # FiLM Transition
        sample_embed = self.sample_embedding(sample_index.squeeze(-1).long())
        z_cycle, z_state = self.split_film_transition(u_cycle, u_state, sample_embed)
        u_base = torch.cat([u_cycle, u_state], dim=-1)
        z_base = torch.cat([z_cycle, z_state], dim=-1)

        # Derive z shift
        z_shift = z_base - u_base
        
        # Final composition
        z_final = torch.cat([z_base, v_ood], dim=-1) 

        # Library Size
        empirical_counts = torch.clamp(x.sum(dim=-1, keepdim=True), min=1.0) # Calculate the reality (empirical sum)
        empirical_lib_log = torch.log(empirical_counts)
        q_lib_params = self.encoder_library(x) # Let the network guess a small TWEAK, not the absolute number
        q_m_lib = empirical_lib_log + q_lib_params[:, 0:1] 
        q_v_lib_raw = torch.clamp(q_lib_params[:, 1:2], min=-5.0, max=2.0) # Tame the variance
        q_v_lib = torch.exp(q_v_lib_raw) + 1e-4 
        library_log = q_m_lib + torch.sqrt(q_v_lib) * torch.randn_like(q_m_lib) # Sample the final latent library size
        
        return {
            "u_cycle": u_cycle, "u_state": u_state, "v_ood": v_ood,
            "q_m_cycle": q_m_cycle, "q_v_cycle": q_v_cycle,
            "q_m_state": q_m_state, "q_v_state": q_v_state,
            "q_m_ood": q_m_ood, "q_v_ood": q_v_ood,
            "u_base": u_base, "z_shift": z_shift, "z_base": z_base,
            "z_cycle": z_cycle, "z_state": z_state,
            "z_final": z_final,
            "pred_cycle": pred_cycle,
            "q_m_lib": q_m_lib,
            "q_v_lib": q_v_lib,
            "library_log": library_log
        }