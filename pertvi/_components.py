import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Normalize the base manifolds FIRST
        u_cycle_norm = self.norm_cycle(u_cycle)
        u_state_norm = self.norm_state(u_state)
        
        # Calculate Independent Modulators
        g_cyc, b_cyc = self.gamma_cycle(drug_embedding), self.beta_cycle(drug_embedding)
        g_st, b_st = self.gamma_state(drug_embedding), self.beta_state(drug_embedding)
        
        # Apply the Momentum Shift AFTER normalization
        z_cycle = ((1.0 + g_cyc) * u_cycle_norm) + b_cyc
        z_state = ((1.0 + g_st) * u_state_norm) + b_st
        
        return z_cycle, z_state

# ==========================================
# Bifurcated decoder: cell cycle head + main head 
# ==========================================

class BifurcatedDecoder(nn.Module):
    def __init__(
        self, n_input: int, n_output: int, n_latent_cycle: int, 
        cycle_gene_mask: torch.Tensor, n_cat_list: list = None, n_hidden: int = 128, n_layers: int = 1
    ):
        super().__init__()
        self.n_latent_cycle = n_latent_cycle
        self.n_latent_main = n_input - n_latent_cycle
        self.n_genes = n_output
        self.register_buffer("cycle_gene_mask", cycle_gene_mask.float())
        
        # Pathway A: Strictly Linear Cycle Head
        self.cycle_head = nn.Linear(self.n_latent_cycle, self.n_genes)
        
        # Pathway B: Dense Main Head & Batch Bypass
        self.use_batch = n_cat_list is not None and len(n_cat_list) > 0
        if self.use_batch:
            self.batch_embed = nn.Embedding(n_cat_list[0], 16)
            main_in_dim = self.n_latent_main + 16
            self.batch_gene_bias = nn.Linear(16, self.n_genes) # Batch bypass for cycle genes
        else:
            main_in_dim = self.n_latent_main

        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(main_in_dim, n_hidden))
            layers.append(nn.BatchNorm1d(n_hidden))
            layers.append(nn.ReLU())
            main_in_dim = n_hidden
            
        self.main_hidden = nn.Sequential(*layers)
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
        
        # --- D. The FiLM Transition ---
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
            "q_m_lib": q_m_lib,
            "q_v_lib": q_v_lib,
            "library_log": library_log
        }