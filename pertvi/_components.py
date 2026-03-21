import torch
import torch.nn as nn
from torch.autograd import Function
from scvi.nn import Encoder, FCLayers

# ==========================================
# Perturbation-specialized MrVI Inference Architecture
# ==========================================

class PertVI_Inference(nn.Module):
    def __init__(
        self, 
        n_input, 
        n_samples,
        n_latent_cycle=10, 
        n_latent_state=20, 
        n_latent_ood=20,
        n_cycle_targets=2
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

        # --- B. Encoders to learn library size ---
        self.encoder_library = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2) 
        )
        
        # --- C. The Adversarial "Stick" ---
        self.cycle_classifier = nn.Sequential(
            nn.Linear(n_latent_cycle, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, n_cycle_targets)
        )
        
        # --- D. The Sample-Aware (z) Transition Layer ---
        # Takes the concatenated u-spaces. 
        self.n_latent_u = n_latent_cycle + n_latent_state
        
        # Use a standard PyTorch Sequential block to avoid scvi-tools FCLayers quirks
        self.z_transition = nn.Sequential(
            nn.Linear(self.n_latent_u + 16, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.n_latent_u)
        )
        
        # --- E. The OOD Escape Hatch (v_ood) ---
        self.encoder_ood = Encoder(
            n_input=n_input,
            n_output=n_latent_ood,
            n_hidden=128,
            n_layers=1
        )

    def forward(self, x, sample_index):
        """
        x: Raw counts tensor
        """
        
        # 1. Extract Cell Cycle state
        q_m_cycle, q_v_cycle, u_cycle = self.encoder_cycle(x)
        
        # 2. Extract Intrinsic Biology (Cancer State / EMT / Metabolism)
        q_m_state, q_v_state, u_state = self.encoder_state(x)
        
        # 3. Extract the Out-of-Distribution shift
        q_m_ood, q_v_ood, v_ood = self.encoder_ood(x)
        
        # 4. Apply the Orthogonality Constraint
        pred_cycle = self.cycle_classifier(u_cycle)
        
        # 5. The Sample-Aware Space (z)
        u_combined = torch.cat([u_cycle, u_state], dim=-1)
        
        # 6. Embed the sample and glue it to the biology
        sample_embed = self.sample_embedding(sample_index.squeeze(-1).long())
        u_and_sample = torch.cat([u_combined, sample_embed], dim=-1)
        
        # 7. Calculate the sample-specific shift and add it to the base biology
        # This mirrors the true MrVI: z = u + f(u, s)
        z_shift = self.z_transition(u_and_sample)
        z_base = u_combined + z_shift 
        
        # 8. Final compositional mapping
        z_final = torch.cat([z_base, v_ood], dim=-1) 

        # 9. Extract and Sample the Library Size
        # Calculate the reality (empirical sum)
        empirical_counts = torch.clamp(x.sum(dim=-1, keepdim=True), min=1.0)
        empirical_lib_log = torch.log(empirical_counts)
        
        # Let the network guess a small TWEAK, not the absolute number!
        q_lib_params = self.encoder_library(x)
        
        # add the network's guess to the true empirical baseline
        q_m_lib = empirical_lib_log + q_lib_params[:, 0:1] 
        
        # Tame the variance
        q_v_lib_raw = torch.clamp(q_lib_params[:, 1:2], min=-5.0, max=2.0)
        q_v_lib = torch.exp(q_v_lib_raw) + 1e-4 
        
        # Sample the final latent library size
        library_log = q_m_lib + torch.sqrt(q_v_lib) * torch.randn_like(q_m_lib)
        
        return {
            "u_cycle": u_cycle, "u_state": u_state, "v_ood": v_ood,
            "q_m_cycle": q_m_cycle, "q_v_cycle": q_v_cycle,
            "q_m_state": q_m_state, "q_v_state": q_v_state,
            "q_m_ood": q_m_ood, "q_v_ood": q_v_ood,
            "z_final": z_final,
            "pred_cycle": pred_cycle,
            "q_m_lib": q_m_lib,
            "q_v_lib": q_v_lib,
            "library_log": library_log
        }