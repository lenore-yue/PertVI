import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.distributions import NegativeBinomial
from ._components import PertVI_Inference, BifurcatedDecoder


class PertVIModule(BaseModuleClass):
    def __init__(
        self,
        n_input,
        cycle_gene_mask,
        n_samples=0,
        n_batch=0,
        n_labels=0,
        n_latent_cycle=3,
        n_latent_state=20,
        n_latent_ood=5, # Reduced capacity for OOD bottleneck
        drug_embed_dim=16, # Added embedding dimension
        **kwargs
    ):
        super().__init__()
        self.n_latent_total = n_latent_cycle + n_latent_state + n_latent_ood
        
        self.inference_net = PertVI_Inference(
            n_input=n_input,
            n_samples=n_samples,
            n_latent_cycle=n_latent_cycle,
            n_latent_state=n_latent_state,
            n_latent_ood=n_latent_ood
        )
        
        # use custom bifurcated decoder
        self.generative_net = BifurcatedDecoder(
            n_input=self.n_latent_total, 
            n_output=n_input, 
            n_latent_cycle=n_latent_cycle,
            cycle_gene_mask=cycle_gene_mask, 
            n_cat_list=[n_batch] if n_batch > 0 else None,
            n_layers=1, 
            n_hidden=128
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
            "generative_outputs": generative_outputs
        }

    # --- FORWARD PASSES ---
    @auto_move_data
    def inference(self, tensors):
        x = tensors["X"]
        x_safe = torch.clamp(x, min=0.0)
        x_log = torch.log(1 + x_safe)
        sample_index = tensors.get("sample")
        
        outputs = self.inference_net(x_log, sample_index)
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
            "px_dropout": px_dropout
        }

    # --- LOSS COMPUTATION ---
    def loss(self, tensors, inference_outputs, generative_outputs):
        
        # --- Unpack tensors & clamp ---
        x = tensors["X"]
        x_safe = torch.clamp(x, min=0.0)        
        is_treated = tensors.get("is_treated", torch.ones(x.shape[0], 1, device=x.device))
        
        # Extract 1D masks for KL calculations
        treated_mask_1d = is_treated.squeeze()
        control_mask_1d = (1.0 - is_treated).squeeze()
        
        # --- Unpack inference outputs ---
        q_m_cycle, q_v_cycle = inference_outputs["q_m_cycle"], inference_outputs["q_v_cycle"]
        q_m_state, q_v_state = inference_outputs["q_m_state"], inference_outputs["q_v_state"]
        q_m_ood, q_v_ood = inference_outputs["q_m_ood"], inference_outputs["q_v_ood"]
        q_m_lib, q_v_lib = inference_outputs["q_m_lib"], inference_outputs["q_v_lib"]
        
        u_cycle = inference_outputs["u_cycle"]
        u_state = inference_outputs["u_state"]
        v_ood = inference_outputs["v_ood"]
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

        # --- Orthogonality constraints ---
        # Compute covariance STRICTLY on the predicted means to eliminate stochastic jitter
        q_m_cycle_c = q_m_cycle - q_m_cycle.mean(dim=0)
        q_m_state_c = q_m_state - q_m_state.mean(dim=0)
        
        # State-Cycle
        cov_matrix_state_cycle = torch.matmul(q_m_state_c.t(), q_m_cycle_c) / (q_m_cycle.shape[0] - 1 + 1e-8)
        disentangle_state_cycle = torch.sum(cov_matrix_state_cycle ** 2)

        # OOD shielded constraints (Only on treated cells)
        treated_mask = is_treated.squeeze().bool()
        if treated_mask.sum() > 1: 
            # Use the means for the shielded covariance as well
            q_m_cycle_treated = q_m_cycle[treated_mask]
            q_m_state_treated = q_m_state[treated_mask]
            q_m_ood_treated = q_m_ood[treated_mask]
            
            u_cyc_t_c = q_m_cycle_treated - q_m_cycle_treated.mean(dim=0)
            u_st_t_c = q_m_state_treated - q_m_state_treated.mean(dim=0)
            q_m_ood_t_c = q_m_ood_treated - q_m_ood_treated.mean(dim=0)
            
            # OOD-Cycle
            cov_matrix_ood_cycle = torch.matmul(q_m_ood_t_c.t(), u_cyc_t_c) / (u_cyc_t_c.shape[0] - 1 + 1e-8)
            disentangle_ood_cycle = torch.sum(cov_matrix_ood_cycle ** 2)
            
            # State-OOD
            cov_matrix_ood_state = torch.matmul(q_m_ood_t_c.t(), u_st_t_c) / (u_st_t_c.shape[0] - 1 + 1e-8)
            disentangle_ood_state = torch.sum(cov_matrix_ood_state ** 2)
        else:
            disentangle_ood_cycle = torch.tensor(0.0, device=x.device)
            disentangle_ood_state = torch.tensor(0.0, device=x.device)

        # --- Single-Tiered L1 Sparsity ---
        control_mask_2d = (1.0 - is_treated.view(-1, 1))
        
        # Crush controls to absolute zero
        ood_control_l1 = torch.sum(torch.abs(q_m_ood * control_mask_2d), dim=-1)

        # --- Soft L1 penalty mask on main head ---
        cycle_mask_float = self.generative_net.cycle_gene_mask.float().unsqueeze(1)
        masked_weights = self.generative_net.main_head.weight * cycle_mask_float
        soft_l1_loss = torch.sum(torch.abs(masked_weights))
        
        # --- Dynamic Weighting & Aggregation ---
        lambda_kl = 1.0
        lambda_adv_state = getattr(self, "current_lambda_adv_state", 100.0)
        lambda_adv_ood = getattr(self, "current_lambda_adv_ood", 5.0)
        lambda_adv_state_ood = 5.0 
        lambda_ood_anchor = getattr(self, "current_lambda_ood", 50.0)
        lambda_soft = 50.0 

        # --- Base loss ---
        base_loss_unaggregated = (
            reconst_loss + 
            (lambda_kl * kl_local_unaggregated) + 
            (lambda_ood_anchor * ood_control_l1)
        )
        
        # --- Total loss ---
        total_loss = (
            torch.mean(base_loss_unaggregated) + 
            (lambda_adv_state * disentangle_state_cycle) + 
            (lambda_adv_ood * disentangle_ood_cycle) + 
            (lambda_adv_state_ood * disentangle_ood_state) + 
            (lambda_soft * soft_l1_loss)
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
                "v_ood_active_fraction": torch.mean((torch.abs(q_m_ood) > 0.5).float()),
            }
        )