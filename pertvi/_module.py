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
        n_latent_cycle=10,
        n_latent_state=20,
        n_latent_ood=10, # Reduced capacity for OOD bottleneck
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
        
        # 1. Unpack Tensors & Clamp
        x = tensors["X"]
        x_safe = torch.clamp(x, min=0.0)        
        is_treated = tensors.get("is_treated", torch.ones(x.shape[0], 1, device=x.device))
        
        # 2. Unpack Inference & Generative Outputs
        q_m_cycle, q_v_cycle = inference_outputs["q_m_cycle"], inference_outputs["q_v_cycle"]
        q_m_state, q_v_state = inference_outputs["q_m_state"], inference_outputs["q_v_state"]
        q_m_ood, q_v_ood = inference_outputs["q_m_ood"], inference_outputs["q_v_ood"]
        q_m_lib, q_v_lib = inference_outputs["q_m_lib"], inference_outputs["q_v_lib"]
        
        u_cycle = inference_outputs["u_cycle"]
        u_state = inference_outputs["u_state"]
        
        px_rate = generative_outputs["px_rate"]
        
        # 3. Base VAE Losses
        px_r_safe = torch.clamp(self.px_r, min=-10.0, max=15.0)
        px_r = torch.exp(px_r_safe) + 1e-4

        reconst_loss = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x_safe).sum(dim=-1)
        
        prior = Normal(0, 1)
        kl_cycle = kl(Normal(q_m_cycle, torch.sqrt(q_v_cycle)), prior).sum(dim=-1)
        kl_state = kl(Normal(q_m_state, torch.sqrt(q_v_state)), prior).sum(dim=-1)
        
        # Symmetric KL for OOD
        kl_ood_var = kl(Normal(torch.zeros_like(q_m_ood), torch.sqrt(q_v_ood)), prior).sum(dim=-1)

        # KL for library size
        empirical_counts = torch.clamp(x_safe.sum(dim=-1, keepdim=True), min=1.0)
        empirical_library_log = torch.log(empirical_counts).detach() 
        
        prior_lib = Normal(empirical_library_log, torch.ones_like(empirical_library_log))
        kl_library = kl(Normal(q_m_lib, torch.sqrt(q_v_lib)), prior_lib).sum(dim=-1)
        
        kl_local_unaggregated = kl_cycle + kl_state + kl_ood_var + kl_library

        # 4. Orthogonality constraint between state/ood and cycle
        u_cycle_c = u_cycle - u_cycle.mean(dim=0)
        u_state_c = u_state - u_state.mean(dim=0)
        
        cov_matrix = torch.matmul(u_state_c.t(), u_cycle_c) / (u_cycle.shape[0] - 1 + 1e-8)
        disentangle_state_cycle = torch.sum(cov_matrix ** 2)

        # NEW: OOD-Cycle Orthogonalization
        treated_mask = is_treated.squeeze().bool()
        
        # Only calculate if we have enough treated cells in the minibatch to form a covariance matrix
        if treated_mask.sum() > 1: 
            u_cycle_treated = u_cycle[treated_mask]
            q_m_ood_treated = q_m_ood[treated_mask]
            
            u_cyc_t_c = u_cycle_treated - u_cycle_treated.mean(dim=0)
            q_m_ood_t_c = q_m_ood_treated - q_m_ood_treated.mean(dim=0)
            
            cov_matrix_ood = torch.matmul(q_m_ood_t_c.t(), u_cyc_t_c) / (u_cyc_t_c.shape[0] - 1 + 1e-8)
            disentangle_ood_cycle = torch.sum(cov_matrix_ood ** 2)
        else:
            disentangle_ood_cycle = torch.tensor(0.0, device=x.device)

        # OOD Masked L1
        control_mask = (1.0 - is_treated.view(-1, 1))
        ood_leakage_loss = torch.sum(torch.abs(q_m_ood * control_mask), dim=-1) ## switched to L1 penalty

        # 5. Soft L1 Penalty Mask on Main Head
        # Extract weights connecting to cycle genes from the main head
        cycle_mask_bool = self.generative_net.cycle_gene_mask.bool()
        main_head_cycle_weights = self.generative_net.main_head.weight[cycle_mask_bool]
        soft_l1_loss = torch.sum(torch.abs(main_head_cycle_weights))
        
        # 6. Dynamic Weighting & Aggregation
        lambda_kl = 1.0
        lambda_adv_state = getattr(self, "current_lambda_adv_state", 100.0)
        lambda_adv_ood = getattr(self, "current_lambda_adv_ood", 5.0)
        lambda_ood = getattr(self, "current_lambda_ood", 50.0)
        lambda_soft = 50.0 # 5 -> 50 to protect main decoder from cell cycle genes

        # base loss
        base_loss_unaggregated = (
            reconst_loss + (lambda_kl * kl_local_unaggregated) + 
            (lambda_ood * ood_leakage_loss)
        )
        
        # Add scalar penalties (Soft L1 and Covariance) to the final objective
        total_loss = (
            torch.mean(base_loss_unaggregated) + 
            (lambda_adv_state * disentangle_state_cycle) + 
            (lambda_adv_ood * disentangle_ood_cycle) + 
            (lambda_soft * soft_l1_loss)
        )

        # 6. Output
        return LossOutput(
            loss=total_loss,
            reconstruction_loss=reconst_loss,       
            kl_local=kl_local_unaggregated,         
            n_obs_minibatch=x.shape[0],             
            extra_metrics={
                "covariance_penalty_state": disentangle_state_cycle,
                "covariance_penalty_ood": disentangle_ood_cycle,
                "ood_leakage_penalty": torch.mean(ood_leakage_loss),
                "soft_l1_penalty": soft_l1_loss,
                "v_ood_active_fraction": torch.mean((torch.abs(q_m_ood) > 0.5).float()),
            }
        )