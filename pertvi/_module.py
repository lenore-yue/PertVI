import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI
from scvi.distributions import NegativeBinomial
from ._components import PertVI_Inference

class PertVIModule(BaseModuleClass):
    def __init__(
        self,
        n_input,
        n_samples=0,
        n_batch=0,
        n_labels=0,
        n_latent_cycle=10,
        n_latent_state=20,
        n_latent_ood=20,
        n_cycle_targets=2,
        **kwargs
    ):
        super().__init__()
        self.n_latent_total = n_latent_cycle + n_latent_state + n_latent_ood
        
        self.inference_net = PertVI_Inference(
            n_input=n_input,
            n_samples=n_samples,
            n_latent_cycle=n_latent_cycle,
            n_latent_state=n_latent_state,
            n_latent_ood=n_latent_ood,
            n_cycle_targets=n_cycle_targets
        )
        
        # Standard initialization (no dispersion kwarg!)
        self.generative_net = DecoderSCVI(
            n_input=self.n_latent_total,
            n_output=n_input,
            n_cat_list=[n_batch] if n_batch > 0 else None,
            n_layers=1,
            n_hidden=128
        )
        
        # Explicitly initialize the global gene dispersion as a trainable PyTorch parameter
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
        
        # Instead of calculating raw_library from 'X', we use the learned one
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
        
        # ---------------------------------------------------------
        # 1. Unpack Tensors & Clamp
        # ---------------------------------------------------------
        x = tensors["X"]
        x_safe = torch.clamp(x, min=0.0)
        
        cycle_targets = tensors["cell_cycle_scores"] 
        cycle_targets = torch.nan_to_num(cycle_targets, nan=0.0)
        
        # We need a mask: 1 for treated (OOD), 0 for control.
        # We use .get() so it doesn't crash if you haven't added it to AnnData yet.
        # It defaults to all 1s (treated) if missing, which just turns off the penalty safely.
        is_treated = tensors.get("is_treated", torch.ones(x.shape[0], 1, device=x.device))
        
        # ---------------------------------------------------------
        # 2. Unpack Inference & Generative Outputs
        # ---------------------------------------------------------
        q_m_cycle, q_v_cycle = inference_outputs["q_m_cycle"], inference_outputs["q_v_cycle"]
        q_m_state, q_v_state = inference_outputs["q_m_state"], inference_outputs["q_v_state"]
        q_m_ood, q_v_ood = inference_outputs["q_m_ood"], inference_outputs["q_v_ood"]
        q_m_lib, q_v_lib = inference_outputs["q_m_lib"], inference_outputs["q_v_lib"]
        
        u_cycle = inference_outputs["u_cycle"]
        u_state = inference_outputs["u_state"]
        v_ood = inference_outputs["v_ood"]
        pred_cycle = inference_outputs["pred_cycle"]
        
        px_rate = generative_outputs["px_rate"]
        
        # ---------------------------------------------------------
        # 3. Base VAE Losses (Reconstruction & KL)
        # ---------------------------------------------------------
        px_r_safe = torch.clamp(self.px_r, min=-10.0, max=15.0)
        px_r = torch.exp(px_r_safe) + 1e-4

        reconst_loss = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x_safe).sum(dim=-1)
        
        # --- Major KL Divergence ---
        prior = Normal(0, 1)
        kl_cycle = kl(Normal(q_m_cycle, torch.sqrt(q_v_cycle)), prior).sum(dim=-1)
        kl_state = kl(Normal(q_m_state, torch.sqrt(q_v_state)), prior).sum(dim=-1)
        kl_ood_var = kl(Normal(torch.zeros_like(q_m_ood), torch.sqrt(q_v_ood)), prior).sum(dim=-1)
        
        # --- Library KL Divergence ---
        # The prior mean is the empirical log sum (reality check)
        empirical_counts = torch.clamp(x_safe.sum(dim=-1, keepdim=True), min=1.0)
        empirical_library_log = torch.log(empirical_counts).detach() 
        
        prior_lib = Normal(empirical_library_log, torch.ones_like(empirical_library_log))
        kl_library = kl(Normal(q_m_lib, torch.sqrt(q_v_lib)), prior_lib).sum(dim=-1)
        
        # total unaggregated kl sum
        kl_local_unaggregated = kl_cycle + kl_state + kl_ood_var + kl_library

        # ---------------------------------------------------------
        # 4. Custom Disentanglement & Constraint Penalties
        # ---------------------------------------------------------
        
        # A. Positive Cycle Loss (Force u_cycle to learn the phases)
        # Shape: (Batch_size,)
        cycle_loss = F.mse_loss(pred_cycle, cycle_targets, reduction="none").sum(dim=-1)

        # B. Covariance Disentanglement (Force u_state to ignore the phases)
        # Center the vectors
        u_cycle_c = u_cycle - u_cycle.mean(dim=0)
        u_state_c = u_state - u_state.mean(dim=0)
        
        # Calculate Covariance Matrix: (Dim_State x Dim_Cycle)
        # We add 1e-8 to the denominator to prevent division by zero in tiny batches
        cov_matrix = torch.matmul(u_state_c.t(), u_cycle_c) / (u_cycle.shape[0] - 1 + 1e-8)
        
        # The penalty is the Frobenius norm squared (scalar value for the whole batch)
        disentangle_loss_scalar = torch.sum(cov_matrix ** 2)

        # C. OOD Leakage Constraint (Force v_ood to be 0 for controls)
        # control_mask is 1 for control, 0 for treated
        control_mask = (1.0 - is_treated.view(-1, 1))
        # Shape: (Batch_size,)
        ood_leakage_loss = torch.sum((v_ood * control_mask) ** 2, dim=-1)

        # ---------------------------------------------------------
        # 5. Dynamic Weighting & Aggregation
        # ---------------------------------------------------------
        lambda_kl = 1.0
        lambda_adv = getattr(self, "current_lambda_adv", 0.0)
        lambda_ood = getattr(self, "current_lambda_ood", 0.0)

        # We aggregate the 1D tensors first
        base_loss_unaggregated = (
            reconst_loss 
            + (lambda_kl * kl_local_unaggregated) 
            + (10.0 * cycle_loss) 
            + (lambda_ood * ood_leakage_loss)
        )

        # Calculate final total loss (Mean over batch + the scalar covariance penalty)
        total_loss = torch.mean(base_loss_unaggregated) + (lambda_adv * disentangle_loss_scalar)

        # ---------------------------------------------------------
        # 6. Output to PyTorch Lightning Logger
        # ---------------------------------------------------------
        return LossOutput(
            loss=total_loss,
            reconstruction_loss=reconst_loss,       
            kl_local=kl_local_unaggregated,         
            n_obs_minibatch=x.shape[0],             
            extra_metrics={
                "cycle_prediction_loss": torch.mean(cycle_loss),
                "covariance_penalty": disentangle_loss_scalar,
                "ood_leakage_penalty": torch.mean(ood_leakage_loss),
                "v_ood_active_fraction": torch.mean((torch.abs(q_m_ood) > 0.5).float()),
                "kl_library": torch.mean(kl_library)
            }
        )