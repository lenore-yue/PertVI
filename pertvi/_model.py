import torch
import numpy as np
import pandas as pd
import scvi

from scvi.model.base import BaseModelClass, RNASeqMixin, VAEMixin
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField, CategoricalObsField, NumericalJointObsField
from scvi.train import TrainRunner, TrainingPlan
from scvi.dataloaders import DataSplitter
from ._module import PertVIModule

class PertVITrainingPlan(TrainingPlan):
    def __init__(
        self, 
        module, 
        max_lambda_adv_state=100.0,  # Heavy penalty for State vs Cycle
        max_lambda_adv_ood=5.0,      # Gentle penalty for OOD vs Cycle
        max_lambda_ood=50.0,         # Max penalty for OOD leakage
        floor_lambda_ood=5.0,        # Floor penalty to prevent garbage dump
        warmup_epochs=50,
        anneal_epochs=100,
        **kwargs
    ):
        super().__init__(module, **kwargs)
        self.max_lambda_adv_state = max_lambda_adv_state
        self.max_lambda_adv_ood = max_lambda_adv_ood
        self.max_lambda_ood = max_lambda_ood
        self.floor_lambda_ood = floor_lambda_ood
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs

    def _compute_standard_weight(self, max_weight):
        """Standard annealing: 0.0 -> max_weight for covariance penalties."""
        epoch = self.current_epoch
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch < self.warmup_epochs + self.anneal_epochs:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            return max_weight * progress
        else:
            return max_weight

    def _compute_floor_weight(self, floor_weight, max_weight):
        """Floor annealing: floor_weight -> max_weight for the OOD L1 penalty."""
        epoch = self.current_epoch
        if epoch < self.warmup_epochs:
            return floor_weight
        elif epoch < self.warmup_epochs + self.anneal_epochs:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            # Linearly interpolate between floor and max
            return floor_weight + (max_weight - floor_weight) * progress
        else:
            return max_weight

    def training_step(self, batch, batch_idx):
        # Calculate dynamically scheduled weights
        current_lambda_adv_state = self._compute_standard_weight(self.max_lambda_adv_state)
        current_lambda_adv_ood = self._compute_standard_weight(self.max_lambda_adv_ood)
        current_lambda_ood = self._compute_floor_weight(self.floor_lambda_ood, self.max_lambda_ood)
        
        # Inject weights directly into the module for the loss function
        self.module.current_lambda_adv_state = current_lambda_adv_state
        self.module.current_lambda_adv_ood = current_lambda_adv_ood
        self.module.current_lambda_ood = current_lambda_ood

        # 3. Proceed with standard forward pass and loss calculation
        return super().training_step(batch, batch_idx)
        
    def on_train_epoch_end(self):
        # Log the dynamic weights so we can track them in TensorBoard/W&B
        self.log("lambda_adv_state", self.module.current_lambda_adv_state, on_epoch=True, prog_bar=False)
        self.log("lambda_adv_ood", self.module.current_lambda_adv_ood, on_epoch=True, prog_bar=False)
        self.log("lambda_ood", self.module.current_lambda_ood, on_epoch=True, prog_bar=False)
        super().on_train_epoch_end()


class PertVI(RNASeqMixin, VAEMixin, BaseModelClass):
    def __init__(
        self,
        adata,
        n_latent_cycle=3,
        n_latent_state=30,
        n_latent_ood=10, 
        cycle_gene_mask_key="is_cycle_gene",
        **model_kwargs
    ):
        super().__init__(adata)
        
        # Extract the boolean mask from AnnData
        if cycle_gene_mask_key not in adata.var.columns:
            raise ValueError(f"Column '{cycle_gene_mask_key}' not found in adata.var.")
        
        cycle_mask_array = adata.var[cycle_gene_mask_key].values.astype(bool)
        cycle_gene_mask = torch.tensor(cycle_mask_array, dtype=torch.bool)
        
        self.module = PertVIModule(
            n_input=self.summary_stats.n_vars,
            n_batch=self.summary_stats.n_batch,
            n_samples=len(self.adata_manager.get_state_registry("sample").categorical_mapping),
            n_latent_cycle=n_latent_cycle,
            n_latent_state=n_latent_state,
            n_latent_ood=n_latent_ood,
            cycle_gene_mask=cycle_gene_mask, # Pass to module
            **model_kwargs
        )
        self._model_summary_string = "Disentangled Multi-resolution Variational Inference for Perturbation Analysis."
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata,
        layer=None,
        batch_key=None,
        sample_key=None,
        treatment_mask_key="is_treated",   
        **kwargs
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        
        # PHASE 2 FIX: Removed the S_score and G2M_score dependencies
        anndata_fields = [
            LayerField("X", layer, is_count_data=True),
            CategoricalObsField("batch", batch_key),
            CategoricalObsField("sample", sample_key),
            NumericalJointObsField("is_treated", [treatment_mask_key])
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs=400,
        warmup_epochs=50,
        anneal_epochs=100,
        max_lambda_adv_state=100.0,
        max_lambda_adv_ood=5.0,
        max_lambda_ood=50.0,
        floor_lambda_ood=5.0,
        train_size=0.8,
        validation_size=None,
        batch_size=256,
        use_gpu=True,
        accelerator="auto",
        devices="auto",
        early_stopping=False,
        plan_kwargs=None,
        datasplitter_kwargs=None,
        **trainer_kwargs
    ):
        import torch
        
        plan_kwargs = plan_kwargs or {}
        datasplitter_kwargs = datasplitter_kwargs or {}

        if accelerator == "auto":
            accelerator = "gpu" if (use_gpu and torch.cuda.is_available()) else "cpu"
            devices = 1 if accelerator == "gpu" else "auto"

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            **datasplitter_kwargs
        )

        training_plan = PertVITrainingPlan(
            self.module,
            warmup_epochs=warmup_epochs,
            anneal_epochs=anneal_epochs,
            max_lambda_adv_state=max_lambda_adv_state,
            max_lambda_adv_ood=max_lambda_adv_ood,
            max_lambda_ood=max_lambda_ood,
            floor_lambda_ood=floor_lambda_ood,
            **plan_kwargs
        )
        
        if early_stopping:
            trainer_kwargs["early_stopping"] = True
            if "early_stopping_monitor" not in trainer_kwargs:
                trainer_kwargs["early_stopping_monitor"] = "reconstruction_loss_validation"

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs
        )
        
        return runner()

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata=None,
        indices=None,
        batch_size=None,
    ):
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        
        # Added z_cycle and z_state extractors
        latent_spaces = {
            "u_cycle": [], "u_state": [], "v_ood": [], 
            "z_cycle": [], "z_state": [],
            "z_shift": [], "z_base": [], "z_final": []
        }
        
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            latent_spaces["u_cycle"].append(outputs["q_m_cycle"].cpu().numpy())
            latent_spaces["u_state"].append(outputs["q_m_state"].cpu().numpy())
            latent_spaces["v_ood"].append(outputs["q_m_ood"].cpu().numpy())
            
            latent_spaces["z_cycle"].append(outputs["z_cycle"].cpu().numpy())
            latent_spaces["z_state"].append(outputs["z_state"].cpu().numpy())
            latent_spaces["z_shift"].append(outputs["z_shift"].cpu().numpy())
            latent_spaces["z_base"].append(outputs["z_base"].cpu().numpy())
            latent_spaces["z_final"].append(outputs["z_final"].cpu().numpy())

        for key in latent_spaces.keys():
            latent_spaces[key] = np.concatenate(latent_spaces[key], axis=0)
            
        return latent_spaces

    @torch.inference_mode()
    def get_normalized_expression(
        self, 
        adata=None, 
        indices=None, 
        transform_batch=None, 
        transform_sample=None, 
        target_sum=10000, 
        batch_size=128
    ):
        """
        Generates denoised / counterfactual expression.
        Use transform_batch to harmonize technical noise.
        Use transform_sample to generate counterfactual drug responses.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        
        # Batch Registry Lookup
        target_batch_idx = None
        if transform_batch is not None:
            batch_mapping = self.adata_manager.get_state_registry("batch").categorical_mapping
            if transform_batch not in batch_mapping:
                raise ValueError(f"Batch '{transform_batch}' not found.")
            target_batch_idx = np.where(batch_mapping == transform_batch)[0][0]

        # Sample (Perturbation) Registry Lookup
        target_sample_idx = None
        if transform_sample is not None:
            sample_mapping = self.adata_manager.get_state_registry("sample").categorical_mapping
            if transform_sample not in sample_mapping:
                raise ValueError(f"Sample '{transform_sample}' not found.")
            target_sample_idx = np.where(sample_mapping == transform_sample)[0][0]

        exprs = []
        for tensors in scdl:
            # Overwrite batch tensor for denoising
            if target_batch_idx is not None:
                tensors[scvi.REGISTRY_KEYS.BATCH_KEY] = torch.full_like(
                    tensors[scvi.REGISTRY_KEYS.BATCH_KEY], target_batch_idx
                )
            
            # Overwrite sample tensor for counterfactual generation
            if target_sample_idx is not None:
                tensors["sample"] = torch.full_like(
                    tensors["sample"], target_sample_idx
                )
                
            inference_inputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_inputs)
            generative_inputs = self.module._get_generative_input(tensors, inference_outputs)
            generative_outputs = self.module.generative(**generative_inputs)
            
            exprs.append(generative_outputs["px_scale"].cpu().numpy() * target_sum)
        
        exprs_array = np.concatenate(exprs, axis=0)
        return pd.DataFrame(exprs_array, index=adata.obs_names, columns=adata.var_names)