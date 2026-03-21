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
        max_lambda_adv=10.0,
        max_lambda_ood=50.0,
        warmup_epochs=50,
        anneal_epochs=100,
        **kwargs
    ):
        # Initialize the standard scvi-tools TrainingPlan
        super().__init__(module, **kwargs)
        
        # Store our custom scheduling hyperparameters
        self.max_lambda_adv = max_lambda_adv
        self.max_lambda_ood = max_lambda_ood
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs

    def _compute_scheduled_weight(self, max_weight):
        """
        Calculates the current penalty weight based on the epoch.
        """
        epoch = self.current_epoch
        
        # Phase 1: Warm-up (Standard VAE training, penalties = 0)
        if epoch < self.warmup_epochs:
            return 0.0
            
        # Phase 2: Linear Annealing (Gradually ramp up the penalties)
        elif epoch < self.warmup_epochs + self.anneal_epochs:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            return max_weight * progress
            
        # Phase 3: Plateau (Max tension applied for the remainder of training)
        else:
            return max_weight

    def training_step(self, batch, batch_idx):
        """
        Intercepts the training step to update the module's penalty weights 
        dynamically before the forward pass and loss computation occur.
        """
        # 1. Calculate the current weights
        current_lambda_adv = self._compute_scheduled_weight(self.max_lambda_adv)
        current_lambda_ood = self._compute_scheduled_weight(self.max_lambda_ood)
        
        # 2. Inject the weights directly into the module
        # This makes them accessible inside compute_loss method
        self.module.current_lambda_adv = current_lambda_adv
        self.module.current_lambda_ood = current_lambda_ood

        # 3. Proceed with the standard scvi-tools forward pass and loss calculation
        return super().training_step(batch, batch_idx)
        
    def on_train_epoch_end(self):
        """
        Log the dynamic weights so we can track them in TensorBoard/W&B.
        """
        # Logging these allows visualization of the annealing curves
        # alongside the reconstruction loss and KL divergence.
        self.log("lambda_adv", self.module.current_lambda_adv, on_epoch=True, prog_bar=False)
        self.log("lambda_ood", self.module.current_lambda_ood, on_epoch=True, prog_bar=False)
        
        super().on_train_epoch_end()


class PertVI(RNASeqMixin, VAEMixin, BaseModelClass):
    """
    Model for disentangling state-driven (cell cycle/differentiation) from 
    state-driving (OOD) perturbation effects.
    """
    def __init__(
        self,
        adata,
        n_latent_cycle=10,
        n_latent_state=20,
        n_latent_ood=20,
        **model_kwargs
    ):
        super().__init__(adata)
        
        # Setup the scVI module
        self.module = PertVIModule(
            n_input=self.summary_stats.n_vars,
            n_batch=self.summary_stats.n_batch,
            n_samples=len(self.adata_manager.get_state_registry("sample").categorical_mapping),
            n_latent_cycle=n_latent_cycle,
            n_latent_state=n_latent_state,
            n_latent_ood=n_latent_ood,
            **model_kwargs
        )
        self._model_summary_string = "Disentangled MrVI for Perturbation Analysis."
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata,
        layer=None,
        batch_key=None,
        sample_key=None,
        s_score_key="S_score",   # Default output from sc.tl.cell_cycle_scoring
        g2m_score_key="G2M_score",   # Default output from sc.tl.cell_cycle_scoring
        treatment_mask_key="is_treated",   # To feed into the penalizer
        **kwargs
    ):
        """Registers the AnnData object for the model."""
        setup_method_args = cls._get_setup_method_args(**locals())
        
        # Define what fields the model needs to pull from AnnData
        anndata_fields = [
            LayerField("X", layer, is_count_data=True),
            CategoricalObsField("batch", batch_key),
            CategoricalObsField("sample", sample_key),
        
            # This is the magic key for multiple continuous covariates.
            # It creates a tensor named "cell_cycle_scores" by concatenating the S and G2M columns.
            NumericalJointObsField(
                "cell_cycle_scores", 
                [s_score_key, g2m_score_key] 
            ),
            NumericalJointObsField(
                "is_treated", 
                [treatment_mask_key]
            )
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)


    def train(
        self,
        max_epochs=400,
        warmup_epochs=50,
        anneal_epochs=100,
        max_lambda_adv=10.0,
        max_lambda_ood=50.0,
        train_size=0.9,
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
        """
        Overrides the standard train method to use our dynamic scheduling plan
        and modern scvi-tools TrainRunner.
        """
        import torch
        
        plan_kwargs = plan_kwargs or {}
        datasplitter_kwargs = datasplitter_kwargs or {}

        # 0. Hardware routing: Translate use_gpu to modern PyTorch Lightning args
        if accelerator == "auto":
            accelerator = "gpu" if (use_gpu and torch.cuda.is_available()) else "cpu"
            devices = 1 if accelerator == "gpu" else "auto"

        # 1. Setup the DataSplitter (Removed use_gpu here, as it is no longer accepted)
        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            **datasplitter_kwargs
        )

        # 2. Instantiate our custom Lightning plan
        training_plan = PertVITrainingPlan(
            self.module,
            warmup_epochs=warmup_epochs,
            anneal_epochs=anneal_epochs,
            max_lambda_adv=max_lambda_adv,
            max_lambda_ood=max_lambda_ood,
            **plan_kwargs
        )
        
        # 3. Handle Early Stopping kwargs
        if early_stopping:
            trainer_kwargs["early_stopping"] = True
            if "early_stopping_monitor" not in trainer_kwargs:
                trainer_kwargs["early_stopping_monitor"] = "reconstruction_loss_validation"

        # 4. Initialize and execute the TrainRunner with modern hardware args
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
        """Returns a dictionary of all disentangled latent spaces."""
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        
        latent_spaces = {
            "u_cycle": [], "u_state": [], "v_ood": [], "z_final": []
        }
        
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            
            latent_spaces["u_cycle"].append(outputs["q_m_cycle"].cpu().numpy())
            latent_spaces["u_state"].append(outputs["q_m_state"].cpu().numpy())
            latent_spaces["v_ood"].append(outputs["q_m_ood"].cpu().numpy())
            
            # This is your combined, sample-aware space!
            latent_spaces["z_final"].append(outputs["z_final"].cpu().numpy())

        import numpy as np
        for key in latent_spaces.keys():
            latent_spaces[key] = np.concatenate(latent_spaces[key], axis=0)
            
        return latent_spaces

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata=None,
        indices=None,
        batch_size=None,
        target_sum=10000, # Standard scaling factor (like CPM)
    ):
        """Returns the batch-corrected, denoised gene expression."""
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        
        exprs = []
        
        for tensors in scdl:
            # 1. Run inference to get the latent states
            inference_inputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_inputs)
            
            # 2. Feed the latent states into the generative decoder
            generative_inputs = self.module._get_generative_input(tensors, inference_outputs)
            generative_outputs = self.module.generative(**generative_inputs)
            
            # 3. Extract px_scale (the denoised, batch-free relative frequencies)
            px_scale = generative_outputs["px_scale"].cpu().numpy()
            
            # Scale to a target sum for easier interpretation 
            exprs.append(px_scale * target_sum)

        import numpy as np
        import pandas as pd
        
        # Concatenate and return as a DataFrame with gene names
        exprs_array = np.concatenate(exprs, axis=0)
        return pd.DataFrame(
            exprs_array, 
            index=adata.obs_names, 
            columns=adata.var_names
        )