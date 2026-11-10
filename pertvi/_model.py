import os
from typing import Dict, List, Optional, Union

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

_DEFAULT_CONTROL_TOKENS = {
    "ntc",
    "dmso",
    "control",
    "nt",
    "ctrl",
    "non-targeting",
    "vehicle",
    "unperturbed",
}


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
        **kwargs,
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
    _loading_legacy_ckpt: bool = False

    def __init__(
        self,
        adata,
        n_latent_cycle: int = 3,
        n_latent_state: int = 30,
        n_latent_ood: int = 10,
        cycle_gene_mask_key: str = "is_cycle_gene",
        control_groups: Optional[Union[str, List[str]]] = None,
        film_mode: str = "m7_block_tt",
        lambda_film_gamma: float = 1.5,
        lambda_film_alpha: float = 1.0,
        basal_gene_dropout_rate: Union[float, str] = 0.15,
        ortho_mode: str = "linear_cov",
        **model_kwargs,
    ):
        super().__init__(adata)

        if getattr(self.__class__, "_loading_legacy_ckpt", False):
            film_mode = "legacy_hadamard"
            basal_gene_dropout_rate = 0.0
            model_kwargs = dict(model_kwargs)
            model_kwargs.setdefault("max_col_norm", None)

        # Extract the boolean mask from AnnData
        if cycle_gene_mask_key not in adata.var.columns:
            raise ValueError(f"Column '{cycle_gene_mask_key}' not found in adata.var.")

        cycle_mask_array = adata.var[cycle_gene_mask_key].values.astype(bool)
        cycle_gene_mask = torch.tensor(cycle_mask_array, dtype=torch.bool)

        # Compute sample counts and auto-detect sample-level control categories
        cat_mapping = np.asarray(self.adata_manager.get_state_registry("sample").categorical_mapping)
        n_samples = len(cat_mapping)

        sample_codes = np.asarray(self.adata_manager.get_from_registry("sample")).reshape(-1).astype(np.int64)
        is_treated_arr = np.asarray(self.adata_manager.get_from_registry("is_treated")).reshape(-1).astype(np.float32)

        counts_np = np.bincount(sample_codes, minlength=n_samples).astype(np.float32)
        sample_counts = torch.tensor(np.maximum(counts_np, 1.0), dtype=torch.float32)

        is_ctrl_np = np.zeros(n_samples, dtype=np.float32)
        if control_groups is not None:
            ctrl_targets = (
                {str(control_groups)}
                if isinstance(control_groups, str)
                else {str(g) for g in control_groups}
            )
            for idx, name in enumerate(cat_mapping):
                if str(name) in ctrl_targets:
                    is_ctrl_np[idx] = 1.0
        else:
            for idx, name in enumerate(cat_mapping):
                if str(name).strip().lower() in _DEFAULT_CONTROL_TOKENS:
                    is_ctrl_np[idx] = 1.0
            untreated_codes = np.unique(sample_codes[is_treated_arr < 0.5])
            if len(untreated_codes) > 0:
                is_ctrl_np[untreated_codes] = 1.0

        is_control_sample = torch.tensor(is_ctrl_np, dtype=torch.float32)
        n_obs_total = int(adata.n_obs)

        self.module = PertVIModule(
            n_input=self.summary_stats.n_vars,
            n_batch=self.summary_stats.n_batch,
            n_samples=n_samples,
            n_latent_cycle=n_latent_cycle,
            n_latent_state=n_latent_state,
            n_latent_ood=n_latent_ood,
            cycle_gene_mask=cycle_gene_mask,
            n_obs_total=n_obs_total,
            sample_counts=sample_counts,
            is_control_sample=is_control_sample,
            film_mode=film_mode,
            lambda_film_gamma=lambda_film_gamma,
            lambda_film_alpha=lambda_film_alpha,
            basal_gene_dropout_rate=basal_gene_dropout_rate,
            ortho_mode=ortho_mode,
            **model_kwargs,
        )
        self._model_summary_string = "Disentangled Multi-resolution Variational Inference for Perturbation Analysis."
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata=None,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        prefix: Optional[str] = None,
        **kwargs,
    ):
        pt_path = os.path.join(dir_path, f"{prefix or ''}model.pt")
        is_legacy = False
        if os.path.exists(pt_path):
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", {})
            if "inference_net.split_film_transition.gamma_state.bias" in sd:
                is_legacy = True
        cls._loading_legacy_ckpt = is_legacy
        try:
            return super().load(
                dir_path,
                adata=adata,
                accelerator=accelerator,
                device=device,
                prefix=prefix,
                **kwargs,
            )
        finally:
            cls._loading_legacy_ckpt = False

    @classmethod
    def setup_anndata(
        cls,
        adata,
        layer=None,
        batch_key=None,
        sample_key=None,
        treatment_mask_key="is_treated",
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())

        # PHASE 2 FIX: Removed the S_score and G2M_score dependencies
        anndata_fields = [
            LayerField("X", layer, is_count_data=True),
            CategoricalObsField("batch", batch_key),
            CategoricalObsField("sample", sample_key),
            NumericalJointObsField("is_treated", [treatment_mask_key]),
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
        **trainer_kwargs,
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
            **datasplitter_kwargs,
        )

        self.module.train()
        training_plan = PertVITrainingPlan(
            self.module,
            warmup_epochs=warmup_epochs,
            anneal_epochs=anneal_epochs,
            max_lambda_adv_state=max_lambda_adv_state,
            max_lambda_adv_ood=max_lambda_adv_ood,
            max_lambda_ood=max_lambda_ood,
            floor_lambda_ood=floor_lambda_ood,
            **plan_kwargs,
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
            **trainer_kwargs,
        )

        return runner()

    @torch.inference_mode()
    def get_film_diagnostics(self) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """Extract learned FiLM SNR gates, effective modulators, and shrunk centroids."""
        was_training = self.module.training
        self.module.eval()
        try:
            return self._get_film_diagnostics_impl()
        finally:
            self.module.train(was_training)

    def _get_film_diagnostics_impl(self) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        sample_names = np.asarray(
            self.adata_manager.get_state_registry("sample").categorical_mapping
        )
        film = self.module.inference_net.split_film_transition

        if getattr(film, "film_mode", "legacy_hadamard") == "legacy_hadamard":
            dev = film.gamma_state.weight.device
            idx = torch.arange(len(sample_names), dtype=torch.long, device=dev)
            emb = self.module.inference_net.sample_embedding(idx)
            gamma_par_state = film.gamma_state(emb).cpu().numpy()
            alpha_state = film.beta_state(emb).cpu().numpy()
            gamma_par_cycle = film.gamma_cycle(emb).cpu().numpy()
            alpha_cycle = film.beta_cycle(emb).cpu().numpy()
            S = len(sample_names)
            D_st = film.u_state_dim
            D_cyc = film.u_cycle_dim
            is_ctrl = np.zeros(S, dtype=np.float32)
            counts = np.ones(S, dtype=np.float32)
            summary_df = pd.DataFrame(
                {
                    "is_control_sample": is_ctrl,
                    "sample_counts": counts,
                    "gamma_par_state_norm": np.linalg.norm(gamma_par_state, axis=1),
                    "alpha_state_norm": np.linalg.norm(alpha_state, axis=1),
                },
                index=pd.Index(sample_names, name="sample"),
            )
            return {
                "sample_names": sample_names,
                "is_control_sample": is_ctrl,
                "sample_counts": counts,
                "snr_state": np.ones(D_st, dtype=np.float32),
                "g_act_state": np.ones(D_st, dtype=np.float32),
                "gamma_par_state": gamma_par_state,
                "gamma_perp_state": np.zeros((S, 0), dtype=np.float32),
                "alpha_state": alpha_state,
                "shrunk_mu_state": np.zeros((S, D_st), dtype=np.float32),
                "snr_cycle": np.ones(D_cyc, dtype=np.float32),
                "g_act_cycle": np.ones(D_cyc, dtype=np.float32),
                "gamma_par_cycle": gamma_par_cycle,
                "gamma_perp_cycle": np.zeros((S, 0), dtype=np.float32),
                "alpha_cycle": alpha_cycle,
                "shrunk_mu_cycle": np.zeros((S, D_cyc), dtype=np.float32),
                "summary_df": summary_df,
            }

        dev = film.emb_gamma.weight.device
        S = film.n_samples
        idx = torch.arange(S, dtype=torch.long, device=dev)
        c_film = (1.0 - film.is_control_sample[idx]).unsqueeze(-1)

        e_gamma = film.emb_gamma(idx)
        e_alpha = film.emb_alpha(idx)

        snr_state = film.get_snr("state").cpu().numpy()
        g_act_state_t, _ = film.get_snr_gates("state")
        g_act_state = g_act_state_t.cpu().numpy()
        gamma_par_state = (c_film * film.gamma_par_state(e_gamma)).cpu().numpy()
        gamma_perp_state = (c_film * film.gamma_perp_state(e_gamma)).cpu().numpy()
        alpha_state = (c_film * film.alpha_state(e_alpha)).cpu().numpy()
        shrunk_mu_state = film.get_shrunk_centroids("state").cpu().numpy()

        snr_cycle = film.get_snr("cycle").cpu().numpy()
        g_act_cycle_t, _ = film.get_snr_gates("cycle")
        g_act_cycle = g_act_cycle_t.cpu().numpy()
        gamma_par_cycle = (c_film * film.gamma_par_cycle(e_gamma)).cpu().numpy()
        gamma_perp_cycle = (c_film * film.gamma_perp_cycle(e_gamma)).cpu().numpy()
        alpha_cycle = (c_film * film.alpha_cycle(e_alpha)).cpu().numpy()
        shrunk_mu_cycle = film.get_shrunk_centroids("cycle").cpu().numpy()

        is_ctrl_np = film.is_control_sample.cpu().numpy()
        counts_np = film.sample_counts.cpu().numpy()

        summary_df = pd.DataFrame(
            {
                "is_control_sample": is_ctrl_np,
                "sample_counts": counts_np,
                "gamma_par_state_norm": np.linalg.norm(gamma_par_state, axis=1),
                "gamma_perp_state_norm": np.linalg.norm(gamma_perp_state, axis=1),
                "alpha_state_norm": np.linalg.norm(alpha_state, axis=1),
                "shrunk_mu_state_norm": np.linalg.norm(shrunk_mu_state, axis=1),
                "gamma_par_cycle_norm": np.linalg.norm(gamma_par_cycle, axis=1),
                "alpha_cycle_norm": np.linalg.norm(alpha_cycle, axis=1),
            },
            index=pd.Index(sample_names, name="sample"),
        )

        return {
            "sample_names": sample_names,
            "is_control_sample": is_ctrl_np,
            "sample_counts": counts_np,
            "snr_state": snr_state,
            "g_act_state": g_act_state,
            "gamma_par_state": gamma_par_state,
            "gamma_perp_state": gamma_perp_state,
            "alpha_state": alpha_state,
            "shrunk_mu_state": shrunk_mu_state,
            "snr_cycle": snr_cycle,
            "g_act_cycle": g_act_cycle,
            "gamma_par_cycle": gamma_par_cycle,
            "gamma_perp_cycle": gamma_perp_cycle,
            "alpha_cycle": alpha_cycle,
            "shrunk_mu_cycle": shrunk_mu_cycle,
            "summary_df": summary_df,
        }

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
            "z_shift": [], "z_base": [], "z_final": [],
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
        batch_size=128,
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
