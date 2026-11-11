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
from ._go_mask import build_go_cycle_and_quarantine_masks, GO_PATHWAY_NAMES

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
        max_lambda_adv_state=120.0,
        max_lambda_adv_ood=120.0,
        max_lambda_ood=100.0,
        floor_lambda_ood=10.0,
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
        epoch = self.current_epoch
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch < self.warmup_epochs + self.anneal_epochs:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            return max_weight * progress
        else:
            return max_weight

    def _compute_floor_weight(self, floor_weight, max_weight):
        epoch = self.current_epoch
        if epoch < self.warmup_epochs:
            return floor_weight
        elif epoch < self.warmup_epochs + self.anneal_epochs:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            return floor_weight + (max_weight - floor_weight) * progress
        else:
            return max_weight

    def training_step(self, batch, batch_idx):
        current_lambda_adv_state = self._compute_standard_weight(self.max_lambda_adv_state)
        current_lambda_adv_ood = self._compute_standard_weight(self.max_lambda_adv_ood)
        current_lambda_ood = self._compute_floor_weight(self.floor_lambda_ood, self.max_lambda_ood)

        self.module.current_lambda_adv_state = current_lambda_adv_state
        self.module.current_lambda_adv_ood = current_lambda_adv_ood
        self.module.current_lambda_ood = current_lambda_ood

        return super().training_step(batch, batch_idx)

    def on_train_epoch_end(self):
        self.log("lambda_adv_state", self.module.current_lambda_adv_state, on_epoch=True, prog_bar=False)
        self.log("lambda_adv_ood", self.module.current_lambda_adv_ood, on_epoch=True, prog_bar=False)
        self.log("lambda_ood", self.module.current_lambda_ood, on_epoch=True, prog_bar=False)
        super().on_train_epoch_end()


class PertVI(RNASeqMixin, VAEMixin, BaseModelClass):
    _loading_legacy_ckpt: bool = False
    _loading_v040_ckpt: bool = False
    _loading_legacy_cyl_ckpt: bool = False

    def __init__(
        self,
        adata,
        n_latent_cycle: int = 3,
        n_latent_state: int = 30,
        n_latent_ood: int = 10,
        cycle_gene_mask_key: str = "is_cycle_gene",
        control_groups: Optional[Union[str, List[str]]] = None,
        cycle_encoder_mode: str = "go_pathway",
        cycle_decoder_mode: Optional[str] = None,
        two_tier_cycle_mask: bool = True,
        split_ood_decoder: Optional[bool] = None,
        direct_pathway_latent: bool = False,
        film_mode: str = "m7_block_tt",
        cycle_norm_mode: Optional[str] = None,
        cycle_prior_mode: Optional[str] = None,
        deleak_radial_cycle: Optional[bool] = None,
        selective_cycle_input: Optional[bool] = None,
        ood_all_cells_hsic: Optional[bool] = None,
        within_pert_hsic_weight: Optional[float] = None,
        lambda_film_gamma: float = 1.5,
        lambda_film_alpha: float = 1.0,
        basal_gene_dropout_rate: Union[float, str] = "auto",
        ood_sub_dropout_rate: Optional[float] = None,
        ood_dropout_rate: Optional[float] = None,
        ortho_mode: Optional[str] = None,
        ood_anchor_mode: Optional[str] = None,
        max_lambda_adv_state: Optional[float] = None,
        max_lambda_adv_ood: Optional[float] = None,
        lambda_soft: Optional[float] = None,
        floor_lambda_ood: Optional[float] = None,
        max_lambda_ood: Optional[float] = None,
        beta_cyc: Optional[float] = None,
        beta_cycle_null: float = 3.0,
        sigma_cycle_z: Optional[float] = None,
        harmonic_basis_mode: Optional[str] = None,
        r0_smooth: float = 0.45,
        lambda_drug_cycle_align: Optional[float] = None,
        lambda_go_pathway: float = 2.5,
        lambda_ribbon_2d: float = 25.0,
        **model_kwargs,
    ):
        super().__init__(adata)

        if getattr(self.__class__, "_loading_legacy_ckpt", False):
            film_mode = "legacy_hadamard"
            cycle_encoder_mode = "standard"
            cycle_decoder_mode = "linear"
            split_ood_decoder = False
            basal_gene_dropout_rate = 0.0
            ood_sub_dropout_rate = 0.0
            ood_dropout_rate = 0.3
            ortho_mode = "linear_cov"
            ood_anchor_mode = "l1"
            selective_cycle_input = False
            model_kwargs = dict(model_kwargs)
            model_kwargs.setdefault("max_col_norm", None)
        elif getattr(self.__class__, "_loading_v040_ckpt", False):
            cycle_encoder_mode = "standard"
            cycle_decoder_mode = "linear"
            split_ood_decoder = False
            cycle_norm_mode = "matched_ln"
            cycle_prior_mode = "standard_normal"
            selective_cycle_input = False
        elif getattr(self.__class__, "_loading_legacy_cyl_ckpt", False):
            if harmonic_basis_mode is None:
                harmonic_basis_mode = "legacy_cylindrical"
            model_kwargs = dict(model_kwargs)
            model_kwargs.setdefault("ood_max_col_norm", None)
            model_kwargs.setdefault("decouple_batch_state", False)

        if cycle_gene_mask_key not in adata.var.columns:
            raise ValueError(f"Column '{cycle_gene_mask_key}' not found in adata.var.")

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

        # Build masks depending on cycle_encoder_mode
        use_go_arch = (cycle_encoder_mode == "go_pathway") and (film_mode == "m7_block_tt")
        if use_go_arch:
            ctrl_cell_mask = is_ctrl_np[sample_codes] > 0.5
            go_info = build_go_cycle_and_quarantine_masks(
                adata,
                cycle_gene_mask_key=cycle_gene_mask_key,
                control_cell_mask=ctrl_cell_mask,
            )
            tirosh_mask = go_info["tirosh_mask"]
            cycle_gene_mask = go_info["cycle_hard_mask"]
            ood_quarantine_mask = go_info["ood_quarantine_mask"]
            pathway_mask = go_info["pathway_mask"]
            state_exclude_mask = tirosh_mask if bool(two_tier_cycle_mask) else cycle_gene_mask
            self.go_pathway_names_ = list(go_info["pathway_names"])
            self.go_gene_audit_df_ = go_info["gene_audit_df"]
        else:
            cycle_mask_array = adata.var[cycle_gene_mask_key].values.astype(bool)
            cycle_gene_mask = torch.tensor(cycle_mask_array, dtype=torch.bool)
            state_exclude_mask = cycle_gene_mask
            ood_quarantine_mask = model_kwargs.pop("ood_quarantine_mask", cycle_gene_mask)
            pathway_mask = model_kwargs.pop("pathway_mask", None)
            self.go_pathway_names_ = list(GO_PATHWAY_NAMES) if pathway_mask is not None else []
            self.go_gene_audit_df_ = None

        # Resolve defaults based on whether v0.4.2 GO architecture is active
        if cycle_decoder_mode is None:
            cycle_decoder_mode = "go_cylindrical" if use_go_arch else "linear"
        if split_ood_decoder is None:
            split_ood_decoder = True if use_go_arch else False
        if cycle_norm_mode is None:
            cycle_norm_mode = "natural_identity" if use_go_arch else "matched_ln"
        if cycle_prior_mode is None:
            cycle_prior_mode = "anisotropic_normal" if use_go_arch else "standard_normal"
        if deleak_radial_cycle is None:
            deleak_radial_cycle = True if use_go_arch else False
        if selective_cycle_input is None:
            selective_cycle_input = True if use_go_arch else False
        if ood_all_cells_hsic is None:
            ood_all_cells_hsic = True if use_go_arch else False
        if within_pert_hsic_weight is None:
            within_pert_hsic_weight = 0.0
        if ood_sub_dropout_rate is None:
            ood_sub_dropout_rate = 0.05 if use_go_arch else 0.0
        if ood_dropout_rate is None:
            ood_dropout_rate = 0.0 if use_go_arch else 0.3
        if ortho_mode is None:
            ortho_mode = "snr_harmonic_hsic" if use_go_arch else "linear_cov"
        if ood_anchor_mode is None:
            ood_anchor_mode = "l2" if use_go_arch else "l1"
        if max_lambda_adv_state is None:
            max_lambda_adv_state = 160.0 if use_go_arch else 100.0
        if max_lambda_adv_ood is None:
            max_lambda_adv_ood = 120.0 if use_go_arch else 5.0
        if lambda_soft is None:
            lambda_soft = 150.0 if use_go_arch else 50.0
        if floor_lambda_ood is None:
            floor_lambda_ood = 12.0 if use_go_arch else 5.0
        if max_lambda_ood is None:
            max_lambda_ood = 140.0 if use_go_arch else 50.0
        if beta_cyc is None:
            beta_cyc = 0.30 if use_go_arch else 0.50
        if sigma_cycle_z is None:
            sigma_cycle_z = 0.03 if use_go_arch else 0.05
        if harmonic_basis_mode is None:
            harmonic_basis_mode = "soft_unit_ribbon"
        if lambda_drug_cycle_align is None:
            lambda_drug_cycle_align = 0.0

        self.module = PertVIModule(
            n_input=self.summary_stats.n_vars,
            n_batch=self.summary_stats.n_batch,
            n_samples=n_samples,
            n_latent_cycle=n_latent_cycle,
            n_latent_state=n_latent_state,
            n_latent_ood=n_latent_ood,
            cycle_gene_mask=cycle_gene_mask,
            state_exclude_mask=state_exclude_mask,
            ood_quarantine_mask=ood_quarantine_mask,
            pathway_mask=pathway_mask,
            cycle_encoder_mode=cycle_encoder_mode,
            cycle_decoder_mode=cycle_decoder_mode,
            split_ood_decoder=split_ood_decoder,
            direct_pathway_latent=direct_pathway_latent,
            n_obs_total=n_obs_total,
            sample_counts=sample_counts,
            is_control_sample=is_control_sample,
            film_mode=film_mode,
            cycle_norm_mode=cycle_norm_mode,
            cycle_prior_mode=cycle_prior_mode,
            beta_cyc=beta_cyc,
            beta_cycle_null=beta_cycle_null,
            sigma_cycle_z=sigma_cycle_z,
            harmonic_basis_mode=harmonic_basis_mode,
            r0_smooth=r0_smooth,
            deleak_radial_cycle=deleak_radial_cycle,
            selective_cycle_input=selective_cycle_input,
            ood_all_cells_hsic=ood_all_cells_hsic,
            within_pert_hsic_weight=within_pert_hsic_weight,
            lambda_film_gamma=lambda_film_gamma,
            lambda_film_alpha=lambda_film_alpha,
            basal_gene_dropout_rate=basal_gene_dropout_rate,
            ood_sub_dropout_rate=ood_sub_dropout_rate,
            ood_dropout_rate=ood_dropout_rate,
            ortho_mode=ortho_mode,
            ood_anchor_mode=ood_anchor_mode,
            max_lambda_adv_state=max_lambda_adv_state,
            max_lambda_adv_ood=max_lambda_adv_ood,
            lambda_soft=lambda_soft,
            floor_lambda_ood=floor_lambda_ood,
            max_lambda_ood=max_lambda_ood,
            lambda_drug_cycle_align=lambda_drug_cycle_align,
            lambda_go_pathway=lambda_go_pathway if use_go_arch else 0.0,
            lambda_ribbon_2d=lambda_ribbon_2d if use_go_arch else 0.0,
            **model_kwargs,
        )
        self._model_summary_string = (
            "Disentangled Multi-resolution Variational Inference with GO Pathway-Masked Cell-Cycle Encoder (v0.4.2)."
        )
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
        is_v040 = False
        is_legacy_cyl = False
        if os.path.exists(pt_path):
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", {})
            if "inference_net.split_film_transition.gamma_state.bias" in sd:
                is_legacy = True
            elif "inference_net.encoder_cycle.pathway_linear.weight" not in sd:
                is_v040 = True
            elif "generative_net.cyl_to_go.bias" in sd:
                is_legacy_cyl = True
        cls._loading_legacy_ckpt = is_legacy
        cls._loading_v040_ckpt = is_v040
        cls._loading_legacy_cyl_ckpt = is_legacy_cyl
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
            cls._loading_v040_ckpt = False
            cls._loading_legacy_cyl_ckpt = False

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
        max_lambda_adv_state=None,
        max_lambda_adv_ood=None,
        max_lambda_ood=None,
        floor_lambda_ood=None,
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
        plan_kwargs = plan_kwargs or {}
        datasplitter_kwargs = datasplitter_kwargs or {}

        if max_lambda_adv_state is None:
            max_lambda_adv_state = self.module.max_lambda_adv_state
        if max_lambda_adv_ood is None:
            max_lambda_adv_ood = self.module.max_lambda_adv_ood
        if max_lambda_ood is None:
            max_lambda_ood = self.module.max_lambda_ood
        if floor_lambda_ood is None:
            floor_lambda_ood = self.module.floor_lambda_ood

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
        was_training = self.module.training
        self.module.eval()
        try:
            adata = self._validate_anndata(adata)
            scdl = self._make_data_loader(
                adata=adata, indices=indices, batch_size=batch_size
            )

            latent_spaces = {
                "u_cycle": [],
                "u_state": [],
                "v_ood": [],
                "u_cycle_src": [],
                "u_state_src": [],
                "z_cycle": [],
                "z_state": [],
                "z_shift": [],
                "z_base": [],
                "z_final": [],
            }
            opt_keys = ("u_cycle_go", "z_cycle_go", "u_cycle_go_enc", "emp_go_scores")
            opt_buffers = {k: [] for k in opt_keys}

            for tensors in scdl:
                inference_inputs = self.module._get_inference_input(tensors)
                outputs = self.module.inference(**inference_inputs)

                latent_spaces["u_cycle"].append(outputs["q_m_cycle"].cpu().numpy())
                latent_spaces["u_state"].append(outputs["q_m_state"].cpu().numpy())
                latent_spaces["v_ood"].append(outputs["q_m_ood"].cpu().numpy())
                latent_spaces["u_cycle_src"].append(outputs["u_cycle_src"].cpu().numpy())
                latent_spaces["u_state_src"].append(outputs["u_state_src"].cpu().numpy())

                latent_spaces["z_cycle"].append(outputs["z_cycle"].cpu().numpy())
                latent_spaces["z_state"].append(outputs["z_state"].cpu().numpy())
                latent_spaces["z_shift"].append(outputs["z_shift"].cpu().numpy())
                latent_spaces["z_base"].append(outputs["z_base"].cpu().numpy())
                latent_spaces["z_final"].append(outputs["z_final"].cpu().numpy())

                for k in opt_keys:
                    if k in outputs and outputs[k] is not None:
                        opt_buffers[k].append(outputs[k].cpu().numpy())

            for key in latent_spaces.keys():
                latent_spaces[key] = np.concatenate(latent_spaces[key], axis=0)
            for k in opt_keys:
                if len(opt_buffers[k]) > 0:
                    latent_spaces[k] = np.concatenate(opt_buffers[k], axis=0)

            return latent_spaces
        finally:
            self.module.train(was_training)

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
        self._check_if_trained(warn=False)
        was_training = self.module.training
        self.module.eval()
        try:
            adata = self._validate_anndata(adata)
            scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

            target_batch_idx = None
            if transform_batch is not None:
                batch_mapping = self.adata_manager.get_state_registry("batch").categorical_mapping
                if transform_batch not in batch_mapping:
                    raise ValueError(f"Batch '{transform_batch}' not found.")
                target_batch_idx = np.where(batch_mapping == transform_batch)[0][0]

            target_sample_idx = None
            if transform_sample is not None:
                sample_mapping = self.adata_manager.get_state_registry("sample").categorical_mapping
                if transform_sample not in sample_mapping:
                    raise ValueError(f"Sample '{transform_sample}' not found.")
                target_sample_idx = np.where(sample_mapping == transform_sample)[0][0]

            exprs = []
            for tensors in scdl:
                if target_batch_idx is not None:
                    tensors[scvi.REGISTRY_KEYS.BATCH_KEY] = torch.full_like(
                        tensors[scvi.REGISTRY_KEYS.BATCH_KEY], target_batch_idx
                    )
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
        finally:
            self.module.train(was_training)
