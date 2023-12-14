from __future__ import annotations

import logging
import warnings

import anndata
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager, fields
from scvi.data._constants import _MODEL_NAME_KEY, _SCVI_VERSION_KEY, _SETUP_ARGS_KEY
from scvi.data._utils import get_anndata_attribute
from scvi.model._utils import parse_device_args
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.model.base._archesmixin import _get_loaded_data
from scvi.model.base._utils import _initialize_model
from scvi.utils import setup_anndata_dsp, track

from ._components import ExtendableEmbedding
from ._fields import ExtendableCategoricalJointObsField
from ._module import EmbeddingVAE

logger = logging.getLogger(__name__)
MIN_VAR_NAME_RATIO = 0.8


class EmbeddingSCVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    def __init__(self, adata: AnnData, **kwargs):
        super().__init__(adata)

        categorical_covariates = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        self.module = EmbeddingVAE(
            n_vars=self.summary_stats.n_vars,
            n_labels=self.summary_stats.n_labels,
            categorical_covariates=categorical_covariates,
            **kwargs,
        )
        self.init_params_ = self._get_init_params(locals())

    def get_covariate_representation(self, covariate_key: str) -> np.ndarray:
        adata = self._validate_anndata(None)
        manager = self.get_anndata_manager(adata, required=True)

        covariate_state_registry = manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
        covariate_field_keys = getattr(
            covariate_state_registry,
            ExtendableCategoricalJointObsField.FIELD_KEYS_KEY,
            {},
        )
        index = list(covariate_field_keys).index(covariate_key)

        return self.module.get_covariate_embeddings(index, as_tensor=False)[0]

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        labels_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_cat_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            fields.CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            ExtendableCategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @classmethod
    def prepare_finetune_anndata(
        cls,
        adata: AnnData,
        pretrained_model: str | BaseModelClass,
    ) -> AnnData:
        _, var_names, _ = _get_loaded_data(pretrained_model, device="cpu")
        var_names = pd.Index(var_names)

        intersection = adata.var_names.intersection(var_names)
        intersection_len = len(intersection)
        if intersection_len == 0:
            raise ValueError("No overlapping genes between the dataset and the pretrained model.")

        ratio = intersection_len / len(var_names)
        logger.info(f"Found {ratio * 100}% reference vars in query data.")
        if ratio < MIN_VAR_NAME_RATIO:
            warnings.warn(
                f"Query data contains less than {MIN_VAR_NAME_RATIO:.0%} of reference "
                "var names. This may result in poor performance.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )

        genes_to_add = var_names.difference(adata.var_names)
        needs_padding = len(genes_to_add) > 0
        if needs_padding:
            padding_mtx = csr_matrix(np.zeros((adata.n_obs, len(genes_to_add))))
            adata_padding = anndata.AnnData(
                X=padding_mtx.copy(),
                layers={layer: padding_mtx.copy() for layer in adata.layers},
            )
            adata_padding.var_names = genes_to_add
            adata_padding.obs_names = adata.obs_names
            # Concatenate object
            adata_out = anndata.concat(
                [adata, adata_padding],
                axis=1,
                join="outer",
                index_unique=None,
                merge="unique",
            )
        else:
            adata_out = adata

        # also covers the case when new adata has more var names than old
        if not var_names.equals(adata_out.var_names):
            adata_out._inplace_subset_var(var_names)

        return adata_out

    @classmethod
    def load_finetune_anndata(
        cls,
        adata: anndata.AnnData,
        pretrained_model: str | BaseModelClass,
        accelerator: str = "auto",
        device: int | str = "auto",
    ) -> BaseModelClass:
        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device="torch",
            validate_single_device=True,
        )

        attr_dict, _, load_state_dict = _get_loaded_data(pretrained_model, device=device)

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError("It appears you are loading a model from a different class.")

        if _SETUP_ARGS_KEY not in registry:
            raise ValueError("Saved model does not contain original setup inputs. " "Cannot load the original setup.")

        cls.setup_anndata(
            adata,
            source_registry=registry,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        model = _initialize_model(cls, adata, attr_dict)
        adata_manager = model.get_anndata_manager(adata, required=True)

        version_split = adata_manager.registry[_SCVI_VERSION_KEY].split(".")
        if int(version_split[1]) < 8 and int(version_split[0]) == 0:
            warnings.warn(
                "Query integration should be performed using models trained with " "version >= 0.8",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )

        model.to_device(device)
        model.module.load_state_dict(load_state_dict)
        model.module.eval()
        model.is_trained_ = False

        return model

    def initialize_finetune_embeddings(
        self,
        adata: anndata.AnnData | None = None,
        batch_size: int | None = None,
    ):
        adata = self._validate_anndata(adata)
        manager = self.get_anndata_manager(adata, required=True)

        covariate_state_registry = manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
        extended_categories = getattr(
            covariate_state_registry,
            ExtendableCategoricalJointObsField.EXTENDED_CATEGORIES_DICT_KEY,
            {},
        )
        covariate_keys = getattr(
            covariate_state_registry,
            ExtendableCategoricalJointObsField.FIELD_KEYS_KEY,
            {},
        )
        covariate_mappings = getattr(
            covariate_state_registry,
            ExtendableCategoricalJointObsField.MAPPINGS_KEY,
            {},
        )

        return self._initialize_embeddings(
            adata,
            covariate_keys,
            extended_categories,
            covariate_mappings,
            batch_size=batch_size,
        )

    @torch.no_grad()
    def _initialize_embeddings(
        self,
        adata: anndata.AnnData,
        covariate_keys: list[str],
        extended_categories: dict[str, list[str]],
        covariate_mappings: dict[str, np.ndarray],
        batch_size: int | None = None,
    ):
        for i, key in enumerate(covariate_keys):
            mappings = covariate_mappings[key]
            new_cats = extended_categories[key]

            n_cats = len(mappings)
            n_new_cats = len(new_cats)
            n_old_cats = n_cats - n_new_cats

            if n_new_cats == 0:
                continue

            field = get_anndata_attribute(adata, "obs", key)
            indices = np.where(np.isin(field, new_cats))[0]
            dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

            losses_new_cats = torch.zeros((n_new_cats, n_old_cats)).to(self.module.device)
            losses_n_obs = torch.zeros((n_new_cats, n_old_cats), dtype=torch.int).to(self.module.device)

            for tensors in track(
                dataloader,
                style="tqdm",
                total=len(dataloader),
                description=f"Transferring embeddings for {key}",
            ):
                X = tensors[REGISTRY_KEYS.X_KEY]  # (n_obs, n_vars)
                covariates = tensors[REGISTRY_KEYS.CAT_COVS_KEY][:, [i]]  # (n_obs, 1)
                counterfactuals = torch.arange(n_old_cats).view(-1, 1).repeat(X.shape[0], 1)  # (n_obs * n_old_cats,)

                # (n_obs, n_vars) -> (n_obs * n_old_cats, n_vars)
                X = X.repeat_interleave(n_old_cats, dim=0)
                # (n_obs, 1) -> (n_obs * n_old_cats, 1)
                covariates = covariates.repeat_interleave(n_old_cats, dim=0)

                tensors[REGISTRY_KEYS.X_KEY] = X
                tensors[REGISTRY_KEYS.CAT_COVS_KEY] = counterfactuals

                inference_inputs = self.module._get_inference_input(tensors)
                inference_outputs = self.module.inference(subset_categorical_covs=i, **inference_inputs)
                generative_inputs = self.module._get_generative_input(tensors, inference_outputs)
                generative_outputs = self.module.generative(**generative_inputs)

                # (n_obs * n_old_cats,)
                loss = self.module.loss(tensors, inference_outputs, generative_outputs).extra_tensors["loss_unreduced"]

                loss_grouped = [loss[i::n_old_cats] for i in range(n_old_cats)]
                covariates_grouped = [covariates[i::n_old_cats] for i in range(n_old_cats)]

                for new_cat in new_cats:
                    index = np.where(mappings == new_cat)[0][0]
                    obs_indexes = [torch.where(c == index)[0] for c in covariates_grouped]
                    losses = [l[j] for l, j in zip(loss_grouped, obs_indexes)]

                    for old_cat, (j, l) in enumerate(zip(obs_indexes, losses)):
                        losses_n_obs[index - n_old_cats, old_cat] += j.shape[0]
                        losses_new_cats[index - n_old_cats, old_cat] += torch.sum(l)

            losses_new_cats /= losses_n_obs
            new_cats_transfer = torch.argmin(losses_new_cats, dim=1).tolist()
            embedding_layer = self.module.covariates_encoder.get_embedding_layer(i)
            embedding_layer = ExtendableEmbedding.extend_embedding(
                embedding_layer,
                new_cats_transfer,
            )
            self.module.covariates_encoder.set_embedding_layer(i, embedding_layer)
