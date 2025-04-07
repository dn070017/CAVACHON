from __future__ import annotations

import os
from typing import Dict, List, Tuple

import anndata
import muon as mu
import numpy as np
import scipy
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from cavachon.dataset.batch_effect_process_configs import BatchEffectProcessConfig
from cavachon.environment.constants import Constants
from cavachon.utils.mudata_utils import MuDataUtils
from cavachon.utils.reflection_handler import ReflectionHandler
from cavachon.utils.tensor_utils import TensorUtils


class DatasetCreator:
    """DatasetCreator

    Creator which aims to create tf.data.Dataset from MuData.

    Attributes
    ----------
    mdata: mu.MuData
        (single-cell) multi-omics data used to create the dataset.

    modality_names: List[str]
        the modality names used to create the dataset.

    distribution_names: Dict[str, str]
        the distribution names of modality (used to perform data
        modification). The keys should be the modality names, the
        values are the distribution names.

    batch_effect_process_configs: Dict[str, List[BatchEffectProcessConfig] | None]
        the batch effect process config for each batch effect column of
        each modality. The keys are the `modality`, the values are
        a list of batch effect process configs. The values are None if
        the modality does not have batch effect.

    batch_effect_total_n_vars: Dict[str, int]
        the number of variables for the batch effect tensor. The keys
        are the `modality`, the values are the numbers of variables
        (dimensionality) of the corresponding batch effect tensor for
        the modality.
    """

    def __init__(
        self,
        mdata: mu.MuData,
        modality_names: List[str] | None = None,
        distribution_names: Dict[str, str] | None = None,
        batch_effect_colnames: Dict[str, List[str]] | None = None,
        batch_effect_encoders: Dict[str, List[LabelEncoder | None]] | None = None,
    ) -> None:
        """Constructor for DataLoader

        Parameters
        ----------
        mdata: mu.MuData
            (single-cell) multi-omics data used to create the dataset.

        modality_names: List[str] | None, optional
            the modality names used to create the dataset. If not
            provided, `adata.uns['cavachon']['modality_names']` will be
            used. If None of these are provided, every modality in
            mdata.mod.keys() will be used.

        distribution_names: Dict[str, str] | None, optional
            the distribution names of modality (used to perform data
            modification). The keys should be the modality names, the
            values are the distribution names. If not provided,
            adata.uns['cavachon']['distribution'] of each modality
            will be used.

        batch_effect_colnames: Dict[str, List[str]] | None, optional
            the keys should be the modality_names, the values are the
            column names of batch effect to correct. If not provided,
            `adata.uns['cavachon']['batch_effect_colnames']`
            of each modality will be used. The order of the
            `batch_effect_colnames` must match the corresponding
            `batch_effect_encoders`. For instance:
            1. `batch_effect_colnames`: `['A', 'B', 'C']`
            2. `batch_effect_encoders`: `[LabelEncoder(...), None, LabelEncoder(...)]`

        batch_effect_encoders: Dict[str, List[LabelEncoder | None]] | None, optional
            the encoders used to create one-hot encoded batch effect
            tensor. The keys should be the modality_names, the
            values are the LabelEncoder stored the mapping between
            categorical batch effect variables and their numerical
            representation. If not provided,
            `adata.uns['cavachon']['batch_effect_encoders_classes']`
            of each modality will be used to reconstruct the
            LabelEncoder. If the batch effect is a numerical variable,
            the LabelEncoder must be set to None. The order of the
            `batch_effect_colnames` must match the corresponding
            `batch_effect_encoders`. For instance:
            1. `batch_effect_colnames`: `['A', 'B', 'C']`
            2. `batch_effect_encoders`: `[LabelEncoder(...), None, LabelEncoder(...)]`

        """
        self.mdata = mdata
        self.modality_names = self.configure_modality_names(self.mdata, modality_names)
        self.distribution_names = self.configure_distribution_names(
            self.mdata, self.modality_names, distribution_names
        )
        self.batch_effect_process_configs = self.configure_batch_effect_process_configs(
            self.mdata,
            self.modality_names,
            batch_effect_colnames,
            batch_effect_encoders,
        )
        # self.batch_effect_total_n_vars = self.configure_batch_effect_total_n_vars(
        #    self.mdata, self.modality_names, self.batch_effect_process_configs
        # )

        return

    def configure_modality_names(
        self,
        mdata: mu.MuData,
        modality_names: List[str] | None = None,
    ) -> None:
        """Configure modality names to be used, and stored the result
        to self.modality_names.

        Parameters
        ----------
        mdata: mu.MuData
            (single-cell) multi-omics data used to create the dataset.

        modality_names: List[str] | None, optional
            the modality names used to create the dataset. If not
            provided, `mdata.uns['cavachon']['modality_names']` will be
            used. If None of these are provided, every modality in
            mdata.mod.keys() will be used.

        Returns
        -------
        List[str]
            the configured modality names used to create the dataset.

        Raises
        ------
        KeyError:
            when the `modality_name` is not found in the mdata.

        """
        resulting_modality_names = []
        if modality_names is not None:
            resulting_modality_names = modality_names
        elif MuDataUtils.check_if_cavachon_config_exists(
            mdata, Constants.MDATA_UNS_FIELD_MODALITY
        ):
            resulting_modality_names = mdata.uns[Constants.MDATA_UNS_FIELD_CAVACHON][
                Constants.MDATA_UNS_FIELD_MODALITY
            ]
        else:
            resulting_modality_names = list(self.mdata.mod.keys())

        for modality_name in resulting_modality_names:
            if modality_name not in mdata.mod.keys():
                raise KeyError(f"Cannot find modality {modality_name} in the mdata.")

        return resulting_modality_names

    def configure_distribution_names(
        self,
        mdata: mu.MuData,
        modality_names: List[str],
        distribution_names: Dict[str, str] | None = None,
    ) -> None:
        """Configure distribution names for each modality, and stored
        the result to self.distribution_names.

        Parameters
        ----------
        mdata: mu.MuData
            (single-cell) multi-omics data used to create the dataset.

        modality_names: List[str]
            the configured modality names used to create the dataset.

        distribution_names: Dict[str, str], optional
            the distribution names for each modality. The keys should
            be the modality names, the values are the distribution of
            the modality. If not provided, extract distribution names
            of each modality from `adata.uns['cavachon']['distribution']`

        Returns
        -------
        Dict[str, str]
            the configured distribution names for each modality.
        """
        resulting_distribution_names = dict()
        for modality_name in modality_names:
            adata = mdata[modality_name]
            if modality_name in distribution_names:
                resulting_distribution_names[modality_name] = distribution_names[
                    modality_name
                ]
            elif MuDataUtils.check_if_cavachon_config_exists(
                adata, Constants.MDATA_UNS_FIELD_DISTRIBUTION
            ):
                resulting_distribution_names[modality_name] = (
                    adata.uns[Constants.MDATA_UNS_FIELD_CAVACHON][
                        Constants.MDATA_UNS_FIELD_DISTRIBUTION
                    ],
                )
            else:
                raise KeyError(
                    f"Cannot find distribution name for modality {modality_name}"
                )

        for distribution_name in resulting_distribution_names.keys():
            if distribution_name not in mdata.mod.keys():
                raise KeyError(
                    f"Cannot find distribution name {distribution_name}"
                    f" for modality {distribution_name} in the mdata."
                )

        return resulting_distribution_names

    def configure_batch_effect_process_configs(
        self,
        mdata: mu.MuData,
        modality_names: List[str],
        batch_effect_colnames: Dict[str, List[str]] | None = None,
        batch_effect_encoders: Dict[str, List[LabelEncoder | None]] | None = None,
    ) -> None:
        """Configure batch effect column process configs.

        mdata: mu.MuData
            (single-cell) multi-omics data used to create the dataset.

        modality_names: List[str]
            the configured modality names used to create the dataset.

        batch_effect_colnames: Dict[str, List[str]] | None, optional
            the keys should be the modality_names, the values are the
            column names of batch effect to correct. If not provided,
            `adata.uns['cavachon']['batch_effect_colnames']`
            of each modality will be used. The order of the
            `batch_effect_colnames` must match the corresponding
            `batch_effect_encoders`. For instance:
            1. `batch_effect_colnames`: `['A', 'B', 'C']`
            2. `batch_effect_encoders`: `[LabelEncoder(...), None, LabelEncoder(...)]`

        batch_effect_encoders: Dict[str, List[LabelEncoder | None]] | None, optional
            the encoders used to create one-hot encoded batch effect
            tensor. The keys should be the modality_names, the
            values are the LabelEncoder stored the mapping between
            categorical batch effect variables and their numerical
            representation. If not provided,
            `adata.uns['cavachon']['batch_effect_encoders_classes']`
            of each modality will be used to reconstruct the
            LabelEncoder. If the batch effect is a numerical variable,
            the LabelEncoder must be set to None. The order of the
            `batch_effect_colnames` must match the corresponding
            `batch_effect_encoders`. For instance:
            1. `batch_effect_colnames`: `['A', 'B', 'C']`
            2. `batch_effect_encoders`: `[LabelEncoder(...), None, LabelEncoder(...)]`

        Returns
        -------
        Dict[str, List[BatchEffectProcessConfig] | None]
            the batch effect process config for each batch effect
            column of each modality. The keys are the `modality`,
            the values are a list of batch effect process configs. The
            values are None if the modality does not have batch effect.
        """
        batch_effect_colnames = (
            batch_effect_colnames if batch_effect_colnames is not None else dict()
        )
        batch_effect_encoders = (
            batch_effect_encoders if batch_effect_encoders is not None else dict()
        )

        resulting_batch_effect_process_configs = dict()
        for modality_name in modality_names:
            batch_effect_colnames_of_modality, batch_effect_encoders_of_modality = (
                self.configure_batch_effect_colnames_and_encoders(
                    adata=mdata[modality_name],
                    modality_name=modality_name,
                    batch_effect_colnames_of_modality=batch_effect_colnames.get(
                        modality_name, None
                    ),
                    batch_effect_encoders_of_modality=batch_effect_encoders.get(
                        modality_name, None
                    ),
                )
            )
            resulting_batch_effect_process_configs[modality_name] = (
                self.configure_batch_effect_process_config_for_modality(
                    adata=mdata[modality_name],
                    modality_name=modality_name,
                    batch_effect_colnames_of_modality=batch_effect_colnames_of_modality,
                    batch_effect_encoders_of_modality=batch_effect_encoders_of_modality,
                )
            )

        return resulting_batch_effect_process_configs

    def configure_batch_effect_colnames_and_encoders(
        self,
        adata: anndata.AnnData,
        modality_name: str,
        batch_effect_colnames_of_modality: List[str] | None = None,
        batch_effect_encoders_of_modality: List[LabelEncoder | None] | None = None,
    ) -> Tuple[List[str] | None, List[LabelEncoder | None] | None]:
        """Extract batch effect colnames and encoders from the
        mdata.uns for the specified if it is not specified in the
        function parameter

        Parameters
        ----------
        adata: anndata.AnnData
            the single modality data needed to be processed for batch
            effect.

        modality_name : str
            modality to process.

        batch_effect_colnames_of_modality : List[str] | None, optional
            the column names of batch effect to correct. If not
            provided, `adata.uns['cavachon']['batch_effect_colnames']`
            of each modality will be used. The order of the
            `batch_effect_colnames` must match the corresponding
            `batch_effect_encoders`. For instance:
            1. `batch_effect_colnames`: `['A', 'B', 'C']`
            2. `batch_effect_encoders`: `[LabelEncoder(...), None, LabelEncoder(...)]`

        batch_effect_encoders_of_modality : List[LabelEncoder  |  None] | None, optional
            thee LabelEncoder stored the mapping between categorical
            batch effect variables and their numerical representation.
            If not provided,
            `adata.uns['cavachon']['batch_effect_encoders_classes']`
            of each modality will be used to reconstruct the
            LabelEncoder. If the batch effect is a numerical variable,
            the LabelEncoder must be set to None. The order of the
            `batch_effect_colnames` must match the corresponding
            `batch_effect_encoders`. For instance:
            1. `batch_effect_colnames`: `['A', 'B', 'C']`
            2. `batch_effect_encoders`: `[LabelEncoder(...), None, LabelEncoder(...)]`

        Returns
        -------
        Tuple[List[str] | None, List[LabelEncoder | None] | None]
            the first element is the batch effect colnames and the
            second element is the batch effect encoders. None if the
            element corresponding element is not configured.
        """

        if batch_effect_colnames_of_modality is None:
            if MuDataUtils.check_if_cavachon_config_exists(
                adata, Constants.MDATA_UNS_FIELD_BATCH_EFFECT_COLNAMES
            ):
                # if batch_effect_colnames is not configured by function parameters,
                # but configured in adata.uns['cavachon']['batch_effect_colnames'].
                result_batch_effect_colnames_of_modality = adata.uns[
                    Constants.MDATA_UNS_FIELD_CAVACHON
                ][Constants.MDATA_UNS_FIELD_BATCH_EFFECT_COLNAMES]
            else:
                result_batch_effect_colnames_of_modality = None
        else:
            result_batch_effect_colnames_of_modality = batch_effect_colnames_of_modality

        if batch_effect_encoders_of_modality is None:
            if MuDataUtils.check_if_cavachon_config_exists(
                adata, Constants.MDATA_UNS_FIELD_BATCH_EFFECT_ENCODER_CLASSES
            ):
                # if batch_effect_encoders is not configured by function parameters,
                # but configured in adata.uns['cavachon']['batch_effect_encoders'].
                result_batch_effect_encoders_of_modality = []
                batch_effect_encoders_of_modality_classes = adata.uns[
                    Constants.MDATA_UNS_FIELD_CAVACHON
                ][Constants.MDATA_UNS_FIELD_BATCH_EFFECT_ENCODER_CLASSES]
                for encoder_class in batch_effect_encoders_of_modality_classes:
                    if encoder_class is None:
                        result_batch_effect_encoders_of_modality.append(None)
                    else:
                        encoder = LabelEncoder()
                        encoder.classes_ = encoder_class
                        result_batch_effect_encoders_of_modality.append(encoder)
            else:
                result_batch_effect_encoders_of_modality = None
        else:
            result_batch_effect_encoders_of_modality = batch_effect_encoders_of_modality

        if (
            result_batch_effect_colnames_of_modality is not None
            and result_batch_effect_encoders_of_modality is not None
        ):
            if len(result_batch_effect_colnames_of_modality) != len(
                result_batch_effect_encoders_of_modality
            ):
                raise ValueError(
                    "batch_effect_colnames and batch_effect_encoders should have "
                    f"the same length for modality {modality_name}."
                )

        return (
            result_batch_effect_colnames_of_modality,
            result_batch_effect_encoders_of_modality,
        )

    def configure_batch_effect_process_config_for_modality(
        self,
        adata: anndata.AnnData,
        modality_name: str,
        batch_effect_colnames_of_modality: List[str] | None = None,
        batch_effect_encoders_of_modality: List[LabelEncoder] | None = None,
    ):
        """Configure batch effect process config for the specified
        modality.

        Parameters
        ----------
        adata: anndata.AnnData
            the single modality data needed to be processed for batch
            effect.

        modality_name: str
            modality name (keys to select adata, also used as keys to
            update self.update n_vars_batch_effect and
            self.batch_effect_encoder)

        batch_effect_colnames_of_modality: List[str] | None, optional
            the column names of batch effect to correct. The keys
            should be the modality names, the values are lists of batch
            effect column names to correct for the corresponding
            modality. If not provided,
            adata.uns['cavachon']['batch_effect_colnames'] will be used.

        batch_effect_encoders_of_modality: LabelEncoder | None, optional
            the encoders used to create one-hot encoded batch effect
            tensor. The keys of the dictionary are formatted as
            "`modality`_`batch_effect_colname`", The LabelEncoder
            stored the mapping between categorical batch effect
            variables and the numerical representation. If not
            provided, adata.uns['cavachon']['batch_effect_encoders_classes']
            will be used.

        Returns
        -------
        Dict[str, List[BatchEffectProcessConfig] | None]
            the batch effect process config for each batch effect column
            of each modality. The keys are the `modality`, the values
            are a list of batch effect process configs. The values are
            a list of batch effect process configs. The values are None
            if the modality does not have batch effect.

        Raises
        ------
        ValueError:
            when the `batch_effect_encoders_of_modality` is configured,
            but the `batch_effect_colnames_of_modality` is not
            configured.
        """
        if (
            batch_effect_colnames_of_modality is None
            and batch_effect_encoders_of_modality is None
        ):
            # assume there's no batch effect
            return []
        elif (
            batch_effect_colnames_of_modality is not None
            and batch_effect_encoders_of_modality is None
        ):
            # fit batch effect encoder to adata.obs[batch_effect_colnames]
            _, encoder_mapping = TensorUtils.create_tensor_from_df(
                adata.obs,
                batch_effect_colnames_of_modality,
            )
            adata.uns.setdefault(Constants.MDATA_UNS_FIELD_CAVACHON, dict())
            adata.uns[Constants.MDATA_UNS_FIELD_CAVACHON][
                Constants.MDATA_UNS_FIELD_BATCH_EFFECT_ENCODER_CLASSES
            ] = []
            resulting_batch_effect_process_configs = []
            for colname, encoder in encoder_mapping.items():
                is_categorical = encoder is not None
                n_vars = len(encoder.classes_) if is_categorical else 1
                # self.batch_effect_total_n_vars[modality_name] += n_vars
                batch_effect_process_config = BatchEffectProcessConfig(
                    colname=colname,
                    categorical=is_categorical,
                    encoder=encoder,
                    n_vars=n_vars,
                )
                resulting_batch_effect_process_configs.append(
                    batch_effect_process_config
                )
            return resulting_batch_effect_process_configs
        elif (
            batch_effect_colnames_of_modality is None
            and batch_effect_encoders_of_modality is not None
        ):
            # not configured properly
            message = "batch_effect_colnames_of_modality is not configured, "
            "but batch_effect_encoders_of_modality is configured for modality "
            f"{modality_name}. Please configure batch_effect_colnames_of_modality "
            "or remove batch_effect_encoders_of_modality configuration"
            raise ValueError(message)
        elif (
            batch_effect_colnames_of_modality is not None
            and batch_effect_encoders_of_modality is not None
        ):
            # self.batch_effect_total_n_vars[modality_name] = 0
            resulting_batch_effect_process_configs = []
            for colname, encoder in zip(
                batch_effect_colnames_of_modality, batch_effect_encoders_of_modality
            ):
                is_categorical = encoder is not None
                n_vars = len(encoder.classes_) if is_categorical else 1
                # self.batch_effect_total_n_vars[modality_name] += n_vars
                batch_effect_process_config = BatchEffectProcessConfig(
                    colname=colname,
                    categorical=is_categorical,
                    encoder=encoder,
                    n_vars=n_vars,
                )
                resulting_batch_effect_process_configs.append(
                    batch_effect_process_config
                )
            return resulting_batch_effect_process_configs

    def create(
        self,
    ) -> tf.data.Dataset:
        """Create a Tensorflow Dataset modified with the default
        modifiers based on the MuData and batch effect configuration
        provided in the `__init__` function.

        Returns
        -------
        tf.data.Dataset:
            created Dataset. The field of the dataset includes:
            1. "`modality`_matrix_model": (tf.SparseTensor)
            2. "`modality`_matrix_observed": (tf.SparseTensor)
            3. "`modality`_batch_effect": (tf.Tensor)

        """
        dataset = self.create_base_dataset()
        dataset = self.modify_dataset(
            dataset=dataset, distribution_names=self.distribution_names
        )

        return dataset

    def create_base_dataset(self) -> tf.data.Dataset:
        """Create a Tensorflow Dataset based on the MuData and batch
        effect configuration provided in the `__init__` function.

        Returns
        -------
        tf.data.Dataset:
            created Dataset. The field of the dataset includes:
            1. "`modality`_matrix_model": (tf.SparseTensor)
            2. "`modality`_matrix_observed": (tf.SparseTensor)
            3. "`modality`_batch_effect": (tf.Tensor)

        """
        resulting_tensors = dict()
        for modality_name in self.modality_names:
            adata = self.mdata[modality_name]
            if isinstance(adata.X, scipy.sparse.spmatrix):
                data_tensor = TensorUtils.spmatrix_to_sparse_tensor(adata.X)
            elif isinstance(adata.X, np.ndarray):
                data_tensor = tf.convert_to_tensor(adata.X)
            else:
                raise NotImplementedError(
                    f"adata.X must be either scipy.sparse.spmatrix or np.ndarray, get {type(adata.X)}"
                )
            resulting_tensors.setdefault(
                f"{modality_name}_{Constants.TENSOR_NAME_X_MODEL}", data_tensor
            )
            resulting_tensors.setdefault(
                f"{modality_name}_{Constants.TENSOR_NAME_X_OBSERVED}", data_tensor
            )
            batch_effect_tensor, _ = TensorUtils.create_tensor_from_df(
                df=adata.obs,
                colnames=[
                    x.colname for x in self.batch_effect_process_configs[modality_name]
                ],
                col_encoders={
                    x.colname: x.encoder
                    for x in self.batch_effect_process_configs[modality_name]
                },
            )

            resulting_tensors.setdefault(
                f"{modality_name}_{Constants.TENSOR_NAME_BATCH}",
                batch_effect_tensor,
            )

        return tf.data.Dataset.from_tensor_slices(resulting_tensors)

    def modify_dataset(
        self, dataset: tf.data.Dataset, distribution_names: Dict[str, str]
    ) -> None:
        """Modify the dataset based on the modifiers of distributions
        inplace. This will modify the observed data (used as the
        ground truth for computing the loss during training) and
        overwrite the result in  "`modality`_matrix_observed".

        Parameters
        ----------
        dataset: tf.data.Dataset
            the dataset to be modified.

        distribution_names: Dict[str, str]
            the distribution names of modality (used to perform data
            modification).

        Returns
        -------
        tf.data.Dataset:
            created Dataset. The field of the dataset includes:
            1. "`modality`_matrix_model": (tf.SparseTensor)
            2. "`modality`_matrix_observed": (tf.SparseTensor)
            3. "`modality`_batch_effect": (tf.Tensor)

        """
        # TODO: write unit test
        modality_names = self.mdata.mod.keys()
        for modality_name in modality_names:
            uns = self.mdata[modality_name].uns
            if issubclass(type(uns), Dict):
                config = uns.get("cavachon", {})
                distribution_name = distribution_names.get(modality_name, "")
                if not distribution_name:
                    distribution_name = config.get("distribution", "")
                if not distribution_name:
                    continue
                modifier_class = ReflectionHandler.get_class_by_name(
                    distribution_name,
                    "layers/modifiers/distribution_observed_data_presets",
                    "ObservedDataModifier",
                )
                modifier = modifier_class(modality_name=modality_name)
                modified_dataset = dataset.map(modifier)

        return modified_dataset

    @classmethod
    def from_h5mu(cls, h5mu_path: str) -> DatasetCreator:
        """Create DataLoader from h5mu file of MuData. Note that (1)
        the different modalities in the MuData needs to be sorted in a
        way that the order of obs DataFrame needs to be the same. (2)
        the `distribution_names` for each modality need to be stored in
        adata.uns['cavachon']['distribution']. (3) the batch
        effect column names need to be stored in
        adata.uns['cavachon']['batch_effect_columns'] if the user
        wish to consider batch effect while using the model.

        Parameters
        ----------
        h5mu_path: str
            path to the h5mu file.

        Returns
        -------
        DataLoader:
            DataLoader created from h5mu file.

        """
        path = os.path.realpath(h5mu_path)
        mdata = mu.read(path)
        mdata.update()
        return cls(mdata=mdata)
