import warnings
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from anndata import AnnData
from muon import MuData
from pandas.api.types import is_numeric_dtype
from scipy.sparse import coo_array

from cavachon.environment.constants import Constants
from cavachon.utils.dataframe_utils import DataFrameUtils


class MuDataUtils:
    """MuDataUtils

    Class containing multiple utility functions for AnnData and MuData

    """

    @staticmethod
    def check_if_cavachon_config_exists(data: MuData | AnnData, field: str) -> bool:
        """Check if the MuData or AnnData object has a cavachon config
        in the uns.

        Parameters
        ----------
        data: MuData | AnnData
            MuData or AnnData.

        field: str
            the field of the config to be checked.

        Returns
        -------
        bool
            whether or not the config exists in the provided MuData or
            AnnData.
        """
        if Constants.MDATA_UNS_FIELD_CAVACHON not in data.uns:
            return False
        if field not in data.uns[Constants.MDATA_UNS_FIELD_CAVACHON]:
            return False

        return True

    @staticmethod
    def merge_mdata_on_obs_annotation(
        mdata: MuData, group_col: str, batch_effect_colnames: Dict[str, List[str]]
    ) -> MuData:
        """Aggregates MuData objects by grouping observations based on
        the provided `group_col`. It computes the mean of the modality
        `X` for each group and aggregates metadata columns, handling
        categorical and numerical batch effect columns separately.
        The resulting MuData object contains grouped data, with each
        group represented as a single observation per modality.

        Parameters
        ----------
        mdata : MuData
            The input MuData object containing multiple modalities.

        group_col : str
            The name of the column in `.obs` used for grouping
            observations.

        batch_effect_colnames : Dict[str, List[str]]
            A dictionary mapping each modality name to a list of column
            names in `.obs` that contain batch effects. Categorical
            batch effect columns are aggregated using the mode, while
            numerical columns are aggregated using the mean.

        Returns
        -------
        MuData
            A new MuData object where each modality contains aggregated
            observations, with the mean expression values and
            summarized metadata for each group.
        """
        group_X_list = defaultdict(list)
        group_obs_list = defaultdict(list)
        first_mod = list(mdata.mod.keys())[0]
        for group in mdata[first_mod].obs[group_col].unique():
            for mod in mdata.mod.keys():
                adata = mdata[mod]
                adata = adata[adata.obs[group_col] == group].copy()
                obs = pd.DataFrame({group_col: [group]})
                for batch_effect_col in batch_effect_colnames[mod]:
                    if DataFrameUtils.check_is_categorical(adata.obs[batch_effect_col]):
                        obs[batch_effect_col] = adata.obs[batch_effect_col].mode()
                    elif is_numeric_dtype(adata.obs[batch_effect_col]):
                        obs[batch_effect_col] = adata.obs[batch_effect_col].mean()

                obs.index = pd.Index([group])
                group_X_list[mod].append(adata.X.mean(axis=0))
                group_obs_list[mod].append(obs)

        group_adata = dict()
        group_X: Dict[str, coo_array] = dict()
        group_obs: Dict[str, pd.DataFrame] = dict()
        for key in group_X_list.keys():
            group_X[key] = coo_array(np.stack(group_X_list[key])).tocsr()
            group_obs[key] = pd.concat(group_obs_list[key], axis=0)
            adata = AnnData(X=group_X[key])
            adata.obs = group_obs[key]
            adata.var = mdata[key].var
            group_adata[key] = adata

        return MuData(group_adata)

    @staticmethod
    def reorder_or_filter_adata_obs(adata: AnnData, obs_index: pd.Index) -> AnnData:
        """Filter and reorder the AnnData using the obs_index so the
        order and index of obs DataFrame in the AnnData is the same as
        the provided obs_index.

        Parameters
        ----------
        adata: anndata.AnnData
            anndata.AnnData to be reordered (or filtered).

        obs_index: pd.Index
            the desired order of index for the obs DataFrame for
            reordering or the kept index for the obs DataFrame for
            filtering.

        Returns
        -------
        anndata.AnnData
            filtered and reordered AnnData.
        """
        if not isinstance(adata, AnnData):
            message = "Provided adata is not an AnnData object, do nothing."
            warnings.warn(message, RuntimeWarning)
            return

        obs_df = adata.obs
        var_df = adata.var
        matrix = adata.X
        n_obs = obs_df.shape[0]
        indices = pd.DataFrame({"IntegerIndex": range(0, n_obs)}, index=obs_df.index)

        selected_indices = indices.loc[obs_index, "IntegerIndex"].values

        selected_adata = AnnData(X=matrix[selected_indices], dtype=np.float32)
        selected_adata.obs = obs_df.iloc[selected_indices]
        selected_adata.var = var_df

        return selected_adata
