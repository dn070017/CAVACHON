import warnings
from collections import defaultdict

import muon as mu
import numpy as np
import pandas as pd
from anndata import AnnData
from pandas.api.types import is_numeric_dtype
from scipy.sparse import coo_array

from cavachon.utils.dataframe_utils import DataFrameUtils


class AnnDataUtils:
    """AnnDataUtils

    Class containing multiple utility functions for anndata.AnnData.

    """

    @staticmethod
    def merge_mdata_on_obs_annotation(mdata, group_col, batch_effect_colnames):
        group_X = defaultdict(list)
        group_obs = defaultdict(list)
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

                obs.index = [group]
                group_X[mod].append(adata.X.mean(axis=0))
                group_obs[mod].append(obs)

        group_adata = dict()
        for key in group_X.keys():
            group_X[key] = coo_array(np.stack(group_X[key])).tocsr()
            group_obs[key] = pd.concat(group_obs[key], axis=0)
            adata = AnnData(X=group_X[key])
            adata.obs = group_obs[key]
            adata.var = mdata[key].var
            group_adata[key] = adata

        return mu.MuData(group_adata)

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
