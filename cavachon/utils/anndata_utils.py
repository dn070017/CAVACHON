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

        # Use the first modality that actually contains the grouping column
        # as the reference. Other modalities are filtered by the same obs
        # indices so cluster labels do not need to be duplicated everywhere.
        ref_mod = None
        for mod in mdata.mod.keys():
            if group_col in mdata[mod].obs.columns:
                ref_mod = mod
                break
        if ref_mod is None:
            available = {mod: list(mdata[mod].obs.columns) for mod in mdata.mod.keys()}
            raise KeyError(
                f"group_col '{group_col}' not found in any modality obs. "
                f"Available columns per modality: {available}"
            )

        ref_adata = mdata[ref_mod]
        for group in ref_adata.obs[group_col].unique():
            ref_indices = ref_adata.obs[ref_adata.obs[group_col] == group].index
            for mod in mdata.mod.keys():
                adata = mdata[mod]
                if group_col in adata.obs.columns:
                    group_adata = adata[adata.obs[group_col] == group].copy()
                else:
                    group_indices = ref_indices.intersection(adata.obs.index)
                    group_adata = adata[group_indices].copy()
                obs = pd.DataFrame({group_col: [group]})
                for batch_effect_col in batch_effect_colnames[mod]:
                    if batch_effect_col not in group_adata.obs.columns:
                        continue
                    if DataFrameUtils.check_is_categorical(
                        group_adata.obs[batch_effect_col]
                    ):
                        obs[batch_effect_col] = group_adata.obs[batch_effect_col].mode()
                    elif is_numeric_dtype(group_adata.obs[batch_effect_col]):
                        obs[batch_effect_col] = group_adata.obs[batch_effect_col].mean()

                obs.index = [group]
                group_X[mod].append(group_adata.X.mean(axis=0))
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
