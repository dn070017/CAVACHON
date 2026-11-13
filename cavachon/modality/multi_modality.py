from collections.abc import Mapping
from typing import Mapping as MappingType
from typing import Union

import anndata
import muon

from cavachon.modality import Modality


class MultiModality(muon.MuData):
    """MultiModality

    Data structure for (single-cell) multi-omics data. Inherit from
    muon.MuData and is compatible with muon and other APIs that expect
    muon.MuData as inputs. The adata in each modality will be sorted so
    the order of the cells will be the same after initialization.

    """

    def __init__(
        self,
        data: Union[
            anndata.AnnData, Modality, MappingType[str, anndata.AnnData], muon.MuData
        ],
        *args,
        **kwargs,
    ):
        """Constructor for MultiModality

        Parameters
        ----------
        data: Union[anndata.AnnData, Modality, MappingType[str, anndata.AnnData], muon.MuData])
            same as the data parameter for muon.MuData.

        args:
            addtional parameters for initializing anndata.AnnData.

        kwargs: Mapping[str, Any]
            addtional parameters for initializing anndata.AnnData.

        """
        if isinstance(data, Modality):
            modality_name = data.uns.get("cavachon", {}).get("name", "modality")
            data = dict(((modality_name, data),))

        if isinstance(data, muon.MuData):
            # the cells are intersected below on copies, so the MuData
            # provided by the caller is not modified (muon.pp.intersect_obs
            # works inplace).
            data = data.mod

        if issubclass(type(data), Mapping):
            index = None
            for key in data.keys():
                adata = data.get(key)
                index = (
                    adata.obs.index
                    if index is None
                    else index.intersection(adata.obs.index)
                )
            # create a new mapping so the caller's mapping is not modified
            data = dict((key, data.get(key)[index].copy()) for key in data.keys())

        super().__init__(data, *args, **kwargs)
