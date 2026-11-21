import copy
import operator
import warnings

import anndata
import pandas as pd
import scanpy

from cavachon.filter.anndata_filter import AnnDataFilter
from cavachon.utils.general_utils import GeneralUtils


class FilterQC(AnnDataFilter):
    """FilterQC

    Filter for AnnData. Used as an adaptor between the
    scanpy.pp.calculate_qc_metrics() and the configs. After the quality
    metrics is computed, the filtering will be performed based on the
    proportion of controlled genes. Note that FilterQC will not be
    performed inplace regardless the value of kwargs['inplace']
    (as assignment is needed for the subset).

    If 'qc_vars' is not provided, the control variables default to
    'ERCC' (features matching ^ERCC-, i.e. ERCC spike-ins) and 'MT'
    (features matching ^MT- case-insensitively, i.e. human MT- and mouse
    mt- mitochondrial genes). The names 'ERCC' and 'MT' are kept so that
    the QC metrics are still named e.g. pct_counts_MT. User-provided
    'qc_vars' are used as both the names and the regex patterns matched
    (from the start of the feature names, case-sensitively). The patterns
    are matched against the adata.var column referenced by the var index
    name ('Modality:colname', as created by FileReader); if the index is
    unnamed or the column does not exist, adata.var_names is used instead.

    If 'percent_top' is not provided, scanpy's default (50, 100, 200, 500)
    is used without the values larger than the number of features (None
    if no value remains). An explicitly provided 'percent_top' is passed
    to scanpy.pp.calculate_qc_metrics() unchanged.

    Attributes
    ----------
    name: str
        name for the filtering step.

    kwargs: Mapping[str, Any]
        additional parameters used for scanpy.pp.calculate_qc_metrics().
        In addition, 'filter_threshold' (optional) could be used to specify
        the filtering thresholds; if it is not provided, the QC metrics are
        computed and all cells are kept. The values for 'filter_threshold'
        is a mapping with keys 'field' (column name for qc metrics in
        adata.obs), 'operator' (string representation for python
        operators) and 'threshold' (float number specifying the
        proportion used to filter cells)

    """

    default_qc_vars = {"ERCC": r"^ERCC-", "MT": r"(?i)^MT-"}
    default_percent_top = (50, 100, 200, 500)

    def __init__(self, name, **kwargs):
        """Constructor for FilterQC

        Parameters
        ----------
        name: str
            name for the filtering step.

        kwargs: Mapping[str, Any]
            additional parameters used for
            scanpy.pp.calculate_qc_metrics(). In addition,
            'filter_threshold' (optional) could be used to specify the filtering
            thresholds. The values for 'filter_threshold' is a
            mapping with keys 'field' (column name for qc metrics in
            adata.obs), 'operator' (string representation for python
            operators) and 'threshold' (float number specifying the
            proportion used to filter cells)

        """
        super().__init__(name, **kwargs)

    def __call__(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Perform preprocessing to provided AnnData.

        Parameters
        ----------
        adata: anndata.AnnData
            AnnData to preprocessed.

        Returns
        -------
        anndata.AnnData
            preprocessed AnnData.

        Raises
        ------
        RuntimeError
            if no cells left after the filtering.

        """
        n_obs = adata.obs.shape[0]
        selected = pd.Series(
            GeneralUtils.duplicate_obj_to_list(True, n_obs), index=adata.obs.index
        )

        # the filter_threshold is not recognized in
        # scanpy.pp.calculate_qc_metrics, so we create a copy of self.args
        # and pop filter_threshold field.
        kwargs_copy = copy.deepcopy(self.kwargs)
        filter_criteria_list = kwargs_copy.pop("filter_threshold", [])

        # the index columns will be 'Modality.name:colname', the colname
        # (usually gene name) will be used to check the control variables
        # (usually mitochondria). Fall back to var_names otherwise.
        index_name = adata.var.index.name
        index_colname = None
        if isinstance(index_name, str) and ":" in index_name:
            index_colname = index_name.split(":")[-1]
        if index_colname in adata.var:
            feature_names = adata.var[index_colname]
        else:
            feature_names = pd.Series(adata.var_names, index=adata.var.index)

        # user-provided qc_vars are used both as the column names and as the
        # regex (prefix) patterns. The default qc_vars use anchored patterns
        # (see FilterQC.default_qc_vars) so that e.g. MTOR is not flagged as
        # a mitochondrial gene.
        if "qc_vars" in kwargs_copy:
            qc_var_patterns = {qc_var: qc_var for qc_var in kwargs_copy["qc_vars"]}
        else:
            qc_var_patterns = dict(self.default_qc_vars)
            kwargs_copy["qc_vars"] = list(qc_var_patterns.keys())

        for qc_var, pattern in qc_var_patterns.items():
            adata.var[qc_var] = feature_names.str.match(pattern)

        # scanpy's default percent_top raises IndexError if any value is
        # larger than the number of features.
        if "percent_top" not in kwargs_copy:
            percent_top = tuple(
                n for n in self.default_percent_top if n <= adata.n_vars
            )
            kwargs_copy["percent_top"] = percent_top if percent_top else None

        # make sure calculate_qc_metrics is run inplace.
        kwargs_copy["inplace"] = True
        scanpy.pp.calculate_qc_metrics(adata, **kwargs_copy)

        for filter_criteria in filter_criteria_list:
            field = filter_criteria.get("field", None)
            threshold = filter_criteria.get("threshold", 0)
            op = getattr(operator, filter_criteria.get("operator", "ge"))
            if field not in adata.obs:
                message = "".join(
                    (
                        f"{field} is not in the obs_df of AnnData, ignore the ",
                        f"process in {self.__class__.__name__} ({self.name}).",
                    )
                )
                warnings.warn(message, RuntimeWarning)
                continue
            selected &= op(adata.obs[field], threshold)

        obs_index = adata.obs.loc[selected].index
        if len(obs_index) == 0:
            message = "".join(
                (
                    "No objects left in the adata, please use a less strict "
                    f"filter in {self.__class__.__name__} ({self.name})."
                )
            )
            raise RuntimeError(message)

        adata = adata[obs_index].copy()

        return adata
