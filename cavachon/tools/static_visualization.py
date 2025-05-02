from typing import Optional, Sequence, Union

import anndata
import matplotlib


class StaticVisualization:
    @staticmethod
    def embedding(
        adata: anndata.AnnData,
        method: str = "tsne",
        filename: Optional[str] = None,
        use_rep: Union[str, np.array] = "z",
        color: Union[str, Sequence[str], None] = None,
        title: Optional[str] = None,
        color_discrete_sequence: Optional[Sequence[str]] = None,
        force: bool = False,
        *args,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        pass
