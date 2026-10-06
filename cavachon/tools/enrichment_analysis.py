from pathlib import Path
import shutil
import tempfile
from typing import Dict, Iterator, List, Mapping, Optional, Union, cast

import gseapy
import pandas as pd
from gseapy.gsea import Prerank
from plotly.graph_objs._figure import Figure

from cavachon.tools.interactive_visualization import InteractiveVisualization


PathLike = Union[str, Path]


class EnrichmentAnalysis:
    """Enrichment analysis and visualization.

    Enrichment analysis is performed with ``gseapy.prerank``. The ranked
    statistics can be obtained from :class:`DifferentialAnalysis`, for
    example by passing ``degs["K(A>B|Z)"]`` to :meth:`run`.

    """

    def __init__(
        self,
        gene_sets: Union[str, Mapping[str, List[str]]] = "KEGG_2019_Mouse",
        organism: Optional[str] = "Mouse",
    ) -> None:
        """Constructor for EnrichmentAnalysis.

        Parameters
        ----------
        gene_sets: Union[str, Mapping], optional
            Enrichr library name or gene-set mapping. Defaults to the
            ``"KEGG_2019_Mouse"`` library used in the README example.

        organism: str, optional
            Organism used when ``gene_sets`` is an Enrichr library name.
            Defaults to ``"Mouse"``.
        """
        self.gene_sets = gene_sets
        self.organism = organism

    @staticmethod
    def get_library(
        name: str, organism: Optional[str] = None
    ) -> Mapping[str, List[str]]:
        """Retrieve a gene-set library from Enrichr.

        Parameters
        ----------
        name: str
            Name of the Enrichr gene-set library, such as
            ``"KEGG_2019_Mouse"``.

        organism: str, optional
            Organism used by Enrichr when retrieving the library. Defaults
            to None.

        Returns
        -------
        Mapping
            Mapping from gene-set names to gene collections.
        """
        if organism is None:
            return gseapy.get_library(name=name)
        return gseapy.get_library(name=name, organism=organism)

    def _run_prerank(
        self,
        prerank: Union[pd.Series, pd.DataFrame],
        gene_sets: Union[str, Mapping[str, List[str]]],
        outdir: Union[str, Path],
        **kwargs,
    ) -> Prerank:
        """Run preranked gene-set enrichment analysis.

        Parameters
        ----------
        prerank: Union[pd.Series, pd.DataFrame]
            Ranked gene statistics. A Series should use gene names as its
            index and statistics as its values.

        gene_sets: Union[str, Mapping]
            Gene-set library name, file path, or mapping accepted by
            ``gseapy.prerank``.

        outdir: Union[str, Path]
            Directory in which gseapy writes its analysis output.

        **kwargs
            Additional keyword arguments passed to ``gseapy.prerank``.

        Returns
        -------
        Prerank
            The completed gseapy prerank result.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        compatible_gene_sets = cast(
            Union[str, List[str], Dict[str, str]], gene_sets
        )
        return gseapy.prerank(
            rnk=prerank,
            gene_sets=compatible_gene_sets,
            outdir=str(outdir),
            **kwargs,
        )

    def _visualize(
        self,
        prerank_result: Prerank,
        outdir: Union[str, Path],
        terms: Optional[List[str]] = None,
        metric: str = "FDR q-val",
        threshold: float = 0.05,
    ) -> Dict[str, Figure]:
        """Create and save interactive enrichment visualizations.

        The method creates one ring plot for all significant terms and one
        enrichment-score plot for each requested term. HTML is used so the
        visualizations remain interactive and do not require a browser
        backend during analysis.

        Parameters
        ----------
        prerank_result: Prerank
            Output of :meth:`run` or ``gseapy.prerank``.

        outdir: Union[str, Path]
            Directory in which visualization files are written.

        terms: List[str], optional
            Terms for which enrichment-score plots are created. If None,
            all terms passing ``threshold`` are used.

        metric: str, optional
            Result metric used to select terms for the ring plot and the
            default term list. Defaults to ``"FDR q-val"``.

        threshold: float, optional
            Maximum metric value for selected terms. Defaults to 0.05.

        Returns
        -------
        Dict[str, Figure]
            Figures keyed by ``"ringplot"`` and ``"enrichment_score:<term>"``.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        figures = {
            "ringplot": InteractiveVisualization.prerank_ringplot(
                prerank_result,
                metric=metric,
                threshold=threshold,
                filename=str(outdir / "enrichment_ringplot.html"),
            )
        }

        result_table = prerank_result.res2d
        if result_table is None:
            raise ValueError("The enrichment result does not contain a result table")
        raw_results = prerank_result.results
        if raw_results is None:
            raise ValueError("The enrichment result does not contain term results")
        results = cast(Dict[str, object], raw_results)

        if terms is None:
            selected_results = result_table.loc[result_table[metric] <= threshold]
            if "Term" in selected_results.columns:
                selected_terms = selected_results["Term"].astype(str).tolist()
            else:
                selected_terms = [str(term) for term in selected_results.index]
        else:
            selected_terms = terms

        for term in selected_terms:
            if term not in results:
                raise ValueError(f"Term is not present in enrichment results: {term}")
            figures[f"enrichment_score:{term}"] = (
                InteractiveVisualization.prerank_enrichment_score(
                    prerank_result,
                    term=term,
                    filename=str(outdir / f"enrichment_score_{_safe_filename(term)}.html"),
                )
            )

        return figures

    def visualize(
        self,
        prerank_result: Prerank,
        outdir: Union[str, Path],
        terms: Optional[List[str]] = None,
        metric: str = "FDR q-val",
        threshold: float = 0.05,
    ) -> Dict[str, Figure]:
        """Create visualizations for a completed enrichment analysis."""
        return self._visualize(
            prerank_result=prerank_result,
            outdir=outdir,
            terms=terms,
            metric=metric,
            threshold=threshold,
        )

    def run(
        self,
        deg_table: pd.DataFrame,
        column: str,
        outdir: Union[str, Path],
        terms: Optional[List[str]] = None,
        metric: str = "FDR q-val",
        threshold: float = 0.05,
        **kwargs,
    ) -> Prerank:
        """Run enrichment analysis and create its visualizations.

        Parameters
        ----------
        deg_table: pd.DataFrame
            Differential-expression result table. Gene names must be the
            index and ``column`` must contain the ranking statistics.

        column: str
            Column in ``deg_table`` used to rank genes.

        outdir: Union[str, Path]
            Directory for gseapy output and interactive visualizations.

        terms: List[str], optional
            Terms for which enrichment-score plots are created. If None,
            all terms passing ``threshold`` are used.

        metric: str, optional
            Result metric used for term selection. Defaults to
            ``"FDR q-val"``.

        threshold: float, optional
            Maximum metric value for selected terms. Defaults to 0.05.

        **kwargs
            Additional keyword arguments passed to ``gseapy.prerank``.

        Returns
        -------
        Prerank
            The completed gseapy prerank result.
        """
        if column not in deg_table.columns:
            raise ValueError(f"Column is not present in DEG table: {column}")

        prerank = deg_table[column].dropna()
        if prerank.empty:
            raise ValueError(f"Column contains no ranking values: {column}")

        gene_sets = self.gene_sets
        if isinstance(gene_sets, str):
            gene_sets = self.get_library(name=gene_sets, organism=self.organism)

        output_path = Path(outdir)
        with tempfile.TemporaryDirectory(prefix="enrichment-analysis-") as staging_dir:
            result = self._run_prerank(
                prerank=prerank,
                gene_sets=gene_sets,
                outdir=staging_dir,
                **kwargs,
            )
            result_table = result.res2d
            if result_table is None:
                raise ValueError(
                    "The enrichment result does not contain a result table"
                )
            if metric not in result_table.columns:
                raise ValueError(f"Result table does not contain metric: {metric}")

            significant_results = result_table.loc[
                result_table[metric] <= threshold
            ]
            if significant_results.empty:
                print(
                    f"No enrichment pathways passed {metric} <= {threshold}; "
                    f"skipping output directory: {output_path}"
                )
                return result

            output_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staging_dir, output_path, dirs_exist_ok=True)

        self._visualize(
            prerank_result=result,
            outdir=output_path,
            terms=terms,
            metric=metric,
            threshold=threshold,
        )
        return result

    @staticmethod
    def _read_table(filename: PathLike, column: str) -> Optional[pd.DataFrame]:
        """Read a candidate table if it contains usable ranking data."""
        table = pd.read_csv(filename, sep=None, engine="python", index_col=0)
        if column not in table.columns:
            return None

        ranking = cast(pd.Series, pd.to_numeric(table[column], errors="coerce"))
        if ranking.dropna().empty:
            return None

        table[column] = ranking
        return table

    @classmethod
    def find_deg_tables(
        cls,
        input_dir: PathLike,
        column: str = "K(A>B|Z)",
        recursive: bool = True,
    ) -> Iterator[tuple[Path, pd.DataFrame]]:
        """Yield DEG/HDEG tables containing usable ranking statistics.

        Parameters
        ----------
        input_dir: str or Path
            Directory containing DEG or HDEG output tables.

        column: str, optional
            Ranking column required for a table to be selected. Defaults to
            ``"K(A>B|Z)"``.

        recursive: bool, optional
            Search nested directories when True. Defaults to True.
        """
        input_path = Path(input_dir)
        if not input_path.is_dir():
            raise ValueError(f"Input directory does not exist: {input_path}")

        patterns = ("*.tsv", "*.csv", "*.txt")
        if recursive:
            files = sorted(
                path for pattern in patterns for path in input_path.rglob(pattern)
            )
        else:
            files = sorted(
                path for pattern in patterns for path in input_path.glob(pattern)
            )

        for filename in files:
            table = cls._read_table(filename, column)
            if table is not None:
                yield filename, table

    @staticmethod
    def _output_name(input_dir: Path, filename: Path) -> str:
        """Build a stable, filesystem-safe output name from a relative path."""
        relative = filename.relative_to(input_dir).with_suffix("")
        name = "__".join(relative.parts)
        return "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in name
        )

    def run_directory(
        self,
        input_dir: PathLike,
        outdir: Optional[PathLike] = None,
        column: str = "K(A>B|Z)",
        recursive: bool = True,
        terms: Optional[List[str]] = None,
        metric: str = "FDR q-val",
        threshold: float = 0.05,
        **kwargs,
    ) -> Dict[Path, Prerank]:
        """Run enrichment for every usable DEG/HDEG table in a directory.

        Each source table is passed to :meth:`run` and receives its own
        output subdirectory. Files without usable ranking statistics are
        skipped, allowing unrelated reports to share the input directory.
        """
        input_path = Path(input_dir)
        output_path = (
            Path(outdir) if outdir is not None else input_path / "enrichment_analysis"
        )

        tables = list(self.find_deg_tables(input_path, column, recursive))
        if not tables:
            message = (
                f"No DEG/HDEG tables with ranking column '{column}' found "
                f"in {input_path}"
            )
            raise ValueError(message)

        output_path.mkdir(parents=True, exist_ok=True)
        results: Dict[Path, Prerank] = {}
        for filename, table in tables:
            table_output = output_path / self._output_name(input_path, filename)
            results[filename] = self.run(
                deg_table=table,
                column=column,
                outdir=table_output,
                terms=terms,
                metric=metric,
                threshold=threshold,
                **kwargs,
            )

        return results

def _safe_filename(value: str) -> str:
    """Convert an enrichment term into a filesystem-safe filename part."""
    filename = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    ).strip("_")
    while "__" in filename:
        filename = filename.replace("__", "_")
    return filename
