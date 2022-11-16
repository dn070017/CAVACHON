from cavachon.tools.AttributionAnalysis import AttributionAnalysis
from cavachon.tools.ClusterAnalysis import ClusterAnalysis
from cavachon.tools.DifferentialAnalysis import DifferentialAnalysis
from gseapy.gsea import Prerank
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, Mapping, Union, Sequence

import anndata
import numpy as np
import muon as mu
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import umap
import warnings

from cavachon.tools.ClusterAnalysis import ClusterAnalysis

class InteractiveVisualization:
  
  @staticmethod
  def bar(
      data: pd.DataFrame,
      x: str,
      y: str,
      group: Optional[str] = None,
      color_discrete_map: Mapping[str, str] = dict(),
      **kwargs) -> plotly.graph_objs._figure.Figure:
    """Create interactive barplot.

    Parameters
    ----------
    data: pd.DataFrame
        input data.
    
    x: str
        column names in the data used as variable in X-axis.
    
    y: str
        column names in the data used as variable in Y-axis.

    group: Optional[str], optional
        column names in the data used to color code the groups. 
        Defaults to None.
    
    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for fig.update_layout.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.

    """
    if group:
      unique_groups = data[group].value_counts().sort_index().index
      fig = go.Figure()
      for subset in unique_groups:
        data_subset = data.loc[data[group] == subset]
        means = data_subset.groupby(x).mean()[y]
        sem = data_subset.groupby(x).sem()[y]
        if color_discrete_map.get(subset, None):
          fig.add_trace(go.Bar(
              name=subset,
              x=means.index, 
              y=means, 
              marker_opacity=0.7,
              marker_line=dict(width=1, color='DarkSlateGrey'),
              marker_color=color_discrete_map.get(subset),
              error_y=dict(type='data', array=sem)))
        else:
            fig.add_trace(go.Bar(
              name=subset,
              x=means.index, 
              y=means,
              marker_opacity=0.7,
              marker_line=dict(width=1, color='DarkSlateGrey'),
              error_y=dict(type='data', array=sem)))
      fig.update_layout(barmode='group', **kwargs)
    else:
      means = data.groupby(x).mean()[y]
      sem = data.groupby(x).sem()[y]
      fig = go.Figure()
      fig.add_trace(go.Bar(
          name='Control',
          x=means.index, y=means,
          marker_opacity=0.7,
          marker_line=dict(width=1, color='DarkSlateGrey'),
          error_y=dict(type='data', array=sem)))

    return fig

  @staticmethod
  def scatter(*args, **kwargs) -> plotly.graph_objs._figure.Figure:
    """Create interactive scatter plot.
   
    Parameters
    ----------
    *args: Optional[Sequence[Any]], optional
        additional positional arguments for px.scatter.

    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for px.scatter.


    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.

    """
    fig = px.scatter(*args, **kwargs)
    fig.update_traces(
        marker=dict(
            opacity=0.7, 
            line=dict(width=1, color='DarkSlateGrey')))    
    return fig

  @staticmethod
  def embedding(
      adata: anndata.AnnData,
      method: str = 'tsne',
      filename: Optional[str] = None,
      use_rep: Union[str, np.array] = 'z',
      color: Union[str, Sequence[str], None] = None,
      title: Optional[str] = None,
      color_discrete_sequence: Optional[Sequence[str]] = None,
      force: bool = False,
      *args,
      **kwargs) -> plotly.graph_objs._figure.Figure:
    """Create interactive visualization for the latent space.

    Parameters
    ----------
    adata: anndata.AnnData
        the AnnData used for the analysis.
    
    method: str, optional
        embedding method for the latent space, support 'pca', 'umap' 
        and 'tsne'. Defaults to 'tsne'.
    
    filename: Optional[str], optional
        filename to save the figure. None if disable saving. Defaults 
        to None.
    
    use_rep: Union[str, np.array], optional
        which representation to used for the latent space. Defaults to 
        'z'. Alternatively, the array will be used if provided with 
        np.array,
    
    color: Union[str, Sequence[str], None], optional
        column names in the adata.obs that used to color code the 
        samples. Alternatively, if provided with `obsm_key`/`obsm_index`
        the color will be set to the `obsm_index` column from the array
        of adata.obsm[`obsm_key`]. The same color for all samples will
        be used if provided with None. Defaults to None.
    
    title: Optional[str], optional
        title for the figure. Defaults to 'Z(name of AnnData)'
    
    color_discrete_sequence: Optional[Sequence[str]], optional
        the discrete color set individually for each sample. This will
        overwrite the color code from `color`. The color code defined
        from `color` argument will be used if provided with none. To 
        change the color palette for `color`, use `color_discrete_map`
        instead. Defaults to None.
    
    force: bool, optional
        force to rerun the embedding. Defaults to False.
    
    *args: Optional[Sequence[Any]], optional
        additional positional arguments for px.scatter.

    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for px.scatter.
    
    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.

    """

    adata_name = adata.uns.get('cavachon', '').get('name', '')
    if title is None:
      title = f'Z({adata_name})'

    if method not in ['pca', 'tsne', 'umap']:
      message = ''.join((
          f"Invalid value for method ({method}). Expected to be one of the following: ",
          f"'pca', 'tsne' or 'umap'. Set to 'tsne'."
      ))
      warnings.warn(message, RuntimeWarning)
      method = 'tsne'
     
    if color is not None:
      if color in adata.obs:
        color = adata.obs[color]
      else:
        color_obsm_key = '/'.join(color.split('/')[0:-1])
        color_obsm_column = int(color.split('/')[-1])
        if color_obsm_key in adata.obsm:
          color = adata.obsm[color_obsm_key][:, color_obsm_column]
        else:
          message = ''.join((
            f"{color} is not in adata.obs, and {color_obsm_key} is not in adata.obsm "
            f"ignore color argument."
          ))
          warnings.warn(message, RuntimeWarning)
          color = None

    if color is None:
      if color_discrete_sequence is None:
        color_discrete_sequence = ['salmon'] * adata.n_obs
    
    if isinstance(use_rep, np.ndarray):
      matrix = use_rep
    else:
      matrix = adata.obsm[use_rep]
    
    # np.ndarray has __str__ implemented. It also works if use_rep is a
    # np.ndarray
    obsm_key = f'{use_rep}_{method}'  
    if force or obsm_key not in adata.obsm.keys():
      if method == 'pca':
        model = PCA(n_components=2, random_state=0)
        transformed_matrix = model.fit_transform(matrix)
      if method == 'tsne':
        model = TSNE(random_state=0)
        transformed_matrix = model.fit_transform(matrix)
      if method == 'umap':
        model = umap.UMAP(random_state=0)
        transformed_matrix = model.fit_transform(matrix)
      
      if isinstance(use_rep, str):
        adata.obsm[obsm_key] = transformed_matrix

    # in case the embedding is not called again.
    if isinstance(use_rep, str):
      transformed_matrix = adata.obsm[obsm_key]

    x = transformed_matrix[:, 0]
    y = transformed_matrix[:, 1]

    if method == 'pca':
      labels = {'x': 'PCA 1', 'y': 'PCA 2'}
    if method == 'tsne':
      labels = {'x': 't-SNE 1', 'y': 't-SNE 2'}
    if method == 'umap':
      labels = {'x': 'Umap 1', 'y': 'Umap 2'}

    fig = InteractiveVisualization.scatter(
        x=x,
        y=y,
        labels=labels,
        title=title,
        color=color,
        color_discrete_sequence=color_discrete_sequence,
        *args, **kwargs)
    fig.show()
    
    if filename:
      if filename.endswith('html'):
        fig.write_html(filename)
      else:
        fig.write_image(filename)
    
    return fig

  @staticmethod
  def neighbors_with_same_annotations(
      mdata: mu.MuData,
      model: tf.keras.Model,
      modality: str,
      use_cluster: str,
      use_rep: Union[str, np.array],
      n_neighbors: Sequence[int] = list(range(5, 25)),
      filename: Optional[str] = None,
      group_by_cluster: bool = False,
      color_discrete_map: Mapping[str, str] = dict(),
      title: str = 'Cluster Nearest Neighbor Analysis',
      **kwargs) -> plotly.graph_objs._figure.Figure:
    """Create interactive visualization for nearest neighbor analysis.

    Parameters
    ----------
    mdata: mu.MuData
        the MuData used for the generative process.
    
    model: tf.keras.Model
        the trained generative model used for the generative process.
    
    modality: str
        the modality to used.
    
    use_cluster: str
        the column name of the clusters in the obs of modality.
    
    use_rep: Union[str, np.array]
        the key of obsm of modality to used to compute the distance 
        within and between clusters. Alternatively, the array will be 
        used if provided with np.array,
    
    n_neighbors: Union[int, Sequence[int]], optional
        the number of neighbors to be analyzed, Defaults to 
        list(range(5, 25))
    
    filename: Optional[str], optional
        filename to save the figure. None if disable saving. Defaults 
        to None.
    
    group_by_cluster: bool, optional
        whether or not to group by the clusters. Defaults to False
    
    color_discrete_map: Mapping[str, str], optional
        the color palette for `group_by_cluster`. Defaults to dict()

    title: str, optional
        title for the figure. Defaults to 'Cluster Nearest Neighbor
        Analysis'
    
    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for fig.update_layout.
    
    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.

    """
    analysis = ClusterAnalysis(mdata, model)
    analysis_result = analysis.compute_neighbors_with_same_annotations(
        modality=modality, 
        use_cluster=use_cluster,
        use_rep=use_rep,
        n_neighbors=n_neighbors)
    
    if group_by_cluster:
      group = 'Cluster'
    else:
      group = None

    fig = InteractiveVisualization.bar(
        analysis_result, 
        x='Number of Neighbors', 
        y='% of KNN Cells with the Same Cluster',
        group=group,
        title=title,
        color_discrete_map=color_discrete_map,
        xaxis_title='Number of Neighbors',
        yaxis_title='% of KNN Cells with the Same Cluster',
        **kwargs)
    fig.show()

    if filename:
      if filename.endswith('html'):
        fig.write_html(filename)
      else:
        fig.write_image(filename)
    
    return fig
  
  @staticmethod
  def attribution_score(
      mdata: mu.MuData,
      model: tf.keras.Model,
      component: str,
      modality: str,
      target_component: str,
      use_cluster: str,
      steps: int = 10,
      selected_variables: Optional[Sequence[str]] = None,    
      batch_size: int = 128,
      color_discrete_map: Mapping[str, str] = dict(),
      filename: Optional[str] = None,
      title: str = 'Attribution Score Analysis',
      **kwargs) -> plotly.graph_objs._figure.Figure:
    """Create interactive visualization for attribution score.

    Parameters
    ----------
    mdata: mu.MuData
        the MuData used for the generative process.
    
    model: tf.keras.Model
        the trained generative model used for the generative process.

    component: str
        the outputs of which component to used.

    modality: str
        which modality of the outputs of the component to used.

    target_component: str
        the latent representation of which component to used.

    use_cluster: str
        the column name of the clusters in the obs of modality.

    steps: int, optional
        steps in integrated gradients. Defaults to 10.

    selected_variables: Optional[Sequence[str]], optional
        the variables to used. The provided variables needs to match
        the indices of mdata[modality].var. All variables will be used 
        if provided with None. Defaults to None.

    batch_size: int, optional
        batch size used for the forward pass. Defaults to 128.

    color_discrete_map: Mapping[str, str], optional
        the color palette for `group_by_cluster`. Defaults to dict()

    filename: Optional[str], optional
        filename to save the figure. None if disable saving. Defaults 
        to None.

    title: str, optional
        title for the figure. Defaults to 'Attribution Score Analysis'
    
    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for fig.update_layout.
    
    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.

    """

    analysis = AttributionAnalysis(mdata, model)
    attribution_score = analysis.compute_integrated_gradient(
        component=component,
        modality=modality,
        target_component=target_component,
        steps=steps,
        selected_variables=selected_variables,
        batch_size=batch_size)

    data = pd.DataFrame({
        'X': np.ones(mdata[modality].n_obs),
        'Cluster': mdata[modality].obs[use_cluster], 
        'Attribution Score': np.mean(np.abs(attribution_score), axis=-1)})

    fig = InteractiveVisualization.bar(
        data, 
        x='X', 
        y='Attribution Score',
        group='Cluster',
        title=title,
        color_discrete_map=color_discrete_map,
        xaxis_title='Cluster',
        yaxis_title='Attribution Score',
        **kwargs)
    fig.update_layout(xaxis_showticklabels=False)
    fig.show()

    if filename:
      if filename.endswith('html'):
        fig.write_html(filename)
      else:
        fig.write_image(filename)
    
    return fig
  
  @staticmethod
  def differential_volcano_plot(
      differential_results: pd.DataFrame = None,
      mdata: mu.MuData = None,
      model: tf.keras.Model = None,
      group_a_index: Union[pd.Index, Sequence[str]] = None,
      group_b_index: Union[pd.Index, Sequence[str]] = None,
      component: str = None,
      modality: str = None,
      significant_threshold: float = 3.2,
      filename: Optional[str] = None,
      title: str = 'Volcano Plot for Differential Analysis',
      **kwargs) -> plotly.graph_objs._figure.Figure:
    """Create interactive visualization for volcano plot for 
    differential analysis

    Parameters
    ----------
    differential_results: pd.DataFrame
        result for the differential analysis.

    mdata: mu.MuData
        the MuData used for the generative process.
    
    model: tf.keras.Model
        the trained generative model used for the generative process.

    group_a_index : Union[pd.Index, Sequence[str]]
        index of group one. Needs to meet the index in the obs of the
        modality.
    
    group_b_index : Union[pd.Index, Sequence[str]]
        index of group two. Needs to meet the index in the obs of the
        modality.
    
    component : str
        generative result of `modality` from which component to used.
    
    modality : str
        which modality to used from the generative result of 
        `component`.

    significant_threshold : float, optional
        threshold for significance of Bayesian factor. Defaults to 3.2.
    
    filename: Optional[str], optional
        filename to save the figure. None if disable saving. Defaults 
        to None.

    title: str, optional
        title for the figure. Defaults to 'Attribution Score Analysis'
    
    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for fig.update_layout.
    
    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.

    """
    if differential_results is None:
      analysis = DifferentialAnalysis(mdata=mdata, model=model)
      differential_results = analysis.between_two_groups(
          group_a_index=group_a_index,
          group_b_index=group_b_index, 
          component=component,
          modality=modality)
        
    differential_results['LogFC(A/B)'] = np.log(differential_results['Mean(A)']/differential_results['Mean(B)'])
    differential_results['K(A>B|Z)'].index = differential_results.index
    differential_results['K(A!=B|Z)'] = (
        (differential_results['LogFC(A/B)'] >= 0).astype(int) * differential_results['K(A>B|Z)'] - 
        (differential_results['LogFC(A/B)'] < 0).astype(int) * differential_results['K(A>B|Z)']
    )#differential_results[['K(A>B|Z)', 'K(B>A|Z)']].abs().max(axis=1)
    differential_results['Significant'] = differential_results['K(A!=B|Z)'].apply(lambda x: 'Significant' if x > significant_threshold else 'Non-significant')
    differential_results['Target'] = differential_results.index
    fig = InteractiveVisualization.scatter(
        differential_results,
        x='LogFC(A/B)',
        y='K(A!=B|Z)',
        color='Significant', 
        labels={'x': 'LogFC(A/B)', 'y': 'K(A!=B|Z)'},
        hover_data=['Target'],
        title=title,
        **kwargs)
    
    fig.show()

    if filename:
      if filename.endswith('html'):
        fig.write_html(filename)
      else:
        fig.write_image(filename)
    
    return fig
  
  @staticmethod
  def prerank_ringplot(
      prerank_result: Prerank,
      metric: str = 'FDR q-val',
      threshold: float = 0.05,
      filename: Optional[str] = None,
      title: str = 'Ring Plot for Prerank Enrichment Analysis') -> plotly.graph_objs._figure.Figure:
    """Create interactive visualization for volcano plot for 
    differential analysis.

    Parameters
    ----------
    prerank_result: Prerank
        outputs of gseapy.prerank.

    metric: str, optional
        metric used to color the dot. Defaults to 'FDR q-val'.
    
    threshold : float, optional
        threshold of `metrics` used to filter the terms. Defaults to
        0.05.
    
    filename: Optional[str], optional
        filename to save the figure. None if disable saving. Defaults 
        to None.

    title: str, optional
        title for the figure. Defaults to 'Ring Plot for Prerank 
        Enrichment Analysis'

    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.

    """
    data = prerank_result.res2d
    data = data.loc[data[metric] <= threshold]
    data = data.loc[data.index[::-1]]
    color = np.log(1 / (data[metric].values.astype(np.float32) + 1e-7))
    size = np.array([float(x[:-2]) for x in data['Gene %']])
    
    data['Hits (%)'] = size
    data[f'log(1/{metric})'] = color

    fig = InteractiveVisualization.scatter(
      data, 
      x='NES', 
      y='Term', 
      size='Hits (%)', 
      # hard-coded to make sure Hits (%) = 100 is the max size
      size_max=21 * max(size) / 100,
      color=f'log(1/{metric})',
      color_continuous_scale='rdpu',
      labels={'x': 'Normalized Enrichment Score', 'y': 'Term'},
      width=800, height=35 * len(data) + 150,
      title=title)
    
    # add ring
    fig.add_trace(
        go.Scatter(
            x=data["NES"],
            y=data["Term"],
            marker_color='rgba(0, 0, 0, 0)',
            hoverinfo='skip',
            marker_size=30,
            mode='markers',
            showlegend=False))
    fig.update_traces(
        marker=dict(
            opacity=0.7, 
            line=dict(width=1, color='DarkSlateGrey'))) 
    
    fig.show()

    if filename:
      if filename.endswith('html'):
        fig.write_html(filename)
      else:
        fig.write_image(filename)
    
    return fig

  @staticmethod
  def prerank_enrichment_score(
      prerank_result: Prerank,
      term: str,
      filename: Optional[str] = None) -> plotly.graph_objs._figure.Figure:
    """Create interactive visualization for enrichment score for the
    provided term.

    Parameters
    ----------
    prerank_result: Prerank
        outputs of gseapy.prerank.

    term: str
        the term to used.

    filename: Optional[str], optional
        filename to save the figure. None if disable saving. Defaults 
        to None.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.


    """
    term_result = prerank_result.results[term]
    rank_metric = prerank_result.ranking

    indices = np.arange(len(rank_metric))
    RES = np.asarray(term_result.get('RES'))
    zero_score_index = np.abs(rank_metric.values).argmin()
    hit_indices = np.array(term_result.get('hits'))

    z_score_label = f"Zero score at {zero_score_index}"
    nes_label = f"NES: {float(term_result.get('nes')):.3f}"
    pval_label = f"Pval: {float(term_result.get('pval')):.3e}"
    fdr_label = f"FDR: {float(term_result.get('fdr')):.3e}"

    data = pd.DataFrame({
        'Rank in Ordered Dataset': indices,
        'Rank List Metric': rank_metric,
        'Enrichment Score': RES,
        'color': rank_metric > 0,
        'Target': rank_metric.index})

    fig = make_subplots(
        rows=4,
        cols=1,
        row_heights=[0.45, 0.05, 0.05, 0.45],
        vertical_spacing=0.00)

    hover_template = ''.join((
        'Target: %{text}<br>Rank in Ordered Dataset: %{x}'
        '<br>Enrichment Score: %{y:.2f}'))
    fig.add_trace(
        go.Scatter(
            x=data.loc[indices < zero_score_index]['Rank in Ordered Dataset'],
            y=data.loc[indices < zero_score_index]['Enrichment Score'],
            text=data['Target'],
            showlegend=False,
            name='',
            marker_color='salmon',
            hovertemplate=hover_template),
        row=1,
        col=1)
    fig.add_trace(
        go.Scatter(
            x=data.loc[indices >= zero_score_index]['Rank in Ordered Dataset'],
            y=data.loc[indices >= zero_score_index]['Enrichment Score'],
            text=data['Target'],
            showlegend=False,
            name='',
            marker_color='royalblue',
            hovertemplate=hover_template),
        row=1,
        col=1)

    hover_template = ''.join((
        'Target: %{text}<br>Rank in Ordered Dataset: %{x}'))
    marker_color = ['salmon' if x else 'royalblue' for x in hit_indices <= zero_score_index]
    fig.add_trace(
        go.Scatter(
            x=hit_indices,
            y=np.zeros_like(hit_indices),
            text=data['Target'].iloc[hit_indices],
            mode='markers',
            showlegend=False,
            name='',
            marker_color=marker_color,
            marker_symbol='line-ns-open',
            marker_size=30,
            hovertemplate=hover_template),
        row=2,
        col=1)

    mid = (zero_score_index - 0) / len(data)
    fig.add_trace(
        go.Scatter(
            x=data['Rank in Ordered Dataset'], 
            y=np.zeros_like(rank_metric.values), 
            text=data['Target'],
            mode='markers', 
            showlegend=False,
            name='',
            hovertemplate='Target: %{text}<br>Rank in Ordered Dataset: %{x}',
            marker_color=data['Rank in Ordered Dataset'],
            marker_symbol='line-ns-open',
            marker_size=30,
            marker_colorscale=[
                (0.0, "salmon"),
                (mid, "white"),
                (1.0, "royalblue")]),
        row=3,
        col=1)

    hover_template = ''.join((
        'Target: %{text}<br>Rank in Ordered Dataset: %{x}'
        '<br>Rank List Metric: %{y:.2f}'))
    fig.add_trace(
        go.Scatter(
            x=data.loc[indices < zero_score_index]['Rank in Ordered Dataset'],
            y=data.loc[indices < zero_score_index]['Rank List Metric'],
            text=data['Target'],
            marker_color='salmon',
            fill='tozeroy',
            showlegend=False,
            name='',
            hovertemplate=hover_template),
        row=4,
        col=1)
    fig.add_trace(
        go.Scatter(
            x=data.loc[indices >= zero_score_index]['Rank in Ordered Dataset'],
            y=data.loc[indices >= zero_score_index]['Rank List Metric'],
            text=data['Target'],
            marker_color='royalblue',
            fill='tozeroy',
            showlegend=False,
            name='',
            hovertemplate=hover_template),
        row=4,
        col=1)

    fig.add_trace(
        go.Scatter(
            x=[zero_score_index],
            y=[0.0],
            mode="markers+text",
            marker_color='red',
            text=[z_score_label],
            textposition="top right",
            showlegend=False,
            hoverinfo='skip'),
        row=4,
        col=1)

    fig.update_layout(
        title=f'{term}<br>{nes_label}, {pval_label}, {fdr_label}',
        hovermode="x unified", 
        yaxis1_title='Enrichment Score',
        xaxis1_showticklabels=False, yaxis2_showticklabels=False, xaxis2_showticklabels=False, 
        xaxis2_range=[0, len(indices)], xaxis3_showticklabels=False, yaxis3_showticklabels=False, 
        xaxis3_range=[0, len(indices)],
        xaxis4_title='Rank in Ordered Dataset', 
        yaxis4_title='Rank Metric', 
        xaxis4_range=[0, len(indices)],
        width=700, height=700)

    fig.show()
    
    if filename:
      if filename.endswith('html'):
        fig.write_html(filename)
      else:
        fig.write_image(filename)
    
    return fig