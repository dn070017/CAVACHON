# %%
import os
import warnings

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.tools.cluster_analysis import ClusterAnalysis
from cavachon.tools.hierarchical_differential_analysis import (
    HierarchicalDifferentialAnalysis,
)
from cavachon.workflow.workflow import Workflow

warnings.simplefilter(action="ignore", category=FutureWarning)

try:
    physical_devices = tf.config.list_physical_devices("GPU")
    for i, _ in enumerate(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)
except Exception:
    print("No GPU detected. Use CPU instead")


filename = os.path.realpath("./sample.yaml")
workflow = Workflow(filename)
workflow.run()

batch_size = workflow.config.dataset.get(Constants.CONFIG_FIELD_MODEL_DATASET_BATCHSIZE)
component = "RNA"
donor_components = [component]
modality = "RNA"
cluster_key = f"cluster_{component}"

cluster_analysis = ClusterAnalysis(workflow.mdata, workflow.model)
cluster_analysis.compute_cluster_log_probability(
    modality=modality,
    component=component,
    batch_effect_colnames=workflow.batch_effect_colnames,
    distribution_names=workflow.distribution_names,
    batch_size=batch_size,
)

obs = workflow.mdata[modality].obs
clusters = sorted(obs[cluster_key].dropna().unique())

if len(clusters) >= 2:
    donor_label, recipient_label = clusters[:2]
else:
    raise RuntimeError(
        f"Need at least 2 clusters in '{cluster_key}' to run hierarchical DEG QA, "
        f"got: {clusters}"
    )

analysis = HierarchicalDifferentialAnalysis(
    mdata=workflow.mdata,
    model=workflow.model,
    batch_effect_colnames=workflow.batch_effect_colnames,
    distribution_names=workflow.distribution_names,
    batch_effect_encoders=workflow.dataloader.batch_effect_encoders,
)

result = analysis.between_clusters(
    donor_cluster=donor_label,
    recipient_cluster=recipient_label,
    component=component,
    modality=modality,
    use_cluster=cluster_key,
    n_samples=2,
    seed=42,
    donor_components=donor_components,
    batch_size=batch_size,
)

output_dir = os.path.realpath("./ceren/differential_analysis")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test_run_qa_hdeg.tsv")
result.to_csv(output_path, sep="\t")

print("\nInput")
print(f"  config:         {filename}")
print(f"  component:      {component}")
print(f"  modality:       {modality}")
print(f"  cluster key:    {cluster_key}")
print(f"  donor cluster:  {donor_label}")
print(f"  recipient:      {recipient_label}")
print(f"  donor components: {donor_components}")

print("\nOutput")
print(f"  file:    {output_path}")
print(f"  shape:   {result.shape}")
print(f"  columns: {list(result.columns)}")
print(result.head(10))
