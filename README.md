# Marmoset MIND
This repository contains code and data accompanying the article [Cortical myelination networks reflect neuronal gene expression and track adolescent age in marmosets](https://www.biorxiv.org/content/10.1101/2025.09.26.678906v1).

## Installation

```bash
# Clone the repository
git clone https://github.com/edhutch1/marm_MIND.git
cd marm_MIND

# Create environment using Python 3.11
python3.11 -m venv marm_MIND
source marm_MIND/bin/activate

# Install required packages
pip install -r requirements.txt

# Install ipykernel to run notebooks using this environment
python -m ipykernel install --user --name=marm_MIND --display-name="marm_MIND"
```

## Analysis code

Jupyter notebooks contain marked-down code necessary to reproduce Main and Supplementary analyses and figures.
- Fig1: T1w/T2w similarity reflects axonal connectivity and cytoarchitectonic similarity
- Fig2: T1w/T2w similarity aligns with inter-areal similarity of myelin basic protein (MBP) gene expression
- Fig3: Myeloarchitectonic similarity is associated with co-expression of genes in glutamatergic neurons and PV+ and VIP+ interneurons
- Fig4: Marmosets show conserved adolescent increases in myelination
- Fig5: T1w/T2w MIND network edges and nodal degrees predict age more accurately than regional mean T1w/T2w
- Fig6: Sensory and association cortex show distinct age-related changes in myelination network properties
- SF2-4: Comparing Histogram and k-NN estimators of KL divergence
- SF7-: Supplementary analysis for Main Figures 2-6

Notebooks utilise additional helper scripts in /code
- MIND_helpers.py (from https://github.com/isebenius/MIND)
- plotting_helpers.py
- preprocessing_helpers.py
- stats_helpers.py
- Sup_MIND_helpers.py

## Data and pre-computed intermediate outputs

Data necessary to run analyses are provided in /data, with the exception of larger datasets, namely T1-weighted and T2-weighted imaging (Hata et al. 2023: https://dataportal.brainminds.jp/marmoset-mri-na216) and single-nucleus RNA-sequencing data (Krienen et al. 2023: https://cellxgene.cziscience.com/collections/0fd39ad7-5d2d-41c2-bda0-c55bde614bdb). For convenience, intermediate outputs are provided so that these large datasets do not need to be downloaded. These intermediate outputs are:
- Covariates in individual marmosets -> ```/output/subj_df/covariate_per_subj.csv```
- Mean regional T1w/T2w in individual marmosets -> ```/output/subj_df/mean_t12_per_subj.csv```
- Edges of T1w/T2w k-NN MIND networks in individual marmosets -> ```/output/subj_df/edge_per_subj.csv```
- Degrees of T1w/T2w k-NN MIND networks in individual marmosets -> ```/output/subj_df/degree_per_subj.csv```
- Edges of coarse T1w/T2w MIND networks in individual marmosets -> ```/output/subj_df/edge_per_subj_coarse.csv```
- Cell type-specific and interneuron subtype-specific correlated gene expression networks -> ```/output/gene_exp/cge```
- Expression of marker genes in interneuron clusters -> ```/output/gene_exp/gad_marker_expression_in_clusters.csv```
- Edges of T1w/T2w Histogram MIND networks in individual marmosets -> ```/output/SF2-4/hist_opt_edge_per_subj.csv```
- Degrees of T1w/T2w Histogram MIND networks in individual marmosets -> ```/output/SF2-4/hist_opt_degree_per_subj.csv```

Outputs of age prediction analyses in Main (Fig5) and Supplement (SF2-4) are also provided as they are computationally intensive to run.

Brain map plots and circle plots used Matlab and R, and are not reproduced here.