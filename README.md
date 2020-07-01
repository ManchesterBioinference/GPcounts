# GPcounts
GPcounts is Gaussian process regression package for counts data with negative binomial 
and zero-inflated negative binomial likelihoods described in the paper "Non-parametric 
modelling of temporal and spatial counts data from RNA-seq experiments". It is implemented
using the GPflow library. 

## Installation:

1. Clone GPcounts repository:
```
git clone git@github.com:ManchesterBioinference/GPcounts.git
```

2. Install:
  * Install requirements and package
```
cd GPcounts
pip install -r requirements.txt
python setup.py install
cd 
```
In order to reproduce the paper results we have recorded the original packages used in a different requirements file
```
cd GPcounts
pip install -r paper_requirements.txt
python setup.py install
cd 
```

# Notebooks to demonstrate GPcounts features: 
Run the GPcounts/demo-notebooks
```
cd GPcounts/demo-notebooks
jupyter notebook
```
| File <br> name | Description | 
| --- | --- | 
| [bulk_time_series](./demo_notebooks/bulk_time_series.ipynb) | Applying GPcounts with negative binomial likelihood on bulk RNA-Seq time course data. We compare with Gaussian likelihoood results and show how to infer trajectories and carry out one-sample and two-samples tests|
| [scRNA-Seq_time_series](./demo_notebooks/scRNA-Seq_time_series.ipynb) | Applying GPcounts with zero-inflated negative binomial, negative binomial and Gaussian likelihoods on scRNA-seq gene expression data to find DE genes. We also demonstrate the use of sparse inference to improve computational efficiency|
| [GPcountsSpatial](./demo_notebooks/GPcountsSpatial.ipynb)| Applying GPcounts with negative binomial likelihood to identify spatially expressed genes on spatial data from Mouse Olfactory Bulb. We demonstrate how to use the 'scaled' version which is based on data normalisation via multiplication of the NB mean by a location specific scale factor.  |
| [BranchingGPcounts](./demo_notebooks/BranchingGPcounts.ipynb)| Applying GPcounts on the single-cell data to estimate the most probable branching locations for individual genes. This notebook demonstrates how to build a GPcounts model and plot the posterior model fit and posterior branching times. The application of this approach can be extended to the bulk time series data to identify the differentiation or the perturbation points,at which the two time-courses start to diverge for the first time. |

# Notebooks to reproduce additional paper results: 

Run the GPcounts/paper-notebooks
```
cd GPcounts/paper-notebooks
jupyter notebook
```
| File <br> name | Description | 
| --- | --- | 
| [Simulate_synthetic_counts](./paper_notebooks/Simulate_synthetic_counts.ipynb) | Simulate synthetic bulk RNA-seq timeseries|
| [One_sample_test](./paper_notebooks/One_sample_test.ipynb) | One sample test on simulated bulk RNA-seq datasets and ROC curves to compare different likelihood functions|
| [DESeq2_scRNA-seq](./paper_notebooks/DESeq2_scRNA-seq.ipynb) | Run DESeq2 R package to normalize scRNA-seq Islet  𝛼  cells gene expression data|
| [Precision_recall_spearman_correlation](./paper_notebooks/Precision_recall_spearman_correlation.ipynb) | Correlate DESeq2 and GPcounts with NB and Gaussian likelihood results on scRNA-seq Islet  𝛼  cells gene expression data|



