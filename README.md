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
GPcounts requires TensorFlow 2.1.0, GPflow 2.0.0 and Python 3.7.

2. Install:
  * Install [GPflow](https://github.com/GPflow/GPflow)
  * Install requirements and package
```
cd GPcounts
pip install -r requirements.txt
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
| bulk_time_series | Applying GPcounts with negative binomial likelihood on bulk RNA-Seq time course data. We compare with Gaussian likelihoood results and show how to infer trajectories and carry out one-sample and two-samples tests|
| scRNA-Seq_time_series | Applying GPcounts with zero-inflated negative binomial, negative binomial and Gaussian likelihoods on scRNA-seq gene expression data to find DE genes. We also demonstrate the use of sparse inference to improve computational efficiency|

# Notebooks to reproduce additional paper results: 
Run the GPcounts/paper-notebooks
```
cd GPcounts/paper-notebooks
jupyter notebook
```
| File <br> name | Description | 
| --- | --- | 
| Simulate_synthetic_counts | Simulate synthetic bulk RNA-seq timeseries|
| Anscombe_transformation | Transform count data using Anscombe's Transformation|
| One_sample_test | One sample test on simulated bulk RNA-seq datasets and ROC curves to compare different likelihood functions|
| ScRNA_seq_DESeq2 | Run DESeq2 R package to normalize scRNA-seq Islet  𝛼  cells gene expression data|
| Precision_recall_spearman_correlation | Correlate DESeq2 and GPcounts with NB and Gaussian likelihood results on scRNA-seq Islet  𝛼  cells gene expression data|



