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
| GPcountsSpatial| Applying GPcounts with negative binomial likelihood to identify spatially expressed genes on spatial data from Mouse Olfactory Bulb. We demonstrate how to use the 'scaled' version which is based on data normalisation via multiplication of the NB mean by a location specific scale factor.  |

# Notebooks to reproduce additional paper results: 
To install paper versions of GPflow and packages 
```
cd GPcounts
pip install -r paper_requirements.txt
python setup.py install
cd 
```
Run the GPcounts/paper-notebooks
```
cd GPcounts/paper-notebooks
jupyter notebook
```
| File <br> name | Description | 
| --- | --- | 
| Simulate_synthetic_counts | Simulate synthetic bulk RNA-seq timeseries|
| One_sample_test | One sample test on simulated bulk RNA-seq datasets and ROC curves to compare different likelihood functions|
| DESeq2_scRNA-seq | Run DESeq2 R package to normalize scRNA-seq Islet  𝛼  cells gene expression data|
| Precision_recall_spearman_correlation | Correlate DESeq2 and GPcounts with NB and Gaussian likelihood results on scRNA-seq Islet  𝛼  cells gene expression data|



