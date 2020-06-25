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
# Examples of GPcounts uses: 
Run the GPcounts/demo-notebooks
```
cd GPcounts/demo-notebooks
jupyter notebook
```
| File <br> name | Description | 
| --- | --- | 
| bulk_time_series | Example use of GPcounts with negative binomial likelihood on bulk RNA-Seq time course data. Comparison with Gaussian likelihood and examples of inferring trajectories and carrying out one sample and two samples tests on on fission yeast dataset.|
| scRNA-Seq_time_series | Example use of GPcounts with zero-inflated negative binomial, negative binomial and Gaussian likelihoods using full inference on ScRNA-seq gene expression data to find differentially expressed genes using infer trajectory and compare it with GPcounts using sparse inferece to obtain to obtain computational efficiency |

# Reproduce paper results: 
Run the GPcounts/paper-notebooks
```
cd GPcounts/paper-notebooks
jupyter notebook
```
| File <br> name | Description | 
| --- | --- | 
| Simulate_synthetic_counts | Simulate synthetic bulk RNA-seq timeseries|
| Anscombe_transformation | Transform count data using Anscombe Transformation. |
| One_sample_test | Application of GPcounts running one sample test on simulated bulk RNA-seq datasets and show Roc curves.|
| ScRNA_seq_DESeq2 | Run DESeq2 R package to normalize scRNA-seq Islet  𝛼  cells gene expression. |
| Precision_recall_spearman_correlation |Correlate DESeq2 and GPcounts with NB and Gaussian likelihood results on scRNAseq dataset.|



