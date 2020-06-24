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
# List of notebooks
To run the notebooks
```
cd GPcounts/notebooks
jupyter notebook
```

| File <br> name | Description | 
| --- | --- | 
| GPcounts | Example use of GPcounts with negative binomial likelihood on bulk RNA-Seq time course data. Comparison with Gaussian likelihood and examples of inferring trajectories and carrying out one sample and two samples tests on on fission yeast dataset.|
| SimulateSyntheticCounts | Simulate synthetic bulk RNA-seq timeseries|
| AnscombeTransformation | Transform count data using Anscombe Transformation. |
| OneSampleTest | Application of GPcounts running one sample test on simulated bulk RNA-seq datasets and show Roc curves.|
| Branchingkernel | Application of GPcounts with branching kernel on Paul dataset. |
| DESeq2 | Run DESeq2 R package to normalize scRNA-seq Islet  𝛼  cells gene expression. |
| GPcountsZINB |Compare GPcounts fit with zero-inflated negative binomial, negative binomial and Gaussian likelihoods on ScRNA-seq data using full and sparse GP.|
| PrecisionRecallSpearmanCorrelation |Correlate DESeq2 and GPcounts with NB and Gaussian likelihood results on scRNAseq dataset.|



