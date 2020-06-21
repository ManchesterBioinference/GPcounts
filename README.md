# GPcounts
GPcounts is Gaussain process regression package for count data with negative binomial and zero-inflated negative binomial likelihoods described in the paper "Gaussian process modelling of temporal and
spatial counts data from RNA-seq experiments".

## Installation:
1. Create and active virtual environment:

```
conda create -n myenv python=3.7 
conda activate myenv
```

2. Install R kernel and required packages to normalise [fission dataset](https://bioconductor.org/packages/release/data/experiment/html/fission.html) using [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html): (Optional)
```
conda install -c r r-irkernel

conda install -c bioconda bioconductor-fission

conda install -c bioconda bioconductor-deseq2
```
3. Clone GPcounts repository:
```
git clone git@github.com:ManchesterBioinference/GPcounts.git
```
4. Install:
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
| RBFKernel | Example of GPcounts with RBF kernel and negative binomial likelihood on synthetic data. |
| SimulateSyntheticCounts | Simulate synthetic bulk RNA-seq timeseries|
| AnscombeTransformation | Transform count data using Anscombe Transformation. |
| OneSampleTest | Application of GPcounts running one sample test on simulated bulk RNA-seq datasets and show Roc curves.|
| Branchingkernel | Application of GPcounts with branching kernel on Paul dataset. |
| TwoSamplesTest | Application of GPcounts running two samples test on fission yeast dataset normalised using DESeq2. |
| DESeq2 | Run DESeq2 R package to normalize scRNA-seq Islet  𝛼  cells gene expression. |
| GPcountsZINB |Compare GPcounts fit with zero-inflated negative binomial, negative binomial and Gaussian likelihoods on ScRNA-seq data using full and sparse GP.|
| CompareDESeq |Correlate DESeq2 and GPcounts with NB and Gaussian likelihood results on scRNAseq dataset.|



