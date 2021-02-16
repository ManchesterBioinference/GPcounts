# GPcounts
GPcounts is Gaussian process regression package for counts data with negative binomial 
and zero-inflated negative binomial likelihoods described in the paper "Non-parametric 
modelling of temporal and spatial counts data from RNA-seq experiments". It is implemented
using the GPflow library. 

Preprint: https://www.biorxiv.org/content/10.1101/2020.07.29.227207v2

## Installation:

1. Clone GPcounts repository:
```
git clone https://github.com/ManchesterBioinference/GPcounts.git
```

2. Install:
  * Install requirements and package
```
cd GPcounts
pip install -r requirements.txt
cd 
git clone https://github.com/markvdw/RobustGP
cd RobustGP
python setup.py install
cd 
cd GPcounts
python setup.py install
cd 
```
In order to reproduce the paper results we have recorded the original packages used in a different requirements file
```
cd GPcounts
pip install -r paper_requirements.txt
cd
git clone https://github.com/markvdw/RobustGP
cd RobustGP
python setup.py install
cd 
cd GPcounts
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
| [scRNA-Seq_time_series](./demo_notebooks/scRNA-Seq_time_series.ipynb) | Applying GPcounts with negative binomial likelihood on scRNA-seq gene expression data to find DE genes. We also demonstrate the use of sparse inference to improve computational efficiency.|
| [GPcounts_spatial](./demo_notebooks/GPcounts_spatial.ipynb)| Applying GPcounts with negative binomial likelihood to identify spatially expressed genes on spatial data from Mouse Olfactory Bulb. We demonstrate how to use the 'scaled' version which is based on data normalisation via multiplication of the NB mean by a location specific scale factor.  |
| [GPcounts_spatial_smf_scales](./demo_notebooks/GPcounts_spatial_smf_scales.ipynb)| Applying GPcounts with negative binomial likelihood to identify spatially expressed genes on spatial data from Mouse Olfactory Bulb. We show how to calculate the scales factor using python's 'statsmodels' module instead of R code used in the above notebook. This way is easier and faster.|
| [Branching_GPcounts](./demo_notebooks/Branching_GPcounts.ipynb)| Applying GPcounts on the single-cell data to estimate the most probable branching locations for individual genes. This notebook demonstrates how to build a GPcounts model and plot the posterior model fit and posterior branching times. The application of this approach can be extended to the bulk time series data to identify the differentiation or the perturbation points,at which the two time-courses start to diverge for the first time. |

# Notebooks to reproduce additional paper results: 

Run the GPcounts/paper-notebooks
```
cd GPcounts/paper-notebooks
jupyter notebook
```
| File <br> name | Description | 
| --- | --- | 
| [Simulate_synthetic_counts](./paper_notebooks/Simulate_synthetic_counts.ipynb) | Simulate synthetic bulk RNA-seq timeseries|
| [Anscombe_transformation](./paper_notebooks/One_sample_test.ipynb) | A modified version of Anscombe transformation notebook from [SpatialDE](https://github.com/Teichlab/SpatialDE) packages|
| [One_sample_test](./paper_notebooks/One_sample_test.ipynb) | One sample test on simulated bulk RNA-seq datasets and ROC curves to compare different likelihood functions|
| [DESeq2_scRNA-seq](./paper_notebooks/DESeq2_scRNA_seq.Rmd) | Run DESeq2 R package to normalize scRNA-seq Islet  𝛼  cells gene expression data|
| [Precision_recall_spearman_correlation](./paper_notebooks/Precision_recall_spearman_correlation.ipynb) | Correlate DESeq2 and GPcounts with NB and Gaussian likelihood results on scRNA-seq Islet  𝛼  cells gene expression data|
| [scRNA-Seq_time_series](./paper_notebooks/scRNA-Seq_time_series.ipynb) | Applying GPcounts with zero-inflated negative binomial, negative binomial and Gaussian likelihoods on scRNA-seq gene expression data to find DE genes. We also demonstrate the use of sparse inference to improve computational efficiency.|
| [Sparse_Precision_recall_spearman_correlation](./paper_notebooks/Sparse_Precision_recall_spearman_correlation.ipynb) | Correlate GPcounts using NB and Gaussian likelihood results on scRNA-seq Islet  𝛼  cells gene expression data with sparse GPcounts using same likelihoods comapring different methods to select the the location of inducing points.|
| [20190403_evaluateCyclicAll](./paper_notebooks/tradeSeq/20190403_evaluateCyclicAll.R) | Modified R script from [tradeSeq](https://github.com/statOmics/tradeSeqPaper/tree/master/simulation/sim2_dyngen_cycle_72) to run tradeSeq and read GPcounts with NB and Gaussian likelihoods results.|
| [performanceTIPlotCyclic_acrossSimulations](./paper_notebooks/tradeSeq/performanceTIPlotCyclic_acrossSimulations.R) | Modified R script from [tradeSeq](https://github.com/statOmics/tradeSeqPaper/tree/master/simulation/sim2_dyngen_cycle_72) to show the results of tradeSeq and GPcounts with NB and Gaussian likelihoods.|
| [time_evaluation](./paper_notebooks/tradeSeq/time_evaluation.ipynb) | GPcounts with one sample test on the tenth simulated cyclic dataset from [tradeSeq](https://www.bioconductor.org/packages/release/bioc/html/tradeSeq.html) package to compare full GP versus sparse GP using different number of inducing points.|
| [evaluate_genes](./paper_notebooks/tradeSeq/evaluate_genes.ipynb) | Example of genes from the first simulated cyclic dataset from [tradeSeq](https://www.bioconductor.org/packages/release/bioc/html/tradeSeq.html) package fitted using GPcounts one-sample test with pseudotime estimated using slingshot package and [slingshot](https://bioconductor.org/packages/release/bioc/html/slingshot.html) and using the true simulated time|
| [spatial_data_pvalue_hist](./paper_notebooks/MOUSE_OB_Spatial/spatial_data_pvalue_hist.ipynb) | p-value histrogram plots for p-values as calculated via a permutation test and p-values as calculated assuming that the null follows a chi-squared with one degree distribution|
| [MOUSE_OB_plots](./paper_notebooks/MOUSE_OB_Spatial/spatial_data_pvalue_hist.ipynb) | Results from null which follows a chi-squared with one degree of freedom distribution and results from permuted null distribution.|









