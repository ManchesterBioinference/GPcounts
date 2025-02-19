# GPcounts
GPcounts is Gaussian process regression package for counts data with negative binomial 
and zero-inflated negative binomial likelihoods described in the paper ["Non-parametric 
modelling of temporal and spatial counts data from RNA-seq experiments"](https://www.biorxiv.org/content/10.1101/2020.07.29.227207v2). It is implemented in python, using the [tensorflow](https://www.tensorflow.org/) and [GPflow](https://github.com/GPflow/GPflow/). 

This is now published in [Bioinformatics](https://academic.oup.com/bioinformatics/article/37/21/3788/6313161).


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5027066.svg)](https://doi.org/10.5281/zenodo.5027066)

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


In order to reproduce the paper results we have recorded the original packages used in a different requirements file [paper results ](https://github.com/ManchesterBioinference/GPcounts/tree/V1.0.0).

