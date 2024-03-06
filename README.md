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
In order to reproduce the paper results we have recorded the original packages used in a different requirements file [paper results ](https://github.com/ManchesterBioinference/GPcounts/tree/V1.0.0).
```
