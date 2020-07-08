'''
Nuha BinTayyash, 2020

This notebook shows how to Simulate synthetic bulk RNA-seq time series of $S =3$ replicate, $D= 600$ genes measured at $T=6$ time points. Half of the genes are differentially expressed across time. We use two classes of generative functions $f$ (sine and cubic splines) to simulate differentially expressed genes and we include a matching non-differentially expressed gene with the same mean in the synthetic data. The sine function is of the form $f(x) = a\sin(2*\pi*f*x)+c$ with $x\in[-.5,1]$. The cubic spline function has the form $f(x) \in C^2 [0,1]$ passing through $n = 6 $ data points $(x,y)$ where $x \in [0,1]$. The $c$ is to vary the mean of counts, $c = 0$ to set the mean of low counts and $c = 6$ to set the mean of high counts. The $a,y \in [-2+c,4+c]$ are drawn from uniform distributions to vary the amplitude for sine and cubic spline functions. The low and high dispersion values are drawn from uniform distributions $\alpha_\mathrm{low} = U[.05,.09]$ and $\alpha_\mathrm{high} = U[.1,1.]$. An exponential inverse-link function is used to determine the mean of count data at each time $\mu(x) = e^{f(x)}$ and we use the Scipy library to sample counts from the negative binomial distribution parametrized by the probability of success $p=\frac{r}{\exp(f)+r}$ and number of failures $r =\frac{1}{dispersion} $. 

Bulk_simulator.py used to simulated four datasets with two levels of the mean of count data (high count and low count datasets) and two levels of dispersion (high dispersion and low dispersion).

'''

import pandas as pd
import numpy as np
from scipy.stats import nbinom
import random
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import matplotlib.pyplot as plt

S = 3 # number of samples/cells  
x = sorted([0. , 0.2, 0.4, 0.6, 0.8, 1. ]*S)# Time points
print(x)

def sin_fun(): 
    f = c+ a* np.sin(b*x+d) 
    return f

def Cubic_spline():
    return f_cs(x)

def sample_from_NegativeBinomal(r,mean): 
    # r  number of failures
    # prob probability of success
    prob = r/(mean+r)
    y =[]
    for i in range(mean.shape[0]):
        y.append(nbinom.rvs(r, prob[i], size=1))
    y = np.array(y)
    return y
    
def sample_count(fun_num):
    
    # sample dispersion parameters for dynamic function then for constant function 
    alpha = np.random.uniform(Dispersion_ranges[dispersion][0],Dispersion_ranges[dispersion][1],1) 
    print('alpha',alpha)
    # sample dynamic and constant function from the generative function
    if fun_num == 0 :
        f = sin_fun()
   
    if fun_num == 1:
        plt.plot(x_spline,y,'o')
        f = Cubic_spline()
        
    f = f 
    #+ np.random.uniform(-1.,1.) # add randomness 
    plt.plot(x,f)
    # exponentiate generative function to set the mean of NB distribution and simulate differentially expressed genes
    f_sample = sample_from_NegativeBinomal(1./alpha,np.exp(f)).T
    
   
    min_fun = np.max(f)/2   
         
    # sample constant function 
    f_con = np.random.uniform(min_fun,np.max(f),1) 
    f_con = f_con * np.ones(len(x))
    plt.plot(x,f_con)
    
    #simulate non-differentially expressed genes from constant function    
    constant_sample = sample_from_NegativeBinomal(1./alpha,np.exp(f_con)).T
       
    return f_sample,constant_sample

def vstack(samples):
    samples = np.vstack(samples)
    return samples

samples = []
samples_constant = []

Dispersion_ranges = [
                  [.01,.1]  # low dispersion range
                  ,[1.,3.] # high dispersion range
                  ]   

max_mean = [2,7] # to change function mean from low count to high count data 
#4
for c in tqdm(max_mean): # c to change the mean 
    for dispersion in range(len(Dispersion_ranges)):
        np.random.seed(0) # reset to sample new dataset
        random.seed(0)
        samples = []
        samples_constant = []
       
        for i in range(10):
            
           
            a = np.random.uniform(-2,4,1) 
            y = np.random.uniform(-2,a+1, 4) + c
            b = np.random.uniform(np.pi/2,np.pi*2,1)
            d = np.random.uniform(0,np.pi*2,1)
            print('a',a)
            print('b',b)
            print('c',c)
            print('d',d)
            
            fig = plt.figure()
            f_sample,f_constant = sample_count(0)
            plt.show()
            samples.append(f_sample)
            samples_constant.append(f_constant)
            
            
            fig = plt.figure()
            x_spline = np.linspace(0.,1,4)
            print(x_spline)
            y = np.random.uniform(-1,a+1, 4) + c
            
            f_cs = CubicSpline(x_spline,y)
            print('y',y)
            #x = np.linspace(x_spline[0],x_spline[5],len(T))
            f_sample,f_constant = sample_count(1)
            plt.show()
            samples.append(f_sample)
            samples_constant.append(f_constant)
            
                
        samples = vstack(samples)
        samples = samples.astype(float)
        samples_constant = vstack(samples_constant)
        samples_constant = samples_constant.astype(float)
     
        if c == max_mean[0]:
            count_level = 'low_counts_'
        else:
            count_level = 'high_counts_'
        
        if dispersion == 0:
            alpha_level = 'low_dispersion'
        else:
            alpha_level = 'high_dispersion'
            
        genes = ['gene_%s' % (s+1) for s in range(samples.shape[0])] 
        col = list(map(str,x))
        samples_df = pd.DataFrame(data=samples,index= genes,columns=col)
        samples_df_constant = pd.DataFrame(data=samples_constant,index= genes,columns=col)
        samples_df_all = pd.concat([samples_df, samples_df_constant])
        samples_df_all.to_csv('../data/'+count_level+alpha_level+'.csv')
        print(count_level+alpha_level)

time_points = pd.DataFrame(data=x,index= col,columns=['times'])
time_points.to_csv('../data/time_points.csv')
