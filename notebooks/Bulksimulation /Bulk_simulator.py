'''
Nuha BinTayyash, 2020

This notebook shows how to Simulate synthetic bulk RNA-seq time series of $S =3$ samples, $D=  400$ genes at $T=6$ time points assuming three generative functions $f$: sine, cosine and cubic splines. The generated data from the three functions is exponentiated using exponential link function to set the mean of count data. Count data is sampled from negative binomial distribution parametrized by the probability of success $p=\frac{r}{\exp(f)+r}$ and number of failures $r =\frac{1}{dispersion} $.
'''

import pandas as pd
import numpy as np
from scipy.stats import nbinom
import random
from scipy.interpolate import CubicSpline
from tqdm import tqdm

S = 3 # number of samples/cells  
T = sorted([0,1,2,3,4,5]*S) # Time points

np.random.seed(0)
random.seed(0)

def sin_fun(x,AM): 
    return AM*np.sin(x) + ms # ms to increase the mean of the function 

def cos_fun(x,AM):
    return AM*np.cos(x) + ms # ms to increase the mean of the function 

def Cubic_spline(x,y):
    return CubicSpline(x, y)


def sample_from_NegativeBinomal(r,mean): 
    # r  number of failures
    # prob probability of success
    
    prob = r/(mean+r)
    y =[]
    for i in range(mean.shape[0]):
        y.append(nbinom.rvs(r, prob[i], size=1))
    y = np.vstack(y)    
    return y
    
def sample_count (fun_num,start):
    
    # sample dispersion parameters for dynamic function then for constant function 
    alpha = np.random.uniform(alpha_intervals[al][0],alpha_intervals[al][1],1) 
    alpha_constant = np.random.uniform(alpha_intervals[al][0],alpha_intervals[al][1],1) 
    
    if fun_num == 0:
        f = sin_fun(xtest,AM)
        
    if fun_num == 1:
        f = cos_fun(xtest,AM)

    if fun_num == 2:
        f = cs(xtest)  
       
    f = f + np.random.uniform(-1.,1.) # add randomness  
    # exponentiate generative function to set the mean of NB distribution and simulate differentially expressed genes
    f_sample = sample_from_NegativeBinomal(1./alpha,np.exp(f)).T
    
    # sample constant function from generative function
    constant_fun = np.linspace(np.min(f),np.max(f),100) 
    # shift constant function to create a variance for low counts
    if np.mean(constant_fun) < 2.0:
        shift = 2.
    else:
        shift = 0.
    
    # take the mean of the generative function to sample constant function 
    constant = np.ones(len(T))*(constant_fun[s]) + shift + (multiplier*.1) 
    #simulate non-differentially expressed genes from constant function    
    constant_sample = sample_from_NegativeBinomal(1./alpha_constant,np.exp(constant)).T
    
    return f_sample,constant_sample

def vstack_mean(samples):
    samples = np.vstack(samples)
    return np.mean(samples,axis=0).astype(int)

samples = []
samples_constant = []

alpha_intervals = [[.01,.2]  # low dispersion range
                  ,[5.,11.] # high dispersion range
                  ]   

mean_scale = [0,5] # low count and/or high count data 

for ms in tqdm(mean_scale):
    for al in range(len(alpha_intervals)):
        
        np.random.seed(0) # reset to sample new dataset
        random.seed(0)
        samples = []
        samples_constant = []

        for multiplier in range(10): # increase mean of count data 
            
            multiplier = multiplier + ms
            
            for i in range(8):
                
                fun_samples = []
                fun_constant_samples = []

                # cubic spline function with five points
                x = np.linspace(0+i,4+i,5)
                y = random.sample(range(ms+-2,ms+6), 5)
                
                cs = Cubic_spline(x,y)
                # Amplifier to extened the rane of sine or cosine function 
                AM = np.array(sorted(np.random.uniform(0., 5.+(multiplier*.1),len(T)))) 
               
                start = np.random.uniform(0.,1000.) # random start point of sine and coise function  
                xtest = np.array(sorted(np.random.uniform(start,start+13,len(T))))
                for j in range(3):
                    
                    if j==2:
                        xtest = np.array(sorted(np.random.uniform(x[0],x[4],len(T))))
                    for s in range(100): # generate 100 samples 

                        f_sample,f_constant = sample_count(j,start)
                        fun_samples.append(f_sample)
                        fun_constant_samples.append(f_constant)
                        
                    samples.append(vstack_mean(fun_samples))
                    samples_constant.append(vstack_mean(fun_constant_samples))
        
        samples = np.array(samples)
        samples = samples.astype(float)
        samples_constant = np.array(samples_constant)
        samples_constant = samples_constant.astype(float)
        
        if ms == 0:
            count_level = 'low_counts_'
        else:
            count_level = 'high_counts_'
        
        if al == 0:
            alpha_level = 'low_dispersion'
        else:
            alpha_level = 'high_dispersion'
            
        genes = ['gene_%s' % (s+1) for s in range(samples.shape[0])] 
        col = list(map(str,T))
        samples_df = pd.DataFrame(data=samples,index= genes,columns=col)
        samples_df_constant = pd.DataFrame(data=samples_constant,index= genes,columns=col)
        samples_df_all = pd.concat([samples_df, samples_df_constant])
        samples_df_all.to_csv(count_level+alpha_level+'.csv')
       
