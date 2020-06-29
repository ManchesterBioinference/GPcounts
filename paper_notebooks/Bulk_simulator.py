'''
Nuha BinTayyash, 2020

This notebook shows how to Simulate synthetic bulk RNA-seq time series of $S =3$ replicate, $D=  400$ genes measured at $T=6$ time points. Half of the genes are differentially expressed across time. We use three classes of generative functions $f$ (sine, cosine and cubic splines) to simulate differentially expressed genes and we include a matching non-differentially expressed gene with the same mean in the synthetic data. The sine and cosine functions are of the form $f(x) = a\sin(xb)+c$ with $x\in[-1,1]$. The cubic spline function has the form $f(x) \in C^2 [a,b]$ passing through $n = 5 $ data points $(x,y)$ where $x \in [a,b]$ and $y \in [-5+c,4+c]$. The $[a, b, c]$ are drawn from uniform distributions to vary the amplitude, lengthscale and mean. The low and high dispersion values are drawn from uniform distributions $\alpha_\mathrm{low} = U[.05,.1]$ and $\alpha_\mathrm{high} = U[8,10]$. An exponential inverse-link function is used to determine the mean of count data at each time $\mu(x) = e^{f(x)}$ and we use the Scipy library to sample counts from the negative binomial distribution parametrized by the probability of success $p=\frac{r}{\exp(f)+r}$ and number of failures $r =\frac{1}{dispersion} $ (\citealp{millman2011python}). 

 
Bulk_simulator.py used to simulated four datasets with two levels of the mean of count data (high count and low count datasets) and two levels of dispersion (high dispersion and low dispersion).

'''

import pandas as pd
import numpy as np
from scipy.stats import nbinom
import random
from scipy.interpolate import CubicSpline
from tqdm import tqdm
#import matplotlib.pyplot as plt

S = 3 # number of samples/cells  
T = sorted([0,1,2,3,4,5]*S) # Time points

# c to increase the mean of the function 
# c_con to shift the constant function 
def sin_fun(): 
    f = a*np.sin(xtest) + c 
    f_con = np.sin(xtest_con) + c 
    return f,f_con 

def cos_fun(): 
    f = a*np.cos(xtest) + c 
    f_con = np.cos(xtest_con) + c 
    return f,f_con 

def Cubic_spline():
    f = CubicSpline(x,y)
    f_con = CubicSpline(x_con,y_con)
    return f(xtest),f_con(xtest_con)

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
    
    # sample dynamic and constant function from the generative function
    if fun_num == 0:
        f,f_con = sin_fun()
              
    if fun_num == 1:
        f,f_con = cos_fun()
        
    if fun_num == 2:
        f,f_con = Cubic_spline()
          
    f = f + np.random.uniform(-1.,1.) # add randomness 
    # exponentiate generative function to set the mean of NB distribution and simulate differentially expressed genes
    f_sample = sample_from_NegativeBinomal(1./alpha,np.exp(f)).T
    # generate random shift to increase the variance of the constant function
    c_con = np.random.uniform(-10,4)
    f_con = f_con +(multiplier*.1)+ c_con  
    #simulate non-differentially expressed genes from constant function    
    constant_sample = sample_from_NegativeBinomal(1./alpha_constant,np.exp(f_con)).T
   
    return f_sample,constant_sample

def vstack_mean(samples):
    samples = np.vstack(samples)
    return np.mean(samples,axis=0).astype(int)

samples = []
samples_constant = []

alpha_intervals = [
                  [.05,.1]  # low dispersion range
                  ,[7.,11.] # high dispersion range
                  ]   

mean_scale = [0,6] # low count and/or high count data 

for c in tqdm(mean_scale): # c to change the mean 
    for al in range(len(alpha_intervals)):
        np.random.seed(0) # reset to sample new dataset
        random.seed(0)
        samples = []
        samples_constant = []

        for multiplier in range(10): # increase mean of count data 
           
            multiplier = multiplier + c
            np.random.seed(multiplier) # reset to sample new dataset
            random.seed(multiplier)
     
            for i in range(10):# 8
               
                fun_samples = []
                fun_constant_samples = []

                # cubic spline function with five points
                x = np.linspace(0+i,4+i,5)
                y = random.sample(range(-5+c,4+c), 5) 
                
                x_con =np.linspace(1*i+1, 1*i+11,15)
                y_con = np.sin(x_con) + c + (multiplier*.1)
                
                # Amplifier to extened the rane of sine or cosine function 
                a = np.array(sorted(np.random.uniform(0., 4.+(multiplier*.1),len(T)))) 
               
                start = np.random.uniform(0.,1000.) # random start point of sine and coise function  
                xtest = np.array(sorted(np.random.uniform(start,start+13,len(T))))
                xtest_con = np.array(sorted(np.random.uniform(start,start+30,len(T))))
                 
                for j in range(3):
                    #fig = plt.figure()
                    if j==2:
                        xtest = np.array(sorted(np.random.uniform(x[0],x[4],len(T))))
                        xtest_con = np.array(sorted(np.random.uniform(x_con[0],x_con[14],len(T))))
                    for s in range(100): # generate 100 samples 
                        f_sample,f_constant = sample_count(j,start)
                        fun_samples.append(f_sample)
                        fun_constant_samples.append(f_constant)
                    
                    samples.append(vstack_mean(fun_samples))
                    samples_constant.append(vstack_mean(fun_constant_samples))
                    #plt.show()
                
        samples = np.array(samples)
        samples = samples.astype(float)
        samples_constant = np.array(samples_constant)
        samples_constant = samples_constant.astype(float)
        
        if c == 0:
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
        samples_df_all.to_csv('../data/'+count_level+alpha_level+'.csv')
       
