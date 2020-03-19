#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import nbinom
import random
from scipy.interpolate import CubicSpline
from tqdm import tqdm
#import matplotlib.pyplot as plt


S = 3 # number of samples 
T = sorted([0,1,2,3,4,5]*S) # Time points

np.random.seed(0)
random.seed(0)

def sin_fun(x,AM):
    return AM*np.sin(x) + ms # to increase the mean of the function 

def cos_fun(x,AM):
    return AM*np.cos(x) + ms # to increase the mean of the function 

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
    
    # alpha level high or low
    alpha = np.random.uniform(alpha_intervals[al][0],alpha_intervals[al][1],1) 
    alpha_constant = np.random.uniform(alpha_intervals[al][0],alpha_intervals[al][1],1) 
    
    if fun_num == 0:
        f = sin_fun(xtest,AM)
        
    if fun_num == 1:
        f = cos_fun(xtest,AM)

    if fun_num == 2:
        f = cs(xtest) 
    
       
    f = f + np.random.uniform(-1.,1.) # add randomness  
    #ax = fig.add_subplot(4,1,fun_num+1)
    #ax.plot(xtest, f) 
    f_sample = sample_from_NegativeBinomal(1./alpha,np.exp(f)).T
        
    constant_fun = np.linspace(np.min(f),np.max(f),100) # 
    #print(np.mean(constant_fun))
    if np.mean(constant_fun) < 2.0:
        shift = 2.
    else:
        shift = 0.
    constant = np.ones(len(T))*(constant_fun[s]) + shift + (multiplier*.1)
    #ax.plot(xtest, constant) 
    constant_sample = sample_from_NegativeBinomal(1./alpha_constant,np.exp(constant)).T
    
    return f_sample,constant_sample

def vstack_mean(samples):
    samples = np.vstack(samples)
    return np.mean(samples,axis=0).astype(int)

samples = []
samples_constant = []

alpha_intervals = [[.01,.2]  # low dispersion
                  ,[5.,11.]
                  ]   # high dispersion

mean_scale = [0,5] # low count and/or high count data 

for ms in tqdm(mean_scale):
    for al in range(len(alpha_intervals)):
        
        np.random.seed(0) # reset for new dataset
        random.seed(0)
        samples = []
        samples_constant = []

        for multiplier in range(10): # increase mean of count data ## 20 or 10
            
            multiplier = multiplier + ms
            
            for i in range(6): ## 5  
                #print(i)
                #fig = plt.figure(figsize=(6,15))
                
                fun_samples = []
                fun_constant_samples = []

                # cubic spline function with four points
                x = np.linspace(0+i,4+i,5)
                y = random.sample(range(ms+-2,ms+6), 5)
                #np.random.uniform(-2+ ms,6.+ ms,4)
                #print(y)
                cs = Cubic_spline(x,y)
                AM = np.array(sorted(np.random.uniform(0., 5.+(multiplier*.1),len(T)))) # Amplifier
                #print(AM)
                start = np.random.uniform(0.,1000.) # random start point of sine and coise function  
                xtest = np.array(sorted(np.random.uniform(start,start+13,len(T))))
                for j in range(3):
                    #print(j)
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
            alpha_level = 'low_dispersion_'
        else:
            alpha_level = 'high_dispersion_'
            
        #print(count_level+alpha_level+'sample') 
        
        genes = ['gene_%s' % (s+1) for s in range(samples.shape[0])] 
        col = list(map(str,T))
        samples_df = pd.DataFrame(data=samples,index= genes,columns=col)
        samples_df.to_csv(count_level+alpha_level+'differentially_expressed_genes.csv')
        samples_df_constant = pd.DataFrame(data=samples_constant,index= genes,columns=col)
        samples_df_constant.to_csv(count_level+alpha_level+'non_differentially_expressed_genes.csv')
        
