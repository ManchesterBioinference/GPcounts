#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import time
import sys
args = sys.argv
percentage = args[1]
print(percentage)

from GPcounts.GPcounts_Module import Fit_GPcounts
X_real = pd.read_csv('pst_real_10.csv',index_col=[0])
Y = pd.read_csv('Cyclic_Index_10.csv',index_col=[0])
likelihood = 'Negative_binomial' 
gp_counts = Fit_GPcounts(X_real,Y,safe_mode = True) 
gp_counts = Fit_GPcounts(X_real,Y,M= int((int(percentage)*(len(X_real)))/100) ,sparse= True,safe_mode = True) 
log_likelihood = gp_counts.One_sample_test(likelihood)
log_likelihood.to_csv("ll_real_"+percentage+"_percentage_"+likelihood+'_pseudoT_Index_10.csv')

