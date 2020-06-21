#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from matplotlib import pyplot as plt
import statsmodels.api as sm
import numpy as np


def plot(gp_counts,test,likelihood,X,Y,sparse = False):
    indexes = Y.index.values # list of genes to be plotted 
    xtest = np.linspace(np.min(X)-.1,np.max(X)+.1,100)[:,None] # points to make prediction

    params = gp_counts.load_models(indexes,test,xtest,likelihood)
    
    for i in range(len(indexes)):
        fig = plt.figure()
        print(indexes[i])
        model_index = 1
        y = Y.iloc[i].values # genes expression data
        if likelihood == 'Gaussian':
                y = np.log(y+1)
        for mean,var,model in zip(params['means'][i],params['vars'][i],params['models'][i]):
           
            plt.tick_params(labelsize='large', width=2)     
            plt.ylabel('Gene Expression', fontsize=16)
            plt.xlabel('Times', fontsize=16)
            c = 'royalblue'

            if model_index == 3:
                c = 'green'

            plt.plot(xtest, mean,color= c, lw=2) 

            if likelihood == 'Gaussian':
                plt.fill_between(xtest[:,0],
                                    mean[:,0] - 1*np.sqrt(var[:,0]),
                                    mean[:,0] + 1*np.sqrt(var[:,0]),color=c,alpha=0.2) # one standard deviation
                plt.fill_between(xtest[:,0],
                                    mean[:,0] - 2*np.sqrt(var[:,0]),
                                    mean[:,0] + 2*np.sqrt(var[:,0]),color=c, alpha=0.1)# two standard deviation
            else:

                lowess = sm.nonparametric.lowess    
                # one standard deviation 68%
                percentile_16 = lowess(np.percentile(var, 16, axis=0),xtest[:,0],frac=1./5, return_sorted=False)
                percentile_16 = [(i > 0) * i for i in percentile_16]
                percentile_84 = lowess(np.percentile(var, 84, axis=0),xtest[:,0],frac=1./5, return_sorted=False)
                percentile_84 = [(i > 0) * i for i in percentile_84]
                plt.fill_between(xtest[:,0],percentile_16,percentile_84,color=c,alpha=0.2)

                # two standard deviation 95%
                percentile_5 = lowess(np.percentile(var, 5, axis=0),xtest[:,0],frac=1./5, return_sorted=False)
                percentile_5 = [(i > 0) * i for i in percentile_5]
                percentile_95 = lowess(np.percentile(var,95, axis=0),xtest[:,0],frac=1./5, return_sorted=False)
                percentile_95 = [(i > 0) * i for i in percentile_95]
                plt.fill_between(xtest[:,0],percentile_5,percentile_95,color=c,alpha=0.1)

            if test == 'Two_samples_test':
                if model_index == 1:
                    plt.scatter(X[0:int(X.shape[0]/2)],y[0:int(X.shape[0]/2)], s=30, marker='o', color= 'royalblue',alpha=1.) #data    
                    plt.scatter(X[int(X.shape[0]/2)::],y[int(X.shape[0]/2)::], s=30, marker='o', color= 'green',alpha=1.) #data
                elif model_index == 2:
                     plt.scatter(X[0:int(X.shape[0]/2)],y[0:int(X.shape[0]/2)], s=30, marker='o', color= 'royalblue',alpha=1.) #data    
                else:
                    plt.scatter(X[int(X.shape[0]/2)::],y[int(X.shape[0]/2)::], s=30, marker='o', color= 'green',alpha=1.) #data
           
            else: 
                plt.scatter(X,y,s=30,marker = 'o',color=c,alpha=1.)
            if sparse:
                inducing_points = model.inducing_variable.Z.numpy() 
                # discard any inducing points outside X range
                inducing_points = inducing_points[inducing_points >= np.min(X)]
                inducing_points = inducing_points[inducing_points <= np.max(X)]
                plt.scatter(inducing_points,np.zeros(inducing_points.shape[0]),s=30,marker = '^',color='red',label='inducing points',alpha=1.) 
                plt.legend(loc='upper center', bbox_to_anchor=(1.20, 0.1))
            
            if not(test == 'Two_samples_test' and model_index == 2):
                plt.show()
            
            model_index = model_index + 1

