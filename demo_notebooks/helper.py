#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from matplotlib import pyplot as plt
import statsmodels.api as sm
import numpy as np
import matplotlib.lines as mlines

def plot(params,test,likelihood,X,Y,sparse = False):
    
    indexes = Y.index.values # list of genes to be plotted 
    xtest = np.linspace(np.min(X)-.1,np.max(X)+.1,100)[:,None] # points to make prediction
    
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

def plotGene(ax, X, Y, label, colors=['b', 'r'], size=10, alpha=0.6):
    cellColors = [colors[int(i) - 1] for i in label]
    ax.scatter(X, Y, s=size, c=cellColors, alpha=alpha)

def plotBranching(d):
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    lowess = sm.nonparametric.lowess

    lik_name = {'Negative_binomial':'Negative binomial', 'Gaussian': 'Gaussian'}

    mu = d['mean']
    var = d['variance']
    Xnew = d['Xnew']
    testTimes = d['test_times']
    pt = d['MAP_model'].data[0]

    lower_lim = len(Xnew)
    upper_lim = 2 * lower_lim

    fig, ax = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [5, 3]})

    if d['likelihood'] == 'Gaussian':
        geneExpressionTitle = 'log(counts + 1)'
        upper = mu + 2 * np.sqrt(var)
        lower = mu - 2 * np.sqrt(var)
        ax[0].plot(Xnew[:, 0], mu[0:lower_lim, :], '-', lw=3, color='b')
        ax[0].fill_between(Xnew[:, 0], lower[0:lower_lim, :].numpy().reshape(-1), upper[0:lower_lim, :].numpy().reshape(-1),
                           color='blue', alpha=0.2)
        ax[0].plot(Xnew[:, 0], mu[lower_lim:upper_lim, :], '-', lw=3, color='r')
        ax[0].fill_between(Xnew[:, 0], lower[lower_lim:upper_lim, :].numpy().reshape(-1), upper[lower_lim:upper_lim, :].numpy().reshape(-1),
                           color='r', alpha=0.2)

    else:
        geneExpressionTitle = 'Counts'
        num_time_points = upper_lim
        percentile_5 = np.asarray(np.percentile(var, 5, axis=0)).reshape([num_time_points, 1])
        percentile_95 = np.asarray(np.percentile(var, 95, axis=0)).reshape([num_time_points, 1])

        ax[0].plot(Xnew[:, 0], mu[0:lower_lim], '-', lw=3, color='b')
        ax[0].fill_between(Xnew[:, 0], lowess(percentile_5[0:lower_lim, 0], Xnew[:, 0], frac=1. / 5, return_sorted=False),
                           lowess(percentile_95[0:lower_lim, 0], Xnew[:, 0], frac=1. / 5, return_sorted=False), color='blue',
                           alpha=0.2)
        ax[0].plot(Xnew, mu[lower_lim:upper_lim], '-', lw=3, color='r')
        ax[0].fill_between(Xnew[:, 0], lowess(percentile_5[lower_lim:upper_lim, 0], Xnew[:, 0], frac=1. / 5, return_sorted=False),
                           lowess(percentile_95[lower_lim:upper_lim, 0], Xnew[:, 0], frac=1. / 5, return_sorted=False), color='r',
                           alpha=0.2)

    plotGene(ax[0], pt[:, 0], d['MAP_model'].data[1], pt[:, 1], size=40, alpha=.6)
    ax[0].set_ylabel(geneExpressionTitle, fontsize=14)

    blue_line = mlines.Line2D([], [], color='blue', linewidth=3., label='Predicted Mean (branch 1)')
    red_line = mlines.Line2D([], [], color='red', linestyle='-', linewidth=3., label='Predicted Mean (branch 2)')
    blue_dot = mlines.Line2D([], [], color='blue', marker='o', markersize='8', linestyle='', label='branch 1')
    red_dot = mlines.Line2D([], [], color='red', marker='o', markersize='8', linestyle='', label='branch 2')
    ax[0].legend(handles=[blue_line, red_line, blue_dot, red_dot], bbox_to_anchor=(1.25, 0.8), loc=10, fontsize=14,
                 frameon=True)
    #     ax[0].legend(handles=[blue_line, red_line, blue_dot, red_dot], loc=2, fontsize=12)

    title = 'Gene: %s, Likelihood: %s, Branching evidence (log Bayes Factor): %.6f' % (
    *d['geneName'], lik_name[d['likelihood']], d['logBayesFactor'])
    ax[0].set_title(title, fontsize=14, pad=20)

    width = testTimes[1] - testTimes[0]
    ax[1].bar(testTimes, d['branching_probability'], color='royalblue', align='center', edgecolor="white", width=width)
    ax[1].set_xlabel('Pseudotime', fontsize=14)
    ax[1].set_ylabel('Branching probability', fontsize=14)
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, ax