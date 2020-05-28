import os 
import numpy as np
import tensorflow as tf
import gpflow
#from branchingKernel import BranchKernel
from GPcounts import NegativeBinomialLikelihood
from sklearn.cluster import KMeans
import scipy.stats as ss
from pathlib import Path
import pandas as pd
from gpflow.utilities import set_trainable
from tqdm import tqdm
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
import statsmodels.api as sm

# Get number of cores reserved by the batch system (NSLOTS is automatically set, or use 1 if not)
NUMCORES=int(os.getenv("NSLOTS",1))
# print("Using", NUMCORES, "core(s)" )
# Create session properties
config=tf.compat.v1.ConfigProto(inter_op_parallelism_threads=NUMCORES,intra_op_parallelism_threads=NUMCORES)
tf.compat.v1.Session.intra_op_parallelism_threads = NUMCORES
tf.compat.v1.Session.inter_op_parallelism_threads = NUMCORES

class Fit_GPcounts(object):
    
    def __init__(self,X = None,Y= None,sparse = False):
        
        
        self.sparse = sparse # use sparse inference 
      
        self.X = None # time points
        self.M = None # number of inducing point
        self.Z = None # inducing points
        self.Y = None # gene expression matrix
        self.Y_copy = None #copy of gene expression matrix
        self.D =  None # number of genes
        self.N = None # number of cells
        self.genes_name = None
        self.cells_name = None  
        
        if (X is None) or (Y is None):
            print('TypeError: GPcounts() missing 2 required positional arguments: X and Y')
        else:
            self.set_X_Y(X,Y) 
       
        self.seed_value = 0 
        self.count_fix = 0 # number of trails to resolve bad 
        self.fail = []
        self.y = None  
        self.index = None
        self.lik_name = None
        self.hyper_parameters = {} # model paramaters initialization
        self.models_number = None # Total number of models to fit for single gene
        self.model_index = None # index the current model
        self.model = None      
        self.branching = None # DE kernel or RBF kernel
        self.xp = -1000. # Put branching time much earlier than zero time
        self.var = None 
        self.mean = None
        self.fix = False # fix kernel hyper-parameters
        # save likelihood parameters to initialize constant model
        self.lik_alpha = None 
        self.lik_km = None
        self.optimize = True # do not optimize model 
        
    def set_X_Y(self,X,Y):
        
        if X.shape[0] == Y.shape[1]:
            self.X = X
            self.cells_name = self.X.index.values # gene expression name
            self.X = X.values.astype(float) # time points 
            self.X = self.X.reshape([-1,1])
            if self.sparse:
                self.M = int((10*(len(self.X)))/100) # number of inducing point
                # set inducing points by Kmean cluster
                kmeans = KMeans(n_clusters= self.M, random_state=0).fit(self.X)
                self.Z = kmeans.cluster_centers_
                self.Z = np.sort(self.Z,axis=None).reshape([self.M,1])
                self.Z = self.Z.reshape([self.Z.shape[0],1])
             
            self.Y = Y
            self.genes_name = self.Y.index.values.tolist() # gene expression name
            self.Y = self.Y.values.astype(int) # gene expression matrix
            self.Y_copy = self.Y
            self.D = Y.shape[0] # number of genes
            self.N = Y.shape[1] # number of cells
        else:
            print('InvalidArgumentError: Dimension 0 in X shape must be equal to Dimension 1 in Y, but shapes are %d and %d.' %(X.shape[0],Y.shape[1]))
    
    def Infer_trajectory(self,lik_name= 'Negative_binomial'):
        log_likelihood = self.run_test(lik_name,1)
        return log_likelihood
        
    def One_sample_test(self,lik_name= 'Negative_binomial'):
        log_likelihood = self.run_test(lik_name,2)
        return log_likelihood
        
    def Two_samples_test(self,lik_name= 'Negative_binomial'):
        log_likelihood = self.run_test(lik_name,3)
        return log_likelihood
    
    def Infer_trajectory_with_branching_kernel(self,cell_labels,bins_num = 50,lik_name= 'Negative_binomial'):
         # note how to implement this    
        cell_labels = np.array(cell_labels)
        self.X = np.c_[self.X,cell_labels[:,None]]
        log_likelihood = self.run_test(lik_name,1,branching = True)
        return log_likelihood
     
    # Run the selected test and get likelihoods for all genes   
    def run_test(self,lik_name,models_number,branching = False):
        genes_log_likelihoods = {}
        self.Y = self.Y_copy
        self.models_number = models_number
        self.lik_name = lik_name
        self.optimize = True
        
        if self.models_number == 1:
            column_name = ['Dynamic_model_log_likelihood']
        elif self.models_number == 2:
            column_name = ['Dynamic_model_log_likelihood','Constant_model_log_likelihood','log_likelihood_ratio']
        else:
            column_name = ['Shared_log_likelihood','model_1_log_likelihood','model_2_log_likelihood','log_likelihood_ratio'] 
            
        
        for i in tqdm(range(self.D)):
            
            self.index = i
            self.y = self.Y[self.index].astype(float)
            self.y = self.y.reshape([-1,1])
      
            self.model_index = 1  
            model_1_log_likelihood = self.fit_model()
            log_likelihood =  [model_1_log_likelihood]
            
            if self.models_number == 2:
                if not(np.isnan(model_1_log_likelihood)):
                    ls , km_1 = self.record_hyper_parameters()

                    self.model_index = 2
                    model_2_log_likelihood = self.fit_model()
                    
                    if not(np.isnan(model_2_log_likelihood)):
                        if self.lik_name == 'Zero_inflated_negative_binomial':
                            km_2 = self.model.likelihood.km.numpy()

                         #test local optima case 2 
                        ll_ratio = model_1_log_likelihood - model_2_log_likelihood

                        if (ls < (10*(np.max(self.X)-np.min(self.X)))/100 and round(ll_ratio) <= 0) or (self.lik_name == 'Zero_inflated_negative_binomial'  and np.abs(km_1 - km_2) > 50.0 ):
                            self.model_index = 1
                            self.init_hyper_parameters(True)
                            model_1_log_likelihood = self.fit_model()
                            
                            if not(np.isnan(model_1_log_likelihood)):
                                ls , km_1 = self.record_hyper_parameters()
                            self.model_index = 2
                            self.init_hyper_parameters(True)
                            model_2_log_likelihood = self.fit_model()
                            
                        ll_ratio = model_1_log_likelihood - model_2_log_likelihood
                        
                if np.isnan(model_1_log_likelihood) or np.isnan(model_2_log_likelihood):
                    model_2_log_likelihood = np.nan
                    ll_ratio = np.nan
                
                log_likelihood =  [model_1_log_likelihood,model_2_log_likelihood,ll_ratio] 
                
            if self.models_number == 3:
                
                X_df = pd.DataFrame(data=self.X,index= self.cells_name,columns= ['times'])
                Y_df = pd.DataFrame(data=self.Y_copy,index= self.genes_name,columns= self.cells_name)  
                
                # initialize X and Y with first time series
                self.set_X_Y(X_df[0 : int(self.N/2)],Y_df.iloc[:,0:int(self.N/2)])
                self.y = self.Y[self.index].astype(float)
                self.y = self.y.reshape([self.N,1])
                
                self.model_index = 2
                model_2_log_likelihood = self.fit_model() 
                
                # initialize X and Y with second time series
                self.set_X_Y(X_df[self.N : :],Y_df.iloc[:,int(self.N) : :])
                self.y = self.Y[self.index].astype(float)
                self.y = self.y.reshape([self.N,1])
                
                self.model_index = 3
                model_3_log_likelihood = self.fit_model()
                
                self.set_X_Y(X_df,Y_df)
                
                if np.isnan(model_1_log_likelihood) or np.isnan(model_2_log_likelihood) or np.isnan(model_3_log_likelihood): 
                    ll_ratio = np.nan
                else:
                    ll_ratio = ((model_2_log_likelihood+model_3_log_likelihood)-model_1_log_likelihood)
                    
                log_likelihood = [model_1_log_likelihood,model_2_log_likelihood,model_3_log_likelihood,ll_ratio]
                
            genes_log_likelihoods[self.genes_name[self.index]] = log_likelihood
            
        return {'likelihood':pd.DataFrame.from_dict(genes_log_likelihoods, orient='index', columns= column_name),'fail':self.fail}
    
    # fit a Gaussian process, get the log likelihood and save the model for a gene 
    def fit_model(self):
         
        successed_fit = self.fit_GP()
        
        if successed_fit:
            if self.sparse:
                log_likelihood = self.model.log_posterior_density((self.X,self.y)).numpy()
            else:
                log_likelihood = self.model.log_posterior_density().numpy()
            filename = self.get_file_name()
            ckpt = tf.train.Checkpoint(model=self.model, step=tf.Variable(1))
            ckpt.write(filename)
            
        else:
            log_likelihood = np.nan  # set to Nan in case of Cholesky decomposition failure
            self.model = np.nan
           
        return log_likelihood
    
    # fit a GP, fix Cholesky decomposition by random initialization if detected and test case1       
    def fit_GP(self,reset = False):
        self.init_hyper_parameters(reset)
        #print(self.hyper_parameters)
        fit = True   
        try:
            fit = self.fit_GP_with_likelihood()
            
        except tf.errors.InvalidArgumentError as e:
            #print(self.count_fix)
            self.fail.append(self.genes_name[self.index])
            if self.count_fix < 10:
                print('Fit Cholesky decomposition was not successful.')
                self.count_fix = self.count_fix +1 
                fit = self.fit_GP(True)
                
            else:
                print('Can not fit a Gaussian process, Cholesky decomposition was not successful.')
                fit = False
                
        if fit and self.optimize:
            self.test_local_optima_case1()
        return fit
    
    # Fit a GP with selected kernel,likelihood,run it as sparse or full GP 
    def fit_GP_with_likelihood(self):
        fit = True
       
        if self.hyper_parameters['ls'] == 1000:
            k = gpflow.kernels.Constant(variance= self.hyper_parameters['var']) 
        else:
            k = gpflow.kernels.RBF( variance= self.hyper_parameters['var'],
                               lengthscales = self.hyper_parameters['ls'])     
        if self.branching: #select kernel
            if self.fix:
                set_trainable(k.lengthscales,False)
                set_trainable(k.variance,False)
            kern = BranchKernel(k,self.xp)
        else:
            kern = k

        #select likelihood
        if self.lik_name == 'Poisson':
            likelihood = gpflow.likelihoods.Poisson()

        if self.lik_name == 'Negative_binomial':
            likelihood = NegativeBinomialLikelihood.NegativeBinomial(self.hyper_parameters['alpha'])
            if self.hyper_parameters['ls'] == 1000:
                set_trainable(likelihood.alpha,False) 

        if self.lik_name == 'Zero_inflated_negative_binomial':
            alpha = self.hyper_parameters['alpha']
            km = self.hyper_parameters['km']
            likelihood = NegativeBinomialLikelihood.ZeroInflatedNegativeBinomial(alpha,km)

        # Run model with selected kernel and likelihood  
        if self.lik_name == 'Gaussian':
            self.y = np.log(self.y+1)
            
            if self.sparse:
                self.model =  gpflow.models.SGPR((self.X,self.y), kernel=kern,inducing_variable=self.Z)
            else:
                self.model = gpflow.models.GPR((self.X,self.y), kern)
                
            training_loss = self.model.training_loss
        else:
            
            
            if self.sparse:
                self.model = gpflow.models.SVGP( kern ,likelihood,self.Z) 
                training_loss = self.model.training_loss_closure((self.X, self.y.astype(float)))
         
            else:
                self.model = gpflow.models.VGP((self.X, self.y) , kern , likelihood) 
                training_loss = self.model.training_loss
                
        if self.optimize:
            o = gpflow.optimizers.Scipy()
            res = o.minimize(training_loss, variables=self.model.trainable_variables,options=dict(maxiter=5000))
            
            if not(res.success):
                self.fail.append(self.genes_name[self.index])
                if self.count_fix < 10:
                    print('Optimization fail.')
                    self.count_fix = self.count_fix +1 
                    fit = self.fit_GP(True)
                    
                else:
                    print('Can not Optimaize a Gaussian process, Optimization fail.')
                    #fit = False
        return fit

    def record_hyper_parameters(self):
        ls = self.model.kernel.lengthscales.numpy() # to detect local optima case2
        km_1 = 0
        # copy likelihood parameters and use them to fit constant model
        if self.lik_name == 'Negative_binomial':
            self.lik_alpha  = self.model.likelihood.alpha.numpy()
        if self.lik_name == 'Zero_inflated_negative_binomial':
            km_1 = self.lik_km = self.model.likelihood.km.numpy()
            self.lik_alpha  = self.model.likelihood.alpha.numpy()
            
        return ls,km_1

    def get_file_name(self):
        dir_name = 'GPcounts_models/'

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        filename = dir_name+self.lik_name+'_'

        if self.sparse:
                filename += 'sparse_'
        
        if self.models_number == 3:
            filename += 'tst_'  
        
        filename += self.genes_name[self.index]+'_model_'+str(self.model_index)
        return filename
    
    # Hyper-parameters initialization
    def init_hyper_parameters(self,reset = False):
       
        self.hyper_parameters = {}
        
        if reset:
            self.seed_value = self.seed_value + 1
            np.random.seed(self.seed_value)
            self.hyper_parameters['ls'] = np.random.uniform((1*(np.max(self.X)-np.min(self.X)))/100 ,(30*(np.max(self.X)-np.min(self.X)))/100)
            self.hyper_parameters['var'] = np.random.uniform((1*(np.max(self.X)-np.min(self.X)))/100 ,(25*(np.max(self.X)-np.min(self.X)))/100)

            self.hyper_parameters['alpha'] = np.random.uniform(0., 10.)
            self.hyper_parameters['km'] = np.random.uniform(0., 100.)       

        else:   
            self.seed_value = 0
            self.count_fix = 0 
            np.random.seed(self.seed_value)
            self.hyper_parameters['ls'] = (10*(np.max(self.X)-np.min(self.X)))/100
            self.hyper_parameters['var'] = (15*(np.max(self.X)-np.min(self.X)))/100
            self.hyper_parameters['alpha'] = 5.
            self.hyper_parameters['km'] = 35.  

            
        if self.model_index == 2 and self.models_number == 2:
            self.hyper_parameters['ls'] = 1000. # constant kernel
            
            if self.optimize:
                if self.lik_name == 'Negative_binomial':
                    self.hyper_parameters['alpha'] = self.lik_alpha
            
        else:
            #save likelihood parameters to initialize constant model
            self.lik_alpha = None 
            self.lik_km = None
            self.fix = False # fix kernel hyper-parameters    
       
        tf.compat.v1.get_default_graph()
        tf.compat.v1.set_random_seed(self.seed_value)
        tf.random.set_seed(self.seed_value)
        gpflow.config.set_default_float(np.float64)
         
        self.y = self.Y[self.index].astype(float)
        self.y = self.y.reshape([-1,1])   
        self.model = None
        self.var = None 
        self.mean = None   
        self.xp = -1000. # Put branching time much earlier than zero time
       
   
    def test_local_optima_case1(self):
        if self.sparse:
            x = self.Z
        else:
            x = self.X 
        xtest = np.linspace(np.min(x),np.max(x),100)[:,None]
        if self.lik_name == 'Gaussian':
            mean, var = self.model.predict_y(xtest)
            self.mean = mean.numpy()
            self.var = var.numpy()
        else:
            # mean of posterior predictive samples
            self.mean,self.var = self.samples_posterior_predictive_distribution(xtest)         
        
        if self.count_fix < 5: # limit number of trial to fix bad solution 
           
            y_mean = np.mean(self.y)
            mean_mean = np.mean(self.mean) 
            #print('y_mean',y_mean)
            #print('mean_mean',mean_mean)
            y_max = np.max(self.y)
            mean_max = np.max(self.mean)
            #print('y_max',y_max)
            #print('mean_max',mean_max)
            y_min = np.abs(np.min(self.y))
            mean_min = np.abs(np.min(self.mean))
            #print('y_min',y_min)
            #print('mean_min',mean_min)
            #print('abs(round((mean_mean-y_mean)/y_mean))',round((mean_mean-y_mean)/y_mean))
                
            if self.N < 100:
                diff = 0
            else:
                diff = 1
                
            if y_mean > 0.0 and (mean_max > y_max or mean_min < y_min):
                if abs(round((mean_mean-y_mean)/y_mean)) > 0 or mean_mean == 0.0:
                    
                    print('local Optima')
                    print(self.model_index)
                    print('y_mean',y_mean)
                    print('mean_mean',mean_mean)
                    print('abs(round((mean_mean-y_mean)/y_mean))',abs(round((mean_mean-y_mean)/y_mean)))
                    
                    self.fail.append(self.genes_name[self.index])
                    #self.plot(xtest)
                    self.count_fix = self.count_fix +1 
                    fit = self.fit_GP(True)
    
    def generate_Samples_from_distribution(self,mean):
        
        y = []
        if self.lik_name == 'Poisson':
            print(mean)
            for i in range(mean.shape[0]):
                y.append(ss.poisson.rvs(mean[i], size = 500))

        if self.lik_name == 'Negative_binomial':
            r = 1./self.model.likelihood.alpha.numpy()  # r  number of failures
            prob = r / (mean+ r)   # p probability of success
            for i in range(mean.shape[0]):
                y.append(ss.nbinom.rvs(r, prob[i], size = 500))

        if self.lik_name == 'Zero_inflated_negative_binomial':
            r = 1./self.model.likelihood.alpha.numpy()  # r  number of failures
            prob = r / (mean+ r)   # p probability of success
            km = self.model.likelihood.km.numpy() # Michaelin-Menten (MM) constant
            psi = 1.- (mean/(km+mean)) # psi probability of zeros
            for i in range(mean.shape[0]):
                B = ss.bernoulli.rvs(size=1,p = 1-psi[i])
                if B == 0:
                    y.append(np.zeros(500))
                else:
                    y.append(ss.nbinom.rvs(r, prob[i], size = 500))       
        y = np.vstack(y)
        return y

    def samples_posterior_predictive_distribution(self,xtest):
        
        var = []
        f_samples = []
        for i in range(20):
            f_samples.append(self.model.predict_f_samples(xtest, 5))
            f = np.vstack(f_samples)
            link_f = np.exp(f[:, :, 0])
            var.append(self.generate_Samples_from_distribution(np.mean(link_f, 0)).T)

        var = np.vstack(var)
        
        mean = savgol_filter(np.mean(var,axis = 0), int(xtest.shape[0]/2)+1, 3)
        mean = [(i > 0) * i for i in mean]
        return mean,var
  
    def load_models(self,genes,test,xtest,lik_name = 'Negative_binomial'):
        # note that RBF kernel and branching kernel is written in same way         
        params = {}
        genes_models = []
        genes_means = []
        genes_vars = []
        self.Y = self.Y_copy
        self.lik_name = lik_name
        
        if test == 'One_sample_test':
            self.models_number = 2
        elif test == 'Two_samples_test':
            self.models_number = 3
        else:
            self.models_number = 1
           
        for gene in tqdm(genes):
            models = []
            means = []
            variances = []
            self.index = self.genes_name.index(gene)
           
            for model_index in range(self.models_number):
               
                self.optimize = False
                self.model_index = model_index + 1
                self.init_hyper_parameters(False) 
                file_name = self.get_file_name()
               
                if self.models_number == 3:
                    X_df = pd.DataFrame(data=self.X,index= self.cells_name,columns= ['times'])
                    Y_df = pd.DataFrame(data=self.Y_copy,index= self.genes_name,columns= self.cells_name)
                    
                    if model_index == 0:
                        self.set_X_Y(X_df,Y_df)
                        
                    if model_index == 1: # initialize X and Y with first time series
                        self.set_X_Y(X_df[0 : int(self.N/2)],Y_df.iloc[:,0:int(self.N/2)])

                    if model_index == 2: # initialize X and Y with second time series
                        self.set_X_Y(X_df[int(self.N/2) : :],Y_df.iloc[:,int(self.N/2) : :])
                    
                self.y = self.Y[self.index]
                self.y = self.y.reshape([self.N,1])
                successed_fit = self.fit_GP()
                # restore check point
                if successed_fit:
                    ckpt = tf.train.Checkpoint(model=self.model, step=tf.Variable(1))
                    ckpt.restore(file_name)

                    if self.lik_name == 'Gaussian':
                        mean, var = self.model.predict_y(xtest)
                        mean = mean.numpy()
                        var = var.numpy()

                    else:
                        mean, var = self.samples_posterior_predictive_distribution(xtest)
                else:
                    mean = var = 0

                means.append(mean)
                variances.append(var)
                models.append(self.model)

                if self.models_number == 3 and model_index > 0:
                    self.set_X_Y(X_df,Y_df)
            
            genes_means.append(means)
            genes_vars.append(variances)
            genes_models.append(models)
            
        params['means']= genes_means
        params['vars'] = genes_vars
        params['models']= genes_models

        return params
'''
    def plot(self,xtest):
        plt.tick_params(labelsize='large', width=2)     
        #plt.ylabel('Gene Expression', fontsize=16)
        #plt.xlabel('Times', fontsize=16)
        c = 'royalblue'

        if self.model_index == 3:
            c = 'green'

        plt.plot(xtest, self.mean,color= c, lw=2) 

        if self.lik_name == 'Gaussian':
            plt.fill_between(xtest[:,0],
                                self.mean[:,0] - 1*np.sqrt(self.var[:,0]),
                                self.mean[:,0] + 1*np.sqrt(self.var[:,0]),color=c,alpha=0.2) # one standard deviation
            plt.fill_between(xtest[:,0],
                                self.mean[:,0] - 2*np.sqrt(self.var[:,0]),
                                self.mean[:,0] + 2*np.sqrt(self.var[:,0]),color=c, alpha=0.1)# two standard deviation
        else:

            lowess = sm.nonparametric.lowess    
            # one standard deviation 68%
            percentile_16 = lowess(np.percentile(self.var, 16, axis=0),xtest[:,0],frac=1./5, return_sorted=False)
            percentile_16 = [(i > 0) * i for i in percentile_16]
            percentile_84 = lowess(np.percentile(self.var, 84, axis=0),xtest[:,0],frac=1./5, return_sorted=False)
            percentile_84 = [(i > 0) * i for i in percentile_84]
            plt.fill_between(xtest[:,0],percentile_16,percentile_84,color=c,alpha=0.2)

            # two standard deviation 95%
            percentile_5 = lowess(np.percentile(self.var, 5, axis=0),xtest[:,0],frac=1./5, return_sorted=False)
            percentile_5 = [(i > 0) * i for i in percentile_5]
            percentile_95 = lowess(np.percentile(self.var,95, axis=0),xtest[:,0],frac=1./5, return_sorted=False)
            percentile_95 = [(i > 0) * i for i in percentile_95]
            plt.fill_between(xtest[:,0],percentile_5,percentile_95,color=c,alpha=0.1)

        if self.models_number == 3 and self.model_index == 1:
            plt.scatter(self.model.data[0][0:int(self.model.data[0].shape[0]/2)],model.data[1][0:int(model.data[0].shape[0]/2)], s=30, marker='o', color= 'royalblue',alpha=1.) #data    
            plt.scatter(model.data[0][int(model.data[0].shape[0]/2)::],model.data[1][int(model.data[0].shape[0]/2)::], s=30, marker='o', color= 'green',alpha=1.) #data

        else: 
            plt.scatter(self.model.data[0],self.model.data[1],s=30,marker = 'o',color=c,alpha=1.)

        if not(self.models_number == 3 and self.model_index == 2):
            plt.show()
'''         




