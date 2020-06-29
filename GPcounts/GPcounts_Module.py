import os 
import numpy as np
import tensorflow as tf
import gpflow
from GPcounts import branchingKernel
from GPcounts import NegativeBinomialLikelihood
from sklearn.cluster import KMeans
import scipy.stats as ss
from pathlib import Path
import pandas as pd
from gpflow.utilities import set_trainable
from tqdm import tqdm
from scipy.signal import savgol_filter
import random
import time
import datetime
from .utilities import qvalue
import scipy as sp
from scipy import interpolate

#from matplotlib import pyplot as plt
#import statsmodels.api as sm

# Get number of cores reserved by the batch system (NSLOTS is automatically set, or use 1 if not)
NUMCORES=int(os.getenv("NSLOTS",1))
# print("Using", NUMCORES, "core(s)" )
# Create session properties
config=tf.compat.v1.ConfigProto(inter_op_parallelism_threads=NUMCORES,intra_op_parallelism_threads=NUMCORES)
tf.compat.v1.Session.intra_op_parallelism_threads = NUMCORES
tf.compat.v1.Session.inter_op_parallelism_threads = NUMCORES

class Fit_GPcounts(object):
    
    def __init__(self,X = None,Y= None,scale = None,sparse = False,scaled=False,nb_scaled=False, grid_search = False, gs = []):
     
        self.gs = gs # initialize Grid search with length scale values 
        self.gs_item = None # items in grid
        if grid_search: # use grid search to find the best length scale
            self.gs = gs
            if len(self.gs) == 0: 
                self.gs = [.5,3,10,15,25,40]
            self.optimize_ls = False
                
        else:         
            self.gs = [10]  
            self.optimize_ls = True # optimize length scale for single value initialization case or 
        # for the best value selected by grid search not optimize in case of grid search 
        
        self.transform = True # to use log(count+1) transformation
        self.sparse = sparse # use sparse or full inference 
        self.scaled = scaled 
        self.nb_scaled = nb_scaled
        self.X = None # time points == cell or samples 
        self.M = None # number of inducing point
        self.Z = None # inducing points
        self.Y = None # gene expression matrix
        self.Y_copy = None #copy of gene expression matrix
        self.D =  None # number of genes
        self.N = None # number of cells
        self.scale = scale
        self.genes_name = None
        self.cells_name = None  
        self.Scale = None
        
        # test information
        self.lik_name = None # the selected likelihood name 
        #self.fail = []  # list of genes failed to be fitted 
        self.models_number = None # Total number of models to fit for single gene for selected test
        self.model_index = None # index the current model 
        self.hyper_parameters = {} # model paramaters initialization
        self.model = None # GP model information    
        self.var = None # GP variance of posterior predictive
        self.mean = None # GP mean of posterior predictive
        self.fix = False # fix hyper-parameters
        # self.Scale = None
        # save likelihood of hyper-parameters of dynamic model to initialize the constant model
        self.lik_alpha = None  
        self.lik_km = None
        self.optimize = True  # optimize or load model
     
        self.branching = None # DE kernel or RBF kernel
        self.xp = -1000. # Put branching time much earlier than zero time
        
        # single gene information  
        self.y = None
        self.index = None
        self.global_seed = 0
        self.seed_value = 0 # initialize seed 
        self.count_fix = 0 # counter of number of trails to resolve either local optima or failure duo to numerical issues
        
        # check the X and Y are not missing
        if (X is None) or (Y is None):
            print('TypeError: GPcounts() missing 2 required positional arguments: X and Y')
        else:
            self.set_X_Y(X,Y) 
        
    def set_X_Y(self,X,Y):
        self.seed_value = 0
        np.random.seed(self.seed_value)
        
        if X.shape[0] == Y.shape[1]:
            self.X = X
            self.cells_name = self.X.index.values
            self.X = X.values.astype(float) 
            # self.X = np.asarray([float(x) for [x] in X.values.astype(float)]) # convert list of list to array of list
            self.X = self.X.reshape([-1,self.X.shape[1]])
            
            if self.sparse:
                self.M = int((5*(len(self.X)))/100) # number of inducing points is 5% of length of time points 
                # set inducing points by K-mean cluster algorithm
                kmeans = KMeans(n_clusters= self.M).fit(self.X)
                self.Z = kmeans.cluster_centers_
                self.Z = np.sort(self.Z,axis=None).reshape([self.M,1])
                self.Z = self.Z.reshape([self.Z.shape[0],1])
              
            self.Y = Y
            self.genes_name = self.Y.index.values.tolist() # gene expression name
            if self.lik_name == 'Gaussian':
                self.Y = self.Y.values # gene expression matrix
            else:
                self.Y = self.Y.values.astype(int)
                self.Y = self.Y.astype(float)
                
            self.Y_copy = self.Y
            self.D = Y.shape[0] # number of genes
            self.N = Y.shape[1] # number of cells
        else:
            print('InvalidArgumentError: Dimension 0 in X shape must be equal to Dimension 1 in Y, but shapes are %d and %d.' %(X.shape[0],Y.shape[1]))
        
    
    def Infer_trajectory(self,lik_name= 'Negative_binomial',transform = True): 
        
        genes_index = range(self.D)
        genes_results = self.run_test(lik_name,1,genes_index)
        
        return genes_results
        
    def One_sample_test(self,lik_name= 'Negative_binomial', transform = True):
        
        genes_index = range(self.D)
        genes_results = self.run_test(lik_name,2,genes_index)
        genes_results['log_likelihood_ratio'] = genes_results['log_likelihood_ratio'].clip(lower=0)
        if self.scaled:
            genes_results['p value'] = 1 - ss.chi2.cdf(genes_results['log_likelihood_ratio'], df=1)
            genes_results['q value']= self.qvalue(genes_results['p value'])

        return genes_results
        
    def Two_samples_test(self,lik_name= 'Negative_binomial',transform = True):
        
        genes_index = range(self.D)
        genes_results = self.run_test(lik_name,3,genes_index)
        
        return genes_results

    def Infer_trajectory_with_branching_kernel(self, cell_labels, bins_num=20, lik_name='Negative_binomial',
                                               branching_point=-1000):
        # note how to implement this
        cell_labels = np.array(cell_labels)
        self.X = np.c_[self.X, cell_labels[:, None]]
        self.branching = True
        self.xp = branching_point
        # return self.X
        genes_index = range(self.D)
        log_likelihood = self.run_test(lik_name, 1, genes_index, branching=True)

        self.branching_kernel_var = self.model.kernel.kern.variance.numpy()
        self.branching_kernel_ls = self.model.kernel.kern.lengthscales.numpy()

        # return log_likelihood
        return self.infer_branch_location(lik_name, bins_num)

    def infer_branch_location(self, lik_name, bin_nums):
        testTimes = np.linspace(min(self.X[:, 0]), max(self.X[:, 0]), bin_nums, endpoint=True)
        # print(testTimes)

        ll = np.zeros(bin_nums)
        models = list()
        genes_index = range(self.D)
        self.fix = True
        X = self.X
        for i in range(0, bin_nums):
            del self.X
            del self.model

            self.xp = testTimes[i]
            self.X = X.copy()
            self.X[np.where(self.X[:, 0] <= testTimes[i]), 1] = 1
            # print(np.unique(self.X[:, 1], return_counts=True))

            log_likelihood = self.run_test(lik_name, 1, genes_index, branching=True)
            ll[i] = self.model.log_posterior_density().numpy()
            models.append(self.model)
            # print(self.xp)
        # del self.X
        del self.model

        # break
        #         ll[i] = m_new.compute_log_likelihood()
        # models.append(m_new)

        return (models, ll)

    # Run the selected test and get likelihoods for all genes   
    def run_test(self,lik_name,models_number,genes_index,branching = False):
        
        genes_results = {}
        self.Y = self.Y_copy
        self.models_number = models_number
        self.lik_name = lik_name
        self.optimize = True
        
        #column names for likelihood dataframe
        if self.models_number == 1:
            column_name = ['Dynamic_model_log_likelihood']
        elif self.models_number == 2:
            column_name = ['Dynamic_model_log_likelihood','Constant_model_log_likelihood','log_likelihood_ratio']
        else:
            column_name = ['Shared_log_likelihood','model_1_log_likelihood','model_2_log_likelihood','log_likelihood_ratio'] 
        
        for self.index in tqdm(genes_index):
          
            self.y = self.Y[self.index].astype(float)
            self.y = self.y.reshape([-1,1])
            
            if len(self.gs) > 1: #grid search 
                self.optimize_ls = False      
                log_likelihood, best_index = self.fit_single_gene(self.gs,column_name)
                if log_likelihood.empty:
                    gs = [10]
                else:
                    gs = [best_index] # set the grid to the best length scale

            else:
                gs = self.gs
            
            if len(gs)>0: # not empty list 
                self.optimize_ls = True # optimize for the best length scale
                results, best_index = self.fit_single_gene(gs,column_name)

            genes_results[self.genes_name[self.index]] = results

        return pd.DataFrame.from_dict(genes_results, orient='index', columns= column_name)
    
    # fit numbers of GPs = models_number to run the selected test
    def fit_single_gene(self,gs,column_name,reset =False):
        if self.models_number == 1:
            col_name = 0
        else:
            col_name = 2
        
        best_index = None # index of the best value of length scales selected by grid search
        local_results = {} # save results of different length scales to select the best one   
        
        for self.gs_item in gs:
            
            self.model_index = 1  
            model_1_log_likelihood = self.fit_model()
            results =  [model_1_log_likelihood]
            
            if self.models_number == 2:
                if not(np.isnan(model_1_log_likelihood)):
                    ls , km_1 = self.record_hyper_parameters()
            
                    self.model_index = 2
                    model_2_log_likelihood= self.fit_model()
                    
                    if not(np.isnan(model_2_log_likelihood)):

                        if self.lik_name == 'Zero_inflated_negative_binomial':
                            km_2 = self.model.likelihood.km.numpy()

                        #test local optima case 2 
                        ll_ratio = model_1_log_likelihood - model_2_log_likelihood

                        if self.optimize_ls:
                            if (ls < (10*(np.max(self.X)-np.min(self.X)))/100 and round(ll_ratio) <= 0) or (self.lik_name == 'Zero_inflated_negative_binomial'  and np.abs(km_1 - km_2) > 50.0 ):
                                self.model_index = 1
                                reset = True
                                model_1_log_likelihood = self.fit_model(reset)

                                if not(np.isnan(model_1_log_likelihood)):
                                    ls , km_1 = self.record_hyper_parameters()
                                self.model_index = 2
                                reset = True
                                model_2_log_likelihood = self.fit_model(reset)

                                ll_ratio = model_1_log_likelihood - model_2_log_likelihood

                if np.isnan(model_1_log_likelihood) or np.isnan(model_2_log_likelihood):
                    model_2_log_likelihood = np.nan
                    ll_ratio = np.nan

                results =  [model_1_log_likelihood,model_2_log_likelihood,ll_ratio] 

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

                results = [model_1_log_likelihood,model_2_log_likelihood,model_3_log_likelihood,ll_ratio]
  
            local_results[self.gs_item]=results

        local_results = pd.DataFrame.from_dict(local_results, orient='index',columns= column_name)
        local_results=local_results.dropna() 
      
        if local_results.empty:
            results = local_results
        else: 
            best_index = local_results[local_results.columns[col_name]].idxmax()
            results = local_results.loc[local_results[local_results.columns[col_name]].idxmax()]
       
        return results,best_index
    
    # Save and get log likelihood of successed fit and set likelihood to Nan in case of failure 
    def fit_model(self,reset = False):
        #start = time.time() # start time to fit a model  
        fit = self.fit_GP(reset)
        if fit: # save the model in case of successeded fit
            if self.sparse and self.lik_name is not 'Gaussian': 
                log_likelihood = self.model.log_posterior_density((self.X,self.y)).numpy()
            else:
                log_likelihood = self.model.log_posterior_density().numpy()
            if self.optimize_ls:
                filename = self.get_file_name()
                ckpt = tf.train.Checkpoint(model=self.model, step=tf.Variable(1))
                ckpt.write(filename)

        else: # set log likelihood to Nan in case of Cholesky decomposition or optimization failure
            log_likelihood = np.nan  
            self.model = np.nan
            
        #end = time.time() # end time to fit a model 
        #Time = end - start
        #Time = str(datetime.timedelta(seconds=end - start))     
        return log_likelihood
    
    # def get_Scale(self):

    #     return self.Scale
    # fit a GP, fix Cholesky decomposition by random initialization if detected and test case1 of local optima      
    def fit_GP(self,reset = False):
        
        self.init_hyper_parameters(reset) 
        #print(self.hyper_parameters)
        fit = True   
        try:
            fit = self.fit_GP_with_likelihood()
            
        except tf.errors.InvalidArgumentError as e:
            #print(self.count_fix)
            
            if self.count_fix < 10 and self.optimize_ls: # fix failure by random restart 
                #print('Fit Cholesky decomposition was not successful.')
                fit = self.fit_GP(True)
                
            else:
                print('Can not fit a Gaussian process, Cholesky decomposition was not successful.')
                fit = False
                
         # Check for positive log likelihood if no numerical issues
        if fit and self.count_fix < 5: # limit number of trial to fix bad solution 
            # in case of sparse inference or non-Gaussian likelihood attache the data to the model
            if self.sparse and self.lik_name is not 'Gaussian': 
                log_likelihood = self.model.log_posterior_density((self.X,self.y)).numpy()
            else:
                log_likelihood = self.model.log_posterior_density().numpy()
                
            if log_likelihood > 0.0:
                fit = self.fit_GP(True)
                
        if fit and self.optimize and self.count_fix < 5 and self.optimize_ls and not self.branching:
            self.test_local_optima_case1()
        return fit
    
    # Fit a GP with selected kernel,likelihood,run it as sparse or full GP 
    def fit_GP_with_likelihood(self):
        fit = True
        
        #select kernel RBF,constant or branching kernel
        if self.hyper_parameters['ls'] == 1000: # flag to fit constant kernel
            kern = gpflow.kernels.Constant(variance= self.hyper_parameters['var']) 
        else:
            kern = gpflow.kernels.RBF( variance= self.hyper_parameters['var'],
                           lengthscales = self.hyper_parameters['ls'])
            set_trainable(kern.lengthscales,self.optimize_ls)         
        
        if self.branching: 
            if self.fix:
                set_trainable(kern.lengthscales,False)
                set_trainable(kern.variance,False)
            kernel = branchingKernel.BranchKernel(kern,self.xp)
        else:
            kernel = kern
            
        #select likelihood
        if self.lik_name == 'Poisson':
            likelihood = gpflow.likelihoods.Poisson()

        if self.lik_name == 'Negative_binomial':
            if self.scaled:
                scale=pd.DataFrame(self.scale)
                self.Scale = self.scale.iloc[:,self.index]
                self.Scale=np.array(self.Scale)
                self.Scale=np.transpose([self.Scale] * 20)
                likelihood = NegativeBinomialLikelihood.NegativeBinomial(self.hyper_parameters['alpha'],scale=self.Scale,nb_scaled=self.nb_scaled)
            else:
                likelihood = NegativeBinomialLikelihood.NegativeBinomial(self.hyper_parameters['alpha'],nb_scaled=self.nb_scaled)

       
        if self.lik_name == 'Zero_inflated_negative_binomial':
            alpha = self.hyper_parameters['alpha']
            km = self.hyper_parameters['km']
            likelihood = NegativeBinomialLikelihood.ZeroInflatedNegativeBinomial(alpha,km)
            
        # Run model with selected kernel and likelihood       
        if self.lik_name == 'Gaussian':
            if self.transform: # use log(count+1) in case of Gaussian likelihood and transform
                self.y = np.log(self.y+1)
            
            if self.sparse:
                self.model =  gpflow.models.SGPR((self.X,self.y), kernel=kernel,inducing_variable=self.Z)
                if self.model_index == 2 and self.models_number == 2:
                    set_trainable(self.model.inducing_variable.Z,False)
            else:
                self.model = gpflow.models.GPR((self.X,self.y), kernel)
                
            training_loss = self.model.training_loss
        else:
                        
            if self.sparse:
                self.model = gpflow.models.SVGP( kernel ,likelihood,self.Z) 
                training_loss = self.model.training_loss_closure((self.X, self.y))
                if self.model_index == 2 and self.models_number == 2:
                    set_trainable(self.model.inducing_variable.Z,False)
                
            else:
                self.model = gpflow.models.VGP((self.X, self.y) , kernel , likelihood) 
                training_loss = self.model.training_loss
              
        if self.optimize:
            o = gpflow.optimizers.Scipy()
            res = o.minimize(training_loss, variables=self.model.trainable_variables,options=dict(maxiter=5000))
            
            if not(res.success): # test if optimization fail
                if self.count_fix < 10 and self.optimize_ls: # fix failure by random restart 
                    #print('Optimization fail.')     
                    
                    fit = self.fit_GP(True)

                else:
                    print('Can not Optimaize a Gaussian process, Optimization fail.')
                    fit = False
        return fit

    # to detect local optima case2
    def record_hyper_parameters(self):
        ls = self.model.kernel.lengthscales.numpy()
        km_1 = 0
        # copy likelihood parameters and use them to fit constant model
        self.kern_var = self.model.kernel.variance.numpy()
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
    
    # Hyper-parameters initialization or restting in case of failure 
    def init_hyper_parameters(self,reset = False):
       
        self.hyper_parameters = {}
        
        # in case of failure change the seed and sample hyper-parameters from uniform distributions
        if reset:
            self.count_fix = self.count_fix +1 
            self.seed_value = self.seed_value + 1
            np.random.seed(self.seed_value)
            self.hyper_parameters['ls'] = np.random.uniform((1*(np.max(self.X)-np.min(self.X)))/100 ,(50*(np.max(self.X)-np.min(self.X)))/100)
            self.hyper_parameters['var'] = np.random.uniform(0 ,10.)
            self.hyper_parameters['alpha'] = np.random.uniform(0., 5.)
            self.hyper_parameters['km'] = np.random.uniform(0., 100.)       

        else:
            self.seed_value = self.global_seed
            self.count_fix = 0 
            np.random.seed(self.seed_value)
            # if len(self.gs) == 1: no grid search
            self.gs = [10] 
            self.gs_item = 10
            self.hyper_parameters['ls'] = (self.gs_item*(np.max(self.X)-np.min(self.X)))/100
            self.hyper_parameters['var'] = 3.
            self.hyper_parameters['alpha'] = 5.
            self.hyper_parameters['km'] = 35.  
       
        # set ls to 1000 in case of one sample test when fit the constant model    
        if self.model_index == 2 and self.models_number == 2:
            self.hyper_parameters['ls'] = 1000.      
            if self.optimize and self.count_fix == 0:
                self.hyper_parameters['var'] = self.kern_var
                if self.lik_name == 'Negative_binomial':
                    self.hyper_parameters['alpha'] = self.lik_alpha
                    
       
        else:
            #save likelihood parameters to initialize constant model
            self.lik_alpha = None 
            self.lik_km = None
            self.kern_var = None
            self.fix = False # fix kernel hyper-parameters    
        
        # reset gpflow graph
        tf.compat.v1.get_default_graph()
        tf.compat.v1.set_random_seed(self.seed_value)
        tf.random.set_seed(self.seed_value)
        gpflow.config.set_default_float(np.float64)
         
        self.y = self.Y[self.index].astype(float)
        self.y = self.y.reshape([-1,1])   
        self.model = None
        self.var = None 
        self.mean = None   
        # self.xp = -1000. # Put branching time much earlier than zero time

    def generate_Samples_from_distribution(self,mean):
        
        y = []
        if self.lik_name == 'Poisson':
            for i in range(mean.shape[0]):
                y.append(ss.poisson.rvs(mean[i], size = 500))

        if self.lik_name == 'Negative_binomial':
            if self.model.likelihood.alpha.numpy() == 0:
                for i in range(mean.shape[0]):
                     y.append(ss.poisson.rvs(mean[i], size = 500))
                
            else:
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
        if self.branching:
            mean = np.mean(link_f, axis=0)
        else:
            mean = np.mean(var,axis = 0)
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
        self.global_seed = 0
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
    
    # Function to Detection Outlier on one-dimentional datasets.
    def find_anomalies(self,data):
        anomalies_index = []
        # Set upper and lower limit to 3 standard deviation
        data_std = np.std(data)
        data_mean = np.mean(data)
        anomaly_cut_off = data_std * 3

        lower_limit  = data_mean - anomaly_cut_off 
        upper_limit = data_mean + anomaly_cut_off
       
        # Generate outliers
        for outlier in range(data.shape[0]):
            if data[outlier] > upper_limit or data[outlier] < lower_limit:
                anomalies_index.append(outlier)
        return anomalies_index
    
    def test_local_optima_case1(self):
        
     # limit number of trial to fix bad solution 
        if self.sparse:
            x = self.Z
        else:
            x = self.X 

        if self.X.shape[1] == 1:   
                xtest = np.linspace(np.min(x),np.max(x),100)[:,None]
        else: xtest = self.X
        if self.lik_name == 'Gaussian':
            mean, var = self.model.predict_y(xtest)
            self.mean = mean.numpy()
            self.var = var.numpy()
        else:
            # mean of posterior predictive samples
            self.mean,self.var = self.samples_posterior_predictive_distribution(xtest)        

        y_mean = np.mean(self.y)
        mean_mean = np.mean(self.mean) 
        y_max = np.max(self.y)
        mean_max = np.max(self.mean)
        y_min = np.abs(np.min(self.y))
        mean_min = np.abs(np.min(self.mean))
        
        if self.N < 100:
            diff = 0
        else:
            diff = 1

        if y_mean > 0.0 and (mean_max > y_max or mean_min < y_min):
            if abs(round((mean_mean-y_mean)/y_mean)) > 0 or mean_mean == 0.0:
                fit = self.fit_GP(True)
                



    def qvalue(self,pv, pi0=None):
        '''
        Estimates q-values from p-values
        This function is modified based on https://github.com/nfusi/qvalue
        Args
        ====
        pi0: if None, it's estimated as suggested in Storey and Tibshirani, 2003.
        '''
        assert(pv.min() >= 0 and pv.max() <= 1), "p-values should be between 0 and 1"

        original_shape = pv.shape
        pv = pv.ravel()  # flattens the array in place, more efficient than flatten()

        m = float(len(pv))

        # if the number of hypotheses is small, just set pi0 to 1
        if len(pv) < 100 and pi0 is None:
            pi0 = 1.0
        elif pi0 is not None:
            pi0 = pi0
        else:
            # evaluate pi0 for different lambdas
            pi0 = []
            lam = sp.arange(0, 0.90, 0.01)
            counts = sp.array([(pv > i).sum() for i in sp.arange(0, 0.9, 0.01)])
            for l in range(len(lam)):
                pi0.append(counts[l]/(m*(1-lam[l])))

            pi0 = sp.array(pi0)

            # fit natural cubic spline
            tck = interpolate.splrep(lam, pi0, k=3)
            pi0 = interpolate.splev(lam[-1], tck)

            if pi0 > 1:
                pi0 = 1.0

        assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

        p_ordered = sp.argsort(pv)
        pv = pv[p_ordered]
        qv = pi0 * m/len(pv) * pv
        qv[-1] = min(qv[-1], 1.0)

        for i in range(len(pv)-2, -1, -1):
            qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])

        # reorder qvalues
        qv_temp = qv.copy()
        qv = sp.zeros_like(qv)
        qv[p_ordered] = qv_temp

        # reshape qvalues
        qv = qv.reshape(original_shape)

        return qv    
