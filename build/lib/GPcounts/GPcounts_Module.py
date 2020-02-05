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
#from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PowerTransformer

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
       
        self.genes_log_likelihoods = {}
        
        if (X is None) or (Y is None):
            print('TypeError: GPcounts() missing 2 required positional arguments: X and Y')
        else:
            self.set_X_Y(X,Y)
        
        self.branching = None# branching kernel or RBF kernel
        self.y = None 
        self.index = None
        self.hyper_parameters = {} # model paramaters initialization
        self.seed_value = 0 
        self.count_local_optima = 0 # number of trails to resolve local optima
        
        self.models_number = None # Total number of models to fit for single gene
        self.model_index = None # to index the current model
        self.model = None
        # save likelihood parameters to initialize constant model
        self.lik_alpha = None 
        self.lik_km = None
        
        self.xp = -1000. # Put branching time much earlier than zero time
        self.fix = False # fix kernel hyper-parameters
        self.load = False
        self.lik_name = None
            
    def set_X_Y(self,X,Y):
        
        if X.shape[0] == Y.shape[1]:
            self.X = X
            self.cells_name = self.X.index.values # gene expression name
            self.X = X.values.astype(float) # time points 
            self.X = self.X.reshape([-1,1])
            if self.sparse:
                self.M = int((len(self.X))/10) # number of inducing point
                # set inducing points by Kmean cluster
                kmeans = KMeans(n_clusters= self.M, random_state=0).fit(self.X)
                self.Z = kmeans.cluster_centers_
                self.Z = np.sort(self.Z,axis=None).reshape([self.M,1])
                self.Z = self.Z.reshape([self.Z.shape[0],1])
           
            self.Y = Y
            
            self.genes_name = self.Y.index.values # gene expression name
            self.Y = self.Y.values.astype(float) # gene expression matrix
            self.Y_copy = self.Y
            self.D = Y.shape[0] # number of genes
            self.N = Y.shape[1] # number of cells
        else:
            print('InvalidArgumentError: Dimension 0 in X shape must be equal to Dimension 1 in Y, but shapes are %d and %d.' %(X.shape[0],Y.shape[1]))
            
    def initialize_parameters_run(self,lik_name,models_number,branching = False):
        
        self.seed_value = 0 
        self.count_local_optima = 0 # number of trails to resolve local optima
        
        np.random.seed(self.seed_value)
        tf.compat.v1.get_default_graph()
        tf.compat.v1.set_random_seed(self.seed_value)
        tf.random.set_seed(self.seed_value)
        gpflow.config.set_default_float(np.float64)
        
        self.branching = branching# branching kernel or RBF kernel
        self.y = None 
        self.index = None
        self.hyper_parameters = {} # model paramaters initialization
       
        self.models_number = models_number # Total number of models to fit for single gene
        self.model_index = None # to index the current model
        self.model = None
        # save likelihood parameters to initialize constant model
        self.lik_alpha = None 
        self.lik_km = None
        
        self.xp = -1000. # Put branching time much earlier than zero time
        self.fix = False # fix kernel hyper-parameters
        self.load = False
        self.lik_name = lik_name
    
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
    def run_test(self,lik_name,models_number):
        
        for i in tqdm(range(self.D)):
            
            self.initialize_parameters_run(lik_name,models_number)
            self.index = i
            self.y = self.Y[self.index].astype(float)
            self.y = self.y.reshape([self.N,1])
            
            #if self.lik_name == 'Gaussian': 
            #    self.y = np.log(self.y+1)
            #print(self.y)   
            column_name = ['Dynamic_model_log_likelihood']
            self.model_index = 1
            model_1_log_likelihood = self.fit_model()
            log_likelihood =  [model_1_log_likelihood] 
           
            if self.models_number == 2:
                column_name = ['Dynamic_model_log_likelihood','Constant_model_log_likelihood','log_likelihood_ratio']
                
                ls = self.model.kernel.lengthscale.numpy() # to detect local optima case2
                
                # copy likelihood parameters and use them to fit constant model
                if self.lik_name == 'Negative_binomial':
                    self.lik_alpha  = self.model.likelihood.alpha.numpy()
                if self.lik_name == 'Zero_inflated_negative_binomial':
                    km_1 = self.model.likelihood.km.numpy()
                    self.lik_alpha  = self.model.likelihood.alpha.numpy()
                    self.lik_km = km_1
                
                self.model_index = 2
                model_2_log_likelihood = self.fit_model()
                
                if self.lik_name == 'Zero_inflated_negative_binomial':
                    km_2 = self.model.likelihood.km.numpy()
                    
                 #test local optima case 2 
                ll_ratio = model_1_log_likelihood - model_2_log_likelihood
              
                if (ls < ((np.max(self.X)-np.min(self.X))/10.) and round(ll_ratio) <= 0) or (self.lik_name == 'Zero_inflated_negative_binomial'  and np.abs(km_1 - km_2) > 50.0 ):
                    self.model_index = 1
                    model_1_log_likelihood = self.fit_model()
                    self.model_index = 2
                    model_2_log_likelihood = self.fit_model()
                if model_1_log_likelihood == np.nan or model_2_log_likelihood == np.nan : 
                    ll_ratio = np.nan
                else:
                    log_likelihood =  [model_1_log_likelihood,model_2_log_likelihood,ll_ratio]               
            
            if self.models_number == 3:
                column_name = ['model_1_log_likelihood','model_2_log_likelihood','model_3_log_likelihood','log_likelihood_ratio'] 
                
                X_df = pd.DataFrame(data=self.X,index= self.cells_name,columns= ['times'])
                Y_df = pd.DataFrame(data=self.Y,index= self.genes_name,columns= self.cells_name)      
                
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
                
                if model_1_log_likelihood == np.nan or model_2_log_likelihood == np.nan or model_3_log_likelihood == np.nan: 
                    ll_ratio = np.nan
                else:
                    ll_ratio = ((model_2_log_likelihood+model_3_log_likelihood)-model_1_log_likelihood)
                
                log_likelihood = [model_1_log_likelihood,model_2_log_likelihood,model_3_log_likelihood,ll_ratio]
            
            self.genes_log_likelihoods[self.genes_name[self.index]] = log_likelihood
        return pd.DataFrame.from_dict(self.genes_log_likelihoods, orient='index', columns= column_name)
    
    # fit a Gaussian process, get the log likelihood and save the model for a gene 
    def fit_model(self):
        self.init_hyper_parameters(False) 
        successed_fit = self.fit_GP()
        if successed_fit:
            log_likelihood = self.model.log_likelihood().numpy()
            ## save the model 
            dir_name = 'GPcounts_models/'
            filename = dir_name+self.lik_name+'_model_'+str(self.model_index)

            if self.sparse:
                    filename += 'sparse_'

            filename += '_'+self.genes_name[self.index]

            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

            ckpt = tf.train.Checkpoint(model=self.model, step=tf.Variable(1))
            ckpt.write(filename)
            
        else:
            log_likelihood = np.nan  # set to Nan in case of Cholesky decomposition failure

        return log_likelihood
    
    # Hyper-parameters initialization
    def init_hyper_parameters(self,reset = False):
        
        self.hyper_parameters = {}
        
        if reset:
            self.seed_value = self.seed_value + 1
            #print('reset seed:',self.seed_value) 
            
        else:   
            self.seed_value = 0
            self.count_local_optima = 0 
            
        np.random.seed(self.seed_value)
        self.hyper_parameters['ls'] = np.random.uniform(0. , (np.max(self.X)-np.min(self.X))/10.)    
        self.hyper_parameters['var'] = np.random.uniform(1., 20.)
        self.hyper_parameters['alpha'] = np.random.uniform(1., 10.)
        self.hyper_parameters['km'] = np.random.uniform(1.0, 50.)
        
        if self.model_index == 2 and self.models_number == 2:
            self.hyper_parameters['ls'] = 1000. # constant kernel
            if not(self.load):
                if self.lik_name == 'Negative_binomial':
                    self.hyper_parameters['alpha'] = self.lik_alpha
                if self.lik_name == 'Zero_inflated_negative_binomial':
                    self.hyper_parameters['alpha'] = self.lik_alpha
                    self.hyper_parameters['km'] = self.lik_km 

    # fit a GP, fix Cholesky decomposition by random initialization if detected and test case1       
    def fit_GP(self):
        fail = False
        count_Cholesky_fail = 0  # number of trial to fix Cholesky decomposition      
        while count_Cholesky_fail < 20:
           
            try:
                if fail:
                    self.init_hyper_parameters(True)
                    fail = False                  
                             
                self.fit_GP_with_likelihood()

            except tf.errors.InvalidArgumentError as e:
                #print('error occured: {}'.format(e))
                fail = True
                count_Cholesky_fail = count_Cholesky_fail+1
                continue
            break
            
        if(count_Cholesky_fail == 20):
            self.model = None
            print('Can not fit a Gaussian process, Cholesky decomposition was not successful.')
            return False
        else:
            #detect local optima in model mean
            self.test_local_optima_case1()
            return True
    
    # Fit a GP with selected kernel,likelihood,run it as sparse or full GP 
    def fit_GP_with_likelihood(self):
        
        if self.hyper_parameters['ls'] == 1000:
            k = gpflow.kernels.Constant(variance= self.hyper_parameters['var']) 
        else:
            k = gpflow.kernels.RBF( variance= self.hyper_parameters['var'],
                               lengthscale = self.hyper_parameters['ls'])
            
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
            if self.hyper_parameters['ls'] == 1000:
                set_trainable(likelihood.alpha,False)
                set_trainable(likelihood.km,False)
                
        # Run model with selected kernel and likelihood
        
        if self.lik_name == 'Gaussian':
            #self.y = np.log(self.y+1) #log transform 
            # self.y = np.log(1+np.exp(-np.abs(self.y))) + np.max(self.y,0) #soft max 
            #pt = PowerTransformer()
            #self.y = pt.fit_transform(self.y)

            if self.sparse:
                self.model = gpflow.models.SGPR((self.X, self.y), kern, self.Z)
            else:
                self.model = gpflow.models.GPR((self.X, self.y), kern)
        else:
            if self.sparse:
                self.model = gpflow.models.SVGP((self.X, self.y) , kern , likelihood,self.Z)
            else:
                self.model = gpflow.models.VGP((self.X, self.y) , kern , likelihood) 
        
        if not(self.load):
            @tf.function(autograph=False)
            def objective():
                return - self.model.log_marginal_likelihood()

            o = gpflow.optimizers.Scipy()
            res = o.minimize(objective, variables=self.model.trainable_variables)
            #print('GP_optimizers_status:',res.success)
            #print(res)
            # fix optimization failure with random initialization
            if not(res.success):
                self.init_hyper_parameters(True)
                self.fit_GP()     

    def test_local_optima_case1(self):
        xtest = np.linspace(np.min(self.X),np.max(self.X),100)[:,None]
        
        if self.lik_name == 'Gaussian':
            mean, var = self.model.predict_y(xtest)
            mean = mean.numpy()
            var = var.numpy()

        else:
            # mean of posterior predictive samples
            mean,var = self.samples_posterior_predictive_distribution(xtest,sample = False) 

        if self.count_local_optima < 5.0: # limit number of trial to fix local optima  
            
            y_max = self.y.max()
            mean_max = mean.max()
            y_mean = self.y.mean()
            mean_mean = mean.mean()
            
            if (mean_max > y_max) and round((mean_mean-y_mean)/y_mean) > 0 or mean.mean() == 0:
                self.count_local_optima = self.count_local_optima+1 
                #print('local optima') 
                #self.plot(self.lik_name,xtest,self.model,mean,var)
                self.init_hyper_parameters(True)
                self.fit_GP()
                            
    def generate_Samples_from_distribution(self,mean):
        
        y = []

        if self.lik_name == 'Poisson':
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

    def samples_posterior_predictive_distribution(self,xtest,bins_num = 50,sample = False):
        
        '''
        if self.branching:
            bins_num = bins_num
            testTimes = np.linspace(min(m.data[0][:,0]), max(m.data[0][:,0]), bins_num , endpoint=True)
            Xnew = np.linspace(min(m.data[0][:,0]), max(m.data[0][:,0]), num_time_points )[:, None]
            x1 = np.c_[Xnew, np.ones(len(Xnew))[:, None]]
            x2 = np.c_[Xnew, ( np.ones(len(Xnew)) * 2 )[:, None]]
            xtest = np.concatenate((x1, x2))
            xtest[np.where(xtest[:,0] <= self.xp),1]=1
        '''
        samples = []
        f_samples = []
        for i in range(100):
            f_samples.append(self.model.predict_f_samples(xtest, 1))
            f = np.vstack(f_samples)
            link_f = np.exp(f[:, :, 0])
            if sample:
                samples.append(self.generate_Samples_from_distribution(np.mean(link_f, 0)).T)
        f_samples = np.vstack(f_samples)
        link_f = np.exp(f_samples[:, :, 0])
     
        mean = np.mean(link_f,axis=0)
        
        if sample:
            samples = np.vstack(samples)
        
        return mean,samples
  
    def load_and_sample_models(self,indexes,test,xtest,lik_name = 'Negative_binomial',num_bins = 50,sample= True):
         # note that RBF kernel and branching kernel is written in same way         
        dir_name = 'GPcounts_models/'
        filename = lik_name+'_model_'
        params = {}
        genes_models = []
        genes_means = []
        genes_vars =[]
        
        if test == 'One_sample_test':
            self.models_number = 2
        elif test == 'Two_samples_test':
            self.models_number = 3
        else:
            self.models_number = 1
            
        genes_name = self.genes_name.tolist()
        
        for index in indexes:
            
            #self.initialize_parameters_run(lik_name,self.models_number)
            models = []
            means = []
            variances = []
            self.index = genes_name.index(index)
            
            for model_num in range(self.models_number):
                self.initialize_parameters_run(lik_name,self.models_number)
                self.index = genes_name.index(index)
                self.model_index = model_num+1
                file_name = filename+str(self.model_index)+'_'+str(index)
                
                if self.models_number == 3:
                    X_df = pd.DataFrame(data=self.X,index= self.cells_name,columns= ['times'])
                    Y_df = pd.DataFrame(data=self.Y,index= self.genes_name,columns= self.cells_name)
                    
                    if model_num == 0:
                        self.set_X_Y(X_df,Y_df)
                        
                    if model_num == 1:
                        # initialize X and Y with first time series
                        self.set_X_Y(X_df[0 : int(self.N/2)],Y_df.iloc[:,0:int(self.N/2)])

                    if model_num == 2:
                        # initialize X and Y with second time series
                        self.set_X_Y(X_df[int(self.N/2) : :],Y_df.iloc[:,int(self.N/2) : :])
 
                self.y = self.Y[self.index].astype(float)
                self.y = self.y.reshape([self.N,1])

                if self.lik_name == 'Gaussian': 
                    self.y = np.log(self.y+1)
                self.load = True
                self.init_hyper_parameters(False) 
                self.fit_GP_with_likelihood()
                # restore check point
                ckpt = tf.train.Checkpoint(model=self.model, step=tf.Variable(1))
                ckpt.restore(dir_name+file_name)
                models.append(self.model)
               
                if sample:
                    if self.lik_name == 'Gaussian':
                        mean, var = self.model.predict_y(xtest)

                    else:
                        mean, var = self.samples_posterior_predictive_distribution(xtest,sample = True)

                    means.append(mean)
                    variances.append(var)

                if self.models_number == 3 and model_num > 0:
                    self.set_X_Y(X_df,Y_df)

            genes_models.append(models)
            if sample:
                genes_means.append(means)
                genes_vars.append(variances)

        if sample:
            params['means']= genes_means
            params['vars'] = genes_vars
        params['models']= genes_models

        return params
    
   