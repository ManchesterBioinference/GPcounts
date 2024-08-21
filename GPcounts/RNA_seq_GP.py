import os
import random
import warnings
from pathlib import Path
import gpflow
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.interpolate as si
import tensorflow as tf
from tqdm import tqdm
from scipy.special import logsumexp

from GPcounts.GP_NB_ZINB import GP_nb_zinb

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Get number of cores reserved by the batch system (NSLOTS is automatically set, or use 1 if not)
NUMCORES = int(os.getenv("NSLOTS", 1))

# Create session properties
config = tf.compat.v1.ConfigProto(
    inter_op_parallelism_threads=NUMCORES, intra_op_parallelism_threads=NUMCORES
)
tf.compat.v1.Session.intra_op_parallelism_threads = NUMCORES
tf.compat.v1.Session.inter_op_parallelism_threads = NUMCORES


class rna_seq_gp(object):
    def __init__(
        self,
        X=None,
        Y=None,
        sparse=False,
        M=0,
        safe_mode=False,
        scale=None,
        save= True,
    ):
        self.X = X  # time points (cells, samples or spatial locations)  
        self.genes_name = None
        self.cells_name = None
       
        self.models_number = None  # Total number of models to fit for single gene for selected test
        self.Y = Y  # gene expression matrix
        self.Y_copy = None  # copy of gene expression matrix
        self.D = None  # number of genes # rows
        self.N = None  # number of cells # columns
        
        # single gene information
        self.index = None
        self.seed_value = 0  # initial seed
        self.transform = True #  do log transformation and False for other Gaussian transformations   
        self.sparse=sparse # sparse or full inference  
        self.M=M # number of inducing points
        self.kmeans = False # set inducing points using kmeans algorithm
        self.safe_mode=safe_mode # detect and fix local optima case1
        self.scale=scale # library scale
        self.save = save # save the GP model
        self.folder_name = "GPcounts_models/"  # folder name for saved GP model
        # GP hyperparameters 
        self.length_scale = None
        self.variance = None
        self.alpha = None
        self.km = None
        self.cell_labels= None
        self.xp = None
        self.bins_num = None
        
        
        
        if (X is None) or (Y is None):
            print(
                "TypeError: GPcounts() missing 2 required positional arguments: X and Y"
            )
        else:
            self.set_X_Y(X,Y)
    '''
    setting inputs
    '''
    def set_X_Y(self, X, Y):         
        self.seed_value = 0
        np.random.seed(self.seed_value)

        if X.shape[0] == Y.shape[1]:
            self.X = X
            self.cells_name = list(map(str, list(X.index.values)))
            
           
            if self.sparse:
                if self.M == 0:
                    self.M = int(
                        (5 * (len(self.X))) / 100
                    )  # number of inducing points is 5% of the length of time points
                
            self.Y = Y
            self.genes_name = self.Y.index.values.tolist()  # gene expression name
            self.Y_copy = self.Y
            self.D = Y.shape[0]  # number of genes
            self.N = Y.shape[1]  # number of cells
           
        else:
            print(
                "InvalidArgumentError: Dimension 0 in X shape must be equal to Dimension 1 in Y, but shapes are %d and %d."
                % (X.shape[0], Y.shape[1])
            )
        
    '''
    RNA-seq analysis methods:
    1- Infer_trajectory
    2- One_sample_test
    3- Two_samples_test
    4- Infer_branching_location
    '''
    
    def Infer_trajectory(self, lik_name="Negative_binomial",kernel_type = 'RBF',transform=True):
        
        genes_results = self.run_test(lik_name, kernel_type,transform, 1 )
        
        return genes_results

    def One_sample_test(self,lik_name="Negative_binomial",kernel_type = 'RBF',transform=True):
        
        genes_results = self.run_test(lik_name, kernel_type,transform, 2)

        return genes_results

    def Two_samples_test(self, lik_name="Negative_binomial",kernel_type = 'RBF',transform=True):
        
        genes_results = self.run_test(lik_name, kernel_type,transform, 3)

        return genes_results
    
    def Infer_branching_location(
        self,
        cell_labels,
        bins_num=50,
        lik_name="Negative_binomial",
        branching_point=-1000,
        transform=True,
    ):
        
        self.cell_labels = np.array(cell_labels)
        self.bins_num = bins_num
        self.xp = branching_point
        genes_results = self.run_test(lik_name,'Branching',transform, 1)
        
        return genes_results
    
    '''    
    Run the selected test and get likelihoods for all genes
    '''
    def run_test(self, lik_name, kernel_type, transform, models_number):
        
        genes_results = {}
        self.Y = self.Y_copy
        self.models_number = models_number
        self.lik_name = lik_name
        self.kernel_type = kernel_type 
        self.optimize = True
        self.transform = transform
        
        # column names for likelihood dataframe
        if self.models_number == 1:
            column_name = ["Dynamic_model_log_likelihood"]
        elif self.models_number == 2:
            column_name = [
                "Dynamic_model_log_likelihood",
                "Constant_model_log_likelihood",
                "log_likelihood_ratio",
            ]
        else:
            column_name = [
                "Shared_log_likelihood",
                "model_1_log_likelihood",
                "model_2_log_likelihood",
                "log_likelihood_ratio",
            ]

        for self.index in tqdm(range(self.D)):
            results = self.fit_single_gene()
            genes_results[self.genes_name[self.index]] = results
            
        if kernel_type == 'Branching':
            return results
        else:
            return pd.DataFrame.from_dict(genes_results, orient="index", columns=column_name)
    
    '''
    fit GPs equal to the models_number to get the result of the selected test
    '''
    def fit_single_gene(self):
        
        # dynamic model
        model_1_log_likelihood, km, alpha, var = self.Run_GP_NB_ZINB(self.kernel_type,1)
        results = [model_1_log_likelihood]

        if self.models_number == 2:
            if not (np.isnan(model_1_log_likelihood)):
                # Constant model
                model_2_log_likelihood, km, alpha, var  = self.Run_GP_NB_ZINB('Constant',2,alpha, km, var)
            else:
                model_2_log_likelihood = np.nan 
            if not (np.isnan(model_2_log_likelihood) and np.isnan(model_2_log_likelihood)):
                ll_ratio = model_1_log_likelihood - model_2_log_likelihood
            else:
                ll_ratio = np.nan 
           
            results = [model_1_log_likelihood, model_2_log_likelihood, ll_ratio]

        # two dymanic models
        if self.models_number == 3:
            X_df = self.X 
            Y_df = self.Y 
            
            # initialize X and Y with first time series
            self.set_X_Y(X_df[0 : int(self.N / 2)], Y_df.iloc[:, 0 : int(self.N / 2)])
            
            model_2_log_likelihood, km, alpha, var = self.Run_GP_NB_ZINB(self.kernel_type,2,)
            
            # initialize X and Y with second time series
            self.set_X_Y(X_df[self.N : :], Y_df.iloc[:, int(self.N) : :])
       
            model_3_log_likelihood, km, alpha, var = self.Run_GP_NB_ZINB(self.kernel_type,3)
            
            self.set_X_Y(X_df, Y_df)

            if (
                np.isnan(model_1_log_likelihood)
                or np.isnan(model_2_log_likelihood)
                or np.isnan(model_3_log_likelihood)
            ):
                ll_ratio = np.nan
            else:
                ll_ratio = (
                    model_2_log_likelihood + model_3_log_likelihood
                ) - model_1_log_likelihood
            results = [
                model_1_log_likelihood,
                model_2_log_likelihood,
                model_3_log_likelihood,
                ll_ratio,
            ]

        return results
    
    '''
    fit GP by calling Run_GP_NB_ZINB module 
    '''
    def Run_GP_NB_ZINB(self,kern_type,model_index,alpha = None,km= None, var=None):
        
        lik_km = None
        lik_alpha = None
        
        if self.scale is not None:
            scale = self.scale.iloc[:,[self.index]]
          
        else:
            scale = None
        
        gp_nb_zinb = GP_nb_zinb(self.X,self.Y.iloc[[self.index]],sparse= self.sparse,safe_mode = self.safe_mode,M= self.M, scale = scale, save= self.save) 
        gp_nb_zinb.initialize_hyper_parameters(self.length_scale, var, alpha, km)
        gp_nb_zinb.folder_name = self.folder_name
        gp_nb_zinb.lik_alpha = alpha
        gp_nb_zinb.lik_km = km
        #gpflow.utilities.print_summary(gp_nb_zinb.model, fmt='notebook')
        
        if kern_type == 'Branching':
            gp_nb_zinb.X = np.c_[gp_nb_zinb.X, self.cell_labels[:, None]]
                 
        if self.kmeans:
            gp_nb_zinb.kmeans_algorithm_inducing_points(self.M)
     
        if self.models_number == 3:
             txt = str(self.genes_name[self.index]) + "_model_" + str(model_index)+'_tst'
        else:
             txt = str(self.genes_name[self.index]) + "_model_" + str(model_index)
        
        
        log_likelihood = gp_nb_zinb.model_log_likelihood(self.lik_name,self.transform,txt,kernel_type = kern_type,models_number = model_index)
        
        if kern_type == 'Branching':
            gp_nb_zinb.branching_kernel_var = gp_nb_zinb.model.kernel.kern.variance.numpy()
            gp_nb_zinb.branching_kernel_ls = gp_nb_zinb.model.kernel.kern.lengthscales.numpy()
            
            log_likelihood = gp_nb_zinb.infer_branching(self.lik_name,self.transform,txt,kern_type,self.bins_num,self.xp)
        
        if self.models_number == 2:
            if not (np.isnan(log_likelihood)):
                var = gp_nb_zinb.var
                if self.lik_name == "Negative_binomial":
                    lik_alpha = gp_nb_zinb.model.likelihood.alpha.numpy()
                if self.lik_name == "Zero_inflated_negative_binomial":
                    lik_km = gp_nb_zinb.model.likelihood.km.numpy()
                    lik_alpha = gp_nb_zinb.model.likelihood.alpha.numpy()
        del gp_nb_zinb

        return log_likelihood,lik_km,lik_alpha, var
    
    '''
    Return models with the parameters, means and variacnes to make a prediction
    '''
    def load_predict_models(self,genes_name,test_name,likelihood,kernel_type='RBF',predict = True):
        params = {}
        genes_models = []
        genes_means = []
        genes_vars = []
        kern_type = kernel_type
        self.Y = self.Y_copy
        self.lik_name = likelihood
        params["test_name"] = test_name
        params["likelihood"] = self.lik_name
        
        if test_name == "One_sample_test":
            self.models_number = 2
        elif test_name == "Two_samples_test":
            self.models_number = 3
        else:
            self.models_number = 1
        
        xtest = np.linspace(np.min(self.X) - 0.1, np.max(self.X) + 0.1, 100)[:, None]
        
        for gene in tqdm(genes_name):
            models = []
            means = []
            variances = []
            
            self.index = self.genes_name.index(gene)
            
            gp_nb_zinb = GP_nb_zinb(self.X,self.Y.iloc[[self.index]],sparse= self.sparse,safe_mode = self.safe_mode,M= self.M, scale = self.scale, save= self.save) 
            gp_nb_zinb.folder_name = self.folder_name
            gp_nb_zinb.lik_name =  self.lik_name
            
            for model_index in range(self.models_number):

                model_index = model_index + 1
                gp_nb_zinb.init_hyper_parameters(reset=False)
                
                if self.models_number == 3:
                    X_df =  self.X
                    Y_df = pd.DataFrame(
                        data=self.Y_copy, index=self.genes_name, columns=self.cells_name
                    )
               
                    if model_index == 1:
                        self.set_X_Y(X_df, Y_df)

                    if model_index == 2:  # initialize X and Y with first time series
                        self.set_X_Y(
                            X_df[0 : int(self.N / 2)], Y_df.iloc[:, 0 : int(self.N / 2)]
                        )

                    if model_index == 3:  # initialize X and Y with second time series
                        self.set_X_Y(
                            X_df[int(self.N / 2) : :], Y_df.iloc[:, int(self.N / 2) : :]
                        )
                    
                    gp_nb_zinb.set_X_y(self.X,self.Y.iloc[[self.index]]) # Assign the correct X and Y to gb_nb_zinb object
                    
                if self.models_number == 3:
                     txt = str(self.genes_name[self.index]) + "_model_" + str(model_index)+'_tst'
                else:
                     txt = str(self.genes_name[self.index]) + "_model_" + str(model_index)   
                    
                p = {}
                if self.models_number == 2 and model_index == 2:
                    kern_type = 'Constant'
                    
                else:
                    kern_type = kernel_type
               
                p = gp_nb_zinb.load_predict_model(self.lik_name,txt, models_number = self.models_number,kernel_type = kern_type, predict = True)
                means.append(p['mean'])
                variances.append(p['var'])
                models.append(p['model'])
                
                if self.models_number == 3 and model_index > 0:
                    self.set_X_Y(X_df, Y_df)
                
            genes_means.append(means)
            genes_vars.append(variances)
            genes_models.append(models)
            del gp_nb_zinb
        
        params["means"] = genes_means
        params["vars"] = genes_vars
        params["models"] = genes_models
        
        return params
   

    def calculate_FDR(self, genes_results):
        genes_results["p_value"] = 1 - ss.chi2.cdf(
            2 * genes_results["log_likelihood_ratio"], df=1
        )
        genes_results["q_value"] = self.qvalue(genes_results["p_value"])

        return genes_results
    
    
    def qvalue(self, pv, pi0=None):
        """
        Estimates q-values from p-values
        This function is modified based on https://github.com/nfusi/qvalue
        Args
        ====
        pi0: if None, it's estimated as suggested in Storey and Tibshirani, 2003.
        """
        assert pv.min() >= 0 and pv.max() <= 1, "p-values should be between 0 and 1"

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
            lam = np.arange(0, 0.90, 0.01)
            counts = np.array([(pv > i).sum() for i in np.arange(0, 0.9, 0.01)])
            for l in range(len(lam)):
                pi0.append(counts[l] / (m * (1 - lam[l])))

            pi0 = np.array(pi0)

            # fit natural cubic spline
            tck = si.splrep(lam, pi0, k=3)
            pi0 = si.splev(lam[-1], tck)

            if pi0 > 1:
                pi0 = 1.0

        assert pi0 >= 0 and pi0 <= 1, "pi0 is not between 0 and 1: %f" % pi0

        p_ordered = np.argsort(pv)
        pv = pv[p_ordered]
        qv = pi0 * m / len(pv) * pv
        qv[-1] = min(qv[-1], 1.0)

        for i in range(len(pv) - 2, -1, -1):
            qv[i] = min(pi0 * m * pv[i] / (i + 1.0), qv[i + 1])

        # reorder qvalues
        qv_temp = np.copy(qv)
        qv = np.zeros_like(qv)
        qv[p_ordered] = qv_temp

        # reshape qvalues
        qv = np.reshape(qv, original_shape)
        return qv
    
    '''
    Run GP model for linear, periodic and RBF kernels and calculate BIC
    '''
    def Model_selection_test(self, lik_name="Negative_binomial", kernel=None, transform=True):
        
        if transform == True:
            self.Y = self.Y.astype(int)
            self.Y = self.Y.astype(float)
        self.Y_copy = self.Y
        
        
        ker_list = ["Linear", "Periodic", "RBF"]
        genes_index = range(self.D)
        selection_results = pd.DataFrame()
        selection_results["Gene"] = 0
        selection_results["Dynamic_model_log_likelihood"] = 0
        selection_results["Constant_model_log_likelihood"] = 0
        selection_results["log_likelihood_ratio"] = 0
        selection_results["p_value"] = 0
        selection_results["q_value"] = 0
        selection_results["log_likelihood_ratio"] = 0
        selection_results["Model"] = 0
        selection_results["BIC"] = 0

        for word in ker_list:
            if word == "Linear":
                K = 3
            else:
                K = 4
            #self.kernel = word
            results = self.run_test(lik_name, word,transform,2)
            results["BIC"] = -2 * results[
                "Dynamic_model_log_likelihood"
            ] + K * np.log(self.X.shape[0])
            
            results["Gene"] = self.genes_name
            results["Model"] = word
            results["p_value"] = 1 - ss.chi2.cdf(
                2 * results["log_likelihood_ratio"], df=1
            )
            results["q_value"] = self.qvalue(results["p_value"])
            selection_results = selection_results.merge(results, how="outer")

            # Model probability estimation based on bic based on SpatialDE:identification of spatially variable genes: https://www.nature.com/articles/nmeth.4636
            tr = (
                selection_results.groupby(["Gene", "Model"])["BIC"].transform(min)
                == selection_results["BIC"]
            )
            # select bic values for each kernel and gene
            bic_values = -selection_results[tr].pivot_table(
                values="BIC", index="Gene", columns="Model"
            )
            restore_these_settings = np.geterr()
            temp_settings = restore_these_settings.copy()
            temp_settings["over"] = "ignore"
            temp_settings["under"] = "ignore"
            np.seterr(**temp_settings)
            log_v = logsumexp(bic_values, 1)
            log_model_prob = (bic_values.T - log_v).T
            model_prob = np.exp(log_model_prob).add_suffix("_probability")

            tr = (
                selection_results.groupby("Gene")["BIC"].transform(min)
                == selection_results["BIC"]
            )
            selection_results_prob = selection_results[tr]
            selection_results_prob = selection_results_prob.join(model_prob, on="Gene")
            transfer_columns = ["p_value", "q_value"]
            np.seterr(**restore_these_settings)
            selection_results_prob = selection_results_prob.drop(
                transfer_columns, axis = 1
            ).merge(selection_results, how="inner")

        return selection_results_prob
