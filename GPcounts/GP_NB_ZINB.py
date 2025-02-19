import os
import random
import warnings
from pathlib import Path

import gpflow
import numpy as np
import pandas as pd
import scipy.stats as ss
import tensorflow as tf
from gpflow.utilities import set_trainable
from robustgp import ConditionalVariance
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from GPcounts import NegativeBinomialLikelihood, branchingKernel

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

class GP_nb_zinb(object):
    def __init__(
        self,
        X=None,
        y=None,
        sparse=False,
        M=0,
        safe_mode=False,
        scale=None,
        save = True,
    ):

        self.safe_mode = safe_mode # check and fix local optima cases however it takes more computational resources
        self.transform = True  # do log transformation and False for other Gaussian transformations   
        self.sparse = sparse  # sparse or full inference
        self.save = save # save GP result as checkpoint  
        self.scale = scale # library scale 
        self.X = None  # regressor
        self.M = M  # number of inducing points
        
        self.y = None  # f(x)+ noise
        self.y_copy = None 
        self.kernel_type = 'RBF' 
        self.lik_name="Negative_binomial"
        self.folder_name = "GPcounts_models/" #save GP models in this folder
        self.seed_value = 0  # initial seed
        self.count_fix = 0  # counter for the number of trails to solve local optima or failure duo to numerical issues
        self.model = None  # model information
        self.var = None  # GP variance 
        self.mean = None  # GP mean 
        self.fix = False  # fix hyper-parameters
        
        # copy dynamic model hyper-parameters to use them to initialize the constant model
        self.lik_alpha = None
        self.lik_km = None
        
        self.optimize = True  # optimize or load model 
        self.xp = -1000.0  # Put branching time much earlier than zero time
        self.Z = None  # inducing points
        self.robustGP = False  # set inducing points using conditional variance from robustGP method
        self.bic = None

        # kernel information
        self.hyper_parameters = {}  # default hyperparameters initialization
        self.user_hyper_parameters = [ # user hyperparameters initialization
            None, 
            None,
            None,
            None,
        ]  
        
        # check X and y are not missing
        if (X is None) or (y is None):
            print(
                "TypeError: GPcounts() missing 2 required positional arguments: X and Y"
            )
        else:
            self.set_X_y(X, y)

    def set_X_y(self, X, y):
    
        self.seed_value = 0
        np.random.seed(self.seed_value)

        if X.shape[0] == y.shape[1]:
            self.X = X
            self.X = X.values.astype(float)

            if len(self.X.shape) > 1:
                self.X = self.X.reshape([-1, self.X.shape[1]])
            else:
                self.X = self.X.reshape([-1, 1])

            if self.sparse:
                if self.M == 0:
                    self.M = int(
                        (5 * (len(self.X))) / 100
                    )  # number of inducing points is 5% of the length of time points
                self.robustGP = True

            self.y = y
            self.y = self.y.values  # gene expression matrix
            self.y_copy = self.y
           
        else:
            print(
                "InvalidArgumentError: Dimension 0 in X shape must be equal to Dimension 1 in Y, but shapes are %d and %d."
                % (X.shape[0], y.shape[1])
            ) 
    '''
    Set inducing points using K-means clustering algorithm
    '''
    def kmeans_algorithm_inducing_points(self, M=0):
        if M != 0:
            self.M = M
        self.robustGP = False
        
        kmeans = KMeans(n_clusters=self.M).fit(self.X)
        self.Z = kmeans.cluster_centers_
        self.Z = np.sort(self.Z, axis=None).reshape([self.M, 1])
        self.Z = self.Z.reshape([self.Z.shape[0], 1])
   
    '''
    Infer branching points
    '''
    def infer_branching(self,lik_name,transform,txt,kernel_type,bins_num,xp):
          
        testTimes = np.linspace(
            min(self.X[:, 0]), max(self.X[:, 0]), bins_num, endpoint=True
        )
        ll = np.zeros(bins_num)
        models = list()
        self.fix = True
        X = self.X
       
        for i in range(0, bins_num):
            del self.X
            del self.model
            self.xp = testTimes[i]
            self.X = X.copy()
            self.X[np.where(self.X[:, 0] <= testTimes[i]), 1] = 1
            _ = self.model_log_likelihood(lik_name,transform,txt = 'test',kernel_type = 'Branching',models_number = 1)
            ll[i] = self.model.log_posterior_density().numpy()
            models.append(self.model)
        del self.model

        # Find MAP model
        log_ll = np.zeros(bins_num)
        i = 0
        for mm in models:
            log_ll[i] = mm.log_posterior_density().numpy()
            i = i + 1
        p = self.CalculateBranchingEvidence({"loglik": log_ll}, testTimes)
        ll = p["posteriorBranching"]

        iMAP = np.argmax(ll)
        self.model = models[iMAP]

        # Prediction
        Xnew = np.linspace(min(self.X[:, 0]), max(self.X[:, 0]), 100).reshape(-1)[
            :, None
        ]
        x1 = np.c_[Xnew, np.ones(len(Xnew))[:, None]]
        x2 = np.c_[Xnew, (np.ones(len(Xnew)) * 2)[:, None]]
        Xtest = np.concatenate((x1, x2))
        Xtest[np.where(Xtest[:, 0] <= self.model.kernel.xp), 1] = 1

        if self.lik_name == "Gaussian":
            mu, var = self.model.predict_y(Xtest)
        else:
            mu, var = self.samples_posterior_predictive_distribution(Xtest)

        del models
       
        return {
            "branching_probability": ll,
            "branching_location": self.model.kernel.xp,
            "mean": mu,
            "variance": var,
            "Xnew": Xnew,
            "test_times": testTimes,
            "MAP_model": self.model,
            "loglik": log_ll,
            "logBayesFactor": p["logBayesFactor"],
            "likelihood": self.lik_name,
        }

    def CalculateBranchingEvidence(self, d, Bsearch):
        """
        :param d: output dictionary from FitModel
        :param Bsearch: candidate list of branching points
        :return: posterior probability of branching at each point and log Bayes factor
        of branching vs not branching
        """
        # Calculate probability of branching at each point
        # o = d['loglik'][:-1]
        o = d["loglik"]
        pn = np.exp(o - np.max(o))
        p = pn / pn.sum()  # normalize

        # Calculate log likelihood ratio by averaging out
        o = d["loglik"]
        Nb = o.size - 1
        if Nb != len(Bsearch) - 1:
            raise NameError(
                "Passed in wrong length of Bsearch is %g- should be %g"
                % (len(Bsearch), Nb)
            )
        obj = o[:-1]
        illmax = np.argmax(obj)
        llmax = obj[illmax]
        lratiostable = (
            llmax
            + np.log(1 + np.exp(obj[np.arange(obj.size) != illmax] - llmax).sum())
            - o[-1]
            - np.log(Nb)
        )

        return {"posteriorBranching": p, "logBayesFactor": lratiostable}
    
    
    '''
    Return GP log likelihood if fit successed or nans in case of failure
    '''
    def model_log_likelihood(self,lik_name="Negative_binomial",transform=True,txt = 'GP',kernel_type = 'RBF',models_number = 1):
        reset = False
       
        self.kernel_type = kernel_type
        self.lik_name = lik_name # likelihood name
        self.models_number = models_number
        self.tansform = transform
        
        if self.transform == True:
            self.y = self.y.astype(int)
            self.y = self.y.astype(float)
            
        fit = self.fit_GP(reset)
        if fit:  
            if self.sparse and self.lik_name != "Gaussian":
                log_likelihood = self.model.log_posterior_density(
                    (self.X, self.y)
                ).numpy()
            else:
                log_likelihood = self.model.log_posterior_density().numpy()

            # fix positive log likelihood by random restart
            if (
                log_likelihood > 0
                and self.count_fix < 10
                and self.safe_mode
                and self.lik_name != "Gaussian"
            ):
                self.count_fix = self.count_fix + 1
                log_likelihood = self.fit_model(True)
             
            # save GPflow model as tensorflow check point  
            if self.save and log_likelihood:
                self.save_GP(log_likelihood,txt)
            
        # set log likelihood to Nan in case of Cholesky decomposition or optimization failure    
        else:  
            log_likelihood = np.nan
            self.model = np.nan

        return log_likelihood

    '''
    Rerun fit_GP to fix the failures by random restart  
    '''
    def fit_GP(self, reset=False):
        self.init_hyper_parameters(reset=reset)

        fit = True
        # catch Cholesky decomposition failure
        try:
            fit = self.GP_Kernel_Likelihood()

        except tf.errors.InvalidArgumentError as e:

            if self.count_fix < 10:  # fix failure by random restart
                fit = self.fit_GP(True)

            else:
                print(
                    "Can not fit a Gaussian process, Cholesky decomposition was not successful."
                )
                fit = False

        if (
            fit
            and self.optimize
            and self.count_fix < 5
            and self.kernel_type != 'Branching'
            and self.safe_mode
        ):
            self.test_local_optima_case1()
        return fit
    
    '''
    Run GPflow model with selected kernel,likelihood as sparse or full GP
    '''
    def GP_Kernel_Likelihood(self):
        fit = True
        if self.kernel_type == 'Constant':
            kernel = gpflow.kernels.Constant(variance=self.hyper_parameters["var"])
            
        elif self.kernel_type == "Linear":
            kernel = gpflow.kernels.Linear(variance=self.hyper_parameters["var"])
            print("Fitting GP with Linear Kernel")
    
        elif self.kernel_type =="Periodic":
            kernel = gpflow.kernels.Periodic(
                (
                    gpflow.kernels.SquaredExponential(
                        variance=self.hyper_parameters["var"],
                        lengthscales=self.hyper_parameters["ls"],
                    )
                )
            )
            print("Fitting GP with Periodic Kernel")
           
        
        elif self.kernel_type == 'Nugget':
            kernel = gpflow.kernels.RBF(
                variance=self.hyper_parameters["var"],
                lengthscales=self.hyper_parameters["ls"],
            )+gpflow.kernels.White(variance=self.hyper_parameters["var"])
        
        # branching or non-branching kernel
        elif self.kernel_type == 'Branching':
            if self.fix:
                kern = gpflow.kernels.RBF(
                    variance=self.branching_kernel_var,
                    lengthscales=self.branching_kernel_ls,
                )
                set_trainable(kern.lengthscales, False)
                set_trainable(kern.variance, False)
            else:
                kern = gpflow.kernels.RBF()
            kernel = branchingKernel.BranchKernel(kern, self.xp)
        
        else:
            kernel = gpflow.kernels.RBF(
                    variance=self.hyper_parameters["var"],
                    lengthscales=self.hyper_parameters["ls"],
                )
           
            
        # select likelihood
        if self.lik_name == "Poisson":
            likelihood = gpflow.likelihoods.Poisson()

        if self.lik_name == "Negative_binomial":
            # with library size scaling
            if self.scale is not None:
                Scale = np.array(self.scale)
                #Scale = np.transpose([Scale] * 20)
                likelihood = NegativeBinomialLikelihood.NegativeBinomial(
                    self.hyper_parameters["alpha"],
                    scale= Scale,
                   
                )
            # without library size scaling    
            else:
                likelihood = NegativeBinomialLikelihood.NegativeBinomial(
                    self.hyper_parameters["alpha"],
                )

        if self.lik_name == "Zero_inflated_negative_binomial":
            likelihood = NegativeBinomialLikelihood.ZeroInflatedNegativeBinomial(
                self.hyper_parameters["alpha"], self.hyper_parameters["km"]
            )

        # select the likelihood
        if self.lik_name == "Gaussian":
            if self.transform: 
                 # use log(count+1) in case of Gaussian likelihood and transform
                self.y = np.log(self.y + 1)
            
            # select sparse model
            if self.sparse:
                if self.robustGP:
                    init_method = ConditionalVariance()
                    self.Z = init_method.compute_initialisation(self.X, self.M, kernel)[
                        0
                    ]

                self.model = gpflow.models.SGPR(
                    (self.X, self.y), kernel=kernel, inducing_variable=self.Z
                )
                if self.kernel_type == 'Constant' and self.models_number == 2:
                    set_trainable(self.model.inducing_variable.Z, False)
            else:
                self.model = gpflow.models.GPR((self.X, self.y), kernel)
            training_loss = self.model.training_loss
        else:

            if self.sparse:
                if self.robustGP:
                    init_method =  ConditionalVariance()
                    ## Fix the number of inducing points for constant kernel 
                    if self.kernel_type == "Constant":
                        self.Z = init_method.compute_initialisation(self.X, 1, kernel)[
                            0
                        ]
                    else : 
                        self.Z = init_method.compute_initialisation(self.X, self.M, kernel)[
                        0
                    ]
                   
                self.model = gpflow.models.SVGP(kernel, likelihood, self.Z)
                training_loss = self.model.training_loss_closure((self.X, self.y))
                if self.kernel_type == 'Constant' and self.models_number == 2:
                    set_trainable(self.model.inducing_variable.Z, False)

            else:
                self.model = gpflow.models.VGP((self.X, self.y), kernel, likelihood,)
                training_loss = self.model.training_loss

        if self.optimize:
            if self.robustGP:
                set_trainable(self.model.inducing_variable.Z, False)
            
            o = gpflow.optimizers.Scipy()
            res = o.minimize(training_loss,self.model.trainable_variables)
            
            #options=dict(maxiter=5000),
            if not (res.success):  # test if optimization fail
                if self.count_fix < 10:  # fix failure by random restart
                    fit = self.fit_GP(True)

                else:
                    print("Can not Optimaize a Gaussian process, Optimization fail.")
                    fit = False
        return fit
    
    '''
    user assign the default values for the hyper_parameters
    '''
    def initialize_hyper_parameters(
        self, length_scale=None, variance=None, alpha=None, km=None , scale = None
    ):
        if length_scale is None:
            self.hyper_parameters["ls"] = (5 * (np.max(self.X) - np.min(self.X))) / 100 # len(X)* 5%
        else:
            self.hyper_parameters["ls"] = length_scale

        if variance is None:
            if self.lik_name == "Gaussian" and not self.transform:
                self.hyper_parameters["var"] = np.mean(self.y + 1 ** 2)
            #****** Improved the initialisation of the RBF scale parameter to use the empirical variance of the data
            else:
                if self.scale is not None:
                    Scale = np.array(self.scale)
                    self.hyper_parameters["var"] = np.var(np.log((y + 1) / Scale))
                    
                else:    
                    self.hyper_parameters["var"] = np.var(np.log((y + 1)))
                #np.mean(np.log(self.y + 1) ** 2)

        else:
            self.hyper_parameters["var"] = variance

        if alpha is None:
            self.hyper_parameters["alpha"] = 1.0
        else:
            self.hyper_parameters["alpha"] = alpha

        if km is None:
            self.hyper_parameters["km"] = 35.0
        else:
            self.hyper_parameters["km"] = km

        self.user_hyper_parameters = [length_scale, variance, alpha, km]
    
    '''
    Hyper-parameters initialization or restting in case of failure
    '''
    def init_hyper_parameters(self, reset=False):
        # donot use default values if user is setting them 
        if not reset:
            self.seed_value = 0
            self.count_fix = 0
            np.random.seed(self.seed_value)
        self.initialize_hyper_parameters(
            self.user_hyper_parameters[0],
            self.user_hyper_parameters[1],
            self.user_hyper_parameters[2],
            self.user_hyper_parameters[3],
        )
        # in case of failure change the seed and sample hyper-parameters from uniform distributions
        if reset:
            self.count_fix = self.count_fix + 1
            self.seed_value = self.seed_value + 1
            np.random.seed(self.seed_value)
            
            # sample values from [len(X)*.25%,len(X)*30%]
            self.hyper_parameters["ls"] = np.random.uniform(
                (0.25 * (np.max(self.X) - np.min(self.X))) / 100,
                (30.0 * (np.max(self.X) - np.min(self.X))) / 100,
            )
            self.hyper_parameters["var"] = np.random.uniform(0.0, 10.0)
            self.hyper_parameters["alpha"] = np.random.uniform(0.0, 10.0)
            self.hyper_parameters["km"] = np.random.uniform(0.0, 100.0)
        
            if self.optimize and self.count_fix == 0:
                if self.lik_name == "Negative_binomial":
                    self.hyper_parameters["alpha"] = self.lik_alpha

        else:
            # save likelihood parameters to initialize constant model
            self.lik_alpha = None
            self.lik_km = None
            if self.kernel_type != 'Branching':
                self.fix = False  # fix kernel hyper-parameters

        # reset gpflow graph
        tf.compat.v1.get_default_graph()
        tf.compat.v1.set_random_seed(self.seed_value)
        tf.random.set_seed(self.seed_value)
        gpflow.config.set_default_float(np.float64)
        
        self.y = self.y_copy
        self.y = self.y.astype(float)
        self.y = self.y.reshape([-1, 1])
        self.model = None
        self.var = None
        self.mean = None

    
    '''
    Detect local optima case1 and fix it with a random restart
    '''
    def test_local_optima_case1(self):
        # limit number of trial to fix bad solution
        if self.sparse:
            x = self.Z
        else:
            x = self.X

        if self.X.shape[1] == 1:
            xtest = np.linspace(np.min(x), np.max(x), 100)[:, None]
        else:
            xtest = self.X
        if self.lik_name == "Gaussian":
            mean, var = self.model.predict_y(xtest)
            self.mean = mean.numpy()
            self.var = var.numpy()
        else:
            # mean of posterior predictive samples
            self.mean, self.var = self.samples_posterior_predictive_distribution(xtest)

        mean_mean = np.mean(self.mean)
        y_max = np.max(self.y)
        mean_max = np.max(self.mean)
        y_min = np.min(self.y)
        mean_min = np.min(self.mean)
        y_mean = np.mean(self.y)
        mean_mean = np.mean(self.mean)

        if self.X.shape[0] < 100:
            diff = 0
        else:
            diff = 1
       
        if self.kernel_type == 'Constant' and self.models_number == 2:
            if mean_min < y_min or mean_max > y_max or mean_mean == 0.0:
                fit = self.fit_GP(True)
        
        if y_mean > 0.0:
            diff_mean = abs(round((mean_mean - y_mean) / y_mean))
            if (
                diff_mean > diff
                and mean_min < y_min
                or diff_mean > diff
                and mean_max > y_max
                or mean_mean == 0.0
            ):
                fit = self.fit_GP(True)
    
    '''
    Sample the count distribution to calculate the variance of non-Gaussian distribution
    '''
    def generate_Samples_from_distribution(self, mean):

        y = []
        if self.lik_name == "Poisson":
            for i in range(mean.shape[0]):
                y.append(ss.poisson.rvs(mean[i], size=500))

        if self.lik_name == "Negative_binomial":
            if self.model.likelihood.alpha.numpy() == 0:
                for i in range(mean.shape[0]):
                    y.append(ss.poisson.rvs(mean[i], size=500))

            else:
                r = 1.0 / self.model.likelihood.alpha.numpy()  # r  number of failures
            prob = r / (mean + r)  # p probability of success
            for i in range(mean.shape[0]):
                y.append(ss.nbinom.rvs(r, prob[i], size=500))

        if self.lik_name == "Zero_inflated_negative_binomial":
            r = 1.0 / self.model.likelihood.alpha.numpy()  # r  number of failures
            prob = r / (mean + r)  # p probability of success
            km = self.model.likelihood.km.numpy()  # Michaelin-Menten (MM) constant
            psi = 1.0 - (mean / (km + mean))  # psi probability of zeros
            for i in range(mean.shape[0]):
                B = ss.bernoulli.rvs(size=1, p=1 - psi[i])
                if B == 0:
                    y.append(np.zeros(500))
                else:
                    y.append(ss.nbinom.rvs(r, prob[i], size=500))
        y = np.vstack(y)
        return y
    
    '''
    Generate smoothed samples from GP for non-Gaussian likelihood to set the mean and variance of the posterior predictive distribution
    '''
    def samples_posterior_predictive_distribution(self, xtest):

        var = []
        f_samples = []
        for i in range(20): # 20 samples
            f_samples.append(self.model.predict_f_samples(xtest, 5)) # sample 5 each time
            f = np.vstack(f_samples)
            link_f = np.exp(f[:, :, 0])
            var.append(self.generate_Samples_from_distribution(np.mean(link_f, 0)).T)

        var = np.vstack(var)
        if self.kernel_type == 'Branching':
            mean = np.mean(link_f, axis=0)
        else:
            mean = np.mean(var, axis=0)
            # smooth samples using savgol_filter
            mean = savgol_filter(np.mean(var, axis=0), int(xtest.shape[0] / 2) + 1, 3) 
            mean = [(i > 0) * i for i in mean]
        return mean, var

    '''
    load and predict single GP function
    '''
    def load_predict_model(self,likelihood="Negative_binomial", txt = 'GP', models_number = 1, kernel_type = 'RBF',predict = True):
        
        self.lik_name = likelihood
        self.kernel_type = kernel_type
        self.models_number = models_number
        
        params = {}
        params["likelihood"] = self.lik_name
        self.y = self.y_copy
        self.y = self.y.reshape([self.X.shape[0], 1])
        xtest = np.linspace(np.min(self.X) - 0.1, np.max(self.X) + 0.1, 100)[:, None]
        
        self.optimize = False
        self.init_hyper_parameters(reset=False)
        file_name = self.get_file_name(txt)
        
        successed_fit = self.fit_GP()
        # restore check point
        if successed_fit:
            ckpt = tf.train.Checkpoint(model=self.model, step=tf.Variable(1))
            ckpt.restore(file_name)

            if predict:
                if self.lik_name == "Gaussian":
                    mean, var = self.model.predict_y(xtest)
                    mean = mean.numpy()
                    var = var.numpy()

                else:
                    mean, var = self.samples_posterior_predictive_distribution(
                        xtest
                    )

            else:
                mean = var = 0
              
        params["mean"] = mean
        params["var"] = var
        params["model"] = self.model
        
        return params
    
    '''
    Save GP as tensorflow check point
    '''
    def save_GP(self,log_likelihood,txt = 'GP'):
        if not np.isnan(log_likelihood):
            filename = self.get_file_name(txt)
            ckpt = tf.train.Checkpoint(model=self.model, step=tf.Variable(1))
            ckpt.write(filename)
    
    
    '''
    Return tensorflow check point file name
    '''
    def get_file_name(self,txt):
       
        if not os.path.exists(self.folder_name):
            os.mkdir(self.folder_name)
     
        filename = self.folder_name + self.lik_name + "_"
        if self.sparse:
            filename += "sparse_"
        
        filename += txt
          
        return filename
    