import tensorflow as tf
import numpy as np
from gpflow.likelihoods import ScalarLikelihood
from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.utilities import positive

class NegativeBinomial(ScalarLikelihood):
    def __init__(self, alpha= 1.0 ,invlink=tf.exp, **kwargs):
        super().__init__( **kwargs)
        self.alpha = Parameter(alpha,
                               transform= positive(),
                               dtype=default_float())
        self.invlink = invlink

    def _scalar_log_prob(self, F, Y):
        """
        P(Y) = Gamma(k + Y) / (Y! Gamma(k)) * (m / (m+k))^Y * (1 + m/k)^(-k)
        """
        '''
        m = self.invlink(F)
        k = 1 / self.alpha
                       
        return tf.lgamma(k + Y) - tf.lgamma(Y + 1) - tf.lgamma(k) + Y * tf.log(m / (m + k)) - k * tf.log(1 + m * self.alpha) 
        '''
        
        return negative_binomial(self.invlink(F), Y, self.alpha)
    
    def _conditional_mean(self, F):
        return self.invlink(F)
      
    def _conditional_variance(self, F):
        m = self.invlink(F)
        return m + m**2 * self.alpha

def negative_binomial(m, Y, alpha):
    k = 1 / alpha
    return tf.math.lgamma(k + Y) - tf.math.lgamma(Y + 1) - tf.math.lgamma(k) + Y * tf.math.log(m / (m + k)) - k * tf.math.log(1 + m * alpha)

class ZeroInflatedNegativeBinomial(ScalarLikelihood):
    def __init__(self, alpha = 1.0,km = 1.0, invlink=tf.exp,  **kwargs):
        super().__init__( **kwargs)
        self.alpha = Parameter(alpha,
                               transform= positive(),
                               dtype=default_float())
        self.km = Parameter(km,
                           transform= positive(),
                           dtype=default_float())
        
        self.invlink = invlink

    def _scalar_log_prob(self, F, Y):
        m = self.invlink(F)
        psi = 1. - (m / (self.km + m))
        comparison = tf.equal(Y, 0)
        nb_zero = - tf.math.log(1. + m * self.alpha) / self.alpha
        log_p_zero = tf.reduce_logsumexp([tf.math.log(psi), tf.math.log(1.-psi) + nb_zero], axis=0)
        log_p_nonzero = tf.math.log(1.-psi) + negative_binomial(m, Y, self.alpha)
        return tf.where(comparison, log_p_zero, log_p_nonzero)

    def _conditional_mean(self, F):
        m = self.invlink(F)
        psi = 1. - (m /(self.km + m))
        return m * (1-psi) 
      
    def _conditional_variance(self, F):
        m = self.invlink(F)
        psi = 1. - (m /(self.km + m))
        return m * (1-psi)*(1 + (m * (psi+self.alpha)))
