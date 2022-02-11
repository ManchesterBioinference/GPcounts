import numpy as np
import tensorflow as tf
from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.likelihoods import ScalarLikelihood
from gpflow.utilities import positive


class NegativeBinomial(ScalarLikelihood):
    """
    alpha: dispersion parameter
    scale: to adjust the mean of the negative binomial by a scale factor
    """
    def __init__(self, alpha=1.0, invlink=tf.exp, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = Parameter(alpha, transform=positive(), dtype=default_float())
        self.scale = np.array(scale, dtype=default_float())
        self.invlink = invlink
  

    def _scalar_log_prob(self, F, Y):
        
        return negative_binomial(self.invlink(F) * self.scale, Y, self.alpha)
        
    def _conditional_mean(self, F):
        return self.invlink(F) * self.scale
       

    def _conditional_variance(self, F):
        m = self.invlink(F) * self.scale
        return m + m ** 2 * self.alpha


def negative_binomial(m, Y, alpha):
    """
    P(Y) = Gamma(k + Y) / (Y! Gamma(k)) * (m / (m+k))^Y * (1 + m/k)^(-k)
    """
    k = 1 / alpha
    return (
        tf.math.lgamma(k + Y)
        - tf.math.lgamma(Y + 1)
        - tf.math.lgamma(k)
        + Y * tf.math.log(m / (m + k))
        - k * tf.math.log(1 + m * alpha)
    )


class ZeroInflatedNegativeBinomial(ScalarLikelihood):
    """
    alpha: dispersion parameter
    km: Michaelis-Menten constant
    """
    def __init__(self, alpha=1.0, km=1.0, invlink=tf.exp, **kwargs):
        super().__init__(**kwargs)
        self.alpha = Parameter(alpha, transform=positive(), dtype=default_float())
        self.km = Parameter(km, transform=positive(), dtype=default_float())

        self.invlink = invlink

    def _scalar_log_prob(self, F, Y):
        """
        P(Y) = psi + (1-psi) * NB(y_i=0)+(1-psi) NB(y_i!= 0)
        
        """
        m = self.invlink(F)
        psi = 1.0 - (m / (self.km + m)) # estimete the zeros using Michaelis-Menten equation
        comparison = tf.equal(Y, 0)
        nb_zero = -tf.math.log(1.0 + m * self.alpha) / self.alpha
        log_p_zero = tf.reduce_logsumexp(
            [tf.math.log(psi), tf.math.log(1.0 - psi) + nb_zero], axis=0
        )
        log_p_nonzero = tf.math.log(1.0 - psi) + negative_binomial(m, Y, self.alpha)
        
        return tf.where(comparison, log_p_zero, log_p_nonzero)

    def _conditional_mean(self, F):
        m = self.invlink(F)
        psi = 1.0 - (m / (self.km + m))
        return m * (1 - psi)

    def _conditional_variance(self, F):
        m = self.invlink(F)
        psi = 1.0 - (m / (self.km + m))
        return m * (1 - psi) * (1 + (m * (psi + self.alpha)))
