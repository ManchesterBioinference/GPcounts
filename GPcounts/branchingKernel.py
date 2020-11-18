import gpflow
import numpy as np
import tensorflow as tf
from gpflow.config import default_float
from gpflow.config import default_jitter

from matplotlib import pyplot as plt


def plotCovarianceMatrix(covMatrix, ax, xLabel='', yLabel='', n=None, replicate_no=1):
    im = ax.matshow(covMatrix, cmap=plt.cm.jet)
    plt.colorbar(im, fraction=0.046, pad=0.04, ax=ax)

    if n is None:
        n = len(covMatrix) / 2

    ax.set_xticks((int(n * replicate_no / 2), int(3 * n * replicate_no / 2)))
    ax.set_yticks((int(n * replicate_no / 2), int(3 * n * replicate_no / 2)))
    ax.set_xticklabels(xLabel)
    ax.set_yticklabels(yLabel)
    ax.tick_params(labelsize=16)

class BranchKernel(gpflow.kernels.Kernel):

    def __init__(self, base_kern, branchingPoint, noise_level=1e-6):
        ''' branchPtTensor is tensor of branch points of size F X F X B where F the number of
        functions and B the number of branching points '''
        super().__init__()
        self.kern = base_kern
        self.xp = branchingPoint
        self.noise_level = noise_level

    def K(self, X, Y=None, DE=None):
        if Y is None:
            Y = X  # hack to avoid duplicating code below
            square = True
        else:
            square = False

        # if DE is None:
        # X[np.where(X[:, 0] <= self.xp), 1] = 1
        # Y[np.where(Y[:, 0] <= self.xp), 1] = 1

        t1s = tf.expand_dims(X[:, 0], 1)  # N X 1
        t2s = tf.expand_dims(Y[:, 0], 1)
        i1s = tf.expand_dims(X[:, 1], 1)
        i2s = tf.expand_dims(Y[:, 1], 1)

        i1s_matrix = tf.tile(i1s, tf.reverse(tf.shape(i2s), [0]))
        i2s_matrix = tf.tile(i2s, tf.reverse(tf.shape(i1s), [0]))
        same_functions = tf.equal(i1s_matrix, tf.transpose(i2s_matrix), name='FiEQFj')
        # i2s_matrixT = tf.transpose(i2s_matrix)

        Ktts = self.kern.K(t1s, t2s)  # N*M X N*M
        # print(Ktts)
        # Bs = tf.expand_dims(tf.expand_dims(self.xp, axis=0), axis=1)
        # Bs = get_Bv([self.xp,], [0,])
        # print(Bs)
        Bs = np.ones((1,1)) * self.xp
        kbb = self.kern.K(Bs) + tf.linalg.diag(tf.ones(tf.shape(Bs)[:1], dtype=default_float())) * default_jitter()
        # print(kbb)
        Kbbs_inv = tf.linalg.inv(kbb, name='invKbb')  # B X B
        # print(Kbbs_inv)
        Kb1s = self.kern.K(t1s, Bs)  # N*m X B
        Kb2s = self.kern.K(t2s, Bs)  # N*m X B
        a = tf.linalg.matmul(Kb1s, Kbbs_inv)
        K_crosss = tf.linalg.matmul(a, tf.transpose(Kb2s), name='Kt1_Bi_invBB_KBt2')
        K_s = tf.where(same_functions, Ktts, K_crosss, name='selectIndex')

        if square:
            return K_s + tf.eye(tf.shape(K_s)[0], dtype=default_float())*self.noise_level
        else:
            return K_s

    def K_diag(self, X, dim=0):
        # diagonal is just single point no branch point relevant
        return tf.linalg.diag_part(self.kern.K(X)) + self.noise_level

