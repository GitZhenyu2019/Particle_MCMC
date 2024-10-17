# ----------------------------------------------------------------------
# Author : Zhenyu Yao
#          Date: 14 Oct 2024
#          Email : 15255137027@163.com/yaozhenyu2019@outlook.com
# ----------------------------------------------------------------------

import numpy as np
import scipy.stats as stats
from utils import multinomial_resampling, stratified_resampling

class CSMC(object):
    """
    params:
        mu-initial density, f-transition density, g-likelihood density, sigma_v-transition std,
        sigma_w-likelihood std, step-length of particle trajectory, N-number of particles, x-matrix of states,
        A-index of the parent (A[i, j] denotes the the parent of x[i, j+1] is x[A[i, j], j]), W-normalised weights
    """
    def __init__(self, mu, f, g, sigma_v, sigma_w):
        self.mu = mu
        self.f = f
        self.g = g
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.step = 50
        self.N = 500
        self.x = np.zeros((self.N, self.step))
        self.A = np.zeros((self.N, self.step), dtype=int)
        self.W = np.zeros((self.N, self.step))
        self.resampling_method = stratified_resampling

    # resampling step
    def resampling(self, t):
        self.A[:, t] = self.resampling_method(self.W[:, t]).astype(int)

    # forward propagation step
    def propagation(self, t):
        # first predict the next state corresponding to the current state without re-permutation
        x_new = self.f(self.x[:, t-1], t-1) + self.sigma_v*stats.norm.rvs(size=self.N)
        # re-permutation states according to indexes of parents
        self.x[:, t] = x_new[self.A[:, t-1]]

    # weights computing step
    def weighting(self, y_t, t):
        # compute log weights, subtract the maximum value and recover to improve numerical stability
        log_w = stats.norm.logpdf(self.g(self.x[:, t]), y_t, self.sigma_w)
        w_max = max(log_w)
        w = np.exp(log_w-w_max)     # unnormalised weights
        self.W[:, t] = w / sum(w)   # normalised weights

    # generate all particles and record their weights and indexes of parents
    # for the prespecified particle, we run the standard SMC as it does not
    def particle_generation(self, x_pre, y, num_particle=500):
        # set parameters
        self.step = len(y)
        self.N = num_particle
        # x, B, W will adjust their shapes automatically
        # initialize states at time 0 by mu
        self.x[:, 0] = self.mu(size=len(self.x[:, 0]))
        # fix the first particle as the prespecified particle
        self.x[0, 0] = x_pre[0]
        # weights computing at time 0
        self.weighting(y[0], 0)

        # start interation
        for t in range(1, self.step):
            # resampling step
            self.resampling(t-1)
            self.A[0, t-1] = 0      # fix A of the first particle

            # propagation step
            self.propagation(t)
            self.x[0, t] = x_pre[t]     # fix the states of the first particle

            # weights computing step
            self.weighting(y[t], t)

        # miss A[:, -1], define A[k, -1] = k according to SMC
        self.A[:, -1] = np.arange(self.N)
        self.A = self.A.astype(int)

    # sample a particular particle from the estimated posterior
    def particle_sampling(self, x_pre, y, num_particle=500):
        self.step = len(y)
        self.N = num_particle

        # apply particle_generation to yield x, A, W
        self.particle_generation(x_pre, y, num_particle)
        # sample a particle
        x_new = np.zeros(self.step)
        indx = multinomial_resampling(self.W[:, self.step-1], 1)[0]

        # backtrack to get the path of particle
        for t in reversed(range(self.step)):
            indx = self.A[indx, t]
            x_new[t] = self.x[indx, t]

        return x_new

















