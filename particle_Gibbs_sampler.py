# ----------------------------------------------------------------------
# Author : Zhenyu Yao
#          Date: 14 Oct 2024
#          Email : 15255137027@163.com/yaozhenyu2019@outlook.com
# ----------------------------------------------------------------------

import numpy as np
import scipy.stats as stats
from utils import multinomial_resampling, stratified_resampling, state_init_dens, state_trans_dens, obs_lik_dens, data_generation
from conditional_SMC import CSMC

class PGSampler(object):
    """
    params:
        mu-initial density, f-transition density, g-likelihood density, sigma_v-transition std,
        sigma_w-likelihood std, step-length of particle trajectory, N-number of particles, x-matrix of states,
        iteration-number of MCMC steps, a and b-parameters of inverse Gamma prior
    """
    def __init__(self, mu, f, g, sigma_v, sigma_w):
        self.csmc = CSMC(mu, f, g, sigma_v, sigma_w)

    def sample_generation(self, y, sigma_v_0, sigma_w_0, a, b, num_particle=500, iteration=500):
        self.N = num_particle
        self.step = len(y)

        # store samples
        self.x = np.zeros((iteration, self.step))
        self.sigma_v_est = np.zeros(iteration)
        self.sigma_w_est = np.zeros(iteration)

        # initialization
        x_pre = np.zeros(self.step) # initialize the prespecified particle as 0
        self.csmc.sigma_v = sigma_v_0   # initialize csmc parameters
        self.csmc.sigma_w = sigma_w_0
        self.x[0, :] = self.csmc.particle_sampling(x_pre, y, self.N)    # initialize the state
        self.sigma_v_est[0] = sigma_v_0
        self.sigma_w_est[0] = sigma_w_0

        # run Gibbs sampling for iteration times and record the estimates
        for t in range(1, iteration):
            # update parameters based on the fact that inverse Gamma is the
            # conjugate prior of Gaussian likelihood, so we have inverse Gamma posterior, formula
            # can be find in https://andrewwang.rbind.io/courses/bayesian_statistics/notes/Ch3_h.pdf
            err_sigma_v = self.x[t-1, 1:self.step] - self.csmc.f(self.x[t-1, np.arange(self.step-1)], np.arange(self.step-1))
            err_sigma_v = sum(err_sigma_v**2)
            self.sigma_v_est[t] = np.sqrt(stats.invgamma.rvs(a=a+(self.step-1)/2, scale=b+err_sigma_v/2, size=1))
            err_sigma_w = y - self.csmc.g(self.x[t-1, :])
            err_sigma_w = sum(err_sigma_w**2)
            self.sigma_w_est[t] = np.sqrt(stats.invgamma.rvs(a=a+self.step/2, scale=b+err_sigma_w/2, size=1))

            # update states
            self.csmc.sigma_v = self.sigma_v_est[t]
            self.csmc.sigma_w = self.sigma_w_est[t]
            self.x[t, :] = self.csmc.particle_sampling(self.x[t-1, :], y, self.N)

    def get_sigma_v(self):
        return self.sigma_v_est

    def get_sigma_w(self):
        return self.sigma_w_est

    def get_x(self):
        return self.x













