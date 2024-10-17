# ----------------------------------------------------------------------
# Author : Zhenyu Yao
#          Date: 14 Oct 2024
#          Email : 15255137027@163.com/yaozhenyu2019@outlook.com
# ----------------------------------------------------------------------

import numpy as np
import scipy.stats as stats
import json
from utils import multinomial_resampling, stratified_resampling, state_init_dens, state_trans_dens, obs_lik_dens, data_generation
from conditional_SMC import CSMC
from particle_Gibbs_sampler import PGSampler
from matplotlib import pyplot as plt

np.random.seed(666)

# use the experimental setting in the paper 3.1
step = 50
N = 500
mu = state_init_dens
f = state_trans_dens
g = obs_lik_dens
sigma_v_gen = 1
sigma_w_gen = 0.5
sigma_v_0 = 5
sigma_w_0 = 5
a = 0.01
b = 0.01
iteration = 500
burnin = 200

# generate ground truth data x, y
x_true, y_true = data_generation(mu=mu, f=f, g=g, sigma_v=sigma_v_gen,
                    sigma_w=sigma_w_gen, step=step, file=None)

# build a particle Gibbs sampler and generate samples from it
pgs = PGSampler(mu=mu, f=f, g=g, sigma_v=sigma_v_0, sigma_w=sigma_w_0)
pgs.sample_generation(y=y_true, sigma_v_0=sigma_v_0, sigma_w_0=sigma_w_0, a=a, b=b, num_particle=N, iteration=iteration)

# read the result
sigma_v_est = pgs.get_sigma_v()
sigma_w_est = pgs.get_sigma_w()

# plot histogram and trace plot of sigma_v
bins = int(np.floor(np.sqrt(iteration - burnin)))
plt.subplot(1, 2, 1)
plt.hist(sigma_v_est[burnin:iteration], bins, density=True, facecolor = 'cyan')
plt.xlabel("sigma_v")
plt.ylabel("estimated posterior density")
plt.axvline(sigma_v_gen, color='k', linestyle='--')

plt.subplot(1, 2, 2)
plt.plot(sigma_v_est[burnin:iteration], color='k', alpha=0.5)
plt.xlabel("iteration")
plt.ylabel("sigma_v")
plt.axhline(sigma_v_gen, color='k', linestyle='--')

plt.tight_layout()

plt.savefig('plot/sigma_v.pdf', format='pdf')
plt.show()
plt.clf()

# plot histogram and trace plot of sigma_w
bins = int(np.floor(np.sqrt(iteration - burnin)))
plt.subplot(1, 2, 1)
plt.hist(sigma_w_est[burnin:iteration], bins, density=True, facecolor = 'cyan')
plt.xlabel("sigma_w")
plt.ylabel("estimated posterior density")
plt.axvline(sigma_w_gen, color='k', linestyle='--')

plt.subplot(1, 2, 2)
plt.plot(sigma_w_est[burnin:iteration], color='k', alpha=0.5)
plt.xlabel("iteration")
plt.ylabel("sigma_w")
plt.axhline(sigma_w_gen, color='k', linestyle='--')

plt.tight_layout()

plt.savefig('plot/sigma_w.pdf', format='pdf')
plt.show()
plt.clf()

# scatter plot of (sigma_v, sigma_w)
plt.figure()
plt.scatter(sigma_v_est[burnin:iteration], sigma_w_est[burnin:iteration], color='k', alpha=0.5, label="samples")
plt.scatter(sigma_v_gen, sigma_w_gen, color='r', marker='o', label="true value")
plt.xlabel('sigma_v')
plt.ylabel('sigma_w')
plt.title('trajectories of samples')
plt.legend()

plt.savefig('plot/scatter_plot.pdf', format='pdf')
plt.show()




