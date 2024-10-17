# ----------------------------------------------------------------------
# Author : Zhenyu Yao
#          Date: 14 Oct 2024
#          Email : 15255137027@163.com/yaozhenyu2019@outlook.com
# ----------------------------------------------------------------------

import numpy as np
import scipy.stats as stats
import json

# define some resampling methods
# multimomial resampling
def multinomial_resampling(W, size=0):
    if size == 0:
        n = len(W)
    else:
        n = size

    index_set = np.random.choice(np.arange(len(W)), size=n, replace=True, p=W)
    return index_set

# stratified resampling
def stratified_resampling(W, size=0):
    if size == 0:
        n = len(W)
    else:
        n = size

    pts = (np.arange(n) + np.random.rand(n)) / n
    index_set = np.digitize(pts, np.cumsum(W))
    return index_set

# state initial density(as described in the paper)
def state_init_dens(size=1):
    return stats.norm.rvs(loc=0, scale=np.sqrt(5), size=size)

# state transition density(as described in the paper)
def state_trans_dens(x, t):
    return 1/2*x + 25*x/(1+x**2) + 8*np.cos(1.2*(t+1))

# observation likelihood density(as described in the paper)
def obs_lik_dens(x):
    return x**2/20

# generate ground truth state values and observations
def data_generation(mu=state_init_dens, f=state_trans_dens, g=obs_lik_dens, sigma_v=1,
                    sigma_w=0.5, step=50, file=None):
    x = np.zeros(step)
    y = np.zeros(step)
    x[0] = mu()   #initial state value
    y[0] = g(x[0]) + sigma_w*stats.norm.rvs()   #initial observation value

    for t in range(1, step):
        x[t] = f(x[t-1], t-1) + sigma_v*stats.norm.rvs()
        y[t] = g(x[t]) + sigma_w*stats.norm.rvs()

    # store data in a json file
    if file is not None:
        data = {'x': x.tolist(), 'y': y.tolist()}
        with open(file, 'w+') as data_file:
            data_file.write(json.dumps(data))

    return x, y











