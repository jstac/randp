"""

Stochastic optimal growth model 

Jeno Pal and John Stachurski, June 2012

This file contains the basic description of the model, the parameters,
and the function implementing the Bellman operator.

The state variable is y = f(k) and the Bellman equation is

 V(y) = max_{0 <= k <= y} [U(y - k) + beta E V(z f(k))]

"""

import numpy as np
from scipy.optimize import fminbound
from scipy.integrate import fixed_quad
from scipy.stats import lognorm

alpha, beta, sigma = 0.33, 0.95, 0.25   # Parameters

def U(c): 
    return np.log(c)                     # Utility
def f(k): 
    return k**alpha                      # Production 

Z = lognorm(sigma).rvs(100)              # lognormal draws with mu = 0 and sd = sigma
phi = lognorm(sigma).pdf                 # lognormal density with mu = 0 and sd = sigma

def v_true(x):
    "The true value function, vectorized"
    theta = alpha * beta  # Optimal share of investment for this model
    gamma = (np.log(1 - theta) + theta * np.log(theta) / (1 - theta)) 
    return np.log(x) / (1 - theta) + gamma / (1 - beta)

def maximum(h, a, b):
    return h(fminbound(lambda x: -h(x), a, b))  

def global_max(h, a, b, numInits = 3):
    """
    Alternative maximization routing that dissects the interval [a,b] into
    subintervals, and optimizes separately with fminbound (in order to make sure
    we have the global maximum).
    """
    inits = np.linspace(a,b,numInits)
    intervals = [(inits[i], inits[i+1]) for i in
            range(len(inits) -1)]
    max_vals = []
    for inter in intervals:
        low, high = inter
        maxval = maximum(h, low, high)
        max_vals.append(maxval)
    return max(max_vals)

def bellman(w, grid, maxfunc=global_max, integration_method='quadrature'):
    """ 
    The approximate Bellman operator.  Regarding numerical integration, the
    default is fixed order Gaussian quadrature and the alternative is Monte
    Carlo.
    Parameters: 
        * w is a (vectorized) function object (a callable object)
        * grid is a numpy array
        * maxfunc: the function used for maximization
    Returns: The values of Tw on grid
    """
    vals = []
    for y in grid:
        if integration_method == 'quadrature':
            h = lambda k: U(y - k) + beta * fixed_quad(lambda z: \
                    w(z * f(k)) * phi(z), 0.0001, 4)[0]
        else:  
            h = lambda k: U(y - k) + beta * np.mean(w(Z * f(k)))
        vals.append(maxfunc(h, 0, y))
    return vals

