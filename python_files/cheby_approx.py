
"""

 A class implementing approximation using Chebyshev regression

 Jeno Pal and John Stachurski, June 2012

The class CR implements Chebyshev regression of a given function on the
specified interval, using either extended or standard Chebyshev roots (as
found in "Note on Chebyshev Regression" by Makoto Nakajima).  The difference
between the extended and standard roots is in the mapping of the interval
[xmin, xmax] onto [-1, 1].

"""

import numpy as np
from scipy.special import t_roots, eval_chebyt
from numpy import cos, pi, linspace


class CR:

    def __init__(self, xmin, xmax, order, gridsize, extended=True):
        """
        Approximation over the interval [xmin, xmax].  
            * gridsize: An integer. The number of collocation points
            * order: An integer. The order of the polynomial.
            * extended: A flag indicating whether to use extended or standard
        The function values of the function to be approximated are set using
        the method set_vals().
        """
        assert gridsize > order
        self.xmin = xmin
        self.xmax = xmax
        self.order = order
        self.gridsize = gridsize
        self.extended = extended
        self.roots = self.compute_roots()
        self.grid = self.from_regular(self.roots)

    def set_vals(self, func_vals):
        self.func_vals = func_vals
        self.theta = self.approx_weights()

    def compute_roots(self):
        "Equivalent to return t_roots(self.gridsize)[0].real"
        i = np.array(range(1, self.gridsize + 1), dtype=int)
        return - cos( ((2 * i - 1) * pi) / (2 * self.gridsize) )

    def from_regular(self, z):
        """
        Maps regular interval [-1,1] into [xmax, xmin].
        """
        m = self.gridsize
        if self.extended == True:
            return (( (1.0 / cos(pi/(2*m))) * z + 1)/2)*(self.xmax - self.xmin) + self.xmin
        else:
            return 0.5 * (z + 1) * (self.xmax - self.xmin) + self.xmin

    def to_regular(self, x):
        """
        Maps from [xmax,xmin] back to [-1,1].
        """
        m = self.gridsize
        if self.extended == True:
            return (cos(pi/(2*m)))*((2.0*(x - self.xmin))/(self.xmax - self.xmin) - 1)
        else:
            return 2 * (x - self.xmin) / (self.xmax - self.xmin) - 1

    def approx_weights(self):
        """
        Returns the weights of the Chebyshev regression using Chebyshev
        polynomials up to 'order'.  Number of approximation nodes = gridsize.
        Returns an array of order + 1 floats.
        """
        theta = np.zeros(self.order + 1)
        theta[0] = np.mean(self.func_vals)
        for j in range(1, self.order + 1):
            tj = eval_chebyt(j, self.roots)
            theta[j] = (2.0 / self.gridsize) * np.sum(tj * self.func_vals)
        return theta

    def __call__(self, eval_grid):
        """
        Evaluate the approximation at the points in array eval_grid.
        Returns an array of floats.
        """
        z_vec = self.to_regular(eval_grid)
        j = np.array(range(self.order + 1), dtype=int)
        K = len(z_vec)
        vals = np.empty(K)
        for k in range(K):
            y = np.sum(self.theta * eval_chebyt(j, z_vec[k]))
            vals[k] = y
        return np.array(vals)


