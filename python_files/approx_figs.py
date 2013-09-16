#
#  Kernel smoother and Chebyshev approximation figures
#  Jeno Pal and John Stachurski, May 2012
#

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from cheby_approx import *
from scipy.interpolate import interp1d

## Choose which figure you want: Kernel smoother approximation or Chebyshev
#flag = 'kernel' 
flag = 'cheby'

# Define the two functions f and g (both quite arbitrary)
base_grid = [0, 0.25, 0.5, 0.75, 1]
shape_vals = [0, 0.25, 0.28, 0.25, 0]
f = interp1d(base_grid, shape_vals, kind='cubic', bounds_error=False, fill_value=0)
g = lambda x: 0.01 * np.sin(-11 * x**2) - 0.1 - 0.01 * x**2

# Set up the plot grid
plot_grid = np.linspace(0, 1, 200)

# Now the plots
if flag == 'kernel':

    Kgrid_len = 5
    Kgrid = np.linspace(0, 1, Kgrid_len)
    h = 7

    def psi(t):
        return np.exp(-(t * h)**2)

    def weights(x):
        ws = psi(abs(x - Kgrid))
        return ws / np.sum(ws)

    def Kf(x):
        return np.sum(f(Kgrid) * weights(x))

    def Kg(x):
        return np.sum(g(Kgrid) * weights(x))

    plt.plot(plot_grid, f(plot_grid), 'k--', lw=2)
    plt.plot(plot_grid, g(plot_grid), 'k--', lw=2, label='function')
    plt.plot(plot_grid, [Kf(p) for p in plot_grid], 'k-', lw=1, label='approximation')
    plt.plot(plot_grid, [Kg(p) for p in plot_grid], 'k-', lw=1)


if flag == 'cheby':
    order = 3
    numNodes = 5
    approx = CR(0, 1, order, numNodes)
    approx.set_vals(f(approx.grid))
    plt.plot(plot_grid, f(plot_grid), 'k--', lw=2, label='function')
    plt.plot(plot_grid, approx(plot_grid), 'k-', lw=1, label='approximation')
    plt.plot(plot_grid, g(plot_grid), 'k--', lw=2)
    approx.set_vals(g(approx.grid))
    plt.plot(plot_grid, approx(plot_grid), 'k-', lw=1)

plt.ylim((-0.2, 0.37))
plt.legend(loc=2, frameon=False)
plt.show()

