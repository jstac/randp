"""

Stability and instability in fitted value iteration when using
different approximation and integration schemes.  

Jeno Pal and John Stachurski, June 2012

For a quick start, run the file and then type "select()" or "gen_plot()".

"""

from __future__ import division
from fitted_val_it import *
from scipy import absolute as abs
from lininterp import LinInterp        
from knn import KNN
from simple_ks import KS1D
from cheby_approx import CR
import matplotlib.pyplot as plt

xmin, xmax = 1e-5, 1   # Grid space
gridsize = 150
grid = np.linspace(xmin, xmax, gridsize)

def iterate(v=U, 
        T=10,             # Number of iterates
        flag='chebyshev', # Approximation type
        cheby_order=12,   # Polynomial order in the Chebyshev case
        h=0.5):           # Smoothing parameter, kernel smoother case
    """
    Implements FVI with a given approximation scheme.  The flag and the code
    below create an object called 'approximator'.  This object must have the
    following two methods: set_vals() and __call__().  The first is used to
    set new function values (for approximating a new function), while the
    second is used to evaluate the approximation, taking grid points, function
    values, etc. as given.  The special name __call__ means that the object
    itself can be treated like a function.  For example, approximator(xvec) is
    equivalent to approximator.__call__(xvec).
    """
    if flag == 'chebyshev':
        approximator = CR(xmin, xmax, cheby_order, gridsize, extended=False)
    if flag == 'lin_interp':
        approximator = LinInterp(grid, grid)
    if flag == 'nearest':
        approximator = KNN(grid, grid)
    if flag == 'ks':
        approximator = KS1D(grid, h=h, vals=grid)
    print """
    i = iteration
    e1 = distance from current iterate to true value function
    e2 = distance between current and last iterate
    """
    count = 0
    error_vals = []
    while count < T:
        # Record the function values of the current iterate, and plot
        current_vals = v(grid)
        grey = 1 - (count + 1.0) / T
        plt.plot(grid, current_vals, 'k-', color=str(grey), lw=2)
        # Update the approximator with new function values 
        if flag == 'chebyshev':
            new_vals = bellman(v, grid, integration_method='quadrature')
        else:
            new_vals = bellman(v, grid, integration_method='monte_carlo')
        approximator.set_vals(new_vals)
        v = approximator
        # Print errors
        e1 = max(abs(v(grid) - v_true(grid)))
        e2 = max(abs(v(grid) - current_vals))
        print "i = ", count, "; e1 = ", e1, "; e2 = ", e2
        error_vals.append(e2)
        count = count + 1
    return error_vals


def gen_plot():
    #iterate(T=8, flag='chebyshev', cheby_order=10)
    iterate(T=25, flag='lin_interp')
    plt.plot(grid, v_true(grid), 'k--', lw=2)
    plt.xlim((xmin, xmax))
    plt.show()


def select():
    """
    A function that gives an illustration for different approximation schemes.
    First select approximation scheme.   """

    while 1:
        prompt_string = """
        Select the approximation scheme:

        * Chebyshev polynomials (1)
        * Linear interpolation (2)
        * Nearest neighbors (3)
        * Kernel smoother (4)
        
        """
        answer = raw_input(prompt_string)
        if '1' in answer:
            flag = 'chebyshev'
            break
        if '2' in answer:
            flag = 'lin_interp'
            break
        if '3' in answer:
            flag = 'nearest'
            break
        if '4' in answer:
            flag = 'ks'
            break
        print "Input not understood."

    iterate(flag=flag)


def gen_table():
    """
    Generates a table showing the sequence of distances between successive
    iterates for several different approximation methods.
    """
    rows = 40
    outfile = "table_data.txt"
    ch = iterate(T=rows, flag='chebyshev')
    lin = iterate(T=rows, flag='lin_interp')
    nn = iterate(T=rows, flag='nearest')
    ks1 = iterate(T=rows, flag='ks', h=0.25)
    ks2 = iterate(T=rows, flag='ks', h=0.5)
    ks3 = iterate(T=rows, flag='ks', h=0.75)
    f = open(outfile, 'w')
    f.write(r"& lin & nn & ks1 & ks2 & ks3 & cheb")
    f.write("\n")
    f.write(r"\\")
    f.write("\n")
    for i in range(rows):
        line = "%d & %f & %f & %f & %f & %f & %f" % (i+1, lin[i], nn[i],
                ks1[i], ks2[i], ks3[i], ch[i])
        f.write(line)
        f.write("\n")
        f.write(r"\\")
        f.write("\n")
    f.close()

