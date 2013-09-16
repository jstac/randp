

from __future__ import division
import numpy as np

class KS1D:

    def __init__(self, grid, h=1, vals=None, func=None):
        self.h, self.grid = h, grid
        self.k = len(grid)
        if func:
            self.vals = func(grid)
        else:
            self.vals = vals

    def set_vals(self, new_vals):
        self.vals = new_vals

    def psi(self, t):
        return np.exp(-t**2)

    def __call__(self, y):
        y = np.array(y)
        y = y.flatten()
        x = self.grid.flatten()
        # Some ugly broadcasting trick to do an outer 'product' with subtraction
        Psi = (x[:, np.newaxis] - y) / self.h
        Psi = self.psi(Psi)
        out = np.dot(self.vals, Psi) / np.dot(np.ones((1, self.k)), Psi)
        return out.flatten()

    def one_point_eval(self, y):
        "Depreciated.  For checking only."
        ws = self.psi((y - self.grid) / self.h)
        weights = ws / np.sum(ws)
        return np.sum(self.vals * weights)



