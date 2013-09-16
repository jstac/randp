
import numpy as np

class KNN:
    """
    Provides nearest neighbor approximation in one dimension.  Note that the
    grid must be increasing.
    """

    def __init__(self, grid, vals):
        """Parameters: grid and vals are sequences or arrays containing the (x,y)
        interpolation points."""
        self.grid, self.vals = np.array(grid), np.array(vals)
        self.k = len(grid)

    def set_vals(self, new_vals):
        self.vals = np.array(new_vals)

    def __call__(self, z):
        indices = self.grid.searchsorted(z) 
        # Some indices will be set to len(grid).  Subtract 1 from those i.
        indices = np.where(indices == self.k, indices - 1, indices)
        return self.vals[indices]
        


