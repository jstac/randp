
from scipy import interp

class LinInterp:
    "Provides linear interpolation in one dimension."

    def __init__(self, grid, vals):
        """Parameters: grid and vals are sequences or arrays containing the (x,y)
        interpolation points."""
        self.grid, self.vals = grid, vals

    def set_vals(self, new_vals):
        self.vals = new_vals

    def __call__(self, z):
        return interp(z, self.grid, self.vals)


