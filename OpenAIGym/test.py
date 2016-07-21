import numpy as np

theta = np.arange(18)
print theta.reshape((9, 2), order="F").ravel(order="F")
print theta.reshape((2, 3, 3))

# theta.reshape((num_row, num_col, num_actions)).swapaxes(1, 2)