import numpy as np
import matplotlib.pyplot as plt

loss1 = np.load('losses_custom.npy')
loss2 = np.load('losses_random.npy')
loss_opt = .0327

plt.semilogy(loss1, 'b', label='Custom')
plt.semilogy(loss2, 'g', label='Random')
plt.plot(loss_opt, 'r--')
plt.xlabel('Iteration (100s)')
plt.ylabel('Loss')
plt.title('Coordinate Descent')
plt.legend()
plt.show()
