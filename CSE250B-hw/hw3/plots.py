import numpy as np
import matplotlib.pyplot as plt

loss1 = np.load('losses_custom.npy')
loss2 = np.load('losses_random.npy')
accuracy1 = np.load('accuracy_custom.npy')
accuracy2 = np.load('accuracy_random.npy')
loss_opt = .0327

plt.figure(1)
plt.semilogy(loss1, 'b', label='Custom')
plt.semilogy(loss2, 'g', label='Random')
plt.plot([0, 1000], [loss_opt, loss_opt], 'r--')
plt.xlabel('Iteration (100s)')
plt.title('Loss')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(accuracy1, 'b', label='Custom')
plt.plot(accuracy2, 'g', label='Random')
plt.xlabel('Iteration (100s)')
plt.title('Accuracy')
plt.legend()
plt.show()
