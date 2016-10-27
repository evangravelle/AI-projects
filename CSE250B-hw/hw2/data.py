import matplotlib.pyplot as plt
import numpy as np

m = [1000, 5000, 10000, 20000]
acc = [.2, .4, .6, .7]
acc_rand = [.1, .3, .5, .7]

plt.figure()
plt.plot(61188, .781, 'o', label='Full vocab')
plt.plot(m, acc, label='Custom vocab')
plt.plot(m, acc_rand, label='Random vocab')
plt.title('Classification Accuracy')
plt.xlabel('log(m)')
plt.ylabel('Probability of accurate classification')
plt.legend(loc='lower right')
plt.show()
