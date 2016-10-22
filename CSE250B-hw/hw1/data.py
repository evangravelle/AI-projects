import matplotlib.pyplot as plt
import numpy as np

# k = 10
data10 = [0.4949, 0.5837, 0.6022]
rand10 = [0.3475, 0.3231, 0.2882]
acc10 = np.mean(data10)
accr10 = np.mean(rand10)
err10 = np.std(data10)
errr10 = np.std(rand10)

# k = 50
data50 = [0.785, 0.7788, 0.7722]
rand50 = [0.5709, 0.6296, 0.5745]
acc50 = np.mean(data50)
accr50 = np.mean(rand50)
err50 = np.std(data50)
errr50 = np.std(rand50)

# k = 100
data100 = [0.8361, 0.8271, 0.8384]
rand100 = [0.7124, 0.6915, 0.733]
acc100 = np.mean(data100)
accr100 = np.mean(rand100)
err100 = np.std(data100)
errr100 = np.std(rand100)

# k = 200
data200 = [0.873, 0.8745, 0.8733]
rand200 = [0.7706, 0.7634, 0.7814]
acc200 = np.mean(data200)
accr200 = np.mean(rand200)
err200 = np.std(data200)
errr200 = np.std(rand200)

# k = 500
data500 = [0.8971, 0.9054, 0.9086]
rand500 = [0.8452, 0.8509, 0.8512]
acc500 = np.mean(data500)
accr500 = np.mean(rand500)
err500 = np.std(data500)
errr500 = np.std(rand500)

# k = 1000
data1000 = [0.9164, 0.9189, 0.9177]
rand1000 = [0.8856, 0.8808, 0.8878]
acc1000 = np.mean(data1000)
accr1000 = np.mean(rand1000)
err1000 = np.std(data1000)
errr1000 = np.std(rand1000)

# k = 5000
acc5000 = 0.9406
accr5000 = 0.9369
err5000 = 0.
errr5000 = 0.

# k = 10000
acc10000 = 0.9521
accr10000 = 0.9479
err10000 = 0.
errr10000 = 0.

x = [10, 50, 100, 200, 500, 1000, 5000, 10000]
acc = [acc10, acc50, acc100, acc200, acc500, acc1000, acc5000, acc10000]
accr = [accr10, accr50, accr100, accr200, accr500, accr1000, accr5000, accr10000]
err = [err10, err50, err100, err200, err500, err1000, err5000, err10000]
errr = [errr10, errr50, errr100, errr200, errr500, errr1000, errr5000, errr10000]
plt.figure()
plt.errorbar(np.log10(x), acc, yerr=err, label='Custom Prototypes')
plt.errorbar(np.log10(x), accr, yerr=errr, label='Random Prototypes')
plt.title('Accuracy with 95% confidence')
plt.xlabel('log(M)')
plt.ylabel('Accuracy of classification')
plt.legend(loc='lower right')
plt.show()
