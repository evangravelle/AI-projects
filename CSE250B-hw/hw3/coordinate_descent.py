import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.optimize import minimize


def prob(w, x, y):
    p = np.zeros(3)
    p[0] = np.exp(np.dot(w[:14], x))
    p[1] = np.exp(np.dot(w[14:28], x))
    p[2] = np.exp(np.dot(w[28:], x))
    p /= np.sum(p)
    return p

num_pts = 178
num_att = 13
num_train = 128
num_test = 50
num_iterations = 10000

data = np.zeros((num_pts, num_att + 1))
with open('wine.data') as data_file:
    for index, line in enumerate(data_file):
        data[index, :] = [float(i) for i in line.split(',')]

np.random.shuffle(data)
train_data = data[:num_train]
test_data = data[num_train:]

w = np.zeros(42)
ind = 0
for iter in xrange(num_iterations):
    func = lambda z: z^2
    res = minimize(func, w[ind], method='Nelder-Mead', tol=1e-6)
    w[ind] = res.x
    ind += 1

regr = linear_model.LogisticRegression(solver='sag', multi_class='multinomial', max_iter=10000)
regr.fit(train_data[:, 1:], train_data[:, 0])
# print regr.coef_, regr.intercept_
# print regr.score(test_data[:, 1:], test_data[:, 0])

predictions = regr.predict(test_data[:, 1:])

accuracy = np.sum(np.equal(predictions, test_data[:, 0])) / float(num_test)
print accuracy

plt.scatter(train_data[:, 1], train_data[:, 2])
plt.show()
