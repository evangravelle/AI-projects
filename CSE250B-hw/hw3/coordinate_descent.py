import numpy as np
from sklearn import linear_model

mode = 1  # 1 is custom coordinate descent, 2 is random
num_pts = 178
num_att = 13
num_train = 128
num_test = 50
num_iterations = 100000
step_size = .05
step_size_final = .0001
gamma = np.exp(np.log(step_size_final / step_size) / 1000.)
print gamma
eps = 1e-8
reg_term = 10.
np.set_printoptions(2)


def prob(_w, _x):
    p = np.zeros(3)
    p[0] = np.exp(np.dot(_w[:14], _x))
    p[1] = np.exp(np.dot(_w[14:28], _x))
    p[2] = np.exp(np.dot(_w[28:], _x))
    # print 'w = ', _w[:14]
    # print 'x = ', _x
    # print 'prob = ', p
    p /= np.sum(p)
    return p


def loss(_w, _train_data, _reg_term):
    l = np.linalg.norm(_w) / _reg_term
    # print 'w = ', w
    # print 'data = ', _train_data[0, 1:]
    # print 'prob = ', prob(_w, _train_data[0, 1:])
    for _i in xrange(num_train):
        l -= np.log(prob(_w, _train_data[_i, 1:])[int(_train_data[_i, 0]) - 1])
    return l


# first column is label, middle columns are data, last column is zero for offset
data = np.ones((num_pts, num_att + 2))
with open('wine.data') as data_file:
    for index, line in enumerate(data_file):
        data[index, :14] = [float(i) for i in line.split(',')]


np.random.shuffle(data)
train_data = data[:num_train, :]
test_data = data[num_train:, :]

# training loop
mid_ctr = 0
ind = 0
ctr = 0
w = np.zeros(42)
dw = np.zeros(42)
loss_change = 1000*np.ones(42)
loss1 = np.zeros(1000)
accuracy = np.zeros(1000)
prev_best = 1000.
predictions = np.zeros(num_test)
for k in xrange(num_iterations):
    dw[ind] = step_size
    lloss = loss(w - dw, train_data, reg_term)
    mloss = loss(w, train_data, reg_term)
    rloss = loss(w + dw, train_data, reg_term)
    # print 'losses = ', [lloss, mloss, rloss]
    if abs(lloss - min([lloss, mloss, rloss])) < eps:
        w[ind] -= step_size
        loss_change[ind] = abs(lloss - mloss)
        prev_best = loss_change[ind]
        mid_ctr = 0
    elif abs(rloss - min([lloss, mloss, rloss])) < eps:
        w[ind] += step_size
        loss_change[ind] = abs(rloss - mloss)
        prev_best = loss_change[ind]
        mid_ctr = 0
    else:
        loss_change[ind] = prev_best / 2.
        mid_ctr += 1

    # if mid_ctr >= 100:
    #     step_size /= 1.5
    #     loss_change /= 1.5
    #     mid_ctr = 0
    #     print 'step_size = ', step_size

    # print 'ind =', ind
    if k % 60 == 0:
        loss_change = 1000 * np.ones(42)
        prev_best = 1000.

    if k % 100 == 0:
        step_size *= gamma
        loss1[ctr] = min([lloss, mloss, rloss])
        print 'loss = ', loss(w, train_data, reg_term)
        print 'step_size = ', step_size

        for i in xrange(num_test):
            predictions[i] = np.argmax(prob(w, test_data[i, 1:]))

        accuracy[ctr] = np.sum(np.equal(predictions, test_data[:, 0] - 1)) / float(num_test)
        ctr += 1

    dw[ind] = 0
    if mode == 1:
        ind = np.argmax(loss_change)
    elif mode == 2:
        ind = np.random.random_integers(0, 41)


print 'w = ', w
print 'loss = ', loss(w, train_data, reg_term)
print 'accuracy = ', accuracy[ctr - 1]

regr = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', C=reg_term, tol=1e-8, max_iter=num_iterations)
regr.fit(train_data[:, 1:14], train_data[:, 0])
w_regr = np.column_stack((regr.coef_, regr.intercept_)).reshape(-1)
print 'w_regr = ', w_regr
print 'regr_loss = ', loss(w_regr, train_data, reg_term)
print 'regr_accuracy = ', regr.score(test_data[:, 1:14], test_data[:, 0])

if mode == 1:
    np.save('losses_custom.npy', loss1)
    np.save('accuracy_custom.npy', accuracy)
elif mode == 2:
    np.save('losses_random.npy', loss1)
    np.save('accuracy_random.npy', accuracy)
