# This script classifies articles using a naive Bayes model on words contained
# in the article. Instead of using the entire training set, a subet of the training set
# is chosen, and performance as a function of number of prototypes is analyzed.
# Evan Gravelle, Fall 2016
import numpy as np
import random
import pickle

num_proto = 1000
num_words = 61188
num_train_docs = 11269
num_test_docs = 7505
num_labels = 20
num_reps = 10

scores_filename = 'scores.p'

train_counts = np.zeros((num_labels, num_words), dtype=np.uint16)
train_labels = np.zeros(num_train_docs, dtype=np.uint16)
test_counts = np.zeros((num_test_docs, num_words), dtype=np.uint16)
test_labels = np.zeros(num_test_docs, dtype=np.uint16)
train_label_prob = np.zeros(num_labels, dtype=np.float64)
train_word_prob = np.zeros((num_labels, num_words), dtype=np.float64)
label_dict = dict()
vocab_dict = dict()

train_lengths = np.sum(train_counts, axis=1)

# Save the data for the prototype set
with open('20news-bydate/train.label') as train_label_file:
    for index, line in enumerate(train_label_file):
        train_labels[index] = int(line) - 1

with open('20news-bydate/train.data') as train_data_file:
    for line in train_data_file:
        doc, word, count = [int(i) for i in line.split()]
        label = train_labels[doc - 1]
        train_counts[label, word - 1] += count

with open('20news-bydate/train.map') as train_map_file:
    for line in train_map_file:
        label_name, label = [i for i in line.split()]
        label_dict[int(label) - 1] = label_name

with open('20news-bydate/test.label') as test_label_file:
    for index, line in enumerate(test_label_file):
        test_labels[index] = int(line) - 1

with open('20news-bydate/test.data') as test_data_file:
    for line in test_data_file:
        doc, word, count = [int(i) for i in line.split()]
        test_counts[doc - 1, word - 1] = count

with open('vocabulary.txt') as vocabulary_file:
    for index, line in enumerate(vocabulary_file):
        vocab_dict[index] = line
print 'here'
# Finds marginal probability of each class
for doc in xrange(num_train_docs):
    train_label_prob[train_labels[doc]] += 1.
train_label_prob /= num_train_docs
print 'train_label_prob = ', train_label_prob

# Finds probability of each word in each class
for label in xrange(num_labels):
    train_word_prob[label, :] = ((train_counts[label, :] + np.ones(num_words, dtype=np.uint16)) /
                                 float(np.sum(train_counts[label, :]) + num_words))

# scores = np.zeros((num_test_docs, num_labels), dtype=np.float64)
# for doc in xrange(num_test_docs):
#     if doc % 100 == 0:
#         print doc
#     for label in xrange(num_labels):
#         scores[doc, label] = (np.log(train_label_prob[label]) +
#                               np.sum(test_counts[doc, :] * np.log(train_word_prob[label, :])))
# with open(scores_filename, 'w') as scores_file:
#     pickle.dump(scores, scores_file)
with open(scores_filename, 'rb') as scores_file:
    scores = pickle.load(scores_file)
# print 'scores[0, :] = ', scores[0, :]
print 'here'
entropies = -np.sum(train_word_prob * np.log(train_word_prob), axis=0)
sorted_words1 = np.argsort(entropies)[0:num_proto]
# print 'sorted_words1[:10] = '
# for i in xrange(10):
#     print vocab_dict[sorted_words1[i]].strip()

variances = np.var(train_word_prob, axis=0)
sorted_words2 = np.argsort(variances)[0:num_proto]
# print 'sorted_words2[:10] = '
# for i in xrange(10):
#     print vocab_dict[sorted_words2[i]].strip()
#     print ' '

# Calculates performance of prototypes
test_predictions = np.argmax(scores, axis=1)
accuracy = np.sum(np.equal(test_predictions, test_labels)) / float(num_test_docs)
print 'here'
# Prints a representative set of words in each class, by taking the words
# with highest probability in each class from the set of prototypes
reps = np.zeros((num_labels, num_reps), dtype=np.uint16)
ctr = np.zeros(num_labels, dtype=np.uint8)
for word in sorted_words1:
    temp = train_word_prob[:, word]
    for ind in xrange(num_labels):
        if ctr[ind] >= 10:
            temp[ind] = -1.
    label = np.argmax(temp)
    reps[label, ctr[label]] = word
    ctr[label] += 1
    if np.min(ctr) >= 10:
        break

# for label in xrange(num_labels):
#     print label_dict[label]
#     for rep in xrange(num_reps):
#         print vocab_dict[reps[label, rep]].strip()
#     print ' '
print 'here'
rand_vocab = random.sample(xrange(num_words), num_proto)

rand_train_counts = np.zeros((num_labels, num_proto), dtype=np.uint16)
red_train_counts = np.zeros((num_labels, num_proto), dtype=np.uint16)
with open('20news-bydate/train.data') as train_data_file:
    for line in train_data_file:
        doc, word, count = [int(i) for i in line.split()]
        if word in sorted_words1:
            label = train_labels[doc - 1]
            # print train_labels
            red_train_counts[label, np.where(sorted_words1 == word)] += count
        if word in rand_vocab:
            label = train_labels[doc - 1]
            # print train_labels
            rand_train_counts[label, np.where(rand_vocab == word)] += count

print 'rand_train_counts[0, :100] = ', rand_train_counts[0, :100]
# Finds probability of each word in each class
rand_train_word_prob = np.zeros((num_labels, num_proto), dtype=np.float64)
red_train_word_prob = np.zeros((num_labels, num_proto), dtype=np.float64)
print 'here'
for label in xrange(num_labels):
    red_train_word_prob[label, :] = ((red_train_counts[label, :] + np.ones(num_proto, dtype=np.uint16)) /
                                     float(np.sum(red_train_counts[label, :]) + num_proto))
    rand_train_word_prob[label, :] = ((rand_train_counts[label, :] + np.ones(num_proto, dtype=np.uint16)) /
                                      float(np.sum(rand_train_counts[label, :]) + num_proto))
print 'np.sum(red_train_word_prob[1, :] = ', np.sum(red_train_word_prob[1, :])
print 'np.sum(rand_train_word_prob[1, :] = ', np.sum(red_train_word_prob[1, :])

rand_test_counts = np.zeros((num_test_docs, num_proto), dtype=np.uint16)
red_test_counts = np.zeros((num_test_docs, num_proto), dtype=np.uint16)
with open('20news-bydate/test.data') as test_data_file:
    for line in test_data_file:
        doc, word, count = [int(i) for i in line.split()]
        if word in sorted_words1:
            red_test_counts[doc - 1, np.where(sorted_words1 == word)] = count
        if word in rand_vocab:
            rand_test_counts[doc - 1, np.where(rand_vocab == word)] = count
print 'here'
rand_scores = np.zeros((num_test_docs, num_labels), dtype=np.float64)
red_scores = np.zeros((num_test_docs, num_labels), dtype=np.float64)
for doc in xrange(num_test_docs):
    if doc % 100 == 0:
        print doc
    for label in xrange(num_labels):
        scores[doc, label] = (np.log(train_label_prob[label]) +
                              np.sum(red_test_counts[doc, :] * np.log(red_train_word_prob[label, :])))
        rand_scores[doc, label] = (np.log(train_label_prob[label]) +
                                   np.sum(rand_test_counts[doc, :] * np.log(rand_train_word_prob[label, :])))
print 'here'
# Calculates performance of prototypes
red_test_predictions = np.argmax(red_scores, axis=1)
rand_test_predictions = np.argmax(rand_scores, axis=1)
red_accuracy = np.sum(np.equal(red_test_predictions, test_labels)) / float(num_test_docs)
rand_accuracy = np.sum(np.equal(rand_test_predictions, test_labels)) / float(num_test_docs)
print 'Full accuracy = ', accuracy
print 'Reduced accuracy = ', red_accuracy
print 'Rand accuracy = ', rand_accuracy
