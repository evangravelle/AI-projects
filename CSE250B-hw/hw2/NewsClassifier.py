# This script classifies articles using a naive Bayes model on words contained
# in the article. Instead of using the entire training set, a subet of the training set
# is chosen, and performance as a function of number of prototypes is analyzed.

import numpy as np
# import matplotlib.pyplot as plt

num_proto = 5000
word_thresh = 0
num_words = 61188
num_train_docs = 11269
num_test_docs = 7505
num_labels = 20
num_reps = 10

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

# Finds marginal probability of each class
for doc in xrange(num_train_docs):
    train_label_prob[train_labels[doc]] += 1.
train_label_prob /= num_train_docs

# Finds probability of each word in each class
for label in xrange(num_labels):
    train_word_prob[label, :] = ((train_counts[label, :] + np.ones(num_words, dtype=np.uint16)) /
                                 float(np.sum(train_counts[label, :]) + num_words))

scores = np.zeros((num_test_docs, num_labels), dtype=np.float64)
for doc in xrange(num_test_docs):
    if doc % 100 == 0:
        print doc
    for label in xrange(num_labels):
        scores[doc, label] = (np.log(train_label_prob[label]) +
                              np.sum(test_counts[doc, :] * np.log(train_word_prob[label, :])))
print 'scores[0, :] = ', scores[0, :]

entropies = -np.sum(train_word_prob * np.log(train_word_prob), axis=0)
sorted_words1 = np.argsort(entropies)[0:num_proto]

variances = np.var(train_word_prob, axis=0)
sorted_words2 = np.argsort(-variances)[0:num_proto]

# Calculates performance of prototypes
test_predictions = np.argmax(scores, axis=1)
accuracy = np.sum(np.equal(test_predictions, test_labels)) / float(num_test_docs)
print 'accuracy = ', accuracy

# Prints a representative set of words in each class, by taking the words
# with highest probability in each class from the set of prototypes
reps = np.zeros((num_labels, num_reps), dtype=np.uint16)
for label in xrange(num_labels):
    ctr = 0
    for word in entropies:
        if label == np.argmax(train_word_prob[:, word]):
            reps[label, :] = word
            ctr += 1
        if ctr >= 10:
            break

print label_dict[0]
for rep in xrange(num_reps):
    print vocab_dict[reps[0, rep]]
