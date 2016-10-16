# This script classifies articles using a naive Bayes model on words contained
# in the article. Instead of using the entire training set, a subet of the training set
# is chosen, and performance as a function of number of prototypes is analyzed.

import numpy as np
import matplotlib.pyplot as plt

num_words = 61188
num_train_docs = 11269
num_test_docs = 7505
num_labels = 20

train_counts = np.zeros((num_train_docs, num_words), dtype=np.uint16)
test_counts = np.zeros((num_test_docs, num_words), dtype=np.uint16)

# Save the data for the prototype set
with open('20news-bydate/train.data') as train_file:
    for line in train_file:
        doc, word, count = [int(i) for i in line.split()]
        train_counts[doc, word] = count

with open('20news-bydate/test.data') as test_file:
    for line in test_file:
        doc, word, count = [int(i) for i in line.split()]
        test_counts[doc, word] = count

