import numpy as np
import nltk
import string
import pickle

# nltk.download()
nltk.data.path.append('./nltk_data/')

# Load and clean up the words
loaded_words = nltk.corpus.brown.words()
num_words = len(loaded_words)
counts = dict()

for i in xrange(num_words):
    # makes word lowercase, and removes all punctuation, adds to "words" if nonempty
    word = str(loaded_words[i].lower()).translate(None, string.punctuation)
    if word:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

# makes a vocabulary V of the 5000 most commonly occuring words
# makes a list C of 1000 of the most commonly occuring words, which are context words
freq_words = sorted(counts, key=counts.get, reverse=True)
V = freq_words[:5000]
C = V[:1000]
