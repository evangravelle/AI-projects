import numpy as np
import nltk
import string
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

num_v = 5000
num_c = 1000

# nltk.download()
nltk.data.path.append('./nltk_data/')

# Load and clean up the words
words_raw = nltk.corpus.brown.words()
num_words = len(words_raw)
counts = dict()
words = []

for word in words_raw:
    # makes word lowercase, and removes all punctuation, adds to "words" if nonempty
    fixed_word = str(word.lower()).translate(None, string.punctuation)
    words.append(fixed_word)
    if word:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

# makes a vocabulary V of the 5000 most commonly occuring words
# makes a list C of 1000 of the most commonly occuring words, which are context words
freq_words = sorted(counts, key=counts.get, reverse=True)
V = freq_words[:num_v]
C = freq_words[:num_c]

print C
context_counts = np.zeros((num_v, num_c), dtype=np.int16)
p_c = np.zeros(num_c)
print 'Progress defining probabilities:'
for ind, word in enumerate(words):
    if word in V:
        v_ind = V.index(word)
        if word in C:
            p_c[C.index(word)] += 1
        if words[ind - 2] in C:
            context_counts[v_ind, C.index(words[ind - 2])] += 1
        if words[ind - 1] in C:
            context_counts[v_ind, C.index(words[ind - 1])] += 1
        if words[ind + 1] in C:
            context_counts[v_ind, C.index(words[ind + 1])] += 1
        if words[ind + 2] in C:
            context_counts[v_ind, C.index(words[ind + 2])] += 1
    if ind % 10000 == 0:
        print '{:.2f}%'.format(100. * ind / num_words)
p_c /= np.sum(p_c)

print p_c
p_cw = np.zeros((num_v, num_c))
for v_ind, v_word in enumerate(V):
    p_cw[v_ind, :] = context_counts[v_ind, :] / counts[v_word]

mut_info = np.zeros((num_v, num_c))
for v_ind, v_word in enumerate(V):
    for c_ind, c_word in enumerate(C):
        if p_c[c_ind] == 0:
            mut_info[v_ind, c_ind] = 0
        else:
            mut_info[v_ind, :] = np.max(np.zeros(num_c), np.log(np.squeeze(p_cw[v_ind, :]) / p_c))

pca = PCA(n_components=100)
pca.fit(mut_info.T)

kmeans = KMeans(n_clusters=100, random_state=0).fit(pca.transform(mut_info.T))
