import matplotlib.pyplot as plt
import os

score_filename = 'run1/score.txt'
Q_filename = 'run1/Q_val.txt'

if os.path.isfile(score_filename):
    scores = [line.rstrip('\n') for line in open(score_filename)]
    scores = [float(line) for line in scores]
    plt.figure(1)
    plt.plot(scores)
    plt.xlabel('Epoch')
    plt.title('Average score per game')

if os.path.isfile(Q_filename):
    Qs = [line.rstrip('\n') for line in open(Q_filename)]
    Qs = [float(line) for line in Qs]
    plt.figure(2)
    plt.plot(Qs)
    plt.xlabel('Epoch')
    plt.title('Average Q_max on fixed policy')

plt.show()
