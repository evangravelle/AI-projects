import matplotlib.pyplot as plt
import os

score_filename = 'pong_scores/score.txt'
Q_filename = 'pong_scores/Q_val.txt'

if os.path.isfile(score_filename):
    scores = [line.rstrip('\n') for line in open(score_filename)]
    scores = [float(line) for line in scores]
    plt.figure(1)
    plt.plot(scores)
    plt.xlabel('Epoch')
    plt.title('Average score per game, 1 epoch = 10 episodes')

if os.path.isfile(Q_filename):
    Qs = [float(line.rstrip('\n'))/10. for line in open(Q_filename)]
    # Qs = [float(line) for line in Qs]
    plt.figure(2)
    plt.plot(Qs)
    plt.xlabel('Iterations*1000')
    plt.title('Average Q_max on fixed policy')

plt.show()
