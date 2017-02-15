import numpy as np

def has_won_tic_tac_toe(player, M):
    if (M[0, :] == player * np.ones(3)).all():
        return True
    elif (M[1, :] == player * np.ones(3)).all():
        return True
    elif (M[2, :] == player * np.ones(3)).all():
        return True
    elif (M[:, 0] == player * np.ones(3)).all():
        return True
    elif (M[:, 1] == player * np.ones(3)).all():
        return True
    elif (M[:, 2] == player * np.ones(3)).all():
        return True
    elif (M[0, 0] == player and M[1, 1] == player and M[2, 2] == player):
        return True
    elif (M[0, 2] == player and M[1, 1] == player and M[2, 0] == player):
        return True
    else:
        return False

if __name__ == "__main__":
    M = np.ndarray((3, 3))
    M[0, :] = [0, 2, 0]
    M[1, :] = [2, 1, 2]
    M[2, :] = [0, 2, 1]
    won = has_won_tic_tac_toe(1, M)
    print won
