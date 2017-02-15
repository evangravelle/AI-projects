import numpy as np

def trailing_zeros(n):
    num = 0
    if type(n) is not int:
        raise TypeError
    for i in xrange(1, n+1):
        if i % 5 == 0:
            num += np.floor(np.log(i) / np.log(5))
    return num


if __name__ == "__main__":
    print trailing_zeros(4)
    print trailing_zeros(5)
    print trailing_zeros(10)
    print trailing_zeros(15)
    print trailing_zeros(100)
