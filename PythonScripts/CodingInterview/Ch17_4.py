# Write a function that prints the max of two numbers, without using if or else or a comparison statement

def minimal_max(a, b):
    return (b/a)*b + (a/b)*a

if __name__ == "__main__":
    A = minimal_max(4, 5)
    B = minimal_max(5, 4)
    print A, B
