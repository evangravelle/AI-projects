def swap_in_place(a, b):
    a = a - b
    b = a + b
    a = b - a
    return (a, b)

if __name__ == "__main__":
    (A, B) = swap_in_place(7, 4)
    print (A, B)