def hash(str, table_size):
    sum = 0
    for char in list(str):
        sum += ord(char)

    return sum % table_size

if __name__ == "__main__":
    table_size = 5
    num_strs = 2
    strs = ['cat','dog']
    arr = [None]*table_size
    keys = [None]*num_strs
    for i in xrange(num_strs):
        keys[i] = hash(strs[i], table_size)
        arr[keys[i]] = strs[i]

    print arr