# Chapter 2 of "Cracking the Coding Interview" covering linked lists

# Determines if a string has all unique characters
# If additional data structures are not allowed, I would do a pairwise comparison until one was true
# FROM SOLUTION: can rule out very large strings immediately
def is_unique(str):
    seen = []
    for char in str.lower():
        if char not in seen:
            seen.append(char)
        else:
            return False
    return True

# Determines if a string is a permutation of another
# Ideas: sort each str, then compare, O(n log(n))
# count number of instances of each letter then compare
def is_perm(str1, str2):
    list1 = list(str1)
    list2 = list(str2)
    list1.sort()
    list2.sort()
    print ''.join(list1), ''.join(list2)
    if list1 == list2:
        return True
    else:
        return False

# Compress a string, i.e. "aaabbc" maps to "a3b2c"
# if resulting string is larger, return uncompressed
def compress(str):
    pass

if __name__ == '__main__':

    str1 = 'evan'
    str2 = 'even'
    print is_unique(str1), is_unique(str2)

    str3 = 'nave'
    print is_perm(str1, str2)
    print is_perm(str1, str3)

