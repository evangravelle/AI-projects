# Various sorting algorithms are implemented and tested
import random

def merge(left, right):
    sorted = []

    # while left or right are nonempty
    while left or right:
        if left and not right:
            sorted.append(left.pop(0))
        elif right and not left:
            sorted.append(right.pop(0))
        elif left[0] <= right[0]:
            sorted.append(left.pop(0))
        else:
            sorted.append(right.pop(0))
    return sorted


def mergesort(arr):
    length = len(arr)
    if length <= 1:
        return arr

    left = mergesort(arr[:length/2])
    right = mergesort(arr[length/2:])
    return merge(left, right)


def partition(arr, low, high):
    random.randint(low, high)

    return 0

def quicksort(arr, low, high):
    if low < high:
        pivot_ind = partition(arr, low, high)
        quicksort(arr, low, pivot_ind-1)
        quicksort(arr, pivot_ind+1, high)


if __name__ == "__main__":
    arr = [5, 3, 7, 4, 1, 1]
    print arr
    sorted_arr = mergesort(arr)
    print sorted_arr
