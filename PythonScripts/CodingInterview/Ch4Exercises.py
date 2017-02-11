# Chapter 4 of "Cracking the Coding Interview" covering trees and graphs


class Node(object):
    """Creates a node."""

    def __init__(self, data=None):
        self.data = data
        self.left = None
        self.right = None


class BinarySearchTree(object):
    """Creates a binary search tree."""

    def __init__(self, data=None):
        self.root = Node(data=data)

    def add(self, data):
        current = self.root
        if self.root.data is None:
            traversing = False
        else:
            traversing = True
        while traversing:
            if data < current.data and current.left is None:
                current.left = Node(data=data)
                current = current.left
                traversing = False
            elif data < current.data and current.left is not None:
                current = current.left
            elif data > current.data and current.right is None:
                current.right = Node(data=data)
                current = current.right
                traversing = False
            elif data > current.data and current.right is not None:
                current = current.right
        current.data = data


    def BFS(self, data):
        pass

if __name__ == "__main__":
    tree = BinarySearchTree()
    tree.add(5)
    tree.add(6)
    tree.add(4)
    print tree.root.left.data, tree.root.data, tree.root.right.data
