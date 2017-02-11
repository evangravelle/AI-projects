# Chapter 4 of "Cracking the Coding Interview" covering trees and graphs


class Node(object):
    """Creates a node."""

    def __init__(self, next=None, data=None):
        self.data = data
        self.next = next


class BST(object):
    """Creates a binary search tree."""

    def __init__(self, data=None):
        self.root = Node(data=data)

    def add(self, data):
        if self.root.data is None:
            self.root.data = data
        else:
            pass
            # compare, make children if they don't exist, etc.
