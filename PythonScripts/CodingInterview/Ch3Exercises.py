# Chapter 3 of "Cracking the Coding Interview" covering stacks and queues


class Node(object):
    """Creates a node."""

    def __init__(self, next=None, data=None):
        self.data = data
        self.next = next


class Stack(object):
    """Creates a stack."""

    def __init__(self):
        self.top = None

    def push(self, data):
        new_node = Node(data=data)
        if self.top is None:
            self.top = new_node
        else:
            new_node.next = self.top
            self.top = new_node

    def pop(self):
        if self.top is None:
            return None
        else:
            val = self.top.data
            self.top = self.top.next
            return val

    def peek(self):
        return self.top.data

    def __str__(self):
        current = self.top
        if current is None:
            return "Stack is empty."
        else:
            string = ""
            while current.next is not None:
                string += str(current.data)
                current = current.next
            string += str(current.data)
            return string


if __name__ == "__main__":
    stack = Stack()
    stack.push('e')
    stack.push('v')
    stack.push('a')
    stack.push('n')
    top = stack.peek()
    print stack.pop()
    print stack.pop()
    print stack

