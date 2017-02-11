# Chapter 2 of "Cracking the Coding Interview" covering linked lists


class Node(object):
    """Creates a node."""

    def __init__(self, next=None, data=None):
        self.data = data
        self.next = next


class LinkedList(object):
    """Creates a linked list."""

    def __init__(self, head=None):
        self.head = head

    def append(self, data):
        new_node = Node(data=data)
        current = self.head
        if current is not None:
            while current.next is not None:
                current = current.next
            current.next = new_node
        else:
            self.head = Node(new_node)

    def insert(self, data):
        new_node = Node(data=data)
        if self.head is None:
            self.head = new_node
        else:
            new_node.next = self.head
            self.head = new_node

    def delete_node(self, data):
        current = self.head
        if current.data == data:
            self.head = current.next
        while current.next is not None:
            if current.next.data == data:
                current.next = current.next.next
            current = current.next

    def delete_dups(self):
        seen = {}
        current = self.head
        if current is None:
            pass
        else:
            seen[current.data] = True
            while current.next is not None:
                if current.next.data in seen:
                    current.next = current.next.next
                else:
                    seen[current.next.data] = True
                    current = current.next

    def __str__(self):
        string = ''
        current = self.head
        if current is not None:
            while current.next is not None:
                string += str(current.data) + ' '
                current = current.next
            string += str(current.data) + ' '
        else:
            string = 'empty'
        return string


if __name__ == "__main__":
    ll = LinkedList()
    ll.insert(data=-5)
    ll.insert(data=10)
    ll.insert(data=10)
    ll.insert(data=11)
    print ll, '\n'
    ll.delete_dups()
    print ll, '\n'
