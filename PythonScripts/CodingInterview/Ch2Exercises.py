# Chapter 2 of "Cracking the Coding Interview" covering linked lists

class Node(object):
    """Creates a linked list."""

    def __init__(self, next=None, data=None):
        self.data = data
        self.next = next

    def set_next(self, next):
        self.next = next

class LinkedList(object):

    def __init__(self):
        self.head = Node()

    def append(self, data):
        current = self.head
        while current.next != None:
            current = current.next
        current.set_next(Node(data=data))

    def delete_node(self, data):
        node = self.head
        if node.data == data:
            node.next =
        while node.next != None:
            if node.next.data == data:
                node.next = node.next.next
                return head
            node = node.next
        return head

    def delete_dups(self):
        seen = {}
        node = self
        while node.next != None:
            if node.data in seen:
                node.next = node.next.next
            else:
                seen[node.data] = True
                node = node.next

    def print_elements(self):
        node = self
        while node.next != None:
            print node.data
            node.next = node.next.next
        print node.data


if __name__ == "__main__":
    ll = Node(data=5)
    ll.append(data=10)
    ll.append(data=11)
    ll.delete_dups()
    ll.print_elements()
