class Node(object):

    def __init__(self, key=None, val=None):
        self.key = key
        self.val = val
        self.nxt = None


class LinkedList(object):

    def __init__(self, head=None):
        self.head = head

    def add(self, node):
        if self.head is None:
            self.head = node
        else:
            self.head.nxt = self.head
            self.head = node

    def find_val(self, key):
        if self is None:
            return None
        else:
            current = self.head
            while current.nxt is not None and current.key != key:
                current = current.nxt
            if current.key == key:
                return current.val
            else:
                return None

    def __str__(self):
        string = ''
        if self is None:
            return string
        else:
            current = self.head
            while current.nxt is not None:
                string += current.val + ' '
                current = current.nxt
        string += str(current.val) + ' '


class HashTable(object):

    def __init__(self, size):
        self.table = [LinkedList() for i in xrange(size)]
        self.size = size

    def hash(self, obj):
        return hash(obj) % self.size

    def add(self, val):
        self.table[self.hash(val)] = val



if __name__ == "__main__":
    table_size = 3
    hash_table = HashTable(size=table_size)
    hash_table.add('entry1')
    hash_table.add('entry2')
    hash_table.add('entry3')
    hash_table.add('entry4')
    hash_table.add('entry5')
    print hash_table.table[hash_table.hash('entry5')]

