# Example taken from internet
class Reverse(object):
    """Iterator for looping over a sequence backwards."""
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def next(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]

def reverse(data):
    for index in xrange(len(data)-1,-1,-1):
        yield data[index]


if __name__ == "__main__":
    rev1 = Reverse('spam')
    iter(rev1)
    for char in rev1:
        pass
        # print char

    for char in reverse('EVAN'):
        pass
        # print char