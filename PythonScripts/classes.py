# Some classes practice

class Point(object):
    """A point."""
    def __init__(self, x=0., y=0.):
        self.x = x
        self.y = y

    def __str__(self):
        pass


class Rectangle(object):
    """Consists of top left and lower right point."""
    def __init__(self, ):
        self.ll = x
        self.ur = y


def print_attributes(obj):
    for attr in obj.__dict__:
        print attr, getattr(obj, attr)
