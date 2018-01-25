class Tree(object):

    def __init__(self):
        self.nodes = {}

    def add_state(self, s):
        if s in self.nodes.keys():
            self.nodes[s]['count'] += 1
            self.nodes[s]['value'] += 1
        else:
            node = {'count': 1, 'value': 0}
            self.nodes[s] = node
