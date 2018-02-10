class SearchTree(object):

    def __init__(self):
        self.nodes = {}

    def add_state(self, s):
        if s in self.nodes.keys():
            self.nodes[s]['count'] += 1
            self.nodes[s]['value'] += 1
            self.nodes[s]['children'] += 'timmy'
        else:
            node = {'count': 1, 'value': 0, 'children': []}
            self.nodes[s] = node

    def sample_action(self, s):


    def search(self, s):
        for it in range(100):
            s
