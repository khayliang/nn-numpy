class Module:
    def __init__(self, children=[]):
        self.training = False
        self.children = children

    def train(self, mode=True):
        self.training = mode
        for module in self.children:
            module.train(mode)
        return self

    def gd_step(self, lrate):
        pass
    