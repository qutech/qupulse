from PulseTemplate import PulseTemplate
class Branch(PulseTemplate):
    """docstring for Branch"""
    def __init__(self):
        super(Branch, self).__init__()
        self.else_branch = None
        self.if_branch = None
        self.condition = None

