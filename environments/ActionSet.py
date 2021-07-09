class ActionSet:
    def __init__(self, array):
        self.array = array
        self.default = None

    def __getitem__(self, key):
        return self.array[key]

    def __len__(self):
        return len(self.array) 
