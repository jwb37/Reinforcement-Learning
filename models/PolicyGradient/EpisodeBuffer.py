from collections import deque

class EpisodeBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.arr = deque([])

    def append(self, d):
        self.arr.append(d)

    def get(self, key, reverse=True):
        if reverse:
            arr = reversed(self.arr)
        else:
            arr = self.arr
        return [rec[key] for rec in arr]

    def __len__(self):
        return len(self.arr)
