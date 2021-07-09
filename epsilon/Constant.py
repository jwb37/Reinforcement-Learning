from .BaseScheduler import BaseScheduler

class ConstantScheduler(BaseScheduler):
    def __init__(self, value):
        self.value = value

    def value(self, episode):
        return self.value
