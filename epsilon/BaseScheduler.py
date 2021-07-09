from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def value(self, episode):
        pass

    def __getitem__(self, episode):
        return self.value(episode)
