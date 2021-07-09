from abc import ABC

class BaseRLEnv(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def read_state(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def prepare_training(self):
        pass

    def prepare_testing(self):
        pass
