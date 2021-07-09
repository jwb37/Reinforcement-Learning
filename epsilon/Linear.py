from .BaseScheduler import BaseScheduler

class LinearScheduler(BaseScheduler):
    def __init__(self, start_eps, final_eps, num_decay_episodes):
        self.start_eps = start_eps
        self.final_eps = final_eps
        self.num_decay_episodes = num_decay_episodes
        self.grad = (final_eps - start_eps) / num_decay_episodes

    def value(self, episode):
        if episode > self.num_decay_episodes:
            return self.final_eps
        else:
            return episode*self.grad + self.start_eps
