import torch

from itertools import accumulate
from collections import deque

from Params import Params
from ..BaseModel import BaseModel
from .EpisodeBuffer import EpisodeBuffer


class ActorCritic(BaseModel):
    def __init__(self):
        super().__init__()
        self.episode_buffer = EpisodeBuffer()


    def decide_action(self, state):
        logits, value = self.net(state)
        m = torch.distributions.Categorical(logits=logits)
        action = m.sample()
        if self.isTrain:
            return (action, value, m.log_prob(action), m.entropy())
        else:
            return action

    def training_step(self):
        values = self.episode_buffer.get('value')
        values = torch.cat(values, dim=0).view(-1)
        log_probs = self.episode_buffer.get('log_prob')
        log_probs = torch.cat(log_probs, dim=0).view(-1)
        entropies = self.episode_buffer.get('entropy')
        entropies = torch.cat(entropies, dim=0).view(-1)

        rewards = self.episode_buffer.get('reward')
        discounted_rewards = accumulate(
            rewards,
            lambda tot, r: tot*Params.discount_rate + r
        )
        discounted_rewards = torch.Tensor(list(discounted_rewards)).view(-1).to(Params.device)
        discounted_rewards = torch.nn.functional.normalize(discounted_rewards, dim=0)

        loss = {
            'actor':    (-1 * log_probs * (discounted_rewards - values.detach()) * Params.lambda_act).mean(),
            'critic':   (torch.pow(values - discounted_rewards, 2) * Params.lambda_crt).mean()
        }

        if entropies.mean().item() > Params.entropy_threshold:
            loss['reg'] = 0.0
        else:
            loss['reg'] = (-entropies * Params.lambda_reg).mean()

        loss['total'] = loss['actor'] + loss['critic'] + loss['reg']

        self.optimizer.zero_grad()
        loss['total'].backward()
        self.optimizer.step()

        return loss

    def run_episode(self):
        self.env.reset()
        self.episode_buffer.clear()

        episode_steps = 0
        episode_reward = 0

        action = self.env.actions.default

        done = False

        while not done:
            state = self.env.read_state().to(Params.device)
            store_experience = False

            if not Params.isTrue('repeat_action'):
                action = self.env.actions.default

            if not Params.isTrue('frame_skip_freq') or (total_steps % Params.frame_skip_freq) == 0:
                action, value, log_prob, entropy = self.decide_action(state)
                store_experience = True

            try:
                reward, done = self.env.update(action)
                episode_reward += reward
            except:
                # Environment crashed! Try and exit the loop and reset
                break

            if store_experience:
                self.episode_buffer.append({
                    'state': state,
                    'log_prob': log_prob,
                    'value': value,
                    'reward': reward,
                    'entropy': entropy
                })

            episode_steps += 1

        return episode_steps, episode_reward

    def train(self):
        loss_record = deque([],Params.print_freq)
        reward_record = deque([],Params.print_freq)

        for episode in range(self.start_episode, Params.num_episodes):
            ep_steps, ep_reward = self.run_episode()
            losses = self.training_step()

            self.total_steps += ep_steps
            loss_record.append(losses)
            reward_record.append(ep_reward)

            self.post_episode_tasks(
                episode,
                ep_reward,
                reward_record,
                loss_record,
                self.episode_buffer.get('state', reverse=False)
            )

        self.save_checkpoint( self.save_path, f"episode_{Params.num_episodes}" )
