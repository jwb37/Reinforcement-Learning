import copy
import time
import torch
import random
import os
import os.path as path

from collections import deque

from Params import Params
from ..BaseModel import BaseModel
from .ExperienceBuffer import ExperienceBuffer

from environments import load_env
from networks import load_network

from utils.img import save_tensor_list_as_video


class DeepQModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.experience_buffer = ExperienceBuffer()


    def prepare_training(self):
        super().prepare_training()

        self.target = copy.deepcopy(self.net)
        self.target.eval()

        self.loss_function = torch.nn.MSELoss()

    def decide_action(self, state):
        if self.isTrain and random.random() < self.epsilon:
            return random.choice(self.env.actions)
        else:
            return torch.argmax(self.net(state),1).item()


    def training_step(self, old_states, new_states, actions, rewards, gameovers):
        batch_size = old_states.shape[0]

        old_q_vals = self.net(old_states)
        actions = actions.unsqueeze(1)
        old_q_vals = old_q_vals.gather(1, actions).squeeze()

        if Params.isTrue('double_dqn'):
            new_q_vals = self.net(new_states)
            new_best_actions = torch.argmax(new_q_vals, 1)
            max_new_q_vals = self.target(new_states).gather(1, new_best_actions.unsqueeze(-1)).squeeze()
        else:
            new_q_vals = self.target(new_states)
            max_new_q_vals = torch.max(new_q_vals, 1).values

        max_new_q_vals = max_new_q_vals.detach()
        updated_q_vals = torch.zeros(batch_size).to(Params.device)
        updated_q_vals[gameovers.logical_not()] = Params.discount_rate * max_new_q_vals[gameovers.logical_not()]
        updated_q_vals += rewards

        loss = self.loss_function( old_q_vals, updated_q_vals )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
#        self.scheduler.step()

        return loss.item()


    def train(self):
        loss_record = []
        reward_record = deque([],Params.print_freq)
        
        for episode in range(self.start_episode, Params.num_episodes):
            done = False
            episode_reward = 0

            self.env.reset()
            before_state = self.env.read_state().to(Params.device)
            action = self.env.actions.default

            self.epsilon = Params.epsilon[episode]

            if Params.isTrue('save_video'):
                video = [before_state]
            else:
                video = []

            while not done:
                self.store_experience = False

                if not Params.isTrue('repeat_action'):
                    action = self.env.actions.default

                if not Params.isTrue('frame_skip_freq') or (self.total_steps % Params.frame_skip_freq) == 0:
                    action = self.decide_action(before_state)
                    self.store_experience = True

                try:
                    reward, done = self.env.update(action)
                    episode_reward += reward
                except:
                    # Environment crashed! Try and exit the loop and reset
                    break

                after_state = self.env.read_state().to(Params.device)

                if Params.isTrue('save_video'):
                    video.append(after_state)

                if self.store_experience:
                    self.experience_buffer.append( (before_state, after_state, action, reward, done) )

                if (self.total_steps%Params.update_working_net_freq) == 0 and self.total_steps >= Params.first_working_net_update:
                    batch = self.experience_buffer.sample(Params.batch_size, Params.device)
                    loss = self.training_step(*batch)
                    loss_record.append( {"Q_MSE": loss} )

                if (self.total_steps%Params.update_target_net_freq) == 0 and self.total_steps >= Params.first_target_net_update:
                    self.target.load_state_dict(self.net.state_dict())

                before_state = after_state
                self.total_steps += 1

            reward_record.append(episode_reward)

            self.post_episode_tasks(
                episode,
                episode_reward,
                reward_record,
                loss_record,
                video
            )
            loss_record.clear()

        self.save_checkpoint( self.save_path, f"episode_{Params.num_episodes}" )


    def pretrain(self, num_batches):
        for n in range(num_batches):
            batch = self.experience_buffer.sample( Params.batch_size, Params.device )
            self.training_step(*batch)

            if (n%500)==0:
                print( f"Pretraining progress: {n} out of {num_batches}" )
