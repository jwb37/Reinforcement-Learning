import copy
import random

import torch
import numpy as np
from torch.autograd import Variable

from collections import deque

from Cube import Cube
from QValNetwork import QValNetwork
from ExperienceBuffer import StandardExperienceBuffer


class QSolver:
    def __init__(self, cube, learning_rate=1e-4):

        self.cube = cube

        #self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'

        self.discount_factor = torch.tensor([0.98], dtype=torch.float).to(self.device)
        #self.discount_factor = 0.98

        self.update_model = QValNetwork(cube)
        self.target_model = QValNetwork(cube)
        self.target_model.eval()

        self.update_model.to(self.device)
        self.target_model.to(self.device)

        self.loss_function = torch.nn.MSELoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.update_model.parameters(), lr = learning_rate)


    def take_action(self,action):
        self.cube.take_action_by_idx(action)
        if self.cube.check_solved():
            return 100
        else:
            return -5


    def epsilon_greedy(self, epsilon, best_action):
        if random.random() < epsilon:
            return random.randint(0,len(Cube.actions)-1)
        else:
            return best_action



    def reset_environment(self, num_scrambles, return_scramble_steps=False):
        already_reset = False
        # Scramble the cube at least once.
        # Check it's not randomly ended up in a solved state - that would be bad!
        while (not already_reset) or self.cube.check_solved():
            already_reset = True
            self.cube.reset()
            actions = self.cube.scramble( num_scrambles, return_scramble_steps )

        if return_scramble_steps:
            return actions



    def build_training_set(self, experience_buffer, size):
        size = min( len(experience_buffer), size)
        experiences = experience_buffer.sample(size)

        old_states, actions, new_states, rewards = zip(*experiences)

        old_states = torch.stack(old_states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        new_states = torch.stack(new_states).to(self.device)
        rewards = torch.cat(rewards).to(self.device)

        old_q_vals = self.update_model(old_states)
        old_q_vals = torch.gather(old_q_vals, 1, actions).squeeze()
        # DQN Implementation below
        max_new_q_vals = self.target_model.max_qval(new_states)
        # DDQN Implementation below
        #max_new_q_val_indices = torch.argmax(self.update_model(new_states), 1).unsqueeze(1)
        #max_new_q_vals = torch.gather(self.target_model(new_states), 1, max_new_q_val_indices).squeeze()

        updated_q_vals = torch.zeros(size).to(self.device)
        updated_q_vals[rewards != 100] = self.discount_factor * max_new_q_vals[rewards != 100]
        updated_q_vals += rewards

        return ( old_q_vals, updated_q_vals )


    # Experiment - does onehot encoding of colours improve learning?
    def state(self):
        cube_state_tensor = torch.tensor(self.cube.state, dtype=torch.int64)
        state_tensor = torch.nn.functional.one_hot(cube_state_tensor, 6)
        return state_tensor.flatten().float().to(self.device)



    def train(self, num_epochs=10000, min_difficulty = 1, max_difficulty=10, StepsBeforeGiveUp=30, EpsilonStart=0.4, EpsilonEnd=0.03):
        StepsBetweenTargetUpdate = 1000
        StepsBetweenTraining = 10
        ExperienceBufferSize = 1000
        MinBufferSize = 100
        BatchSize = 60
        EpsilonDecay = float(EpsilonEnd - EpsilonStart) / num_epochs

        experience_buffer = StandardExperienceBuffer(ExperienceBufferSize)
        epsilon = EpsilonStart
        total_steps = 0

        # Track how many attempts over the last 100 succeeded
        recent_successes = deque([0]*100,100)

        # Copy parameters from update model across to target model
        self.target_model.load_state_dict(self.update_model.state_dict())

        # Set update_model to training mode
        self.update_model.train()

        for n in range(num_epochs):
            if min_difficulty == max_difficulty:
                self.reset_environment( num_scrambles=min_difficulty )
            else:
                self.reset_environment(
                    num_scrambles = np.random.randint( min_difficulty, max_difficulty )
                )

            num_steps = 0
            finish = False
            succeeded = False
            state = self.state()

            while not finish:
                num_steps += 1

                # Copy parameters from update model to target model
                if (total_steps % StepsBetweenTargetUpdate) == 0:
                    self.target_model.load_state_dict(self.update_model.state_dict())

                best_action = self.update_model.best_action_idx(state).item()
                action = self.epsilon_greedy( epsilon, best_action )
                reward = torch.tensor( [self.take_action(action)] ).to(self.device)
                action = torch.tensor( [action] ).to(self.device)

                if reward == 100:
                    finish = True
                    succeeded = True
                new_state = self.state()

                experience_buffer.append( (state, action, new_state, reward) )

                if len(experience_buffer) > MinBufferSize and (num_steps % StepsBetweenTraining) == 0:
                    # Build a training set and train on it
                    predicted_q_set, updated_q_set = self.build_training_set( experience_buffer, BatchSize )

                    loss = self.loss_function(
                        predicted_q_set,
                        updated_q_set
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Limit max number of steps before giving up on a puzzle
                # Is this a good idea?
                if num_steps > StepsBeforeGiveUp:
                    finish = True
                    succeeded = False

                state = new_state

            recent_successes.append( int(succeeded) )

            # Update us on progress every 10 epochs
            if (n % 100) == 0 and n > 100:
                print( "Epoch: %d"%(n) )
                print( "In last 100 attempts, there were %d successes"%(sum(recent_successes)) )
                
            total_steps += num_steps
            epsilon += EpsilonDecay

    def save(self, path):
        checkpoint = {
            'model_state_dict': self.update_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.update_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def test(self, print_solutions=True):
        MaxDifficulty = 15
        AttemptsPerDifficulty = 10
        StepsBeforeGiveUp = 120

        for difficulty in range(1,MaxDifficulty+1):

            failures = 0
            accumulated_reward = 0

            for attempt in range(AttemptsPerDifficulty):
                scrambling_actions = self.reset_environment( num_scrambles = difficulty, return_scramble_steps=True )

                num_steps = 0
                reward = 0
                finish = False
                succeeded = False
                solving_actions = []

                while not finish:
                    num_steps += 1

                    state = self.state()
                    action = self.update_model.best_action_idx(state).item()
                    new_reward = self.take_action(action)
                    reward += new_reward

                    solving_actions.append( self.cube.actions[action] )

                    if new_reward == 100:
                        succeeded = True
                        finish = True

                    if num_steps > StepsBeforeGiveUp:
                        reward = -20
                        finish = True
                        succeeded = False

                if not succeeded:
                    failures += 1
                elif print_solutions:
                    print("Solution")
                    print("Scramble steps:")
                    print(scrambling_actions)
                    print("Solving steps:")
                    print(solving_actions)

                accumulated_reward += reward

            print( "At difficulty level %d: %d successes, %d failures. Mean reward %.2f"%(
                difficulty,
                AttemptsPerDifficulty-failures,
                failures,
                float(accumulated_reward)/AttemptsPerDifficulty)
            )
