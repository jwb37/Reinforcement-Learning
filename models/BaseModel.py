import os
import time
import torch
import os.path as path

from abc import ABC

from Params import Params
from networks import load_network
from environments import load_env

from utils.img import save_tensor_list_as_video

class BaseModel(ABC):
    def __init__(self):
        self.env = load_env(Params.environment)
        self.net = load_network(self.env)
        self.optimizer = None
        self.start_episode = 0

        self.isTest = self.isTrain = False

        self.save_path = path.join( 'checkpoints/', Params.save_name )
        os.makedirs( self.save_path, exist_ok = True )

    def prepare_training(self):
        self.isTrain = True
        self.isTest = False

        self.net.to(Params.device)
        self.net.train()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=Params.initial_lr)

        self.load_latest()

        self.env.prepare_training()

        if Params.isTrue('save_best'):
            os.makedirs( path.join(self.save_path, 'best'), exist_ok = True )
        if Params.isTrue('save_video'):
            os.makedirs( path.join(self.save_path, 'videos'), exist_ok = True )

        self.log_path = path.join( self.save_path, 'training.log' )

        self.best_reward = Params.min_reward_to_save
        self.total_steps = 0
        self.tic = time.perf_counter()


    def prepare_testing(self):
        self.isTest = True
        self.isTrain = False

        self.net.to(Params.device)
        self.net.eval()

        self.load_latest()

        self.testsave_path = path.join('test_results/', Params.save_name)
        os.makedirs( self.testsave_path, exist_ok=True )

        self.env.prepare_testing()


    def test(self):
        done = False
        steps = 0

        self.env.reset()
        action = self.env.actions.default

        video = []
        total_reward = 0

        while not done:
            state = self.env.read_state().to(Params.device)
            video.append(state)

            if not Params.isTrue('repeat_action'):
                action = self.env.actions.default

            if not Params.isTrue('frame_skip_freq') or (steps % Params.frame_skip_freq) == 0:
                action = self.decide_action(state)

            reward, done = self.env.update(action)

            steps += 1
            total_reward += reward

        if Params.isTrue('save_video'):
            vid_fname = time.strftime('(%Y-%m-%d %H.%M.%S)', time.localtime())
            vid_fname += f" steps({steps}) reward({total_reward}).gif"
            vid_fullpath = path.join(self.testsave_path, vid_fname)
            save_tensor_list_as_video(video, vid_fullpath)


    def save_checkpoint(self, save_path, name, save_opt=True):
        model_save_name = path.join(save_path, f"{name}.pth")
        torch.save(self.net.state_dict(), model_save_name)

        if save_opt:
            optim_save_name = path.join(save_path, f"{name}.opt")
            torch.save(self.optimizer.state_dict(), optim_save_name)


    def load_checkpoint(self, save_path, name):
        net_fname = path.join(save_path, f"{name}.pth")
        self.net.load_state_dict( torch.load(net_fname, map_location=Params.device) )

        if self.optimizer:
            opt_fname = path.join(save_path, f"{name}.opt")
            self.optimizer.load_state_dict( torch.load(opt_fname) )


    def load_latest(self):
        saved_models = [ (int(filename[8:-4]),filename) for filename in os.listdir(self.save_path) if filename.endswith('.pth') and filename.startswith('episode_') ]

        if not saved_models:
            print( "No checkpoints found. Training will begin from scratch" )
            return False

        saved_models.sort(key = lambda t: t[0], reverse=True)

        print( f"Loading model at episode {saved_models[0][0]}. Filename is {saved_models[0][1]}" )
        self.start_episode = saved_models[0][0]
        fname = saved_models[0][1][:-4]

        self.load_checkpoint(self.save_path, fname)
        return True

    def post_episode_tasks(self, ep_num, ep_reward, batch_rewards=[], batch_losses=[], video=[]):
        if Params.isTrue('save_freq') and (ep_num%Params.save_freq) == 0 and ep_num > 1:
            self.save_checkpoint( self.save_path, f"episode_{ep_num}" )

        if ep_reward > self.best_reward:
            if Params.isTrue('save_best'):
                self.save_checkpoint( path.join(self.save_path, 'best'), f"episode_{ep_num}_reward_{ep_reward:.1f}", save_opt=False )
                if Params.isTrue('save_video') and video:
                    vid_filename = path.join(self.save_path, 'videos', f"video_{ep_num}_reward_{ep_reward:.1f}.gif")
                    save_tensor_list_as_video( video, vid_filename )
            self.best_reward = ep_reward

        if Params.isTrue('print_freq') and (ep_num%Params.print_freq) == 0 and ep_num > 1:
            toc = time.perf_counter() - self.tic
            self.tic = time.perf_counter()

            summary_arr = [
                f"Episodes {ep_num}",
                f"Time {toc:.1f}s",
                f"Steps {self.total_steps:.2e}",
                f"Reward {sum(batch_rewards)/len(batch_rewards):.2f}"
            ]

            n = len(batch_losses)
            if n > 0:
                batch_losses = {
                    name: sum(r[name] for r in batch_losses)/n
                    for name in batch_losses[0].keys()
                }
                summary_arr += [ f"L_{name} {val:.3f}" for name, val in batch_losses.items() ]

            summary_str = " ".join(summary_arr)

            print(summary_str)

            with open(self.log_path, 'a') as log_file:
                log_file.write(summary_str + "\n")
