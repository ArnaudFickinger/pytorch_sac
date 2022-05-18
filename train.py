#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
sys.path.insert(0,'/private/home/arnaudfickinger/pytorch_sac')
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils_sac

import dmc2gym
import hydra
import wandb

import pickle

os.environ["WANDB_API_KEY"] = "0253801d5a4a70a326be214e03ac4f97c1d0beb1"


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=False)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             log_frequency=cfg.log_frequency)

        utils_sac.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils_sac.make_env(cfg)

        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device,
                                          universe = self)

        self.video_recorder = VideoRecorder(wandb=self.cfg.wandb)
        self.step = 0
        self.duration = 0
        self.episode = 0
        self.start = time.time()

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils_sac.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(self.evaluate_sample)
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)
        if self.cfg.wandb:
            wandb.log({'Testing reward': average_episode_reward, 'Samples': self.step, 'Time': self.duration, 'Episodes': self.episode})

    def save_expert_demo(self):
        episode_rewards = []
        episode_trajectories = []
        episode_videos = []

        for episode in range(self.cfg.num_eval_episodes):
            episode_trajectories.append(
                {'state': [], 'obs_distraction': [], 'obs_no_distraction': [], 'action': [], 'reward': []})
            # episode_trajectories[-1]['dmc_obs'] = {}
            episode_videos.append([])
            obs = self.env.reset()

            # import pdb; pdb.set_trace()
            # for key in self.env.timestep.observation:
            #     if key not in episode_trajectories[-1]['dmc_obs']:
            #         episode_trajectories[-1]['dmc_obs'][key] = []
            #     episode_trajectories[-1]['dmc_obs'][key].append(self.env.timestep.observation[key])
            # if 'walker' in self.cfg.env:
            #     if 'position' not in episode_trajectories[-1]['dmc_obs']:
            #         episode_trajectories[-1]['dmc_obs']['position'] = []
            #     episode_trajectories[-1]['dmc_obs']['position'].append(self.env.physics.data.qpos[:].copy())
            # if 'cheetah' in self.cfg.env:
            #     # import pdb; pdb.set_trace()
            #     if 'orientations' not in episode_trajectories[-1]['dmc_obs']:
            #         episode_trajectories[-1]['dmc_obs']['orientations'] = []
            #     episode_trajectories[-1]['dmc_obs']['orientations'].append(
            #         self.env.physics.named.data.xmat[1:, ['xx', 'xz']].ravel())

            # episode_videos[-1].append(self.env.render(mode='rgb_array',
            #                                           height=256,
            #                                           width=256).transpose(2, 0, 1))
            self.agent.reset()
            obs_no_distraction, obs_distraction = self.env.get_extra()

            episode_trajectories[-1]['obs_distraction'].append(obs_distraction)
            episode_trajectories[-1]['obs_no_distraction'].append(obs_no_distraction)
            done = False

            episode_reward = 0
            while not done:
                with utils_sac.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                episode_trajectories[-1]['state'].append(obs)

                obs, reward, done, info = self.env.step(action)
                # import pdb; pdb.set_trace()
                # episode_videos[-1].append(self.env.render(mode='rgb_array',
                #                                           height=256,
                #                                           width=256).transpose(2, 0, 1))

                episode_trajectories[-1]['obs_distraction'].append(info['obs_distraction'])
                episode_trajectories[-1]['obs_no_distraction'].append(info['obs_no_distraction'])

                episode_reward += reward

                # if not done:
                    # for key in info['dmc_obs']:
                    #     if key not in episode_trajectories[-1]['dmc_obs']:
                    #         episode_trajectories[-1]['dmc_obs'][key] = []
                    #     episode_trajectories[-1]['dmc_obs'][key].append(info['dmc_obs'][key])
                    # if 'walker' in self.cfg.env:
                    #     if 'position' not in episode_trajectories[-1]['dmc_obs']:
                    #         episode_trajectories[-1]['dmc_obs']['position'] = []
                    #     # print(self.env.physics.data.qpos[0])
                    #     episode_trajectories[-1]['dmc_obs']['position'].append(self.env.physics.data.qpos[:].copy())
                    # if 'cheetah' in self.cfg.env:
                    #     # import pdb; pdb.set_trace()
                    #     if 'orientations' not in episode_trajectories[-1]['dmc_obs']:
                    #         episode_trajectories[-1]['dmc_obs']['orientations'] = []
                    #     episode_trajectories[-1]['dmc_obs']['orientations'].append(
                    #         self.env.physics.named.data.xmat[1:, ['xx', 'xz']].ravel())

                # episode_trajectories[-1]['nobs'].append(obs)
                # episode_trajectories[-1]['pixel_nobs'].append(self.env.render(mode='rgb_array',
                #                                                               height=84,
                #                                                               width=84).transpose(2, 0, 1))
                episode_trajectories[-1]['action'].append(action)
                episode_trajectories[-1]['reward'].append(reward)

            # assert False

            episode_rewards.append(episode_reward)

        best_episode = np.argmax(episode_rewards)
        # print(f"best episode {best_episode}")
        best_trajectory = episode_trajectories[best_episode]
        best_trajectory.update({"cumulative_reward": episode_rewards[best_episode]})

        best_trajectory['state'] = np.stack(best_trajectory['state'])
        # best_trajectory['nobs'] = np.stack(best_trajectory['nobs'])
        best_trajectory['obs_distraction'] = np.stack(best_trajectory['obs_distraction'])
        best_trajectory['obs_no_distraction'] = np.stack(best_trajectory['obs_no_distraction'])
        # best_trajectory['pixel_nobs'] = np.stack(best_trajectory['pixel_nobs'])
        best_trajectory['action'] = np.stack(best_trajectory['action'])

        # for key in best_trajectory['dmc_obs']:
        #     best_trajectory['dmc_obs'][key] = np.stack(best_trajectory['dmc_obs'][key])

        # print(best_trajectory)

        # import pdb; pdb.set_trace()

        with open(f'/private/home/arnaudfickinger/hbisim/expert_demonstrations/expert_trajectory_{self.cfg.env}_{self.save_demo_sample}.pickle',
                  'wb') as handle:
            pickle.dump(best_trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.cfg.wandb:
            wandb.log({f"Expert Demonstration Distraction {self.save_demo_sample}": wandb.Video(best_trajectory['obs_distraction'], fps=30,
                                                            format="mp4")})
            wandb.log({f"Expert Demonstration No Distraction {self.save_demo_sample}": wandb.Video(
                best_trajectory['obs_no_distraction'], fps=30,
                format="mp4")})

    def run(self):
        episode_reward, done = 0, True
        to_evaluate = False
        to_save_demo = False
        while self.step < self.cfg.num_train_steps:

            if done:
                self.duration = time.time() - self.start
                if self.step > 0:
                    self.logger.log('train/duration',
                                    self.duration, self.step)
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                    if to_evaluate:
                        self.logger.log('eval/episode', self.episode, self.step)
                        self.evaluate()
                        to_evaluate = False

                    if to_save_demo:
                        self.save_expert_demo()
                        to_save_demo = False

                    self.logger.log('train/episode_reward', episode_reward,
                                    self.step)

                    if self.cfg.wandb:
                        wandb.log({'Training reward': episode_reward, 'Samples': self.step, 'Time': self.duration, 'Episodes': self.episode})

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                self.episode += 1

                self.logger.log('train/episode', self.episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils_sac.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1

            if self.cfg.save_demo_frequency> 0 and self.step % self.cfg.save_demo_frequency == 0:
                to_save_demo = True
                self.save_demo_sample = self.step

            self.step += 1

            if self.cfg.eval_frequency > 0 and self.step % self.cfg.eval_frequency == 0:
                to_evaluate = True
                self.evaluate_sample = self.step

            if self.cfg.save_expert and self.step == self.cfg.num_train_steps:
                torch.save(self.agent.actor.state_dict(), f'sac_actor_{self.cfg.env}_{self.step}.pth')


@hydra.main(config_path='config', config_name='train.yaml')
def main(cfg):
    if cfg.wandb:
        wandb.init(project=cfg.project_name, name=cfg.experiment, sync_tensorboard=False, config=cfg)
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
