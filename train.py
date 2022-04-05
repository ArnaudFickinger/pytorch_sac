#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
sys.path.insert(0,'/private/home/arnaudfickinger/gw_il')
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils_sac

import dmc2gym
import hydra
import wandb


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
                       visualize_reward=True)
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
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils_sac.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils_sac.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
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

    def run(self):
        episode_reward, done = 0, True
        to_evaluate = False
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
            self.step += 1

            if self.cfg.eval_frequency > 0 and self.step % self.cfg.eval_frequency == 0:
                to_evaluate = True
                self.evaluate_sample = self.step

            if self.cfg.save_expert and self.step == self.cfg.num_train_steps:
                torch.save(self.agent.actor.state_dict(), f'sac_actor_{self.cfg.env}_{self.step}.pth')


@hydra.main(config_path='config', config_name='train.yaml')
def main(cfg):
    if cfg.wandb:
        wandb.init(project=cfg.project_name, name=f'train_{cfg.env}', sync_tensorboard=False, config=cfg)
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
