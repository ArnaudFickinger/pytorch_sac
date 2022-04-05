import imageio
import os
import numpy as np
import sys

import wandb


class VideoRecorder(object):
    def __init__(self, height=256, width=256, camera_id=0, fps=30, wandb=False):
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []
        if wandb:
            self.frames_wdb = []
        self.wandb = wandb

    def init(self, enabled=True):
        self.frames = []
        if self.wandb:
            self.frames_wdb = []
        self.enabled = enabled

    def record(self, env):
        # import pdb; pdb.set_trace()
        if self.enabled:
            frame = env.render(mode='rgb_array',
                               height=self.height,
                               width=self.width)
            self.frames.append(frame)
            if self.wandb:
                self.frames_wdb.append(frame.transpose(2,0,1))

    def save(self, samples):
        if self.enabled:
            if self.wandb:
                wandb.log({f"Agent after {samples} samples": wandb.Video(np.stack(self.frames_wdb, axis=0), fps=30,
                                                                format="mp4")})
