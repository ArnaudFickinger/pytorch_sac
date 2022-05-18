# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import imageio
import os
import numpy as np
import glob

from os import listdir
from os.path import isfile, join

import wandb

from dmc2gym import natural_imgsource

class VideoRecorder(object):
    def __init__(self, dir_name=None, wandb=True, resource_files='', img_source=None, total_frames=1000, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []
        self.wandb=wandb
        if wandb:
            self.frames_wdb = []
        if img_source is not None:
            shape2d = (height, width)
            if img_source == "color":
                self._bg_source = natural_imgsource.RandomColorSource(shape2d)
            elif img_source == "noise":
                self._bg_source = natural_imgsource.NoiseSource(shape2d)
            else:
                # files = glob.glob(os.path.expanduser(resource_files))

                files = [f'{resource_files}/{f}' for f in listdir(resource_files) if isfile(join(resource_files, f))]
                # import pdb;pdb.set_trace()
                assert len(files), "Pattern {} does not match any files".format(
                    resource_files
                )
                if img_source == "images":
                    self._bg_source = natural_imgsource.RandomImageSource(shape2d, files, grayscale=True,
                                                                          total_frames=total_frames)
                elif img_source == "video":
                    self._bg_source = natural_imgsource.RandomVideoSource(shape2d, files, grayscale=True,
                                                                          total_frames=total_frames)
                else:
                    raise Exception("img_source %s not defined." % img_source)
        else:
            self._bg_source = None

    def init(self, enabled=True):
        self.frames = []
        if self.wandb:
            self.frames_wdb = []
        # self.enabled = self.dir_name is not None and enabled
        self.enabled = enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            # import pdb;pdb.set_trace()
            if self._bg_source:
                mask = np.logical_and((frame[:, :, 2] > frame[:, :, 1]), (frame[:, :, 2] > frame[:, :, 0]))  # hardcoded for dmc
                bg = self._bg_source.get_image()
                frame[mask] = bg[mask]
            self.frames.append(frame)
            if self.wandb:
                # import pdb;pdb.set_trace()
                self.frames_wdb.append(frame.transpose(2,0,1))

    def save(self, file_name, str=''):
        if self.enabled:
            # path = os.path.join(self.dir_name, file_name)
            # imageio.mimsave(path, self.frames, fps=self.fps)
            if self.wandb:
                wandb.log({f"{str}Agent after {file_name} samples": wandb.Video(np.stack(self.frames_wdb, axis=0), fps=30,
                                                                format="mp4")})
