import os, sys
import random
import pickle
import re
import numpy as np
import torch
import math
import numpy
import glob
import pdb
import ipdb
import scipy.io
import torch.utils.data as data

class GridWorldLoader(object):

    def __init__(self, path, gridsize):
        super(GridWorldLoader, self).__init__()
        self.data = scipy.io.loadmat(path)
        self.world = torch.FloatTensor()
        self.state = torch.LongTensor()
        self.n_samples = len(self.data['im_data'])
        self.gridsize = gridsize

        if 'im_data' not in self.data:
            assert('all_im_data' in self.data)
            self.data['im_data'] = self.data['all_im_data']
        if 'value_data' not in self.data:
            assert('all_value_data' in self.data)
            self.data['value_data'] = self.data['all_value_data']
        ipdb.set_trace()
        # if 'im_data' not in self.data:
        #     assert('all_im_data' in self.data)
        #     self.data['im_data'] = self.data['all_im_data']
    # def get_batch(self, batch_size):
    #     self.world.resize_(batch_size, 2, self.gridsize, self.gridsize)
    #     self.state.resize_(batch_size, 2)
    #     for b in range(batch_size):
    #         indx = random.randint(0, self.n_samples - 1)
    #         world = 1-self.data['im_data'][indx].reshape(self.gridsize, self.gridsize)
    #         goal = self.data['value_data'][indx].reshape(self.gridsize, self.gridsize)
    #         world = torch.from_numpy(world)
    #         goal = torch.from_numpy(goal)
    #         state = torch.from_numpy(self.data['state_xy_data'][indx])
    #         assert(world[state[0]][state[1]] == 0)
    #         self.world[b][0].copy_(world)
    #         self.world[b][1].copy_(goal)
    #         self.state[b].copy_(state)

    #     return self.world, self.state

    def __getitem__(self, index):
        world = 1 - self.data['im_data'][index].reshape(
            self.gridsize, self.gridsize)
        goal = self.data['value_data'][index].reshape(self.gridsize, self.gridsize)
        world = torch.from_numpy(world)
        goal = torch.from_numpy(goal)
        state = torch.from_numpy(self.data['state_xy_data'][index])
        assert(world[state[0]][state[1]] == 0)
        # self.world[b][0].copy_(world)
        # self.world[b][1].copy_(goal)
        # ipdb.set_trace()
        return torch.stack([world, goal]).float(), state.long()
        # self.state[b].copy_(state)

    def __len__(self):
        return len(self.data['im_data'])
