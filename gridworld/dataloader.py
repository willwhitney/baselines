import os, sys
import random
import cPickle as pickle
import re
import numpy as np
import torch
import math
import numpy
import glob
import pdb
import scipy.io


class GridWorldLoader(object):

    def __init__(self, path, gridsize):
        super(GridWorldLoader, self).__init__()
        self.data = scipy.io.loadmat(path)
        self.world = torch.FloatTensor()
        self.state = torch.LongTensor()
        self.n_samples = len(self.data['im_data'])
        self.gridsize = gridsize

    def get_batch(self, batch_size):
        self.world.resize_(batch_size, 2, self.gridsize, self.gridsize)
        self.state.resize_(batch_size, 2)
        for b in range(batch_size):
            indx = random.randint(0, self.n_samples - 1)
            world = 1-self.data['im_data'][indx].reshape(self.gridsize, self.gridsize)
            goal = self.data['value_data'][indx].reshape(self.gridsize, self.gridsize)
            world = torch.from_numpy(world)
            goal = torch.from_numpy(goal)
            state = torch.from_numpy(self.data['state_xy_data'][indx])
            assert(world[state[0]][state[1]] == 0)
            self.world[b][0].copy_(world)
            self.world[b][1].copy_(goal)
            self.state[b].copy_(state)

        return self.world, self.state
