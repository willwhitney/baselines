import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import ipdb
from numpy import unravel_index
import random
import gym
from gym import core, spaces
from gridworld.dataset import *
from gridworld.gridworld import *
from gym.envs.registration import register

register(
    id='GridWorld-v0',
    entry_point='gridworld.world:StatefulEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 50},
)

# here states are represented as 2d vectors encoding location
class Environment(nn.Module):
    def __init__(self):
        super(Environment, self).__init__()

    # same as below, in batch mode
    def forward(self, s0, a, world, T):
        batch_size = s0.size(0)
        assert(a.size(0) == batch_size, '{} != {}'.format(a.size(0), batch_size))
        assert(world.size(0) == batch_size, '{} != {}'.format(world.size(0), batch_size))
        s = []
        r = []
        for b in range(batch_size):
            sb, rb = self.forward_single(s0[b], a[b], world[b], T)
            s.append(sb)
            r.append(rb)

        s = torch.stack(s)
        r = torch.stack(r)
        assert(s.size(0) == batch_size)
        assert(r.size(0) == batch_size)
        assert(s.size(1) == T + 1)
        assert(r.size(1) == T + 1)
        return s, r

    # s0: 2d vector representing agent's coordinates on the grid
    # world: 2 x imsize x imsize tensor, first channel indicates wall, second the goal. 
    # a: action sequence
    def forward_single(self, s0, a, world, T):
        walls = world[0]
        goals = world[1]            
        s = []
        r = []
        s.append(s0)
        r.append(0)
        for t in range(0, T):
            s_new = s[t].clone()
            if a[t] == 1:
                s_new[0] -= 1 # up
            elif a[t] == 2:
                s_new[0] += 1 # down
            elif a[t] == 3:
                s_new[1] -= 1 # left
            elif a[t] == 4:
                s_new[1] += 1 # right

            if goals[s_new[0]][s_new[1]] == 10:
                r.append(1) # reached goal, get reward
            elif walls[s_new[0]][s_new[1]] == 1:
                r.append(-1) # bump into wall, do not move and get negative reward
                s_new = s[t].clone()
            else:
                # move, get small negative reward
                r.append(-0.01)

            s.append(s_new)
                
        s = torch.stack(s)
        r = torch.Tensor(r)
        return s, r

class StatefulEnv(core.Env):
    def __init__(self, flatten=False):
        size = 8
        dataset = GridworldData(
            'gridworld/gridworld_{}x{}.npz'.format(size, size), 
            imsize=size, train=True, transform=None)
        self.dataset = dataset
        self.flatten = flatten
        self.action_space = spaces.Discrete(4)

        if flatten:
            img_dims = 3 * size * size
        else:
            img_dims = (dataset.imsize, dataset.imsize, 3)
        self.observation_space = spaces.Box(low=0, high=1, shape=img_dims)
        self.index = 0
        self.max_difficulty = 1
        self.reset()

    def _step(self, action):
        walls = self.world[0]
        goals = self.world[1]

        new_location = self.agent_loc.clone()
        if action == 0:
            new_location[0] -= 1 # up
        if action == 1:
            new_location[0] += 1 # down
        if action == 2:
            new_location[1] -= 1 # left
        if action == 3:
            new_location[1] += 1 # right
        
        done = False
        if goals[new_location[0]][new_location[1]] == 10:
            reward = 1.0 # reached goal, get reward
            done = True
        elif walls[new_location[0]][new_location[1]] == 1:
            reward = -1.0 # bump into wall, do not move and get negative reward
            new_location = self.agent_loc.clone()
            # done = True
        else:
            # move, get small negative reward
            reward = -0.01

        self.agent_loc = new_location

        # print("Distance: ", self.get_current_distance())
        # print("Action: ", action)
        # print(self._get_human_obs())
        # ipdb.set_trace()

        return self._get_obs(), reward, done, {}

    def get_current_distance(self):
        w = self.world
        # get goal
        row = torch.max(torch.max(w[1], 0)[1])
        col = torch.max(torch.max(w[1], 1)[1])
        gridworld_walls = 1 - w[0]
        M = gridworld(gridworld_walls.cpu().numpy(), row, col)
        goal_s = M.map_ind_to_state(M.targetx, M.targety)
        G, W = M.get_graph_inv()
        g_dense = W
        g_sparse = csr_matrix(g_dense)
        d, pred = dijkstra(g_sparse, indices=goal_s, return_predecessors=True)

        current_loc_index = M.map_ind_to_state(self.agent_loc[0],
                                                self.agent_loc[1])
        return d[current_loc_index]

    def advance_curriculum(self):
        self.max_difficulty += 1

    def _get_obs(self):
        agent_map = self.world[0].clone().zero_()
        agent_map[self.agent_loc[0], self.agent_loc[1]] = 1
        full_obs = torch.cat([self.world, agent_map.unsqueeze(0)], 0)
        
        # agent_map[self.agent_loc[0], self.agent_loc[1]] = -10
        # full_obs = torch.cat([self.world, agent_map.unsqueeze(0)], 0)
        # full_obs = torch.sum(full_obs, 0)

        if self.flatten:
            full_obs.resize_(int(torch.Tensor(list(full_obs.size())).prod()))
        else:
            full_obs.transpose_(0, 1).transpose_(1,2)
        # return full_obs.cpu().numpy().transpose()
        return full_obs.cpu().numpy()

    def _get_human_obs(self):
        agent_map = self.world[0].clone().zero_()
        
        agent_map[self.agent_loc[0], self.agent_loc[1]] = 100
        full_obs = torch.cat([self.world, agent_map.unsqueeze(0)], 0)
        full_obs = torch.sum(full_obs, 0)

        return full_obs

    def _reset(self):
        self.index = random.randint(0, len(self.dataset))
        self.world = self.dataset[self.index % len(self.dataset)][0]
        self.agent_loc = self.dataset[self.index % len(self.dataset)][1]
        # ipdb.set_trace()
        if self.get_current_distance() > self.max_difficulty:
            self._reset()
        # else:
            # return self._get_obs()
        # print("Distance: ", self.get_current_distance())
        # print(self._get_human_obs())
        # ipdb.set_trace()
        return self._get_obs()

    def _render(self, *args, **kwargs):
        return self._get_obs()

    def _close(self, *args, **kwargs):
        pass

    def _seed(self, seed=None):
        if seed != None:
            self.index = seed % len(self.dataset)
        else:
            max_index = len(self.dataset) - 1
            self.index = random.randint(0, max_index)
