from os import path

import gymnasium
import pygame
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class Gridworld(FrozenLakeEnv):

    def __init__(self, **kwargs):
        super(Gridworld, self).__init__(**kwargs)

        elfs = [
            path.join(path.dirname(__file__), "img/ghost_left.png"),
            path.join(path.dirname(__file__), "img/ghost_down.png"),
            path.join(path.dirname(__file__), "img/ghost_right.png"),
            path.join(path.dirname(__file__), "img/ghost_up.png"),
        ]

        self.elf_images = [
            pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
            for f_name in elfs
        ]


class Gridworld1D(Gridworld):

    def __init__(self, nS=7, **kwargs):
        desc = ['H' + nS // 2 * 'F' + 'S' + nS // 2 * 'F' + 'G']
        super(Gridworld1D, self).__init__(
            desc=desc, is_slippery=False, **kwargs)

        self.action_space = gymnasium.spaces.Discrete(2, seed=42)
        self.elf_images[DOWN] = self.elf_images[RIGHT]
        for s in range(self.observation_space.n):  # type: ignore
            self.P[s][DOWN] = self.P[s][RIGHT].copy()
            del self.P[s][RIGHT]
            del self.P[s][UP]
