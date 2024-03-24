from os import path

import gymnasium
import pygame
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class Gridworld(FrozenLakeEnv):
    """
    The same as the FrozenLakeEnv but with elf images switched to ghost images
    """

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
    """
    1D deterministic gridworld with actions {LEFT, RIGHT},
    All episodes start in the center state and terminate in the leftmost
    or rightmost state, transition to the right terminal state gives reward
    of 1 and all other transitions give reward of 0.

    Reference:
        Reinforcement Learning: An Introduction. Sutton, R. and Barto, A. (2018)
        Example 6.2 - TD vs MC Methods

    """

    def __init__(self, nS=7, **kwargs):
        desc = ["H" + nS // 2 * "F" + "S" + nS // 2 * "F" + "G"]
        super(Gridworld1D, self).__init__(desc=desc, is_slippery=False, **kwargs)

        self.action_space = gymnasium.spaces.Discrete(2, seed=42)
        self.elf_images[DOWN] = self.elf_images[RIGHT]
        for s in range(self.observation_space.n):  # type: ignore
            self.P[s][DOWN] = self.P[s][RIGHT].copy()
            del self.P[s][RIGHT]
            del self.P[s][UP]


class ClassicGridworld(Gridworld):
    """
    Simple deterministic 4x4 gridworld with actions {LEFT, DOWN, RIGHT, UP},
    for all transitions are reward is -1, the task is undiscounted and episodic
    There are two terminal states

    Reference:
        Reinforcement Learning: An Introduction. Sutton, R. and Barto, A. (2018)
        Example 4.1 - Dynamic Programming for Gridworld
    """

    def __init__(self, **kwargs):
        desc = ["GFFF"]
        desc.extend(["FFFF" for _ in range(2)])
        desc.append("SFFG")
        super(ClassicGridworld, self).__init__(desc=desc, is_slippery=False, **kwargs)

        # Adjust the rewards (in original env reward is 0 not -1)
        for s in self.P:
            for a in self.P[s]:
                for p in range(len(self.P[s][a])):
                    P, S, R, done = self.P[s][a][p]
                    if s not in [0, 15]:
                        self.P[s][a][p] = (P, S, -1, done)
