import numpy as np
import gymnasium as gym


class ElevatorEnv(gym.Env):
    """_summary_

    Args:
        gym (_type_): _description_
    """
    def __init__(self, num_floors=10, num_elevators=1):
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.reset()
