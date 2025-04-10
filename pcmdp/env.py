from gymnasium import Env, spaces
import matplotlib.pyplot as plt
import numpy as np
from .simulator.elevator import Elevator
from .rewards import average_waiting_time

class ElevatorEnv(Env):
    
    def __init__(self, 
                 settings:dict,
                 arrival_distributions:list,
                 init_elevator_pos:int = None,
                 total_time:int = 3600,
                 ):
        metadata = {'render_modes': ['human']}
        
        self.elevator = Elevator(**settings)
        self.init_elevator_pos = init_elevator_pos
        self.arrival_distributions = arrival_distributions
        
        self.current_time = 0
        self.total_time = total_time
          
        self.observation_space = spaces.Dict({
            'current_position': spaces.Discrete(int(self.elevator.floor_height * self.elevator.max_floor / self.elevator.movement_speed + 1)),
            'n_passangers': spaces.Discrete(self.elevator.max_capacity + 1),
            'speed': spaces.Discrete(3),
            'floor_queues': spaces.MultiDiscrete([5] * (self.elevator.max_floor - self.elevator.min_floor)),
            'arrivals': spaces.MultiDiscrete([5] * (self.elevator.max_floor - self.elevator.min_floor)),
        })
        self.action_space = spaces.Discrete(3)
    
    def _get_obs(self):
        return {
            'current_position': self.elevator.current_position,
            'n_passangers': len(self.elevator.passengers),
            'speed': self.elevator.speed,
            # No ground floor because (ground floor == goal)
            'floor_queues': [len(queue) for queue in self.elevator.queues if queue.floor != 0],     
            'arrivals': [len(queue.futures) for queue in self.elevator.queues if queue.floor != 0],
        }
    
    def _get_info(self):
        return {
            'elevator_status': self.elevator.status(),
            'current_time': self.current_time,
            'min_floor': self.elevator.min_floor,
            'max_floor': self.elevator.max_floor,
            'floor_height': self.elevator.floor_height,
        }
    
    def reset(self, seed=None, options=None):
        """

        Args:
            seed (_type_, optional): 
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        super().reset(seed=seed)
        
        # Reset the elevator and queues
        self.current_time = 0
        self.elevator.reset()
        
        if self.init_elevator_pos is not None:
            self.elevator.current_position = self.init_elevator_pos
        
        # Init the queues with the arrival distributions
        for i, arrivals in enumerate(self.arrival_distributions):
            self.elevator.queues[i].set_arrivals(arrivals=arrivals)
        
        self.elevator.update_queues(current_time=self.current_time)
        return self._get_obs(), self._get_info()
        
    def step(self, action):
        """
        Perform one step in the environment.

        Args:
            action (np.ndarray): The action to take.
            0: move up
            1: move down
            2: stay still

        Returns:
            tuple: (observation, reward, done, truncated, info)
                observation (dict): The current state of the environment.
                reward (float): The reward received after taking the action.
                done (bool): Whether the episode has ended.
                truncated (bool): Whether the episode was truncated.
                info (dict): Additional information about the step.
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        served = []
        
        # 0: move up, 1: move down, 2: stay still
        if action == 0:
            self.elevator.move(1)
        elif action == 1:
            self.elevator.move(-1)
        elif action == 2: 
            served = self.elevator.open_doors()
        else:
            raise ValueError(f"Invalid action {action}")
        
        # Calculate the reward 
        reward = 0
        if len(served) > 0:
            reward += len(served) * 10
        else:
            reward -= 1
        
        # Check arrivals at the current time
        self.current_time += 1
        self.elevator.update_queues(current_time=self.current_time)
        
        done = self.current_time >= self.total_time
        
        return self._get_obs(), reward, done, False, self._get_info()
        
    def render(self, mode='human'):
        """
        Render the environment.

        Args:
            mode (str): The mode of rendering. Currently, only 'human' is supported.
        """
        if mode != 'human':
            raise NotImplementedError("Only 'human' rendering mode is supported.")

        # Create a blank canvas
        fig, ax = plt.subplots(figsize=(5, 10))
        ax.set_xlim(-1, 1)
        ax.set_ylim(self.elevator.min_floor - 1, self.elevator.max_floor + 1)
        ax.set_title("Elevator Environment")
        ax.set_xlabel("Elevator Shaft")
        ax.set_ylabel("Floors")

        # Draw floors
        for floor in range(self.elevator.min_floor, self.elevator.max_floor + 1):
            ax.hlines(y=floor, xmin=-0.5, xmax=0.5, color='gray', linestyle='--', linewidth=0.5)
            ax.text(-0.8, floor, f"Floor {floor}", verticalalignment='center')

        # Draw elevator
        elevator_y = self.elevator.current_position / self.elevator.floor_height
        ax.add_patch(plt.Rectangle((-0.4, elevator_y - 0.2), 0.8, 0.4, color='blue', alpha=0.7))
        ax.text(0, elevator_y, f"Elevator\n({len(self.elevator.passengers)} passengers)", 
                color='white', ha='center', va='center', fontsize=8)

        # Draw floor queues
        for floor, queue in enumerate(self.elevator.queues):
            ax.text(0.6, floor, f"Queue: {len(queue)}", verticalalignment='center', fontsize=8)

        plt.show()
    
    def close(self):
        plt.close()

