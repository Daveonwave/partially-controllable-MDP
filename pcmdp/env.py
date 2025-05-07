from gymnasium import Env, spaces
import matplotlib.pyplot as plt
import numpy as np
import pygame
from .simulator.elevator import Elevator
from .rewards import average_waiting_time

class ElevatorEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, 
                 settings:dict,
                 arrival_distributions:list,
                 init_elevator_pos:int = None,
                 total_time:int = 3600,
                 render_mode:str = None
                 ):
        super().__init__()
        
        self._elevator = Elevator(**settings)
        self.init_elevator_pos = init_elevator_pos
        self.arrival_distributions = arrival_distributions
        
        self.current_time = 0
        self.total_time = total_time
        
        self.observation_space = spaces.Dict({
            'current_position': spaces.Discrete(int(self._elevator.floor_height * self._elevator.max_floor / self._elevator.movement_speed) + 1, start=0),
            'n_passengers': spaces.Discrete(self._elevator.max_capacity + 1, start=0),
            'speed': spaces.Discrete(3, start=-1),  # -1: down, 0: still, 1: up (multiply by movement speed)
            'floor_queues': spaces.MultiDiscrete(np.array([settings['max_queue_length'] + 1] * (self._elevator.max_floor - self._elevator.min_floor)), start=[0] * (self._elevator.max_floor - self._elevator.min_floor)),
            'arrivals': spaces.MultiDiscrete(np.array([settings['max_arrivals'] + 1] * (self._elevator.max_floor - self._elevator.min_floor)), start=[0] * (self._elevator.max_floor - self._elevator.min_floor))
        })
       
        # Action space -> 1: stay still, 2: move up, 0: move down
        self.action_space = spaces.Discrete(3)
        
         # Pygame setup
        self.render_mode = render_mode
        if self.render_mode is not None and self.render_mode in self.metadata["render_modes"]:
            self._init_pygame()
        
    def _init_pygame(self):
        self.window_size = 512
        self.size = 5  
        
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
        else:
            self.window = None
            self.clock = None

    def set_state(self, state):
        self._elevator.set_status(state, self.current_time)
    
    def _get_obs(self):
        return {
            'arrivals': [len(queue.futures) for queue in self._elevator.queues if queue.floor != 0],
            'current_position': self._elevator.current_position // self._elevator.movement_speed,
            'floor_queues': [len(queue) for queue in self._elevator.queues if queue.floor != 0],  
            'n_passengers': len(self._elevator.passengers),
            'speed': self._elevator.speed,            
        }
    
    def _get_info(self):
        return {
            'elevator_status': self._elevator.status(),
            'current_time': self.current_time,
            'min_floor': self._elevator.min_floor,
            'max_floor': self._elevator.max_floor,
            'floor_height': self._elevator.floor_height,
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
        self._elevator.reset()
        
        if self.init_elevator_pos is not None:
            self._elevator.current_position = self.init_elevator_pos
        
        # Init the queues with the arrival distributions
        for i, arrivals in enumerate(self.arrival_distributions):
            self._elevator.queues[i].set_arrivals(arrivals=arrivals)
        
        self._elevator.update_queues(current_time=self.current_time)
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), self._get_info()
        
    def step(self, action):
        """
        Perform one step in the environment.

        Args:
            action (np.ndarray): The action to take.
            0: move down
            1: stay still
            2: move up
            
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
        if action == 2:
            self._elevator.move('up')
        elif action == 0:
            self._elevator.move('down')
        elif action == 1: 
            served = self._elevator.open_doors()
        else:
            raise ValueError(f"Invalid action {action}")
        
        # Calculate the reward 
        reward = 0
        if len(served) > 0:
            reward += len(served) * 10
        else:
            reward -= 1 * (len(self._elevator.passengers) + sum([len(queue) for queue in self._elevator.queues]))
        
        # Check arrivals at the current time
        self.current_time += 1
        self._elevator.update_queues(current_time=self.current_time)
        
        done = self.current_time >= self.total_time
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), reward, done, False, self._get_info()
        
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
            
    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        # The size of a single grid square in pixels
        pix_square_size = (self.window_size / self.size)  
        elevator_square_size = pix_square_size - 1

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                elevator_square_size * np.array([0,0]),
                (elevator_square_size, elevator_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.array([1,1]) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        
        # Vertical line
        pygame.draw.line(
                canvas,
                0,
                (pix_square_size, 0),
                (pix_square_size, self.window_size),
                width=3,
            )
        
        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
                
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

