from pcmdp.simulator.passenger import generate_arrival_distribution
from pcmdp.env import ElevatorEnv
from algo import *
from rich.pretty import pprint
import gymnasium as gym


if __name__ == "__main__":
    
    settings = {
        'movement_speed': 3,
        'max_capacity': 4,
        'min_floor': 0,
        'max_floor': 3,
        'floor_height': 6,
        'max_arrivals': 2,
        'max_queue_length': 3
        }
    
    # The ground floor has no arrivals
    arrival_distributions = [[]]
    total_time = 3600
    
    for length in range(settings['min_floor'], settings['max_floor']):
        arrival_distributions.append(generate_arrival_distribution(lambd=0.05, total_time=total_time, goal_floor=0))
    
    
    params = {"settings": settings, 
              "arrival_distributions": arrival_distributions, 
              "total_time": total_time, 
              "init_elevator_pos": None
              }
    render_mode = None #'human'
    env = gym.make("pcmdp/elevator-v0", render_mode=render_mode, **params)
    
    q_learning(env)