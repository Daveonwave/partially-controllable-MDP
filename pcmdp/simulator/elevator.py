from .passenger import Passenger, FloorQueue, generate_generic_passengers


class Elevator:
    """
    A class to represent an elevator in a building.

    Attributes:
        id (int): The unique identifier for the elevator.
        min_floor (int): The minimum floor the elevator can reach.
        max_floor (int): The maximum floor the elevator can reach.
        current_floor (int): The current floor of the elevator.
        direction (str or None): The current direction of the elevator ('up', 'down', or None if stationary).
        passengers (list): The list of passengers currently in the elevator.
        speed (float): The speed of the elevator.
        max_capacity (int): The maximum number of passengers the elevator can hold.
    """
    def __init__(self, movement_speed, max_capacity, min_floor, max_floor, floor_height, max_queue_length=5, max_arrivals=5):
        """
        Initializes an Elevator instance.

        Args:
            movement_speed (float): The speed of the elevator when mooving.
            max_capacity (int): The maximum number of passengers the elevator can hold.
            min_floor (int): The minimum floor the elevator can reach.
            max_floor (int): The maximum floor the elevator can reach.
            floor_height (float): The height of each floor.
        """
        assert floor_height % movement_speed == 0, "Floor height must be divisible by movement speed"
        
        self.current_position = 0 
        self.min_floor = min_floor
        self.max_floor = max_floor
        self.floor_height = floor_height    
        
        self.speed = 0
        self.passengers = []
        
        self.max_capacity = max_capacity
        self.movement_speed = movement_speed
        
        self.queues = [FloorQueue(floor=floor, max_queue_length=max_queue_length, max_arrivals=max_arrivals) for floor in range(min_floor, max_floor + 1)]

    def reset(self):
        """
        Resets the elevator to its initial state.
        """
        self.current_position = 0
        self.speed = 0
        self.passengers = []
        
        for queue in self.queues:
            queue.reset()
    
    def move(self, direction:str):
        """
        Moves the elevator in the specified direction. 
        The elevator can only move up or down by its movement speed.
        The elevator cannot move beyond its min and max floor limits.

        Args:
            direction (int): _description_
        """
        assert direction in ['up', 'down'], "Direction must be 'down' 'up'"
        
        if direction == 'up' and self.current_position / self.floor_height < self.max_floor:
            self.current_position += self.movement_speed
        elif direction == 'down' and self.current_position / self.floor_height > self.min_floor:   
            self.current_position -= self.movement_speed
        else:
            pass
            #print("Elevator cannot move in this direction.")
        return 0
    
    def open_doors(self):
        """
        Opens the elevator doors.
        """
        if self.current_position % 1 != 0:
            return 0

        floor = int(self.current_position / self.floor_height)
        
        served = []
        
        for passanger in self.passengers:
            if passanger.goal_floor == floor:
                served.append(passanger)
                self.remove_passenger(passanger)
        
        for _ in range(len(self.queues[floor])):
            if len(self.passengers) < self.max_capacity and len(self.queues[floor]) > 0:
                self.add_passenger(self.queues[floor].waitings.pop(0))
        
        return served

    def add_passenger(self, passanger: Passenger):
        """
        Add a passenger to the elevator if it is not full.
        """
        self.passengers.append(passanger)
            
    def remove_passenger(self, passanger: Passenger):
        """
        Removes a passenger from the elevator.
        """
        if len(self.passengers) > 0:
            self.passengers.remove(passanger)
        else:
            raise ValueError("The elevator is empty and no passanger can be removed!")
    
    def update_queues(self, current_time: int):
        """
        Check the arrival of passengers at each floor and add them to the queue.

        Args:
            current_time (int): The current time in the simulation.
        """
        for queue in self.queues:
            queue.update_waitings()
            queue.check_arrivals(current_time=current_time)
    
    def status(self):
        """
        Returns the current status of the elevator.

        Returns:
            dict: A dictionary containing the elevator's id, current floor, direction,
                  passenger count, and maximum capacity.
        """
        return {
            'current_position': self.current_position,
            'speed': self.speed,
            'passenger_count': len(self.passengers),
            'queues': [len(queue) for queue in self.queues],
        }
        
    def set_status(self, status, current_time):
        """
        Sets the current status of the elevator.

        Args:
            status (dict): A dictionary containing the current floor, direction,
                           passenger count, and maximum capacity.
        """
        self.current_position = status['current_position']
        self.speed = status['speed']
        self.passengers = generate_generic_passengers(n=status['n_passengers'], global_goal_floor=0)
        
        for i, queue_len in enumerate(status['floor_queues']):
            self.queues[i+1].set_queue(generate_generic_passengers(n=queue_len, global_goal_floor=0))
            self.queues[i+1].check_arrivals(current_time=current_time - 1)
        