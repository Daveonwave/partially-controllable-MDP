import numpy as np

def generate_arrival_distribution(lambd: float, total_time: int, goal_floor: int = None) -> list:
    """
    Generate a list of random arrival times based on a Poisson distribution.

    :param lambd: The rate (lambda) parameter for the Poisson distribution.
    :param total_time: The total time period over which to generate arrivals.
    :return: A list of random arrival times.
    """
    person_list = []
    person_id = 0
    
    for t in range(total_time):
        # Draw the number of people arriving at time t from a Poisson distribution
        num_new = np.random.poisson(lambd)
        
        for i in range(num_new):
            person = Passenger(_id=person_id, 
                               arrival_time=t, 
                               goal_floor=goal_floor if goal_floor is not None else np.random.randint(0, 4))
            person_list.append(person)
            person_id += 1
    
    return person_list


class Passenger:
    def __init__(self, _id:int, arrival_time:int, goal_floor:int):
        """
        Initialize a Passenger instance.

        :param _id: The unique identifier for the passenger.
        :param arrival_time: The time at which the passenger arrives.
        :param goal_floor: The floor the passenger wants to go to.
        """
        self._id = _id
        self.arrival_time = arrival_time
        self.goal_floor = goal_floor

    def __repr__(self):
        """
        Return a string representation of the Passenger instance.
        """
        return f"Passenger(id={self._id}, arrival_time={self.arrival_time}, goal_floor={self.goal_floor})"
    
    @property
    def id(self):
        """
        Return the ID of the queue.
        """
        return self._id
        
        
class FloorQueue:
    def __init__(self, floor:int, max_waiting:int = 5):
        """
        Initialize a FloorQueue instance.
        """
        self.floor = floor
        self.max_waiting = max_waiting
        
        self.waitings = []
        self.futures = []
        self.arrivals = {}
    
    def __repr__(self):
        """
        Return a string representation of the PassengerQueue instance.
        """
        return f"FloorQueue(floor={self.floor}, waiting={self.waitings}, futures={self.futures})"
    
    def __len__(self):
        """
        Return the number of passengers in the queue.
        """
        return len(self.waitings)
    
    def reset(self):
        """
        Reset the queue to an empty state.
        """
        self.waitings = []
        self.futures = []
        self.arrivals = {}
    
    def set_arrivals(self, arrivals: list):
        """
        Set the arrivals for the queue.

        :param arrivals: A list of Passenger instances.
        """
        times = set([person.arrival_time for person in arrivals])
        self.arrivals = {time:[person for person in arrivals if person.arrival_time == time] for time in times}
                
    def update_waitings(self):
        """
        Update the queue by moving passengers from the futures list to the waiting list.
        """
        for i, _ in enumerate(self.futures):
            if len(self) >= self.max_waiting:
                break
            self.waitings.append(self.futures.pop(i))
            print(f"Queue {self.floor}: passenger {self.waitings[-1].id} in queue at {self.waitings[-1].arrival_time} seconds")
    
        self.futures = []
                
    def check_arrivals(self, current_time: int):
        """
        Update the queue by removing passengers who have arrived.

        :param current_time: The current time in the simulation.
        """
        if current_time in self.arrivals.keys():
            for person in self.arrivals[current_time]:
                self.futures.append(person)