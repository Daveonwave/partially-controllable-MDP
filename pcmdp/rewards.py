def average_waiting_time(passangers: list, current_time: float) -> float:
    """
    Calculate the average waiting time of passengers.

    Args:
        passangers (list): List of passengers with their arrival times.
        current_time (float): The current time.

    Returns:
        float: The average waiting time of passengers.
    """
    if not passangers:
        return 0.0
    total_waiting_time = sum(current_time - p['arrival_time'] for p in passangers)
    return total_waiting_time / len(passangers)