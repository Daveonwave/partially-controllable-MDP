from gymnasium.envs.registration import register

register(
    id='pcmdp/elevator-v0',
    entry_point='pcmdp.elevator.elevator_env:ElevatorEnv'
    )

register(
    id='pcmdp/taxi-v0',
    entry_point='pcmdp.taxi.taxi_env:TaxiEnv'
    )