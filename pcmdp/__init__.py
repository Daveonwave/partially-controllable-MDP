from gymnasium.envs.registration import register

register(
    id='pcmdp/elevator-v0',
    entry_point='pcmdp.env:ElevatorEnv',
)