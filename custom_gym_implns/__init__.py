from gymnasium.envs.registration import register

register(
     id="SamSegEnv-v0",
     entry_point="custom_gym_implns.envs:SamSegEnv",
)