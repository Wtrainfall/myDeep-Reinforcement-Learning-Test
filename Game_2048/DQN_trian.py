from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from Network import CustomCNN
from Game_env import GameEnv

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[256]   
)

def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

lr_schedule = linear_schedule(3e-4, 5e-6)

def make_env():
    def _init():
        env = GameEnv()
        env.reset()
        return env
    return _init

def main():
    env_num = 16
    env = SubprocVecEnv([make_env() for _ in range(env_num)])

    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=lr_schedule,
        buffer_size=300_000,
        learning_starts=int(1e6),
        batch_size=512,
        gamma=0.94,
        target_update_interval=5000,
        exploration_fraction=0.5,
        exploration_final_eps=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    model.learn(total_timesteps=int(2e7), callback=ProgressBarCallback())
    model.save("Game_2048\\2048_model_DQN.zip")

if __name__ == '__main__':
    main()