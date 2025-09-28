from Game_env import GameEnv
from sb3_contrib import MaskablePPO as PPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import ProgressBarCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

lr_schedule = linear_schedule(5e-4, 5e-6)

def mask_action_fn(env):
    return env.get_action_mask()

def make_env():
    def _init():
        env = GameEnv()
        env = ActionMasker(env, mask_action_fn)
        env.reset()
        return env
    return _init

if __name__ == '__main__':
    n_envs = 16

    env_fns = [make_env() for _ in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    progress = ProgressBarCallback()
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        learning_rate=lr_schedule,
        gamma=0.94,
        device='cuda',
        batch_size=8192,
        n_steps=256,
        ent_coef=0.01,
        gae_lambda=0.95,
        clip_range=0.2,
        n_epochs=10,
    )

    model.learn(
        total_timesteps=int(1e7),
        callback=progress,
    )

    model.save('Game_2048/2048_model_PPO')