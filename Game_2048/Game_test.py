from Game_env import GameEnv
from stable_baselines3 import DQN
from sb3_contrib import MaskablePPO as PPO
import pygame
import time
import torch

env = GameEnv()

PATH = "Game_2048\\2048_model_DQN.zip"
PATH = "Game_2048\\2048_model_PPO.zip"
# 
model = DQN.load(PATH)
model = PPO.load(PATH)

def predict_with_mask(model, obs, mask):
    with torch.no_grad():
        obs_t = torch.as_tensor(obs).unsqueeze(0).to(model.device)
        q_values = model.q_net(obs_t)      
        q_values[:, mask == 0] = -torch.inf    
    return q_values.argmax(dim=1).item()

def DQN_test():
    obs, info = env.reset()
    pygame.init()
    while True:
        for event in pygame.event.get():
            pass
        action, state = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        print(reward)
        # time.sleep(0.1)
        env.render()
        # if reward == -10:
            # done = True
        if env.game.win:
            print("win")
            time.sleep(2)
            done = True
        if done:
            obs, info = env.reset()

def PPO_test():
    obs, info = env.reset()
    pygame.init()
    while True:
        for event in pygame.event.get():
            pass
        action_masks = env.get_action_mask()  
        action, _ = model.predict(
            obs,
            action_masks=action_masks,            
            deterministic=True,
        )
        obs, reward, done, _, info = env.step(action)
        env.render()
        if done:
            obs, info = env.reset()

if __name__ == '__main__':
    DQN_test()
    # PPO_test()