from Game import Game2048
import gymnasium as gym
import numpy as np
import time
import pygame

class GameEnv(gym.Env):
    def __init__(self):
        self.game = Game2048()
        self.game.reset()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 2048, shape=(16,), dtype=int)
        self.done = False
        self.info = {}
    
    def reset(self, seed=None, options=None):
        super().reset()
        self.game.reset()
        self.done = False
        self.info = {}
        obs = self._generate_observation()
        return obs, self.info

    def _generate_observation(self):
        board = self.game.board
        obs = []
        for i in range(4):
            for j in range(4):
                if board[i][j] == 0:
                    obs.append(0)
                else:
                    obs.append(int(np.log2(board[i][j])))
        return obs
       
    def step(self, action):
        reward, self.done = self.game.step(action)
        obs = self._generate_observation()
        return obs, reward, self.done, self.done, self.info
    
    def render(self, mode='human'):
        self.game.render()

    def get_action_mask(self):
        action_mask = self.game.get_action_mask()
        return np.array(action_mask)

if __name__ == '__main__':
    #测试环境
    env = GameEnv()
    env.reset()
    env.render()
    while True:
        for event in pygame.event.get():
            pass
        time.sleep(0.5)
        #选择动作
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(obs, reward, done, info)
        env.render()
        if done:
            break
    env.close()
