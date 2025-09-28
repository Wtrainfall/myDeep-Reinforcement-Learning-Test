import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import pygame


class DQN(nn.Module):
    def __init__(self, input_size, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, env, gamma=0.99, lr=0.001, buffer_size=10000, batch_size=32, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.num_actions = env.action_space.n
        self.input_size = env.observation_space.shape[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(self.input_size, self.num_actions).to(self.device)
        self.target_net = DQN(self.input_size, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.buffer_size)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        current_q = self.policy_net(state).gather(1, action)
        max_next_q = self.target_net(next_state).detach().max(1)[0].unsqueeze(1)
        expected_q = reward + self.gamma * max_next_q * (1 - done)

        loss = F.mse_loss(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, num_episodes=1000):
        rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                self.train_step()

            rewards.append(episode_reward)
            print(f"Episode: {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {self.epsilon:.4f}")

            if episode % 10 == 0:
                self.update_target_net()

        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.show()

def test(env, agent):
    pygame.init()
    agent.policy_net.load_state_dict(torch.load('flappybird_model.pth'))
    agent.policy_net.eval()
    agent.epsilon = 0
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        for event in pygame.event.get():
            pass                                    #byd不把事件Get掉游戏会崩
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        # print(f"Action: {action}, Reward: {reward}")
        state = next_state
        episode_reward += reward
        env.render()
        
    print(f"Test Reward: {episode_reward}")

if __name__ == "__main__":
    from flappybird import FlappyBird
    # env = FlappyBird()
    # agent = DQNAgent(env)
    # agent.train(num_episodes=1000)

    # torch.save(agent.policy_net.state_dict(), 'Flappy_bird/flappybird_model.pth')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for i in range(100):
        env = FlappyBird()
        agent = DQNAgent(env)
        test(env, agent)