import pygame
import random
import numpy as np

pygame.init()

SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

BIRD_WIDTH = 32
BIRD_HEIGHT = 22
PIPE_WIDTH = 52
PIPE_GAP = 100
PIPE_SPEED = 5
GRAVITY = 1
JUMP_VELOCITY = -9

class FlappyBird:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()
        self.reset()

        self.action_space = type('ActionSpace', (object,), {'n': 2})()  # 两个动作：0（不跳），1（跳）
        self.observation_space = type('ObservationSpace', (object,), {'shape': (5,)})()  # 状态是一个5维向量
        self.background = pygame.image.load('flappybird\\flappybird\\bg_day.png')
        self.pipe_up_img = pygame.image.load('flappybird\\flappybird\pipe_down.png')
        self.pipe_down_img = pygame.image.load('flappybird\\flappybird\\pipe_up.png')

        self.bird_frames = [pygame.image.load(f'flappybird\\flappybird\\bird0_0.png'),
                            pygame.image.load(f'flappybird\\flappybird\\bird0_1.png'),
                            pygame.image.load(f'flappybird\\flappybird\\bird0_2.png'),
                            pygame.image.load(f'flappybird\\flappybird\\bird0_1.png'),
                            pygame.image.load(f'flappybird\\flappybird\\bird0_0.png'),
                            ]

    def reset(self):
        self.bird_x = SCREEN_WIDTH // 3
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.pipes = []
        self.add_pipe()
        self.score = 0
        self.game_over = False
        self.state = self.get_state()
        self.font = pygame.font.SysFont(None, 36)
        return self.state

    def add_pipe(self):
        gap_y = random.randint(50, SCREEN_HEIGHT - PIPE_GAP - 50)
        self.pipes.append({
            'x': SCREEN_WIDTH,
            'y': gap_y,
            'passed': False
        })

    def step(self, action):
        reward = 0.1  
        done = False

        if action == 1:
            self.bird_velocity = JUMP_VELOCITY

        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity

        for pipe in self.pipes:
            pipe['x'] -= PIPE_SPEED
            if pipe['x'] < -PIPE_WIDTH:
                self.pipes.remove(pipe)
                self.add_pipe()

            if (self.bird_x + BIRD_WIDTH > pipe['x'] and
                self.bird_x < pipe['x'] + PIPE_WIDTH and
                (self.bird_y < pipe['y'] or self.bird_y + BIRD_HEIGHT > pipe['y'] + PIPE_GAP)):
                reward = -1
                done = True

            if not pipe['passed'] and self.bird_x > pipe['x'] + PIPE_WIDTH:
                pipe['passed'] = True
                self.score += 1
                reward = 1

        if self.bird_y < 0 or self.bird_y + BIRD_HEIGHT > SCREEN_HEIGHT:
            reward = -1
            done = True

        self.state = self.get_state()
        self.game_over = done
        return self.state, reward, done, {}
 
    def get_state(self):
        closest_pipe = None
        for pipe in self.pipes:
            if pipe['x'] > self.bird_x:
                closest_pipe = pipe
                break
        if closest_pipe is None:
            closest_pipe = self.pipes[-1]

        state = [
            self.bird_x,
            self.bird_y,
            self.bird_velocity,
            closest_pipe['x'],
            closest_pipe['y']
        ]
        return np.array(state, dtype=np.float32)

    def render(self):

        self.screen.blit(self.background, (0, 0))

        bird = self.bird_frames[self.bird_velocity // -5 % len(self.bird_frames)]
        self.screen.blit(bird, (self.bird_x, self.bird_y))
        # pygame.draw.rect(self.screen, RED, (self.bird_x, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT))
        # print(self.pipe_up_img.get_height())

        for pipe in self.pipes:
            self.screen.blit(self.pipe_up_img, (pipe['x'], pipe['y'] - self.pipe_up_img.get_height()))
            self.screen.blit(self.pipe_down_img, (pipe['x'], pipe['y'] + PIPE_GAP))
            
        img = self.font.render(f'Score: {self.score}', True, BLACK)
        self.screen.blit(img, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = FlappyBird()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                _, _, done, _ = env.step(1)
                if done:
                    env.reset()
        _, _, done, _ = env.step(0)
        if done:
            env.reset()
        env.render()
    env.close()