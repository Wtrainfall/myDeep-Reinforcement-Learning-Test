import pygame
import random
import numpy as np
WINDOW_SIZE   = (500, 550)         
MARGIN        = 10                 
GRID_PADDING  = 6                  
GRID_SIZE     = 4                   
BG_COLOR      = (250, 248, 239)     
EMPTY_COLOR   = (205, 192, 180)
FONT_COLOR    = (119, 110, 101)

TILE_COLORS = {
    2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121),
    16: (245, 149, 99), 32: (246, 124, 95), 64: (246, 94, 60),
    128: (237, 207, 114), 256: (237, 204, 97), 512: (237, 200, 80),
    1024: (237, 197, 63), 2048: (237, 194, 46)
}

def font_color(val):
    return (249, 246, 242) if val > 4 else FONT_COLOR

def get_font(size):
    return pygame.font.Font(pygame.font.match_font("freesansbold"), size)

class Game2048:
    def __init__(self):
        self.board = None
        self.score = 0
        self.game_over = False
        self.win = False
        self.action = None
        self.screen = None
        self.clock = None
        
    def reset(self):
        self.board = [[0 for i in range(4)] for j in range(4)]
        self._generate_new_tile()
        self.score = 0
        self.game_over = False
        self.action = None
        self.win = False
        # self.render()

    def _generate_new_tile(self):
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.board[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def _move_left(self):
        reward = 0
        for i in range(4):
            line = self.board[i]
            line = [x for x in line if x != 0]
            merged = []
            j = 0
            while j < len(line):
                if j+1 < len(line) and line[j] == line[j+1]:
                    merged.append(line[j]*2)
                    self.score += line[j]*2
                    reward = max(reward, np.log2(line[j]*2))
                    j += 2
                else:
                    merged.append(line[j])
                    j += 1
            merged += [0] * (4 - len(merged))
            self.board[i] = merged
        return int(reward)

    def _move_right(self):
        reward = 0
        for i in range(4):
            line = self.board[i]
            line = [x for x in line if x != 0]
            merged = []
            j = len(line)-1
            while j >= 0:
                if j-1 >= 0 and line[j] == line[j-1]:
                    merged.append(line[j]*2)
                    self.score += line[j]*2
                    reward = max(reward, np.log2(line[j]*2))
                    j -= 2
                else:
                    merged.append(line[j])
                    j -= 1
            merged = merged[::-1]
            merged = [0] * (4 - len(merged)) + merged
            self.board[i] = merged
        return int(reward)

    def _move_up(self):
        reward = 0
        for j in range(4):
            col = [self.board[i][j] for i in range(4) if self.board[i][j]!= 0]
            merged = []
            i = 0
            while i < len(col):
                if i+1 < len(col) and col[i] == col[i+1]:
                    merged.append(col[i]*2)
                    self.score += col[i]*2
                    reward = max(reward, np.log2(col[i]*2))
                    i += 2
                else:
                    merged.append(col[i])
                    i += 1
            merged += [0] * (4 - len(merged))
            for i in range(4):
                self.board[i][j] = merged[i]
        return int(reward)

    def _move_down(self):
        reward = 0
        for j in range(4):
            col = [self.board[i][j] for i in range(4) if self.board[i][j] != 0]
            merged = []
            i = len(col)-1
            while i >= 0:
                if i-1 >= 0 and col[i] == col[i-1]:
                    merged.append(col[i]*2)
                    self.score += col[i]*2
                    reward = max(reward, np.log2(col[i]*2))
                    i -= 2
                else:
                    merged.append(col[i])
                    i -= 1
            merged = merged[::-1]
            merged = [0] * (4 - len(merged)) + merged
            for i in range(4):
                self.board[i][j] = merged[i]
        return int(reward)

        
    def _check_game_over(self):
        if any(2048 in row for row in self.board):
            self.win = True
            return True

        if any(0 in row for row in self.board):
            return False

        for i in range(4):
            for j in range(4):
                if j+1 < 4 and self.board[i][j] == self.board[i][j+1]:
                    return False
                if i+1 < 4 and self.board[i][j] == self.board[i+1][j]:
                    return False
        return True
    
    def init_window(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("2048")
        self.clock = pygame.time.Clock()
    
    def render(self):
        if self.screen is None:
            self.init_window()

        self.screen.fill(BG_COLOR)

        board_side = WINDOW_SIZE[0] - 2 * MARGIN
        board_rect = pygame.Rect(MARGIN, MARGIN, board_side, board_side)
        pygame.draw.rect(self.screen, EMPTY_COLOR, board_rect, border_radius=6)

        cell_total = board_side - (GRID_SIZE + 1) * GRID_PADDING
        cell_size  = cell_total // GRID_SIZE

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                val = self.board[r][c]
                x = MARGIN + GRID_PADDING + c * (cell_size + GRID_PADDING)
                y = MARGIN + GRID_PADDING + r * (cell_size + GRID_PADDING)

                color = TILE_COLORS.get(val, (60, 58, 50))
                pygame.draw.rect(self.screen, color,
                                 (x, y, cell_size, cell_size), border_radius=3)
                if val != 0:
                    if val < 100:
                        font_size = 48
                    elif val < 1000:
                        font_size = 40
                    else:
                        font_size = 32
                    txt = get_font(font_size).render(str(val), True, font_color(val))
                    txt_rect = txt.get_rect(center=(x + cell_size // 2,
                                                    y + cell_size // 2))
                    self.screen.blit(txt, txt_rect)

        score_surf = get_font(24).render(f"Score: {self.score}", True, FONT_COLOR)
        self.screen.blit(score_surf, (MARGIN, WINDOW_SIZE[1] - 40))

        if self.game_over and not self.win:
            overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 240))  
            self.screen.blit(overlay, (0, 0))
            msg = get_font(56).render("Game Over!", True, (119, 110, 101))
            rect = msg.get_rect(center=(WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2))
            self.screen.blit(msg, rect)
        elif self.win:
            overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 240))  
            self.screen.blit(overlay, (0, 0))
            msg = get_font(56).render("You Win!", True, (119, 110, 101))
            rect = msg.get_rect(center=(WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2))
            self.screen.blit(msg, rect)

        pygame.display.flip()

    def get_action_mask(self):

        mask = [0, 0, 0, 0]
        b = self.board

        for i in range(4):
            left_valid = 0
            right_valid = 0
            for j in range(1, 4):
                if b[i][j] != 0 and b[i][j-1] == 0:          
                    left_valid = 1
                if b[i][j-1] != 0 and b[i][j] == 0:         
                    right_valid = 1
                if b[i][j] == b[i][j-1] and b[i][j] != 0:
                    left_valid = right_valid = 1
            mask[2] |= left_valid
            mask[3] |= right_valid

            up_valid = 0
            down_valid = 0
            for j in range(1, 4):
                if b[j][i] != 0 and b[j-1][i] == 0:
                    up_valid = 1
                if b[j-1][i] != 0 and b[j][i] == 0:
                    down_valid = 1
                if b[j][i] == b[j-1][i] and b[j][i] != 0:
                    up_valid = down_valid = 1
            mask[0] |= up_valid
            mask[1] |= down_valid

        return [float(i) for i in mask]
    
    def _evaluate_board(self, old_board, move_score):

        merge_bonus = float(move_score) * 0.1 

        empty_cnt = sum(1 for r in range(4) for c in range(4) if self.board[r][c] == 0)
        empty_bonus = 0.2 * empty_cnt

        def monotonicity_smoothness(mat):
            mono = [0, 0, 0, 0]  # up,down,left,right
            smooth = 0
            for r in range(4):
                for c in range(4):
                    if r < 3 and mat[r][c] >= mat[r+1][c]:
                        mono[0] += mat[r][c] - mat[r+1][c]
                    if r > 0 and mat[r][c] >= mat[r-1][c]:
                        mono[1] += mat[r][c] - mat[r-1][c]
                    if c < 3 and mat[r][c] >= mat[r][c+1]:
                        mono[2] += mat[r][c] - mat[r][c+1]
                    if c > 0 and mat[r][c] >= mat[r][c-1]:
                        mono[3] += mat[r][c] - mat[r][c-1]
                    if r < 3:
                        smooth -= abs(mat[r][c] - mat[r+1][c])
                    if c < 3:
                        smooth -= abs(mat[r][c] - mat[r][c+1])
            return max(mono) * 1e-3 + smooth * 1e-4

        struct_bonus = monotonicity_smoothness(self.board)

        def get_max_tile_postion(mat):
            postion = np.argmax(mat)
            max_val = max(mat[i][j] for i in range(4) for j in range(4) if mat[i][j] != 0)
            if postion in [0, 3, 12, 15]:
                return 0.2 * np.log2(max_val)
            else:
                return 0.0
        position_reward = get_max_tile_postion(self.board)

        get_max_tile_postion(self.board)

        max_now = max(max(row) for row in self.board)
        max_old = max(max(row) for row in old_board)
        max_bonus = 0.0
        if max_now > max_old:
            max_bonus = 2.0 + 0.5 * np.log2(max_now) 

        return merge_bonus + empty_bonus + struct_bonus + max_bonus + position_reward
        
    def step(self, action):
        self.game_over = self._check_game_over()
        if self.game_over:
            return -10 + (60 if self.win else 0), self.game_over

        mask = self.get_action_mask()
        if mask[action] == 0:    
            return -10, self.game_over
    
        move_score = (self._move_up, self._move_down,
                    self._move_left, self._move_right)[action]()
        
        reward = self._evaluate_board(self.board, move_score)
        
        self._generate_new_tile()

        self.game_over = self._check_game_over()

        if self.game_over:
            return -10 + (60 if self.win else 0), self.game_over
        
        return reward, self.game_over

if __name__ == "__main__":
    game = Game2048()
    game.reset()
    # game.board = [[1024, 0, 0, 0], [1024, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    running = True
    while running:
        game.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:    action = 0
                elif event.key == pygame.K_DOWN:  action = 1
                elif event.key == pygame.K_LEFT:  action = 2
                elif event.key == pygame.K_RIGHT: action = 3
                else:
                    continue
                reward = game.step(action)
                print(f"Action: {action}, Reward: {reward}")
                print(game.game_over)
        game.clock.tick(60)
    pygame.quit()