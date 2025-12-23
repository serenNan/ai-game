# -*- coding: utf-8 -*-
"""
Pygame GUI for AlphaZero Gomoku
Human vs AI with graphical interface

@author: Claude Code
"""

import pygame
import pickle
import threading
from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy

# 颜色定义
BOARD_COLOR = (220, 179, 92)      # 棋盘木色
LINE_COLOR = (50, 50, 50)          # 线条颜色
BLACK_PIECE = (20, 20, 20)         # 黑棋
WHITE_PIECE = (240, 240, 240)      # 白棋
HIGHLIGHT_COLOR = (255, 80, 80)    # 最后落子高亮
BUTTON_COLOR = (100, 150, 100)     # 按钮颜色
BUTTON_HOVER = (120, 180, 120)     # 按钮悬停
TEXT_COLOR = (255, 255, 255)       # 文字颜色
BG_COLOR = (60, 60, 60)            # 背景色


class GomokuGUI:
    def __init__(self, width=8, height=8, n_in_row=5):
        self.board_width = width
        self.board_height = height
        self.n_in_row = n_in_row

        # GUI 参数
        self.cell_size = 60
        self.margin = 40
        self.board_pixel_size = self.cell_size * (max(width, height) - 1) + self.margin * 2
        self.info_height = 80
        self.window_width = self.board_pixel_size
        self.window_height = self.board_pixel_size + self.info_height

        # 初始化 Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(f'AlphaZero Gomoku {width}x{height}')
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 28)

        # 游戏状态
        self.board = Board(width=width, height=height, n_in_row=n_in_row)
        self.board.init_board(start_player=0)  # 人类先手 (player 1)
        self.human_player = 1  # 人类是 player 1 (黑棋)
        self.ai_player_id = 2  # AI 是 player 2 (白棋)
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.ai_thinking = False
        self.message = "Your turn (Black)"
        self.move_history = []

        # 加载 AI
        self._load_ai()

    def _load_ai(self):
        """加载 AlphaZero AI"""
        model_file = f'best_policy_{self.board_width}_{self.board_height}_{self.n_in_row}.model'
        try:
            try:
                policy_param = pickle.load(open(model_file, 'rb'))
            except:
                policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')

            best_policy = PolicyValueNetNumpy(self.board_width, self.board_height, policy_param)
            self.ai_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
            self.ai_player.set_player_ind(self.ai_player_id)  # AI 是 player 2 (白棋)
            print(f"Loaded AI model: {model_file}")
        except FileNotFoundError:
            print(f"Model file {model_file} not found, using pure MCTS")
            from mcts_pure import MCTSPlayer as MCTS_Pure
            self.ai_player = MCTS_Pure(c_puct=5, n_playout=1000)
            self.ai_player.set_player_ind(self.ai_player_id)

    def pos_to_pixel(self, x, y):
        """棋盘坐标转像素坐标"""
        px = self.margin + x * self.cell_size
        py = self.margin + (self.board_height - 1 - y) * self.cell_size
        return px, py

    def pixel_to_pos(self, px, py):
        """像素坐标转棋盘坐标"""
        x = round((px - self.margin) / self.cell_size)
        y = self.board_height - 1 - round((py - self.margin) / self.cell_size)
        if 0 <= x < self.board_width and 0 <= y < self.board_height:
            return x, y
        return None, None

    def draw_board(self):
        """绘制棋盘"""
        # 背景
        self.screen.fill(BG_COLOR)

        # 棋盘区域
        board_rect = pygame.Rect(0, 0, self.board_pixel_size, self.board_pixel_size)
        pygame.draw.rect(self.screen, BOARD_COLOR, board_rect)

        # 网格线
        for i in range(self.board_width):
            start_pos = self.pos_to_pixel(i, 0)
            end_pos = self.pos_to_pixel(i, self.board_height - 1)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos, end_pos, 2)

        for j in range(self.board_height):
            start_pos = self.pos_to_pixel(0, j)
            end_pos = self.pos_to_pixel(self.board_width - 1, j)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos, end_pos, 2)

        # 星位点 (如果棋盘足够大)
        if self.board_width >= 9 and self.board_height >= 9:
            star_points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
            for x, y in star_points:
                if x < self.board_width and y < self.board_height:
                    px, py = self.pos_to_pixel(x, y)
                    pygame.draw.circle(self.screen, LINE_COLOR, (px, py), 5)

    def draw_pieces(self):
        """绘制棋子"""
        for move, player in self.board.states.items():
            # move = h * width + w, 所以 h = move // width, w = move % width
            y = move // self.board_width  # h
            x = move % self.board_width   # w
            px, py = self.pos_to_pixel(x, y)

            # player 1 = 黑棋, player 2 = 白棋
            color = BLACK_PIECE if player == 1 else WHITE_PIECE
            pygame.draw.circle(self.screen, color, (px, py), self.cell_size // 2 - 4)

            # 棋子边框
            border_color = (100, 100, 100) if player == 1 else (150, 150, 150)
            pygame.draw.circle(self.screen, border_color, (px, py), self.cell_size // 2 - 4, 2)

        # 高亮最后一步
        if self.last_move is not None:
            y = self.last_move // self.board_width  # h
            x = self.last_move % self.board_width   # w
            px, py = self.pos_to_pixel(x, y)
            pygame.draw.circle(self.screen, HIGHLIGHT_COLOR, (px, py), 8)

    def draw_info(self):
        """绘制信息栏"""
        info_rect = pygame.Rect(0, self.board_pixel_size, self.window_width, self.info_height)
        pygame.draw.rect(self.screen, BG_COLOR, info_rect)

        # 状态信息
        text = self.font.render(self.message, True, TEXT_COLOR)
        text_rect = text.get_rect(center=(self.window_width // 2, self.board_pixel_size + 25))
        self.screen.blit(text, text_rect)

        # 按钮
        self.restart_btn = pygame.Rect(self.window_width // 2 - 120, self.board_pixel_size + 45, 100, 30)
        self.undo_btn = pygame.Rect(self.window_width // 2 + 20, self.board_pixel_size + 45, 100, 30)

        mouse_pos = pygame.mouse.get_pos()

        # 重新开始按钮
        btn_color = BUTTON_HOVER if self.restart_btn.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, btn_color, self.restart_btn, border_radius=5)
        restart_text = self.small_font.render("Restart", True, TEXT_COLOR)
        self.screen.blit(restart_text, restart_text.get_rect(center=self.restart_btn.center))

        # 悔棋按钮
        btn_color = BUTTON_HOVER if self.undo_btn.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, btn_color, self.undo_btn, border_radius=5)
        undo_text = self.small_font.render("Undo", True, TEXT_COLOR)
        self.screen.blit(undo_text, undo_text.get_rect(center=self.undo_btn.center))

    def restart_game(self):
        """重新开始游戏"""
        self.board.init_board(start_player=0)
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.message = "Your turn (Black)"
        self.move_history = []
        self.ai_player.reset_player()

    def undo_move(self):
        """悔棋 (撤销人类和AI各一步)"""
        if len(self.move_history) >= 2 and not self.ai_thinking:
            # 撤销两步 (AI + 人类)
            for _ in range(2):
                if self.move_history:
                    move = self.move_history.pop()
                    del self.board.states[move]
                    self.board.availables.add(move)

            self.board.current_player = self.human_player  # 人类回合
            self.last_move = self.move_history[-1] if self.move_history else None
            self.game_over = False
            self.winner = None
            self.message = "Your turn (Black)"
            self.ai_player.reset_player()

    def check_winner(self):
        """检查是否有赢家"""
        end, winner = self.board.game_end()
        if end:
            self.game_over = True
            self.winner = winner
            if winner == -1:
                self.message = "Draw!"
            elif winner == self.human_player:
                self.message = "You Win!"
            else:
                self.message = "AI Wins!"
            return True
        return False

    def human_move(self, x, y):
        """人类落子"""
        if self.game_over or self.ai_thinking:
            return

        # location_to_move 期望 [h, w] 即 [y, x]
        move = self.board.location_to_move([y, x])
        if move in self.board.availables:
            self.board.do_move(move)
            self.last_move = move
            self.move_history.append(move)

            if not self.check_winner():
                self.message = "AI thinking..."
                self.ai_thinking = True
                # 在后台线程运行 AI
                threading.Thread(target=self.ai_move, daemon=True).start()

    def ai_move(self):
        """AI 落子"""
        move = self.ai_player.get_action(self.board)
        self.board.do_move(move)
        self.last_move = move
        self.move_history.append(move)
        self.ai_thinking = False

        if not self.check_winner():
            self.message = "Your turn (Black)"

    def run(self):
        """主循环"""
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos

                    # 检查按钮点击
                    if self.restart_btn.collidepoint(mx, my):
                        self.restart_game()
                    elif self.undo_btn.collidepoint(mx, my):
                        self.undo_move()
                    # 检查棋盘点击
                    elif my < self.board_pixel_size:
                        x, y = self.pixel_to_pos(mx, my)
                        if x is not None and self.board.current_player == self.human_player:
                            self.human_move(x, y)

            # 绘制
            self.draw_board()
            self.draw_pieces()
            self.draw_info()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def main():
    # 可以修改棋盘大小: 6x6(4连), 8x8(5连), 等
    gui = GomokuGUI(width=8, height=8, n_in_row=5)
    gui.run()


if __name__ == '__main__':
    main()
