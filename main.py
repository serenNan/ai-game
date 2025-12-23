"""
AlphaZero 五子棋 Pygame 图形界面
支持人机对战和机器对战模式，可选择先手或后手
"""

import pygame
import pickle
import threading
import time
import os
from board import GameState
from neural_search import TreeSearchAgent
from model_inference import NumpyNetworkEvaluator

# Color palette
BOARD_BG = (220, 179, 92)
GRID_LINE = (50, 50, 50)
BLACK_STONE = (20, 20, 20)
WHITE_STONE = (240, 240, 240)
HIGHLIGHT = (255, 80, 80)
BTN_NORMAL = (100, 150, 100)
BTN_HOVER = (120, 180, 120)
BTN_DISABLED = (80, 80, 80)
TEXT_WHITE = (255, 255, 255)
TEXT_DARK = (40, 40, 40)
PANEL_BG = (60, 60, 60)
MENU_BG = (45, 45, 55)
MENU_ACCENT = (70, 130, 180)


def loadChineseFont(size):
    """加载支持中文的字体"""
    # 尝试常见的中文字体路径
    fontPaths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
        "/System/Library/Fonts/PingFang.ttc",  # macOS
    ]
    for fontPath in fontPaths:
        if os.path.exists(fontPath):
            try:
                return pygame.font.Font(fontPath, size)
            except:
                continue
    # 回退到默认字体
    return pygame.font.Font(None, size)


class GomokuInterface:
    def __init__(self, boardCols=8, boardRows=8, winLength=5):
        self.boardCols = boardCols
        self.boardRows = boardRows
        self.winLength = winLength

        # Display parameters
        self.cellSize = 60
        self.margin = 40
        self.boardPixelSize = self.cellSize * (max(boardCols, boardRows) - 1) + self.margin * 2
        self.infoHeight = 80
        self.windowWidth = self.boardPixelSize
        self.windowHeight = self.boardPixelSize + self.infoHeight

        # Initialize Pygame
        pygame.init()
        self.display = pygame.display.set_mode((self.windowWidth, self.windowHeight))
        pygame.display.set_caption(f'AI 五子棋 {boardCols}x{boardRows}')
        self.titleFont = loadChineseFont(42)
        self.normalFont = loadChineseFont(28)
        self.smallFont = loadChineseFont(22)

        # Game state
        self.gameState = GameState(width=boardCols, height=boardRows, n_in_row=winLength)
        self.isGameOver = False
        self.victor = None
        self.lastMoveIdx = None
        self.isAiThinking = False
        self.statusMessage = ""
        self.moveLog = []

        # Game mode settings
        self.gameMode = None  # 'human_vs_ai' or 'ai_vs_ai'
        self.humanGoesFirst = True
        self.humanPlayerId = 1
        self.aiPlayerId = 2
        self.showingMenu = True
        self.aiDelay = 0.5  # Delay for AI vs AI mode

        # AI agents
        self.aiAgent1 = None
        self.aiAgent2 = None
        self._initializeAI()

    def _initializeAI(self):
        """Load the AlphaZero AI model"""
        modelFile = f'{self.boardCols}_{self.boardRows}_{self.winLength}.model'
        try:
            try:
                modelParams = pickle.load(open(modelFile, 'rb'))
            except:
                modelParams = pickle.load(open(modelFile, 'rb'), encoding='bytes')

            network = NumpyNetworkEvaluator(self.boardCols, self.boardRows, modelParams)
            self.aiAgent1 = TreeSearchAgent(network.evaluatePosition,
                                            explorationWeight=5, numSimulations=400)
            # Create second AI for AI vs AI mode
            self.aiAgent2 = TreeSearchAgent(network.evaluatePosition,
                                            explorationWeight=5, numSimulations=400)
            print(f"AI model loaded: {modelFile}")
        except FileNotFoundError:
            print(f"Model file {modelFile} not found, using pure MCTS")
            from random_search import PureSearchAgent
            self.aiAgent1 = PureSearchAgent(explorationWeight=5, numSimulations=1000)
            self.aiAgent2 = PureSearchAgent(explorationWeight=5, numSimulations=1000)

    def boardPosToPixel(self, x, y):
        """Convert board coordinates to pixel coordinates"""
        px = self.margin + x * self.cellSize
        py = self.margin + (self.boardRows - 1 - y) * self.cellSize
        return px, py

    def pixelToBoardPos(self, px, py):
        """Convert pixel coordinates to board coordinates"""
        x = round((px - self.margin) / self.cellSize)
        y = self.boardRows - 1 - round((py - self.margin) / self.cellSize)
        if 0 <= x < self.boardCols and 0 <= y < self.boardRows:
            return x, y
        return None, None

    def renderMenuScreen(self):
        """Render the game mode selection menu"""
        self.display.fill(MENU_BG)

        # Title
        title = self.titleFont.render("AI 五子棋", True, TEXT_WHITE)
        titleRect = title.get_rect(center=(self.windowWidth // 2, 60))
        self.display.blit(title, titleRect)

        # Subtitle
        subtitle = self.smallFont.render(f"{self.boardCols}x{self.boardRows} 棋盘，{self.winLength}子连珠获胜",
                                         True, (180, 180, 180))
        subtitleRect = subtitle.get_rect(center=(self.windowWidth // 2, 100))
        self.display.blit(subtitle, subtitleRect)

        mousePos = pygame.mouse.get_pos()

        # Game Mode Selection
        modeLabel = self.normalFont.render("游戏模式：", True, TEXT_WHITE)
        self.display.blit(modeLabel, (self.windowWidth // 2 - 150, 150))

        # Human vs AI - First (Black) button
        self.btnHumanFirst = pygame.Rect(self.windowWidth // 2 - 120, 190, 240, 45)
        btnColor = MENU_ACCENT if self.btnHumanFirst.collidepoint(mousePos) else BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnHumanFirst, border_radius=8)
        btnText = self.normalFont.render("人机对战（先手）", True, TEXT_WHITE)
        self.display.blit(btnText, btnText.get_rect(center=self.btnHumanFirst.center))

        # Human vs AI - Second (White) button
        self.btnHumanSecond = pygame.Rect(self.windowWidth // 2 - 120, 250, 240, 45)
        btnColor = MENU_ACCENT if self.btnHumanSecond.collidepoint(mousePos) else BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnHumanSecond, border_radius=8)
        btnText = self.normalFont.render("人机对战（后手）", True, TEXT_WHITE)
        self.display.blit(btnText, btnText.get_rect(center=self.btnHumanSecond.center))


    def renderBoard(self):
        """Render the game board"""
        self.display.fill(PANEL_BG)

        # Board background
        boardRect = pygame.Rect(0, 0, self.boardPixelSize, self.boardPixelSize)
        pygame.draw.rect(self.display, BOARD_BG, boardRect)

        # Grid lines
        for i in range(self.boardCols):
            startPos = self.boardPosToPixel(i, 0)
            endPos = self.boardPosToPixel(i, self.boardRows - 1)
            pygame.draw.line(self.display, GRID_LINE, startPos, endPos, 2)

        for j in range(self.boardRows):
            startPos = self.boardPosToPixel(0, j)
            endPos = self.boardPosToPixel(self.boardCols - 1, j)
            pygame.draw.line(self.display, GRID_LINE, startPos, endPos, 2)

        # Star points for larger boards
        if self.boardCols >= 9 and self.boardRows >= 9:
            starPoints = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
            for x, y in starPoints:
                if x < self.boardCols and y < self.boardRows:
                    px, py = self.boardPosToPixel(x, y)
                    pygame.draw.circle(self.display, GRID_LINE, (px, py), 5)

    def renderStones(self):
        """Render the stones on the board"""
        for moveIdx, playerId in self.gameState.positions.items():
            row = moveIdx // self.boardCols
            col = moveIdx % self.boardCols
            px, py = self.boardPosToPixel(col, row)

            color = BLACK_STONE if playerId == 1 else WHITE_STONE
            pygame.draw.circle(self.display, color, (px, py), self.cellSize // 2 - 4)

            borderColor = (100, 100, 100) if playerId == 1 else (150, 150, 150)
            pygame.draw.circle(self.display, borderColor, (px, py), self.cellSize // 2 - 4, 2)

        # Highlight last move
        if self.lastMoveIdx is not None:
            row = self.lastMoveIdx // self.boardCols
            col = self.lastMoveIdx % self.boardCols
            px, py = self.boardPosToPixel(col, row)
            pygame.draw.circle(self.display, HIGHLIGHT, (px, py), 8)

    def renderInfoPanel(self):
        """Render the information panel"""
        infoRect = pygame.Rect(0, self.boardPixelSize, self.windowWidth, self.infoHeight)
        pygame.draw.rect(self.display, PANEL_BG, infoRect)

        # Status message
        statusText = self.normalFont.render(self.statusMessage, True, TEXT_WHITE)
        statusRect = statusText.get_rect(center=(self.windowWidth // 2, self.boardPixelSize + 25))
        self.display.blit(statusText, statusRect)

        # Buttons
        self.btnRestart = pygame.Rect(self.windowWidth // 2 - 120, self.boardPixelSize + 45, 100, 30)
        self.btnMenu = pygame.Rect(self.windowWidth // 2 + 20, self.boardPixelSize + 45, 100, 30)

        mousePos = pygame.mouse.get_pos()

        # Restart button
        btnColor = BTN_HOVER if self.btnRestart.collidepoint(mousePos) else BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnRestart, border_radius=5)
        restartText = self.smallFont.render("重新开始", True, TEXT_WHITE)
        self.display.blit(restartText, restartText.get_rect(center=self.btnRestart.center))

        # Menu button
        btnColor = BTN_HOVER if self.btnMenu.collidepoint(mousePos) else BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnMenu, border_radius=5)
        menuText = self.smallFont.render("返回菜单", True, TEXT_WHITE)
        self.display.blit(menuText, menuText.get_rect(center=self.btnMenu.center))

    def resetGame(self):
        """Reset the game to initial state"""
        # 黑方(玩家1)总是先手
        # humanGoesFirst=True: 人类执黑(1)，AI执白(2)
        # humanGoesFirst=False: AI执黑(1)，人类执白(2)
        if self.gameMode == 'human_vs_ai':
            self.humanPlayerId = 1 if self.humanGoesFirst else 2
            self.aiPlayerId = 2 if self.humanGoesFirst else 1

        # 黑方总是先手
        self.gameState.initState(0)
        self.isGameOver = False
        self.victor = None
        self.lastMoveIdx = None
        self.moveLog = []
        self.aiAgent1.resetState()
        self.aiAgent2.resetState()

        if self.gameMode == 'human_vs_ai':
            if self.humanGoesFirst:
                self.statusMessage = "轮到你了（执黑）"
            else:
                self.statusMessage = "AI 思考中..."
                self.isAiThinking = True
                threading.Thread(target=self._executeAiMove, daemon=True).start()
        else:
            self.statusMessage = "AI 对战 - 黑方回合"
            self.isAiThinking = True
            threading.Thread(target=self._runAiVsAi, daemon=True).start()

    def returnToMenu(self):
        """Return to the main menu"""
        self.showingMenu = True
        self.isGameOver = False
        self.isAiThinking = False

    def _checkGameEnd(self):
        """Check if the game has ended"""
        finished, winner = self.gameState.isTerminal()
        if finished:
            self.isGameOver = True
            self.victor = winner
            if winner == -1:
                self.statusMessage = "平局！"
            elif self.gameMode == 'human_vs_ai':
                if winner == self.humanPlayerId:
                    self.statusMessage = "恭喜你赢了！"
                else:
                    self.statusMessage = "AI 获胜！"
            else:
                self.statusMessage = f"{'黑方' if winner == 1 else '白方'}获胜！"
            return True
        return False

    def _handleHumanMove(self, x, y):
        """Process human player's move"""
        if self.isGameOver or self.isAiThinking:
            return

        moveIdx = self.gameState.coordToIndex([y, x])
        if moveIdx in self.gameState.openPositions:
            self.gameState.applyMove(moveIdx)
            self.lastMoveIdx = moveIdx
            self.moveLog.append(moveIdx)

            if not self._checkGameEnd():
                self.statusMessage = "AI 思考中..."
                self.isAiThinking = True
                threading.Thread(target=self._executeAiMove, daemon=True).start()

    def _executeAiMove(self):
        """Execute AI's move"""
        agent = self.aiAgent1 if self.gameState.activePlayer == self.aiPlayerId else self.aiAgent2
        agent.assignPlayerId(self.gameState.activePlayer)
        moveIdx = agent.selectMove(self.gameState)
        self.gameState.applyMove(moveIdx)
        self.lastMoveIdx = moveIdx
        self.moveLog.append(moveIdx)
        self.isAiThinking = False

        if not self._checkGameEnd():
            pieceColor = "黑" if self.gameState.activePlayer == 1 else "白"
            self.statusMessage = f"轮到你了（执{pieceColor}）"

    def _runAiVsAi(self):
        """Run AI vs AI game"""
        self.aiAgent1.assignPlayerId(1)
        self.aiAgent2.assignPlayerId(2)

        while not self.isGameOver and self.gameMode == 'ai_vs_ai':
            currentPlayer = self.gameState.activePlayer
            agent = self.aiAgent1 if currentPlayer == 1 else self.aiAgent2

            pieceColor = "黑方" if currentPlayer == 1 else "白方"
            self.statusMessage = f"AI（{pieceColor}）思考中..."

            moveIdx = agent.selectMove(self.gameState)
            self.gameState.applyMove(moveIdx)
            self.lastMoveIdx = moveIdx
            self.moveLog.append(moveIdx)

            if self._checkGameEnd():
                break

            time.sleep(self.aiDelay)

        self.isAiThinking = False

    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        isRunning = True

        while isRunning:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    isRunning = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouseX, mouseY = event.pos

                    if self.showingMenu:
                        # Menu button handling - click to start directly
                        if self.btnHumanFirst.collidepoint(mouseX, mouseY):
                            self.gameMode = 'human_vs_ai'
                            self.humanGoesFirst = True
                            self.showingMenu = False
                            self.resetGame()
                        elif self.btnHumanSecond.collidepoint(mouseX, mouseY):
                            self.gameMode = 'human_vs_ai'
                            self.humanGoesFirst = False
                            self.showingMenu = False
                            self.resetGame()
                    else:
                        # Game button handling
                        if self.btnRestart.collidepoint(mouseX, mouseY):
                            self.resetGame()
                        elif self.btnMenu.collidepoint(mouseX, mouseY):
                            self.returnToMenu()
                        elif mouseY < self.boardPixelSize:
                            # Board click
                            if self.gameMode == 'human_vs_ai':
                                x, y = self.pixelToBoardPos(mouseX, mouseY)
                                if x is not None and self.gameState.activePlayer == self.humanPlayerId:
                                    self._handleHumanMove(x, y)

            # Render
            if self.showingMenu:
                self.renderMenuScreen()
            else:
                self.renderBoard()
                self.renderStones()
                self.renderInfoPanel()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def main():
    gui = GomokuInterface(boardCols=8, boardRows=8, winLength=5)
    gui.run()


if __name__ == '__main__':
    main()
