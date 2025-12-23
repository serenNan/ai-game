# -*- coding: utf-8 -*-
"""
Pygame GUI for AlphaZero Gomoku
Supports Human vs AI and AI vs AI modes with first/second player selection

@author: Kevin Chen
"""

import pygame
import pickle
import threading
import time
from game import GameState
from mcts_alphaZero import TreeSearchAgent
from policy_value_net_numpy import NumpyNetworkEvaluator

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
        pygame.display.set_caption(f'AlphaZero Gomoku {boardCols}x{boardRows}')
        self.titleFont = pygame.font.Font(None, 48)
        self.normalFont = pygame.font.Font(None, 36)
        self.smallFont = pygame.font.Font(None, 28)

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
        modelFile = f'best_policy_{self.boardCols}_{self.boardRows}_{self.winLength}.model'
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
            from mcts_pure import PureSearchAgent
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
        title = self.titleFont.render("AlphaZero Gomoku", True, TEXT_WHITE)
        titleRect = title.get_rect(center=(self.windowWidth // 2, 60))
        self.display.blit(title, titleRect)

        # Subtitle
        subtitle = self.smallFont.render(f"{self.boardCols}x{self.boardRows} Board, {self.winLength} in a row to win",
                                         True, (180, 180, 180))
        subtitleRect = subtitle.get_rect(center=(self.windowWidth // 2, 100))
        self.display.blit(subtitle, subtitleRect)

        mousePos = pygame.mouse.get_pos()

        # Game Mode Selection
        modeLabel = self.normalFont.render("Game Mode:", True, TEXT_WHITE)
        self.display.blit(modeLabel, (self.windowWidth // 2 - 150, 150))

        # Human vs AI button
        self.btnHumanVsAi = pygame.Rect(self.windowWidth // 2 - 120, 190, 240, 45)
        btnColor = MENU_ACCENT if self.btnHumanVsAi.collidepoint(mousePos) else BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnHumanVsAi, border_radius=8)
        btnText = self.normalFont.render("Human vs AI", True, TEXT_WHITE)
        self.display.blit(btnText, btnText.get_rect(center=self.btnHumanVsAi.center))

        # AI vs AI button
        self.btnAiVsAi = pygame.Rect(self.windowWidth // 2 - 120, 250, 240, 45)
        btnColor = MENU_ACCENT if self.btnAiVsAi.collidepoint(mousePos) else BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnAiVsAi, border_radius=8)
        btnText = self.normalFont.render("AI vs AI", True, TEXT_WHITE)
        self.display.blit(btnText, btnText.get_rect(center=self.btnAiVsAi.center))

        # First/Second player selection (only for Human vs AI)
        orderLabel = self.normalFont.render("Play as:", True, TEXT_WHITE)
        self.display.blit(orderLabel, (self.windowWidth // 2 - 150, 320))

        # First player (Black) button
        self.btnPlayFirst = pygame.Rect(self.windowWidth // 2 - 120, 360, 115, 45)
        isSelected = self.humanGoesFirst
        btnColor = MENU_ACCENT if isSelected else (BTN_HOVER if self.btnPlayFirst.collidepoint(mousePos) else BTN_NORMAL)
        pygame.draw.rect(self.display, btnColor, self.btnPlayFirst, border_radius=8)
        btnText = self.smallFont.render("First (Black)", True, TEXT_WHITE)
        self.display.blit(btnText, btnText.get_rect(center=self.btnPlayFirst.center))

        # Second player (White) button
        self.btnPlaySecond = pygame.Rect(self.windowWidth // 2 + 5, 360, 115, 45)
        isSelected = not self.humanGoesFirst
        btnColor = MENU_ACCENT if isSelected else (BTN_HOVER if self.btnPlaySecond.collidepoint(mousePos) else BTN_NORMAL)
        pygame.draw.rect(self.display, btnColor, self.btnPlaySecond, border_radius=8)
        btnText = self.smallFont.render("Second (White)", True, TEXT_WHITE)
        self.display.blit(btnText, btnText.get_rect(center=self.btnPlaySecond.center))

        # Start button
        self.btnStart = pygame.Rect(self.windowWidth // 2 - 80, 440, 160, 50)
        btnColor = MENU_ACCENT if self.btnStart.collidepoint(mousePos) else BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnStart, border_radius=10)
        btnText = self.normalFont.render("Start Game", True, TEXT_WHITE)
        self.display.blit(btnText, btnText.get_rect(center=self.btnStart.center))

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
        restartText = self.smallFont.render("Restart", True, TEXT_WHITE)
        self.display.blit(restartText, restartText.get_rect(center=self.btnRestart.center))

        # Menu button
        btnColor = BTN_HOVER if self.btnMenu.collidepoint(mousePos) else BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnMenu, border_radius=5)
        menuText = self.smallFont.render("Menu", True, TEXT_WHITE)
        self.display.blit(menuText, menuText.get_rect(center=self.btnMenu.center))

    def resetGame(self):
        """Reset the game to initial state"""
        if self.gameMode == 'human_vs_ai':
            startPlayer = 0 if self.humanGoesFirst else 1
            self.humanPlayerId = 1 if self.humanGoesFirst else 2
            self.aiPlayerId = 2 if self.humanGoesFirst else 1
        else:
            startPlayer = 0

        self.gameState.initState(startPlayer)
        self.isGameOver = False
        self.victor = None
        self.lastMoveIdx = None
        self.moveLog = []
        self.aiAgent1.resetState()
        self.aiAgent2.resetState()

        if self.gameMode == 'human_vs_ai':
            if self.humanGoesFirst:
                self.statusMessage = "Your turn (Black)"
            else:
                self.statusMessage = "AI thinking..."
                self.isAiThinking = True
                threading.Thread(target=self._executeAiMove, daemon=True).start()
        else:
            self.statusMessage = "AI vs AI - Black's turn"
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
                self.statusMessage = "Draw!"
            elif self.gameMode == 'human_vs_ai':
                if winner == self.humanPlayerId:
                    self.statusMessage = "You Win!"
                else:
                    self.statusMessage = "AI Wins!"
            else:
                self.statusMessage = f"Player {winner} ({'Black' if winner == 1 else 'White'}) Wins!"
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
                self.statusMessage = "AI thinking..."
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
            pieceColor = "Black" if self.gameState.activePlayer == 1 else "White"
            self.statusMessage = f"Your turn ({pieceColor})"

    def _runAiVsAi(self):
        """Run AI vs AI game"""
        self.aiAgent1.assignPlayerId(1)
        self.aiAgent2.assignPlayerId(2)

        while not self.isGameOver and self.gameMode == 'ai_vs_ai':
            currentPlayer = self.gameState.activePlayer
            agent = self.aiAgent1 if currentPlayer == 1 else self.aiAgent2

            pieceColor = "Black" if currentPlayer == 1 else "White"
            self.statusMessage = f"AI ({pieceColor}) thinking..."

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
                        # Menu button handling
                        if self.btnHumanVsAi.collidepoint(mouseX, mouseY):
                            self.gameMode = 'human_vs_ai'
                        elif self.btnAiVsAi.collidepoint(mouseX, mouseY):
                            self.gameMode = 'ai_vs_ai'
                        elif self.btnPlayFirst.collidepoint(mouseX, mouseY):
                            self.humanGoesFirst = True
                        elif self.btnPlaySecond.collidepoint(mouseX, mouseY):
                            self.humanGoesFirst = False
                        elif self.btnStart.collidepoint(mouseX, mouseY) and self.gameMode:
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
