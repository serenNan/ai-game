import pygame
import pickle
import threading
import time
import os
from board import GameState
from neural_search import TreeSearchAgent
from model_inference import NumpyNetworkEvaluator

# 颜色配置
BOARD_BG = (220, 179, 92)
GRID_LINE = (50, 50, 50)
BLACK_STONE = (20, 20, 20)
WHITE_STONE = (240, 240, 240)
HIGHLIGHT = (255, 80, 80)
BTN_NORMAL = (100, 150, 100)
BTN_HOVER = (120, 180, 120)
BTN_SELECTED = (70, 130, 180)
TEXT_WHITE = (255, 255, 255)
TEXT_DARK = (40, 40, 40)
PANEL_BG = (60, 60, 60)
MENU_BG = (45, 45, 55)
MENU_ACCENT = (70, 130, 180)


def loadChineseFont(size):
    """加载支持中文的字体"""
    fontPaths = [
        # Linux - Noto Sans CJK (常见)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        # Linux - 文泉驿
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        # Linux - 文鼎
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        # 用户字体目录
        os.path.expanduser("~/.fonts/LXGWWenKai-Regular.ttf"),
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for fontPath in fontPaths:
        if os.path.exists(fontPath):
            try:
                return pygame.font.Font(fontPath, size)
            except:
                continue
    return pygame.font.Font(None, size)


class GomokuInterface:
    # 棋盘配置选项
    BOARD_CONFIGS = [
        {"cols": 6, "rows": 6, "win": 4, "label": "6x6 (4子连珠)"},
        {"cols": 8, "rows": 8, "win": 5, "label": "8x8 (5子连珠)"},
    ]

    def __init__(self):
        # 默认选择 8x8
        self.selectedBoardConfig = 1
        self.boardCols = 8
        self.boardRows = 8
        self.winLength = 5

        # 先后手选择
        self.humanGoesFirst = True

        # 显示参数
        self.cellSize = 60
        self.margin = 40
        self._updateWindowSize()

        # 初始化 Pygame
        pygame.init()
        self.display = pygame.display.set_mode((self.windowWidth, self.windowHeight))
        pygame.display.set_caption('AI 五子棋')
        self.titleFont = loadChineseFont(42)
        self.normalFont = loadChineseFont(28)
        self.smallFont = loadChineseFont(22)

        # 游戏状态
        self.gameState = None
        self.isGameOver = False
        self.victor = None
        self.lastMoveIdx = None
        self.isAiThinking = False
        self.statusMessage = ""
        self.moveLog = []

        # 游戏模式
        self.gameMode = None
        self.humanPlayerId = 1
        self.aiPlayerId = 2
        self.showingMenu = True

        # AI 代理
        self.aiAgent = None

    def _updateWindowSize(self):
        """根据棋盘大小更新窗口尺寸"""
        self.boardPixelSize = self.cellSize * (max(self.boardCols, self.boardRows) - 1) + self.margin * 2
        self.infoHeight = 80
        self.windowWidth = max(500, self.boardPixelSize)
        self.windowHeight = max(520, self.boardPixelSize + self.infoHeight)

    def _initializeGame(self):
        """初始化游戏状态和AI"""
        # 更新棋盘配置
        config = self.BOARD_CONFIGS[self.selectedBoardConfig]
        self.boardCols = config["cols"]
        self.boardRows = config["rows"]
        self.winLength = config["win"]

        # 更新窗口大小
        self._updateWindowSize()
        self.display = pygame.display.set_mode((self.windowWidth, self.windowHeight))
        pygame.display.set_caption(f'AI 五子棋 {self.boardCols}x{self.boardRows}')

        # 创建游戏状态
        self.gameState = GameState(width=self.boardCols, height=self.boardRows, n_in_row=self.winLength)

        # 加载 AI
        self._loadAI()

    def _loadAI(self):
        """加载对应棋盘的 AI 模型"""
        modelFile = f'{self.boardCols}_{self.boardRows}_{self.winLength}.model'
        try:
            try:
                modelParams = pickle.load(open(modelFile, 'rb'))
            except:
                modelParams = pickle.load(open(modelFile, 'rb'), encoding='bytes')

            network = NumpyNetworkEvaluator(self.boardCols, self.boardRows, modelParams)
            self.aiAgent = TreeSearchAgent(network.evaluatePosition,
                                           explorationWeight=5, numSimulations=400)
            print(f"AI 模型已加载: {modelFile}")
        except FileNotFoundError:
            print(f"模型文件 {modelFile} 未找到，使用纯 MCTS")
            from random_search import PureSearchAgent
            self.aiAgent = PureSearchAgent(explorationWeight=5, numSimulations=1000)

    def boardPosToPixel(self, x, y):
        """棋盘坐标转像素坐标"""
        px = self.margin + x * self.cellSize
        py = self.margin + (self.boardRows - 1 - y) * self.cellSize
        return px, py

    def pixelToBoardPos(self, px, py):
        """像素坐标转棋盘坐标"""
        x = round((px - self.margin) / self.cellSize)
        y = self.boardRows - 1 - round((py - self.margin) / self.cellSize)
        if 0 <= x < self.boardCols and 0 <= y < self.boardRows:
            return x, y
        return None, None

    def renderMenuScreen(self):
        """渲染菜单界面"""
        self.display.fill(MENU_BG)
        mousePos = pygame.mouse.get_pos()

        # 标题
        title = self.titleFont.render("AI 五子棋", True, TEXT_WHITE)
        titleRect = title.get_rect(center=(self.windowWidth // 2, 50))
        self.display.blit(title, titleRect)

        # 棋盘选择
        boardLabel = self.normalFont.render("选择棋盘：", True, TEXT_WHITE)
        self.display.blit(boardLabel, (self.windowWidth // 2 - 180, 110))

        self.btnBoards = []
        startX = self.windowWidth // 2 - 160
        for i, config in enumerate(self.BOARD_CONFIGS):
            btnRect = pygame.Rect(startX + i * 170, 150, 150, 45)
            self.btnBoards.append(btnRect)

            if i == self.selectedBoardConfig:
                btnColor = BTN_SELECTED
            elif btnRect.collidepoint(mousePos):
                btnColor = BTN_HOVER
            else:
                btnColor = BTN_NORMAL

            pygame.draw.rect(self.display, btnColor, btnRect, border_radius=8)
            btnText = self.smallFont.render(config["label"], True, TEXT_WHITE)
            self.display.blit(btnText, btnText.get_rect(center=btnRect.center))

        # 先后手选择
        orderLabel = self.normalFont.render("选择先后手：", True, TEXT_WHITE)
        self.display.blit(orderLabel, (self.windowWidth // 2 - 180, 220))

        self.btnFirst = pygame.Rect(self.windowWidth // 2 - 160, 260, 150, 45)
        self.btnSecond = pygame.Rect(self.windowWidth // 2 + 10, 260, 150, 45)

        # 先手按钮
        if self.humanGoesFirst:
            btnColor = BTN_SELECTED
        elif self.btnFirst.collidepoint(mousePos):
            btnColor = BTN_HOVER
        else:
            btnColor = BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnFirst, border_radius=8)
        btnText = self.normalFont.render("先手（黑）", True, TEXT_WHITE)
        self.display.blit(btnText, btnText.get_rect(center=self.btnFirst.center))

        # 后手按钮
        if not self.humanGoesFirst:
            btnColor = BTN_SELECTED
        elif self.btnSecond.collidepoint(mousePos):
            btnColor = BTN_HOVER
        else:
            btnColor = BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnSecond, border_radius=8)
        btnText = self.normalFont.render("后手（白）", True, TEXT_WHITE)
        self.display.blit(btnText, btnText.get_rect(center=self.btnSecond.center))

        # 开始游戏按钮
        self.btnStart = pygame.Rect(self.windowWidth // 2 - 100, 340, 200, 55)
        btnColor = MENU_ACCENT if self.btnStart.collidepoint(mousePos) else BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnStart, border_radius=10)
        btnText = self.normalFont.render("开始游戏", True, TEXT_WHITE)
        self.display.blit(btnText, btnText.get_rect(center=self.btnStart.center))

        # 当前配置提示
        config = self.BOARD_CONFIGS[self.selectedBoardConfig]
        order = "先手（执黑）" if self.humanGoesFirst else "后手（执白）"
        hint = self.smallFont.render(f"配置: {config['label']}，{order}", True, (150, 150, 150))
        hintRect = hint.get_rect(center=(self.windowWidth // 2, 420))
        self.display.blit(hint, hintRect)

    def renderBoard(self):
        """渲染棋盘"""
        self.display.fill(PANEL_BG)

        # 棋盘背景
        boardRect = pygame.Rect(0, 0, self.boardPixelSize, self.boardPixelSize)
        pygame.draw.rect(self.display, BOARD_BG, boardRect)

        # 网格线
        for i in range(self.boardCols):
            startPos = self.boardPosToPixel(i, 0)
            endPos = self.boardPosToPixel(i, self.boardRows - 1)
            pygame.draw.line(self.display, GRID_LINE, startPos, endPos, 2)

        for j in range(self.boardRows):
            startPos = self.boardPosToPixel(0, j)
            endPos = self.boardPosToPixel(self.boardCols - 1, j)
            pygame.draw.line(self.display, GRID_LINE, startPos, endPos, 2)

    def renderStones(self):
        """渲染棋子"""
        for moveIdx, playerId in self.gameState.positions.items():
            row = moveIdx // self.boardCols
            col = moveIdx % self.boardCols
            px, py = self.boardPosToPixel(col, row)

            color = BLACK_STONE if playerId == 1 else WHITE_STONE
            pygame.draw.circle(self.display, color, (px, py), self.cellSize // 2 - 4)

            borderColor = (100, 100, 100) if playerId == 1 else (150, 150, 150)
            pygame.draw.circle(self.display, borderColor, (px, py), self.cellSize // 2 - 4, 2)

        # 高亮最后一步
        if self.lastMoveIdx is not None:
            row = self.lastMoveIdx // self.boardCols
            col = self.lastMoveIdx % self.boardCols
            px, py = self.boardPosToPixel(col, row)
            pygame.draw.circle(self.display, HIGHLIGHT, (px, py), 8)

    def renderInfoPanel(self):
        """渲染信息面板"""
        infoRect = pygame.Rect(0, self.boardPixelSize, self.windowWidth, self.infoHeight)
        pygame.draw.rect(self.display, PANEL_BG, infoRect)

        # 状态信息
        statusText = self.normalFont.render(self.statusMessage, True, TEXT_WHITE)
        statusRect = statusText.get_rect(center=(self.windowWidth // 2, self.boardPixelSize + 25))
        self.display.blit(statusText, statusRect)

        # 按钮
        self.btnRestart = pygame.Rect(self.windowWidth // 2 - 120, self.boardPixelSize + 45, 100, 30)
        self.btnMenu = pygame.Rect(self.windowWidth // 2 + 20, self.boardPixelSize + 45, 100, 30)

        mousePos = pygame.mouse.get_pos()

        # 重新开始按钮
        btnColor = BTN_HOVER if self.btnRestart.collidepoint(mousePos) else BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnRestart, border_radius=5)
        restartText = self.smallFont.render("重新开始", True, TEXT_WHITE)
        self.display.blit(restartText, restartText.get_rect(center=self.btnRestart.center))

        # 返回菜单按钮
        btnColor = BTN_HOVER if self.btnMenu.collidepoint(mousePos) else BTN_NORMAL
        pygame.draw.rect(self.display, btnColor, self.btnMenu, border_radius=5)
        menuText = self.smallFont.render("返回菜单", True, TEXT_WHITE)
        self.display.blit(menuText, menuText.get_rect(center=self.btnMenu.center))

    def startGame(self):
        """开始游戏"""
        self._initializeGame()
        self.showingMenu = False
        self.gameMode = 'human_vs_ai'
        self.resetGame()

    def resetGame(self):
        """重置游戏状态"""
        self.humanPlayerId = 1 if self.humanGoesFirst else 2
        self.aiPlayerId = 2 if self.humanGoesFirst else 1

        self.gameState.initState(0)
        self.isGameOver = False
        self.victor = None
        self.lastMoveIdx = None
        self.moveLog = []
        self.aiAgent.resetState()

        if self.humanGoesFirst:
            self.statusMessage = "轮到你了（执黑）"
        else:
            self.statusMessage = "AI 思考中..."
            self.isAiThinking = True
            threading.Thread(target=self._executeAiMove, daemon=True).start()

    def returnToMenu(self):
        """返回菜单"""
        self.showingMenu = True
        self.isGameOver = False
        self.isAiThinking = False
        # 重置窗口大小
        self._updateWindowSize()
        self.display = pygame.display.set_mode((self.windowWidth, self.windowHeight))

    def _checkGameEnd(self):
        """检查游戏是否结束"""
        finished, winner = self.gameState.isTerminal()
        if finished:
            self.isGameOver = True
            self.victor = winner
            if winner == -1:
                self.statusMessage = "平局！"
            elif winner == self.humanPlayerId:
                self.statusMessage = "恭喜你赢了！"
            else:
                self.statusMessage = "AI 获胜！"
            return True
        return False

    def _handleHumanMove(self, x, y):
        """处理玩家落子"""
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
        """执行 AI 落子"""
        self.aiAgent.assignPlayerId(self.gameState.activePlayer)
        moveIdx = self.aiAgent.selectMove(self.gameState)
        self.gameState.applyMove(moveIdx)
        self.lastMoveIdx = moveIdx
        self.moveLog.append(moveIdx)
        self.isAiThinking = False

        if not self._checkGameEnd():
            pieceColor = "黑" if self.gameState.activePlayer == 1 else "白"
            self.statusMessage = f"轮到你了（执{pieceColor}）"

    def run(self):
        """主游戏循环"""
        clock = pygame.time.Clock()
        isRunning = True

        while isRunning:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    isRunning = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouseX, mouseY = event.pos

                    if self.showingMenu:
                        # 棋盘选择
                        for i, btnRect in enumerate(self.btnBoards):
                            if btnRect.collidepoint(mouseX, mouseY):
                                self.selectedBoardConfig = i

                        # 先后手选择
                        if self.btnFirst.collidepoint(mouseX, mouseY):
                            self.humanGoesFirst = True
                        elif self.btnSecond.collidepoint(mouseX, mouseY):
                            self.humanGoesFirst = False

                        # 开始游戏
                        if self.btnStart.collidepoint(mouseX, mouseY):
                            self.startGame()
                    else:
                        # 游戏按钮
                        if self.btnRestart.collidepoint(mouseX, mouseY):
                            self.resetGame()
                        elif self.btnMenu.collidepoint(mouseX, mouseY):
                            self.returnToMenu()
                        elif mouseY < self.boardPixelSize:
                            # 棋盘点击
                            x, y = self.pixelToBoardPos(mouseX, mouseY)
                            if x is not None and self.gameState.activePlayer == self.humanPlayerId:
                                self._handleHumanMove(x, y)

            # 渲染
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
    gui = GomokuInterface()
    gui.run()


if __name__ == '__main__':
    main()
