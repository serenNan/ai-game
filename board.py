"""
五子棋游戏引擎实现
处理棋盘状态管理和游戏流程控制
"""

from __future__ import print_function
import numpy as np


class GameState(object):
    """表示五子棋棋盘的当前状态"""

    def __init__(self, **kwargs):
        self.cols = int(kwargs.get('width', 8))
        self.rows = int(kwargs.get('height', 8))
        # 存储落子位置：键=位置索引，值=玩家ID
        self.positions = {}
        # 获胜所需的连续棋子数
        self.winCondition = int(kwargs.get('n_in_row', 5))
        self.playerIds = [1, 2]

    def initState(self, firstPlayer=0):
        """初始化棋盘状态"""
        if self.cols < self.winCondition or self.rows < self.winCondition:
            raise Exception('棋盘尺寸必须至少为 {}'.format(self.winCondition))
        self.activePlayer = self.playerIds[firstPlayer]
        # 可落子的位置列表
        self.openPositions = list(range(self.cols * self.rows))
        self.positions = {}
        self.previousMove = -1

    def indexToCoord(self, idx):
        """
        将一维索引转换为 (行, 列) 坐标
        棋盘布局示例 (3x3):
        6 7 8
        3 4 5
        0 1 2
        位置 5 -> (1, 2)
        """
        row = idx // self.cols
        col = idx % self.cols
        return [row, col]

    def coordToIndex(self, coord):
        """将 (行, 列) 坐标转换为一维索引"""
        if len(coord) != 2:
            return -1
        row = coord[0]
        col = coord[1]
        idx = row * self.cols + col
        if idx not in range(self.cols * self.rows):
            return -1
        return idx

    def getStateArray(self):
        """
        从当前玩家视角返回棋盘状态的 numpy 数组
        形状: 4 x cols x rows
        通道 0: 当前玩家的棋子
        通道 1: 对手的棋子
        通道 2: 上一步落子位置
        通道 3: 颜色标识（先手方为1，后手方为0）
        """
        stateArray = np.zeros((4, self.cols, self.rows))
        if self.positions:
            moves, players = np.array(list(zip(*self.positions.items())))
            currentMoves = moves[players == self.activePlayer]
            opponentMoves = moves[players != self.activePlayer]
            stateArray[0][currentMoves // self.cols,
                         currentMoves % self.cols] = 1.0
            stateArray[1][opponentMoves // self.cols,
                         opponentMoves % self.cols] = 1.0
            stateArray[2][self.previousMove // self.cols,
                         self.previousMove % self.cols] = 1.0
        if len(self.positions) % 2 == 0:
            stateArray[3][:, :] = 1.0
        return stateArray[:, ::-1, :]

    def applyMove(self, moveIdx):
        """执行落子操作"""
        self.positions[moveIdx] = self.activePlayer
        self.openPositions.remove(moveIdx)
        self.activePlayer = (
            self.playerIds[0] if self.activePlayer == self.playerIds[1]
            else self.playerIds[1]
        )
        self.previousMove = moveIdx

    def checkVictory(self):
        """检查是否有玩家获胜（横向/纵向/对角线四个方向）"""
        w = self.cols
        h = self.rows
        positions = self.positions
        n = self.winCondition

        playedMoves = list(set(range(w * h)) - set(self.openPositions))
        if len(playedMoves) < self.winCondition * 2 - 1:
            return False, -1

        for m in playedMoves:
            row = m // w
            col = m % w
            player = positions[m]

            # 检查横向
            if (col in range(w - n + 1) and
                    len(set(positions.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            # 检查纵向
            if (row in range(h - n + 1) and
                    len(set(positions.get(i, -1) for i in range(m, m + n * w, w))) == 1):
                return True, player

            # 检查对角线（左上到右下）
            if (col in range(w - n + 1) and row in range(h - n + 1) and
                    len(set(positions.get(i, -1) for i in range(m, m + n * (w + 1), w + 1))) == 1):
                return True, player

            # 检查对角线（右上到左下）
            if (col in range(n - 1, w) and row in range(h - n + 1) and
                    len(set(positions.get(i, -1) for i in range(m, m + n * (w - 1), w - 1))) == 1):
                return True, player

        return False, -1

    def isTerminal(self):
        """检查游戏是否结束"""
        hasWinner, victor = self.checkVictory()
        if hasWinner:
            return True, victor
        elif not len(self.openPositions):
            return True, -1
        return False, -1

    def getCurrentPlayer(self):
        """获取当前执棋玩家"""
        return self.activePlayer


class GameController(object):
    """控制游戏流程并管理玩家交互"""

    def __init__(self, gameState, **kwargs):
        self.state = gameState

    def renderBoard(self, gameState, agent1, agent2):
        """在终端显示棋盘"""
        w = gameState.cols
        h = gameState.rows

        print("玩家", agent1, "执 X")
        print("玩家", agent2, "执 O")
        print()
        for x in range(w):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(h - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(w):
                loc = i * w + j
                p = gameState.positions.get(loc, -1)
                if p == agent1:
                    print('X'.center(8), end='')
                elif p == agent2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def runMatch(self, agent1, agent2, firstPlayer=0, displayBoard=1):
        """运行两个玩家之间的对局"""
        if firstPlayer not in (0, 1):
            raise Exception('firstPlayer 参数应为 0 或 1')
        self.state.initState(firstPlayer)
        p1, p2 = self.state.playerIds
        agent1.assignPlayerId(p1)
        agent2.assignPlayerId(p2)
        agents = {p1: agent1, p2: agent2}
        if displayBoard:
            self.renderBoard(self.state, agent1.playerId, agent2.playerId)
        while True:
            currentPlayerId = self.state.getCurrentPlayer()
            activeAgent = agents[currentPlayerId]
            move = activeAgent.selectMove(self.state)
            self.state.applyMove(move)
            if displayBoard:
                self.renderBoard(self.state, agent1.playerId, agent2.playerId)
            finished, victor = self.state.isTerminal()
            if finished:
                if displayBoard:
                    if victor != -1:
                        print("游戏结束。获胜者:", agents[victor])
                    else:
                        print("游戏结束。平局")
                return victor

    def runSelfPlay(self, agent, displayBoard=0, temperature=1e-3):
        """运行自我对弈游戏以收集训练数据"""
        self.state.initState()
        p1, p2 = self.state.playerIds
        stateHistory, probHistory, playerHistory = [], [], []
        while True:
            move, moveProbs = agent.selectMove(self.state,
                                               temperature=temperature,
                                               returnProb=1)
            stateHistory.append(self.state.getStateArray())
            probHistory.append(moveProbs)
            playerHistory.append(self.state.activePlayer)
            self.state.applyMove(move)
            if displayBoard:
                self.renderBoard(self.state, p1, p2)
            finished, victor = self.state.isTerminal()
            if finished:
                # 从每个状态的视角计算奖励
                rewards = np.zeros(len(playerHistory))
                if victor != -1:
                    rewards[np.array(playerHistory) == victor] = 1.0
                    rewards[np.array(playerHistory) != victor] = -1.0
                agent.resetState()
                if displayBoard:
                    if victor != -1:
                        print("游戏结束。获胜者:", victor)
                    else:
                        print("游戏结束。平局")
                return victor, zip(stateHistory, probHistory, rewards)
