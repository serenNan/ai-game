"""
五子棋游戏引擎实现
处理棋盘状态管理和游戏流程控制
"""

from __future__ import print_function
import numpy as np


class GameState(object):
    """Represents the current state of a Gomoku board"""

    def __init__(self, **kwargs):
        self.cols = int(kwargs.get('width', 8))
        self.rows = int(kwargs.get('height', 8))
        # Store positions as dict: key=position index, value=player id
        self.positions = {}
        # Number of consecutive pieces needed to win
        self.winCondition = int(kwargs.get('n_in_row', 5))
        self.playerIds = [1, 2]

    def initState(self, firstPlayer=0):
        if self.cols < self.winCondition or self.rows < self.winCondition:
            raise Exception('Board dimensions must be at least {}'.format(self.winCondition))
        self.activePlayer = self.playerIds[firstPlayer]
        # Available positions for moves
        self.openPositions = list(range(self.cols * self.rows))
        self.positions = {}
        self.previousMove = -1

    def indexToCoord(self, idx):
        """
        Convert linear index to (row, col) coordinates
        Board layout example for 3x3:
        6 7 8
        3 4 5
        0 1 2
        Position 5 -> (1, 2)
        """
        row = idx // self.cols
        col = idx % self.cols
        return [row, col]

    def coordToIndex(self, coord):
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
        Return board state as numpy array from current player's perspective
        Shape: 4 x cols x rows
        Channel 0: current player's pieces
        Channel 1: opponent's pieces
        Channel 2: last move position
        Channel 3: color indicator
        """
        stateArray = np.zeros((4, self.cols, self.rows))
        if self.positions:
            moves, players = np.array(list(zip(*self.positions.items())))
            currentMoves = moves[players == self.activePlayer]
            opponentMoves = moves[players != self.activePlayer]
            stateArray[0][currentMoves // self.cols,
                         currentMoves % self.rows] = 1.0
            stateArray[1][opponentMoves // self.cols,
                         opponentMoves % self.rows] = 1.0
            stateArray[2][self.previousMove // self.cols,
                         self.previousMove % self.rows] = 1.0
        if len(self.positions) % 2 == 0:
            stateArray[3][:, :] = 1.0
        return stateArray[:, ::-1, :]

    def applyMove(self, moveIdx):
        self.positions[moveIdx] = self.activePlayer
        self.openPositions.remove(moveIdx)
        self.activePlayer = (
            self.playerIds[0] if self.activePlayer == self.playerIds[1]
            else self.playerIds[1]
        )
        self.previousMove = moveIdx

    def checkVictory(self):
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

            # Check horizontal
            if (col in range(w - n + 1) and
                    len(set(positions.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            # Check vertical
            if (row in range(h - n + 1) and
                    len(set(positions.get(i, -1) for i in range(m, m + n * w, w))) == 1):
                return True, player

            # Check diagonal (top-left to bottom-right)
            if (col in range(w - n + 1) and row in range(h - n + 1) and
                    len(set(positions.get(i, -1) for i in range(m, m + n * (w + 1), w + 1))) == 1):
                return True, player

            # Check diagonal (top-right to bottom-left)
            if (col in range(n - 1, w) and row in range(h - n + 1) and
                    len(set(positions.get(i, -1) for i in range(m, m + n * (w - 1), w - 1))) == 1):
                return True, player

        return False, -1

    def isTerminal(self):
        """Check if game has ended"""
        hasWinner, victor = self.checkVictory()
        if hasWinner:
            return True, victor
        elif not len(self.openPositions):
            return True, -1
        return False, -1

    def getCurrentPlayer(self):
        return self.activePlayer


class GameController(object):
    """Controls game flow and manages player interactions"""

    def __init__(self, gameState, **kwargs):
        self.state = gameState

    def renderBoard(self, gameState, agent1, agent2):
        """Display the board in terminal"""
        w = gameState.cols
        h = gameState.rows

        print("Player", agent1, "with X".rjust(3))
        print("Player", agent2, "with O".rjust(3))
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
        """Run a game between two players"""
        if firstPlayer not in (0, 1):
            raise Exception('firstPlayer should be 0 or 1')
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
                        print("Game over. Winner is", agents[victor])
                    else:
                        print("Game over. Draw")
                return victor

    def runSelfPlay(self, agent, displayBoard=0, temperature=1e-3):
        """Run self-play game for training data collection"""
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
                # Compute rewards from each state's perspective
                rewards = np.zeros(len(playerHistory))
                if victor != -1:
                    rewards[np.array(playerHistory) == victor] = 1.0
                    rewards[np.array(playerHistory) != victor] = -1.0
                agent.resetState()
                if displayBoard:
                    if victor != -1:
                        print("Game over. Winner is player:", victor)
                    else:
                        print("Game over. Draw")
                return victor, zip(stateHistory, probHistory, rewards)
