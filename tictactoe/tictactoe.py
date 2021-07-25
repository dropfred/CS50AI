"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x = 0
    for r in board:
        for c in r:
            if   c == X : x += 1
            elif c == O : x -= 1
    return X if x == 0 else O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    s = set()
    for r in range(3):
        for c in range(3):
            if board[r][c] is None:
                s.add((r, c))
    return s


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    r, c = action
    if r < 0 or r > 2 or c < 0 or c > 2 or board[r][c] is not None:
        raise ValueError
    b = copy.deepcopy(board)
    b[r][c] = player(board)
    return b


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not None: return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] is not None: return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None: return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None: return board[0][2]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None: return True
    for r in board:
        if r.count(None) > 0: return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    w = winner(board)
    return 1 if w == X else -1 if w == O else 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    def minimax_value(player, board, abp, low=-math.inf, high=math.inf):
        if terminal(board): return utility(board)
        if player == X:
            v = low
            for a in actions(board):
                mv = minimax_value(O, result(board, a), (abp[0], v))
                if mv > v: v = mv
                if v >= high or abp[0] == O and v >= abp[1]: break

        else:
            v = high
            for a in actions(board):
                mv = minimax_value(X, result(board, a), (abp[0], v))
                if mv < v: v = mv
                if v <= low or abp[0] == X and v <= abp[1]: break
        return v

    action = None

    if player(board) == X:
        v = -math.inf
        for a in actions(board):
            mv = minimax_value(O, result(board, a), (X, v), -1, 1)
            if mv > v:
                v = mv
                action = a
            if v >= 1: break
    else:
        v = math.inf
        for a in actions(board):
            mv = minimax_value(X, result(board, a), (O, v), -1, 1)
            if mv < v:
                v = mv
                action = a
            if v <= -1: break

    return action
    