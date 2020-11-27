import chess
import numpy as np


_POSITIONS = 64
_PIECES = 'PNBRQKpnbrqk'


class State(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def serialize(self, turn: bool = False):
        """
        Vector of 8x8x12 with 1 if piece k in position j + positions * k,
        with j in [1, 64], k in [1, 12].

        Note that Turn is False if it's white and True if it's black.
        """
        x = np.zeros(_POSITIONS * len(_PIECES), dtype=np.int8)
        for idx, piece in enumerate(_PIECES):
            for pos in range(_POSITIONS):
                if turn:
                    pos = _POSITIONS - 1 - pos
                board_piece = self.board.piece_at(pos)
                if board_piece and piece == board_piece.symbol():
                    x[pos + idx * _POSITIONS] = 1
        return x
