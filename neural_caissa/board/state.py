import chess
import numpy as np

_BOARD_DIM = 8
_POSITIONS = 64
_PIECES = 'PNBRQKpnbrqk'


class State(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def key(self):
        return self.board.board_fen(), self.board.turn, self.board.castling_rights, self.board.ep_square

    def serialize(self, turn: bool = False):
        """
        Vector of 768 (= 8*8*12) with 1 if piece k in position j + positions * k,
        with j in [1, 64], k in [1, 12], else 0.

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

    def serialize_conv(self, turn: bool = False):
        """
        Tensor of 8x8x12 with 1 in (k, i, j) if piece k is in position (i,j) else 0.

        Note that Turn is False if it's white and True if it's black.
        """
        x = np.zeros((len(_PIECES), _BOARD_DIM, _BOARD_DIM), np.uint8)
        for idx, piece in enumerate(_PIECES):
            piece_state = np.zeros(_POSITIONS, dtype=np.int8)
            for pos in range(_POSITIONS):
                if turn:
                    pos = _POSITIONS - 1 - pos
                board_piece = self.board.piece_at(pos)
                if board_piece and piece == board_piece.symbol():
                    piece_state[pos] = 1
            piece_state = piece_state.reshape(8, 8)
            x[idx] = piece_state
        return x
