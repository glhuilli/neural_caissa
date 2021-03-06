import chess
import numpy as np

from neural_caissa.ply.valuators.baseline_valuator import BaselineValuator
from neural_caissa.ply.valuators.neural_valuator import NeuralValuator

_BOARD_DIM = 8
_POSITIONS = 64
_PIECES = 'PNBRQKpnbrqk'
_VALUATORS = {'BaselineValuator': BaselineValuator, 'NeuralValuator': NeuralValuator}


class State:
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board
        self.valuator = self._init_valuator('BaselineValuator')

    def set_valuator(self, valuator_name, model_file: str = None):
        self.valuator = self._init_valuator(valuator_name, model_file)

    def key(self):
        return self.board.board_fen(
        ), self.board.turn, self.board.castling_rights, self.board.ep_square

    def serialize(self, turn: bool = False):
        """
        Tensor of 12 x 64 dimensions (=12 x (8 x 8)) with 1 if piece k in position i,
        with i in [1, 64], else 0.

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

    @staticmethod
    def _init_valuator(valuator_name, model_file=None):
        if valuator_name == 'BaselineValuator':
            return _VALUATORS[valuator_name]()
        return _VALUATORS[valuator_name](model_file)
