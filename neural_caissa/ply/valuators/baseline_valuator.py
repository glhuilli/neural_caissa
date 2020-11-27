import logging

import chess

from neural_caissa.ply.explore import MAX_VALUE
from neural_caissa.ply.valuators.valuator import Valuator

logger = logging.getLogger(__name__)

# Using AlphaZero's valuation https://en.wikipedia.org/wiki/Chess_piece_relative_value
_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3.05,
    chess.BISHOP: 3.33,
    chess.ROOK: 5.63,
    chess.QUEEN: 9.5
}


class BaselineValuator(Valuator):
    def __init__(self):
        self.memory = {}
        self.count = 0

    def reset(self):
        self.count = 0

    def __call__(self, state):
        self.count += 1
        key = state.key()
        if key not in self.memory:
            self.memory[key] = self._value(state)
        return self.memory[key]

    @staticmethod
    def _value(state):
        """
        if game_over
            return MAX_VALUE
        else
            value = SUM(piece in white)
                        - SUM(piece in black)
                        + White pieces mobility score
                        - White pieces mobility score

        TODO: Improve evaluation function
           https://www.chessprogramming.org/Evaluation

        Notes:
            - From the computer's perspective (black by default), the lower the value the best is the move
            - Mobility score is fundamental to make the computer play in a +1000 ELO score range
        """
        b = state.board

        if b.is_game_over():
            if b.result() == '1-0':
                return MAX_VALUE
            elif b.result() == '0-1':
                return -MAX_VALUE
            else:
                raise Exception

        score = 0.0

        # Material score
        # King should be ignored from this score
        pieces = [v for (k, v) in state.board.piece_map().items() if v.piece_type != chess.KING]
        for piece in pieces:
            piece_val = _VALUES.get(piece.piece_type, 0.0)
            if piece.color == chess.WHITE:
                score += piece_val
            else:
                score -= piece_val

        # Mobility score
        # + 0.1 * number of white legal moves - 0.1 * number of black legal moves
        turn_backup = b.turn
        b.turn = chess.WHITE
        score += 0.1 * b.legal_moves.count()

        b.turn = chess.BLACK
        score -= 0.1 * b.legal_moves.count()
        b.turn = turn_backup

        return score
