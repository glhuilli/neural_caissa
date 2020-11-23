import chess

from neural_caissa.ply.explore import MAX_VALUE


_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


class ClassicValuator:

    def __init__(self):
        self.reset()
        self.memo = {}
        self.count = 0

    def reset(self):
        self.count = 0

    # writing a simple value function based on pieces
    # good ideas:
    # https://en.wikipedia.org/wiki/Evaluation_function#In_chess
    def __call__(self, state):
        self.count += 1
        key = state.key()
        if key not in self.memo:
            self.memo[key] = self.value(state)
        return self.memo[key]

    @staticmethod
    def value(state):
        b = state.board

        # game over values
        if b.is_game_over():
            if b.result() == "1-0":
                return MAX_VALUE
            elif b.result() == "0-1":
                return -MAX_VALUE
            else:
                return 0

        val = 0.0

        # piece values
        pm = state.board.piece_map()
        for x in pm:
            tval = _VALUES[pm[x].piece_type]
            if pm[x].color == chess.WHITE:
                val += tval
            else:
                val -= tval

        # add a number of legal moves term
        bak = b.turn
        b.turn = chess.WHITE
        val += 0.1 * b.legal_moves.count()
        b.turn = chess.BLACK
        val -= 0.1 * b.legal_moves.count()
        b.turn = bak

        return val
