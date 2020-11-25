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

    # TODO: Improve evaluation function
    #    https://en.wikipedia.org/wiki/Evaluation_function#In_chess
    #    https://www.chessprogramming.org/Evaluation
    def __call__(self, state):
        self.count += 1
        key = state.key()
        if key not in self.memo:
            self.memo[key] = self.value(state)
        return self.memo[key]

    @staticmethod
    def value(state):
        """
        if game_over
            return MAX_VALUE
        else
            value = SUM(piece in white)
                        - SUM(piece in black)
                        + 0.1 * TOTAL white legal moves
                        - 0.1 * TOTAL black legal moves
        """
        b = state.board

        if b.is_game_over():
            if b.result() == "1-0":
                return MAX_VALUE
            elif b.result() == "0-1":
                return -MAX_VALUE
            else:
                return 0

        val = 0.0

        # SUM(piece in white) - SUM(piece in black)
        pm = state.board.piece_map()
        for x in pm:
            tval = _VALUES.get(pm[x].piece_type)
            if pm[x].color == chess.WHITE:
                val += tval
            else:
                val -= tval

        # + 0.1 * TOTAL white legal moves - 0.1 * TOTAL black legal moves
        bak = b.turn
        b.turn = chess.WHITE
        val += 0.1 * b.legal_moves.count()

        b.turn = chess.BLACK
        val -= 0.1 * b.legal_moves.count()
        b.turn = bak

        return val
