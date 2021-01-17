import logging

import chess.pgn

from neural_caissa.board.state import State

logger = logging.getLogger(__name__)


class PuzzleState(State):
    def __init__(self, games_dataset_path, board=None):
        self.games_dataset_path = games_dataset_path
        State.__init__(self, board=board)
        self.puzzles = []

    def set_puzzle(self):
        logger.debug(f'PUZZLES: {self.puzzles}')
        black_wins = True
        with open(self.games_dataset_path, 'r') as pgn:
            game_idx = 0
            while black_wins:
                game = chess.pgn.read_game(pgn)
                if not game:
                    logger.debug('No more games.')
                    break

                if self._valid_game(game, game_idx):
                    black_wins = False

                if not black_wins:
                    logger.debug(game)
                    self.puzzles.append(game_idx)
                    self.board = game.board()
                    for move in game.mainline_moves():
                        self.board.push(move)
                    self.board.pop()
                    self.board.pop()
                    self.board.pop()
                    break
                game_idx += 1

    def _valid_game(self, game, game_idx):
        """
        Game is not None, white wins, there were no more legal moves for the black, and
        game_idx not in the already reviewed puzzles.
        """
        return game \
               and game.headers['Result'] == '1-0' \
               and self._no_more_legal_moves(game) \
               and game_idx not in self.puzzles

    @staticmethod
    def _no_more_legal_moves(game):
        """
        Iterate over a game until there are no more moves and by the last move
        there are no more legal_moves for the next player.
        """
        board = chess.Board()
        for move in game.mainline_moves():
            board.push(move)
        return board.legal_moves.count() == 0
