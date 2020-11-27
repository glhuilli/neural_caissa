import os
import random

import chess.pgn
import numpy as np

from neural_caissa.board.state import State


_OUTCOME = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}


def _generate_dataset(data_path, samples=None):
    X, X_random, Y = [], [], []
    games_counter = 0
    for fn in os.listdir(data_path):
        pgn = open(os.path.join(data_path, fn))
        while 1:
            if samples and games_counter > samples:
                break
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            result = game.headers['Result']
            if result not in _OUTCOME:
                continue
            _y = _OUTCOME.get(result)

            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                x_move = State(board).serialize(turn=board.turn)
                board.pop()

                legal_moves = list(board.legal_moves)
                random_move = random.choice(legal_moves)
                board.push(random_move)
                x_random = State(board).serialize(turn=board.turn)
                board.pop()
                board.push(move)

                X.append(x_move)
                X_random.append(x_random)
                Y.append(_y)
            games_counter += 1

    X = np.array(X)
    X_random = np.array(X_random)
    Y = np.array(Y)
    return X, X_random, Y


if __name__ == "__main__":
    X, X_random, Y = _generate_dataset('data/raw_data', 100_000)
    np.savez('data/serialized_data/dataset_100k.npz', X, X_random, Y)
