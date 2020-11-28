import random

import click
import chess.pgn
import numpy as np
from tqdm import tqdm

from neural_caissa.board.state import State


_OUTCOME = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}


def _generate_dataset(data_file_path, samples=None):
    X_origin, X_move, X_random, Y = [], [], [], []
    games_counter = 0
    progress_bar = tqdm(total=games_counter + 1)
    with open(data_file_path, 'r') as pgn:
        while games_counter < samples:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            result = game.headers['Result']
            if result not in _OUTCOME:
                continue
            _y = _OUTCOME.get(result)

            board = game.board()
            for move in game.mainline_moves():
                x_origin = State(board).serialize_conv(turn=board.turn)

                board.push(move)
                x_move = State(board).serialize_conv(turn=board.turn)
                board.pop()

                legal_moves = list(board.legal_moves)
                random_move = random.choice(legal_moves)
                board.push(random_move)
                x_random = State(board).serialize_conv(turn=board.turn)
                board.pop()
                board.push(move)

                X_origin.append(x_origin)
                X_move.append(x_move)
                X_random.append(x_random)
                Y.append(_y)
            games_counter += 1
            progress_bar.update(1)
    progress_bar.close()
    X_origin = np.array(X_origin)
    X_move = np.array(X_move)
    X_random = np.array(X_random)
    Y = np.array(Y)
    return X_origin, X_move, X_random, Y


@click.command()
@click.option('--input_data_file',
              default='data/raw_data/1k_caissadb_data.pgn',
              help='Input PGN data file.')
@click.option('--output_data', default='data/serialized_data/dataset_1k.npz', help='Output serialized data file.')
def main(input_data_file, output_data):
    X_origin, X_move, X_random, Y = _generate_dataset(input_data_file, 100_000)
    np.savez(output_data, X_origin, X_move, X_random, Y)
    print("saved", X_origin.shape, X_move.shape, X_random.shape, Y.shape)


if __name__ == "__main__":
    main()
