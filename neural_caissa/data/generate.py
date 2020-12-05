from typing import Any, NamedTuple, Optional
import random

import click
import chess.pgn
import numpy as np
from tqdm import tqdm

from neural_caissa.board.state import State

_OUTCOME = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}


class CaissaData(NamedTuple):
    # TODO: Make numpy typing work: import numpy.typing as npt
    X_origin: Any
    X_move: Optional[Any]
    X_random: Optional[Any]
    Y: Any


def _generate_dataset(data_file_path, full=False, samples=None) -> CaissaData:
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

                if full:
                    x_move, x_random = _get_next_and_random_moves(board, move)
                    X_move.append(x_move)
                    X_random.append(x_random)

                board.push(move)
                X_origin.append(x_origin)
                Y.append(_y)

            games_counter += 1
            progress_bar.update(1)
    progress_bar.close()

    X_origin = np.array(X_origin)
    Y = np.array(Y)
    X_move = np.array(X_move)
    X_random = np.array(X_random)
    output = CaissaData(X_origin=X_origin, Y=Y, X_move=X_move, X_random=X_random)
    return output


def _get_next_and_random_moves(board, move):
    board.push(move)
    x_move = State(board).serialize_conv(turn=board.turn)
    board.pop()

    legal_moves = list(board.legal_moves)
    random_move = random.choice(legal_moves)
    board.push(random_move)
    x_random = State(board).serialize_conv(turn=board.turn)
    board.pop()
    return x_move, x_random


@click.command()
@click.option('--input_data_file',
              default='data/raw_data/1k_caissadb_data.pgn',
              help='Input PGN data file.')
@click.option('--output_data',
              default='data/serialized_data/dataset_1k.npz',
              help='Output serialized data file.')
@click.option('--full', default=False, help='Add random move and next move to data serialization.')
@click.option('--samples', default=1_000, help='Number of games to serialize from ')
def main(input_data_file, output_data, samples, full):
    d = _generate_dataset(input_data_file, full, int(samples))
    np.savez(output_data, d.X_origin, d.X_move, d.X_random, d.Y)
    print("saved", d.X_origin.shape, d.X_move.shape, d.X_random.shape, d.Y.shape)


if __name__ == "__main__":
    main()
