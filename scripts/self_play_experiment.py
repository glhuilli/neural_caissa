import random
import traceback

import chess
import click

from neural_caissa.board.move import computer_move
from neural_caissa.board.state import State

_VALUATOR_MODEL_FILE_MAPPING = {
    'neuralcaissa100k': {
        'valuator': 'NeuralValuator',
        'model_file': 'nets/neural_score_100.pth'
    },
    'neuralcaissa1k': {
        'valuator': 'NeuralValuator',
        'model_file': 'nets/neural_score.pth'
    },
    'baseline': {
        'valuator': 'BaselineValuator'
    }
}

state = State()


def _move_white(initial=True):
    if initial:
        legal_moves = list(state.board.legal_moves)
        random_move = random.choice(legal_moves)

    return None, None


def _move(promotion, source, target):
    move = state.board.san(
        chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

    if move:
        try:
            state.board.push_san(move)
            game_over = computer_move(state)

        except Exception:
            traceback.print_exc()


@click.command()
@click.option('--computer_black_model',
              default='../nets/neural_score.pth',
              help='Input model to play white.')
@click.option('--computer_white_model',
              default='../nets/neural_score.pth',
              help='Input model to play black.')
def main(computer_black_model, computer_white_model):
    """
        1. Find first more from all available moves in state
        2.
    """
    print(f'Using white: {computer_white_model}')
    print(f'Using black: {computer_black_model}')

    source, target = _get_initial_move()
    promotion = False

    while not state.board.is_game_over():
        _move(promotion, source, target)
        next_p, next_s, next_t = _move_white(promotion, source, target)

        p = next_p
        s = next_s
        t = next_t

if __name__ == "__main__":
    main()
