import logging
import traceback

import chess
from flask import Flask, request

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

logging.basicConfig(filename='record.log',
                    level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
app = Flask(__name__)

state = State()


@app.route("/move_coordinates")
def move_coordinates():
    """
    If is not game_over, then find the best computer move given
    the state of the board and a valuator strategy.
    """
    game_over = False
    if not state.board.is_game_over():
        source = int(request.args.get('from', default=''))
        target = int(request.args.get('to', default=''))
        promotion = True if request.args.get('promotion', default='') == 'true' else False

        # TODO: allow to pick promotion. Automatically promoting to Queen for now.
        move = state.board.san(
            chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

        if move:
            app.logger.debug(f'Human moved (standard algebraic notation): {move}')

            try:
                state.board.push_san(move)
                game_over = computer_move(state)
            except ValueError:
                app.logger.error('Human tried to do an illegal move!')
            except Exception:
                traceback.print_exc()

            app.logger.debug(f'Response state.board.fen(): {state.board.fen()}')
            if game_over:
                response = app.response_class(response='game over', status=200)
            else:
                response = app.response_class(response=state.board.fen(), status=200)
            return response

    response = app.response_class(response='game over', status=200)
    return response


@app.route("/newgame")
def newgame():
    state.board.reset()
    state.set_valuator('BaselineValuator')
    response = app.response_class(response=state.board.fen(), status=200)
    return response


@app.route("/valuator")
def valuator():
    mapping = _VALUATOR_MODEL_FILE_MAPPING.get(request.args.get('valuator', 'baseline'))
    state.board.reset()
    state.set_valuator(mapping['valuator'], mapping.get('model_file', None))
    response = app.response_class(response=state.board.fen(), status=200)
    return response


@app.route("/")
def index():
    ret = open("index.html").read()
    return ret.replace('start', state.board.fen())


def main():
    app.run(port=9000, debug=True)
