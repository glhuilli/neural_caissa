import logging
import os
import traceback

import chess
from flask import Flask, request

from neural_caissa.board.move import computer_move
from neural_caissa.puzzle.state import PuzzleState

_FILE_PATH = os.path.dirname(__file__)

logging.basicConfig(filename='record_puzzle.log',
                    level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
app = Flask(__name__)

# TODO: Allow passing down the file with data from the web UI
puzzle_state = PuzzleState(games_dataset_path=os.path.join(_FILE_PATH, '../data/raw_data/100k_caissadb_data.pgn'))


@app.route("/move_coordinates")
def move_coordinates():
    """
    If is not game_over, then find the best computer move given
    the state of the board and a valuator strategy.
    """
    game_over = False
    if not puzzle_state.board.is_game_over():
        source = int(request.args.get('from', default=''))
        target = int(request.args.get('to', default=''))
        promotion = bool(request.args.get('promotion', default='') == 'true')

        # TODO: allow to pick promotion. Automatically promoting to Queen for now.
        move = puzzle_state.board.san(
            chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

        if move:
            app.logger.debug(f'Human moved (standard algebraic notation): {move}')

            try:
                puzzle_state.board.push_san(move)
                game_over = computer_move(puzzle_state)
            except ValueError:
                app.logger.error('Human tried to do an illegal move!')
            except Exception:
                traceback.print_exc()

            app.logger.debug(f'Response state.board.fen(): {puzzle_state.board.fen()}')
            if game_over:
                response = app.response_class(response='game over', status=200)
            else:
                response = app.response_class(response=puzzle_state.board.fen(), status=200)
            return response

    response = app.response_class(response='game over', status=200)
    return response


@app.route("/nextpuzzle")
def nextpuzzle():
    puzzle_state.board.reset()
    puzzle_state.set_puzzle()  # just a random puzzle for now
    app.logger.debug(puzzle_state.board)
    response = app.response_class(response=puzzle_state.board.fen(), status=200)
    return response


@app.route("/")
def index():
    ret = open("index_puzzle.html").read()
    return ret.replace('start', puzzle_state.board.fen())


def main():
    app.run(port=9000, debug=True)
