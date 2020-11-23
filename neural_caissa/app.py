from flask import Flask, request
import traceback
import chess


from neural_caissa.board.state import State
from neural_caissa.board.move import computer_move
from neural_caissa.ply.valuator import ClassicValuator


app = Flask(__name__)
state = State()
valuator = ClassicValuator()


@app.route("/move_coordinates")
def move_coordinates():
    if not state.board.is_game_over():
        source = int(request.args.get('from', default=''))
        target = int(request.args.get('to', default=''))
        promotion = True if request.args.get('promotion', default='') == 'true' else False

        # TODO: allow to pick promotion. Automatically promoting to Queen for now.
        move = state.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

        if move is not None and move != "":

            try:
                state.board.push_san(move)
                computer_move(state, valuator)
            except Exception:
                traceback.print_exc()
            response = app.response_class(
              response=state.board.fen(),
              status=200
            )
            return response

    response = app.response_class(
        response="game over",
        status=200
    )
    return response


@app.route("/newgame")
def newgame():
    state.board.reset()
    response = app.response_class(
        response=state.board.fen(),
        status=200
    )
    return response


@app.route("/")
def index():
    ret = open("index.html").read()
    return ret.replace('start', state.board.fen())


def main():
    app.run(port=5000, debug=True)
