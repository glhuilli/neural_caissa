from typing import List, Optional
from datetime import datetime
import logging
import random
import uuid

import chess
import chess.pgn

from neural_caissa.ply.explore import explore_leaves

logger = logging.getLogger(__name__)


def computer_move(state) -> bool:
    """
    Method to decide which movements
    """
    sorted_movements = sorted(explore_leaves(state), key=lambda x: x[0], reverse=state.board.turn)

    if len(sorted_movements) == 0:
        logger.debug('No moves to make: GAME OVER?')
        return True
    _log_moves(sorted_movements)

    # use the first movement as these are already sorted
    state.board.push(sorted_movements[0][1])
    return False


def return_computer_move(state, random_state: Optional[int] = None, top_choices: int = 3):
    if random_state:
        random.seed(random_state)
    move_choices = sorted(explore_leaves(state), key=lambda x: x[0], reverse=state.board.turn)
    return random.choice(move_choices[:top_choices])[1]


def move_player(state, random_state: Optional[int] = None, top_choices: int = 3) -> str:
    movement = return_computer_move(state, random_state, top_choices)
    chess_move = chess.Move(movement.from_square, movement.to_square, promotion=None)
    move = state.board.san(chess_move)
    state.board.push_san(move)
    return str(movement)


def get_pgn_from_moves(moves: List[str], white: str, black: str, result: str):
    game = chess.pgn.Game()
    game.headers['Event'] = str(uuid.uuid4())
    game.headers['Site'] = 'https://github.com/glhuilli/neural_caissa'
    game.headers['Date'] = datetime.now().date().isoformat()
    game.headers['White'] = white
    game.headers['Black'] = black
    game.headers['Round'] = '1'
    game.headers['Result'] = result

    node = game.add_variation(chess.Move.from_uci(moves[0]))
    for move in moves[1:]:
        node = node.add_variation(chess.Move.from_uci(move))
    return game


def _log_moves(move, top_moves: int = 3) -> None:
    for i, m in enumerate(move[0:top_moves]):
        logger.debug(f'Top {i} move: {m}')
    logger.debug(f'computer moving {move[0][1]}')
