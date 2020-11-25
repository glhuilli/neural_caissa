import logging

from neural_caissa.ply.explore import explore_leaves

logger = logging.getLogger(__name__)


def computer_move(state, valuator) -> None:
    """
    Method to decide w
    """
    move = sorted(explore_leaves(state, valuator), key=lambda x: x[0], reverse=state.board.turn)
    if len(move) == 0:
        logger.error('computer_move: no moves to make')
        return

    _log_moves(move)
    state.board.push(move[0][1])


def _log_moves(move, top_moves: int = 3) -> None:
    for i, m in enumerate(move[0:top_moves]):
        logger.debug(f'Top {i} move: {m}')
    logger.debug(f'computer moving {move[0][1]}')
