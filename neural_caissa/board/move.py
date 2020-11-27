import logging

from neural_caissa.ply.explore import explore_leaves

logger = logging.getLogger(__name__)


def computer_move(state, valuator) -> bool:
    """
    Method to decide which movements
    """
    sorted_movements = sorted(explore_leaves(state, valuator),
                              key=lambda x: x[0],
                              reverse=state.board.turn)

    if len(sorted_movements) == 0:
        logger.debug('No moves to make: GAME OVER?')
        return True
    _log_moves(sorted_movements)

    # use the first movement as these are already sorted
    state.board.push(sorted_movements[0][1])
    return False


def _log_moves(move, top_moves: int = 3) -> None:
    for i, m in enumerate(move[0:top_moves]):
        logger.debug(f'Top {i} move: {m}')
    logger.debug(f'computer moving {move[0][1]}')
