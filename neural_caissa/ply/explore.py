import logging
import time

import chess

MAX_VALUE = float('inf')
_SEARCH_DEPTH = 2
_MAX_MOVES = 10

logger = logging.getLogger(__name__)


def explore_leaves(state, valuator):
    start = time.time()
    valuator.reset()
    current_score = valuator(state)
    movement_score, movements = compute_minimax(state,
                                                valuator,
                                                0,
                                                alpha=-MAX_VALUE,
                                                beta=MAX_VALUE)
    eta = time.time() - start

    valuator_type = type(valuator).__name__
    if valuator_type == 'BaselineValuator':
        logger.debug(
            f"Score transition: {current_score} -> {movement_score}\n "
            f"explored {valuator.count} nodes in {eta} seconds {int(valuator.count / eta)}/sec")
    elif valuator_type == 'NeuralValuator':
        logger.debug(f"Score transition: {current_score} -> {movement_score} ({eta} seconds)")

    return movements


def compute_minimax(state, valuator, depth, alpha, beta):
    """
    Minimax algorithm:
        https://towardsdatascience.com/create-ai-for-your-own-board-game-from-scratch-minimax-part-2-517e1c1e3362

    Note that this version is using the alpha-beta pruning
    """
    if depth >= _SEARCH_DEPTH or state.board.is_game_over():
        return valuator(state), []

    turn = state.board.turn
    if turn == chess.WHITE:
        best_value = -MAX_VALUE
    else:
        best_value = MAX_VALUE

    movements = []

    # TODO: Explore finding best options with beam search
    #   https://medium.com/@dhartidhami/beam-search-in-seq2seq-model-7606d55b21a5
    options = []
    for move in state.board.legal_moves:
        state.board.push(move)
        options.append((valuator(state), move))
        state.board.pop()

    moves = sorted(options, key=lambda x: x[0], reverse=state.board.turn)
    if depth >= _SEARCH_DEPTH - 1:  # TODO: improve pruning strategy, this is shameful
        moves = moves[:_MAX_MOVES]

    for move in [x[1] for x in moves]:
        state.board.push(move)
        move_score, _ = compute_minimax(state, valuator, depth + 1, alpha, beta)

        state.board.pop()

        movements.append((move_score, move))

        # Alpha-beta optimization
        if turn == chess.WHITE:
            best_value = max(best_value, move_score)
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
        else:
            best_value = min(best_value, move_score)
            beta = min(beta, best_value)
            if alpha >= beta:
                break

    return best_value, movements
