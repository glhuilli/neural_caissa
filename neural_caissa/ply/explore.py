import chess
import time


MAX_VALUE = 10_000


def explore_leaves(state, valuator):
    start = time.time()
    valuator.reset()
    b_val = valuator(state)
    c_val, ret = compute_minimax(state, valuator, 0, a=-MAX_VALUE, b=MAX_VALUE, big=True)
    eta = time.time() - start

    print("%.2f -> %.2f: explored %d nodes in %.3f seconds %d/sec" %
          (b_val, c_val, valuator.count, eta, int(valuator.count / eta)))

    return ret


def compute_minimax(state, valuator, depth, a, b, big=False):
    if depth >= 5 or state.board.is_game_over():
        return valuator(state)

    # white is maximizing player
    turn = state.board.turn
    if turn == chess.WHITE:
        ret = -MAX_VALUE
    else:
        ret = MAX_VALUE

    bret = []

    # TODO: Prune tree options with beam search
    isort = []
    for e in state.board.legal_moves:
        state.board.push(e)
        isort.append((valuator(state), e))
        state.board.pop()

    move = sorted(isort, key=lambda x: x[0], reverse=state.board.turn)

    # beam search beyond depth 3
    if depth >= 3:
        move = move[:10]

    for e in [x[1] for x in move]:
        state.board.push(e)
        tval, _ = compute_minimax(state, valuator, depth + 1, a, b)
        state.board.pop()
        if big:
            bret.append((tval, e))
        if turn == chess.WHITE:
            ret = max(ret, tval)
            a = max(a, ret)
            if a >= b:
                break  # b cut-off
        else:
            ret = min(ret, tval)
            b = min(b, ret)
            if a >= b:
                break  # a cut-off

    return ret, bret
