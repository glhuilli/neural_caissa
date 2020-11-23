from neural_caissa.ply.explore import explore_leaves


def computer_move(state, valuator, top_moves: int = 3, verbose: bool = True) -> None:
    """
    Method to decide
    """
    move = sorted(explore_leaves(state, valuator), key=lambda x: x[0], reverse=state.board.turn)
    if len(move) == 0:
        return

    if verbose:
        show_moves(state, move, top_moves)

    state.board.push(move[0][1])


def show_moves(state, move, top_moves):
    for i, m in enumerate(move[0:top_moves]):
        print(f"Top {i} move: ", m)
    print(state.board.turn, "moving", move[0][1])
