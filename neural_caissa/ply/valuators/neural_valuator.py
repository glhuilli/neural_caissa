import torch

from neural_caissa.model.chess_conv_net import ChessConvNet
from neural_caissa.ply.valuators.valuator import Valuator
from neural_caissa.board.state import State


class NeuralValuator(Valuator):
    def __init__(self, model_path):
        torch_model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model = ChessConvNet()
        self.model.load_state_dict(torch_model)

    def __call__(self, state: State) -> float:
        board = state.serialize_conv()[None]
        output = self.model(torch.tensor(board).float())
        return float(output.data[0][0])
