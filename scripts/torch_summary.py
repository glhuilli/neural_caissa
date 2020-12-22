import click
import torch
from torchsummary import summary

from neural_caissa.model.chess_conv_net import ChessConvNet


@click.command()
@click.option('--model_path', default='nets/neural_score.pth', help='Model file.')
def main(model_path):
    torch_model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = ChessConvNet()
    model.load_state_dict(torch_model)
    summary(model, (12, 8, 8))


if __name__ == "__main__":
    main()
