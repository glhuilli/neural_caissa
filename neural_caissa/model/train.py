import click

from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim

from neural_caissa.data.load import ChessDataset
from neural_caissa.model.chess_conv_net import ChessConvNet


_PIECES = 12
_BOARD_DIM = 8
_EPOCHS = 100


@click.command()
@click.option('--input_data_file', default='data/serialized_data/dataset_1k.npz', help='Input data file.')
@click.option('--output_model', default='nets/neural_score.pth', help='Output model file.')
def main(input_data_file, output_model):
    """
    Following this basic example:
         https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    """
    chess_dataset = ChessDataset(input_data_file)
    train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=256, shuffle=True)

    model = ChessConvNet()
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.MSELoss()

    model.train()

    for epoch in tqdm(range(_EPOCHS)):
        all_loss = 0
        num_loss = 0
        for batch_idx, (data_origin, data_move, data_random, target) in enumerate(train_loader):
            target = target.unsqueeze(-1)
            data_origin = data_origin.float()

            target = target.float()

            optimizer.zero_grad()
            output = model(data_origin)

            loss = loss_function(output, target)
            loss.backward()

            optimizer.step()

            all_loss += loss.item()
            num_loss += 1

        if epoch % 10 == 0:
            print("%3d: %f" % (epoch, all_loss / num_loss))

        torch.save(model.state_dict(), output_model)


if __name__ == "__main__":
    main()
