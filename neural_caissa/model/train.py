import click
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from neural_caissa.data.load import ChessDataset
from neural_caissa.model.chess_conv_net import ChessConvNet

_PIECES = 12
_BOARD_DIM = 8
_EPOCHS = 100
_BATCH_SIZE = 256


@click.command()
@click.option('--input_data_file',
              default='data/serialized_data/dataset_1k.npz',
              help='Input data file.')
@click.option('--output_model', default='nets/neural_score.pth', help='Output model file.')
@click.option('--checkpoint_path', default='', help='Model checkpoint.')
@click.option('--checkpoint/--no-checkpoint', ' /-C', default=False)
def main(input_data_file, output_model, checkpoint_path, checkpoint):
    """
    Following this basic example:
         https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    """
    chess_dataset = ChessDataset(input_data_file)
    train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=_BATCH_SIZE, shuffle=True)

    model = ChessConvNet()
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.MSELoss()
    recorded_epoch = 0

    if checkpoint_path and checkpoint:
        checkpnt = torch.load(checkpoint_path)
        model.load_state_dict(checkpnt['model_state_dict'])
        optimizer.load_state_dict(checkpnt['optimizer_state_dict'])
        recorded_epoch = checkpnt['epoch']
        recorded_loss = checkpnt['loss']
        print(f'recorded current loss: {recorded_loss}')

    model.train()

    for epoch in tqdm(range(_EPOCHS-recorded_epoch)):
        all_loss = 0
        num_loss = 0
        for batch_idx, (data_origin, data_move, data_random, target) in tqdm(enumerate(train_loader)):
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

        current_loss = all_loss / num_loss
        print("%3d: %f" % (epoch, current_loss))

        torch.save(model.state_dict(), output_model)
        if checkpoint_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                }, checkpoint_path)


if __name__ == "__main__":
    main()
