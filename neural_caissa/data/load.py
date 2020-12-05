import numpy as np
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, input_file):
        """
        Assuming it has 4 input objects
            - origin board state
            - next move board state
            - random next valid move board state
            - target (1 if won, 0 if draw, -1 if lost)
        """
        data = np.load(input_file)
        self.dim = len([k for k, v in data.items() if v.shape[0] != 0])
        self.X_origin = data['arr_0']
        self.X_move = data['arr_1']
        self.X_random = data['arr_2']
        self.Y = data['arr_3']
        print(f'loaded {self.dim} non 0 objects: ', self.X_origin.shape, self.X_move.shape,
              self.X_random.shape, self.Y.shape)

    def __len__(self):
        return self.X_origin.shape[0]

    def __getitem__(self, idx):
        if self.dim == 4:
            return self.X_origin[idx], self.X_move[idx], self.X_random[idx], self.Y[idx]
        return self.X_origin[idx], [], [], self.Y[idx]
