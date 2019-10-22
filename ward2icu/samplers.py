import torch
from ward2icu.utils import tile as tile_func


class IdentitySampler:
    def __init__(self, X, y, tile=False):
        self.X = X
        self.y = tile_func(y, X.shape[1]) if tile else y
        self.tile = tile
        self.device = X.device

    def sample(self):
        return self.X, self.y

    def __len__(self):
        return self.y.shape[0]


class BinaryBalancedSampler:
    def __init__(self, X, y, tile=False, batch_size=None):
        assert len(y.unique()) <= 2
        self.sequence_length = X.shape[1]
        self.batch_size = batch_size
        self.tile = tile
        self.device = X.device

        self.majority_class = y.mode().values
        self.minority_class = y[y != self.majority_class].mode().values

        self.majority_mask = (y == self.majority_class)
        self.minority_mask = (y == self.minority_class)

        self.majority_count = int(self.majority_mask.sum())
        self.minority_count = int(self.minority_mask.sum())

        if batch_size:
            assert batch_size/2 < self.minority_count

        self.X_majority = X[self.majority_mask]
        self.X_minority = X[self.minority_mask]

        self.y_majority = y[self.majority_mask]
        self.y_minority = y[self.minority_mask]
        self.should_sample_minority_class = self.batch_size is not None

    def sample(self):
        batch_size_half = (int(self.batch_size/2)
                           if self.batch_size is not None 
                           else self.minority_count)

        idx_maj, idx_min = self._create_idxs(batch_size_half)

        X_majority_batch = self.X_majority[idx_maj]
        y_majority_batch = self.y_majority[idx_maj]

        X_minority_batch = (self.X_minority[idx_min]
                            if self.should_sample_minority_class
                            else self.X_minority)
        y_minority_batch = (self.y_minority[idx_min]
                            if self.should_sample_minority_class
                            else self.y_minority)

        X = torch.cat((X_majority_batch, X_minority_batch), dim=0)
        y = torch.cat((y_majority_batch, y_minority_batch), dim=0)

        y = tile_func(y, self.sequence_length) if self.tile else y
        return X, y

    def __len__(self):
        return int(self.batch_size or 2*self.minority_count)
    
    def _create_idxs(self, batch_size):
        idx_maj = torch.randperm(self.majority_count)[:batch_size]
        idx_min = (torch.randperm(self.minority_count)[:batch_size]
                   if self.should_sample_minority_class else None)
        return idx_maj, idx_min
