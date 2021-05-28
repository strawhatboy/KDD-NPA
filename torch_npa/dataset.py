import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import get_train_data, load_data
import numpy as np

def val_collate(batch):
    uid = [data[0] for data in batch]
    history = [data[1] for data in batch]
    candidate = [data[2] for data in batch]
    label = [data[3] for data in batch]

    uid = torch.LongTensor(uid)
    history = torch.LongTensor(history)
    # candidate = torch.LongTensor(candidate)
    label = torch.LongTensor(label)

    return [uid, history, candidate, label]

class MindSDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return np.array(self.data[index][0], dtype='long'), \
            np.array(self.data[index][1], dtype='long'), \
            np.array(self.data[index][2], dtype='long'), \
            np.array(self.data[index][3], dtype='long')


def get_dataloader(batch_size=128, ):
    train_data, val_data, user_len, word_len, embedding_mat = load_data()
    train_dataset = MindSDataset(train_data)
    val_dataset = MindSDataset(val_data)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=8)
    return train_loader, val_loader, user_len, word_len, embedding_mat
