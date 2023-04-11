import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def create_sequence(train_data, train_target, seq_len=24, out_length=3):
    data_x = []
    data_y = []
    for i in range(len(train_data) - seq_len):
        x = train_data[i:i + seq_len]
        y = train_target[i + seq_len:i + seq_len+out_length]
        data_x.append(x)
        data_y.append(y)
    return np.array(data_x), np.array(data_y)


def create_data_loader(df, seq_len, batch_size, device='cpu', normalize_func=None):
    # 标准化
    mm = normalize_func[0]
    ss = normalize_func[1]
    y = df['SWC_F_MDS_1'].to_numpy().reshape(-1, 1)
    X_trans = ss.fit_transform(df)
    # y_trans = mm.fit_transform(y.reshape(-1, 1))
    y_trans = mm.fit_transform(y)
    # 滞后特征
    train_, target_ = create_sequence(X_trans, y_trans, seq_len=seq_len, out_length=3)
    ds = SoilDataset(
        x=train_,
        y=target_,
        device=device,
        seq_len=seq_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
    )


class SoilDataset(Dataset):

    def __init__(self, x, y, device='cpu', seq_len=24):
        self.device = device
        self.x = x
        self.y = y
        self.seq_len = seq_len

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x, dtype=torch.float32).to(self.device), torch.tensor(y, dtype=torch.float32).to(self.device)

    def __len__(self):
        length = len(self.x) if len(self.x) == len(self.y) else 0
        return length
