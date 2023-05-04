"""
Created by xiedong
@Date: 2023/4/19 22:16
"""
from torch.utils.data import DataLoader, Dataset


class BertDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.nums = len(self.features)

    def __len__(self):
        return self.nums

    def __getitem__(self, item):
        data = {
            'input_ids': self.features[item].input_ids.long(),
            'attention_mask': self.features[item].attention_mask.long(),
            'token_type_ids': self.features[item].token_type_ids.long(),
            'seq_label_ids': self.features[item].seq_label_ids.long(),
            'token_label_ids': self.features[item].token_label_ids.long(),
        }
        return data
