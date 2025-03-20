'''
Author: Jiaxin Zheng
Date: 2024-04-20 15:33:41
LastEditors: Jiaxin Zheng
LastEditTime: 2024-04-20 21:07:48
Description: 
'''
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self,seqs,labels,name):
        # Args.seq_len = seqs.shape[1]  # Args.seq_len
        self.seqs = seqs
        self.labels = labels
        self.name = name
    def __len__(self):
        return self.seqs.shape[0]
    def __getitem__(self,idx):
        if self.labels is not None:
            return self.seqs[idx],self.labels[idx],self.name[idx]
        else:
            return self.seqs[idx],None,self.name[idx]
    @staticmethod
    def collate_fn(batch_list):
        batch_size = len(batch_list)
        seqs = torch.cat([item[0] for item in batch_list]).reshape(batch_size, -1)
        labels = torch.tensor([item[1] if item[1] is not None else [] for item in batch_list])
        name = torch.tensor([item[2] for item in batch_list])
        pos_idx = torch.arange(seqs.shape[1])
        
        return seqs,labels,name,pos_idx