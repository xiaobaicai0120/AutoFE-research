'''
Author: Jiaxin Zheng
Date: 2024-04-20 18:51:15
LastEditors: Jiaxin Zheng
LastEditTime: 2024-04-20 21:22:57
Description: 
'''
from collections import Counter

import torch
import pandas as pd

import rootutils
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from src.tokenizer.tokenizer import Tokenizer

def process_data(df, site):
    try:
        site_name_dict = {
            '4mC': {'C.equisetifolia': 0, 'F.vesca': 1, 'S.cerevisiae': 2, 'Tolypocladium': 3},
            '5hmC': {'H.sapiens': 0, 'M.musculus': 1},
            '6mA': {'A.thaliana': 0, 'C.elegans': 1, 'C.equisetifolia': 2, 'D.melanogaster': 3, 'F.vesca': 4, 'H.sapiens': 5, 'R.chinensis': 6, 'S.cerevisiae': 7, 'T.thermophile': 8, 'Tolypocladium': 9, 'Xoc BLS256': 10}
        }
        name_dict = site_name_dict[site]
        tokenizer = Tokenizer([i for i in "ATCG"],WINDOWS_SIZE=1,stride=1)
        
        df["usage_idx"] = 0
        df["name_idx"] = df["species"].apply(lambda x:name_dict[x])
        all_seqs = df['seq'].values.tolist()
        encode_res = tokenizer.encode(all_seqs)
        
        seqs = encode_res[1]
        labels = torch.tensor(df['label']) if 'label' in df.columns else None
        name = torch.LongTensor(df["name_idx"])
    except:
        return None, None, None, None
    return tokenizer.rna_dict, seqs, labels, name
    
if __name__=='__main__':
    file_path = 'data/4mC_all_data.csv'
    df = pd.read_csv(file_path).reset_index(drop=True)
    rna_dict, seqs, labels, name = process_data(df)
    print(seqs, labels, name)
    