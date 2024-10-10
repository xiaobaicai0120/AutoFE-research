'''
Author: Jiaxin Zheng
Date: 2024-04-03 20:29:35
LastEditors: Jiaxin Zheng
LastEditTime: 2024-04-20 21:55:06
Description: 
'''
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import rootutils


rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.dataset.dataset import CustomDataset
from src.utils.process_df import process_data
from src.constants import SPECIES2ABBR
from src.model.site_pred import get_model as site_get_model
from src.utils.utils import create_object

def predict_main(df, type, site, species):
    
    
    rna_dict, seqs, labels, name = process_data(df, site)
    if rna_dict is None:
        return None, 'Please verify your model selection and data.\nClick the Data format example button to see the correct data format and the supported sites and species types.'
    dataset = CustomDataset(seqs, labels, name)
    dataloader = DataLoader(dataset,batch_size=16,shuffle=False,collate_fn=CustomDataset.collate_fn)
    
    if type =='site':
        Args, model = site_get_model(site)
    else:
        species_abbr = SPECIES2ABBR[species]
        site_species_get_model = create_object(f'src.model.{site}.{species_abbr}.load_model')
        Args, model = site_species_get_model()
    model.eval()
    
    res=[]
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            seqs,labels,name,pos_idx=batch_data
            pred_res = model(seqs.to(Args.device), pos_idx.to(Args.device), name.to(Args.device))
            softmax_res = torch.softmax(pred_res[1], dim=-1)
            max_values, max_indices = torch.max(softmax_res, dim=1)
            
            pred_idx = max_indices.detach().cpu().long().tolist()
            res = res + pred_idx
    
    df['label']=res
    return df[['seq', 'species', 'label']], 'sucess'