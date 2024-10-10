'''
Author: Jiaxin Zheng
Date: 2024-04-20 18:55:56
LastEditors: Jiaxin Zheng
LastEditTime: 2024-04-20 18:57:14
Description: 
'''
from tqdm import tqdm
import numpy as np
import torch
class Tokenizer():
    def __init__(self, CODE_SETS, WINDOWS_SIZE,stride=1):
        self.windows_size = WINDOWS_SIZE
        assert type(CODE_SETS)==list or CODE_SETS =="fit"
        
        if type(CODE_SETS)==list:
            self.code_sets = CODE_SETS
        elif CODE_SETS=="fit":
            self.code_sets = None
        
        if self.code_sets:
            self.rna_dict,self.dict_size = self.get_rna_dict()
            self.stride = stride
    def __get_all_word(self, windows_size):
        from functools import reduce
        a, b = self.code_sets, windows_size  
        res = reduce(lambda x, y:[z0 + z1 for z0 in x for z1 in y], [a] * b)
        return res    
    
    def get_rna_dict(self):
        rna_dict = {'<PAD>':0,'<CLS_label>':1,'<CLS_name>':2,'<MASK>':3}
        num = len(rna_dict)
        for j in range(self.windows_size,0,-1):
            for word in self.__get_all_word(j):
                rna_dict[word] = num
                num += 1
        return rna_dict,len(rna_dict)
    @staticmethod
    def pad_code(ngram_encode):
        ngram_encode2 = np.zeros([len(ngram_encode),len(max(ngram_encode,key = lambda x: len(x)))]) # 生成样本数*最长序列长度
        for i,j in enumerate(ngram_encode):
            ngram_encode2[i][0:len(j)] = j
        return ngram_encode2
    
    def encode(self,rna_seq,padding=True,return_pt=True,concer_tail=False):
        
        mRNA_dic,_ = self.get_rna_dict()

        n_gram = []
        n_gram_encode = []
        n_gram_len = []
        len_rna_seq = len(rna_seq)
        for i in tqdm(range(len_rna_seq)):
            cur_rna_seq = []
            cur_rna_encode = []
            if "U" in rna_seq[i]:
                rna_seq[i] = rna_seq[i].replace("U","C")
            for j in range(0,len(rna_seq[i]),self.stride):
#                 print(self.windows_size)
                len_win = len(rna_seq[i][j:j+self.windows_size])
#                 print(len_win)
                if not concer_tail:
                    if  len_win == self.windows_size:
                        try:
                            cur_rna_seq.append(rna_seq[i][j:j+self.windows_size].upper())
                            cur_rna_encode.append(mRNA_dic[rna_seq[i][j:j+self.windows_size].upper()])
                        except Exception as e:
                            print(e)
                            print(rna_seq[i],i)
                else:
                        cur_rna_seq.append(rna_seq[i][j:j+self.windows_size].upper())
                        cur_rna_encode.append(mRNA_dic[rna_seq[i][j:j+self.windows_size].upper()])
            n_gram.append(cur_rna_seq)
            n_gram_encode.append(cur_rna_encode)
            n_gram_len.append(len(cur_rna_encode))
        if padding:
            n_gram_encode = Tokenizer.pad_code(n_gram_encode)
        if return_pt:
            n_gram_encode = torch.LongTensor(n_gram_encode)  
        return n_gram, n_gram_encode, torch.LongTensor(n_gram_len), mRNA_dic