'''
Author: Jiaxin Zheng
Date: 2024-04-03 20:26:17
LastEditors: Jiaxin Zheng
LastEditTime: 2024-04-20 21:49:26
Description: 
'''
SITE_MODEL_PATH = {
    '4mC': 'model/site/4mC.pkl',
    '5hmC': 'model/site/5hmC.pkl',
    '6mA': 'model/site/6mA.pkl',
}

SITE_SPECIES = {
    '4mC': [
        'C. equisetifolia',
        'F. vesca',
        'S. cerevisiae',
        'Tolypocladium'
        ],
    '5hmC': [
        'M. musculus',
        'H. sapiens'
        ],
    '6mA':  [ 
        'A. thaliana',
        'C. elegans',
        'C. equisetifolia',
        'D. melanogast',
        'F. vesca',
        'H. sapiens',
        'R. chinensis',
        'S. cerevisiae',
        'T. thermophil',
        'Tolypocladium',
        'Xoc',
        # 'BLS25'
        ]
}

SPECIES2FILE = {
    'C. equisetifolia':'Ceq_acc.pkl',
    'F. vesca':'Fve_Acc.pkl',
    'S. cerevisiae':'Sce_Acc.pkl',
    'Tolypocladium':'Tol_ACC.pkl',
    
    'M. musculus':'Mmusculus_Acc.pkl',
    'H. sapiens':'Hsapiens_Acc.pkl',
    
    'A. thaliana': 'Ath_acc.pkl',
    'C. elegans': 'Cel_acc.pkl',
    'C. equisetifol': 'Ceq_acc.pkl',
    'D. melanogast': 'Dmelanogast_Acc.pkl',
    'F. vesca': 'Fvesca_Acc.pkl',
    'H. sapiens': 'Hsapiens_Acc.pkl',
    'R. chinensis': 'Rchinensis_ACC.pkl',
    'S. cerevisiae': 'Scerevisiae_Acc.pkl',
    'T. thermophil': 'Tthermophil_Acc.pkl',
    'Tolypocladiu': 'Tolypocladiu_ACC.pkl',
    'Xoc': 'Xoc_Acc.pkl',
    # 'BLS25': 'BLS25_Acc.pkl'
}
SPECIES = ['C. equisetifolia', 'F. vesca', 'S. cerevisiae', 'Tolypocladium', 'M. musculus', 'H. sapiens', 'A. thaliana', 'C. elegans', 'C. equisetifol', 'D. melanogast', 'R. chinensis', 'T. thermophil', 'Tolypocladiu', 'Xoc', 'BLS25']
SPECIES2ABBR = {
    'C. equisetifolia':'Ceq',
    'F. vesca':'Fve',
    'S. cerevisiae':'Sce',
    'Tolypocladium':'Tol',
    
    'M. musculus':'Mmusculus',
    'H. sapiens':'Hsapiens',
    
    'A. thaliana': 'Ath',
    'C. elegans': 'Cel',
    'C. equisetifol': 'Ceq',
    'D. melanogast': 'Dmelanogast',
    # 'F. vesca': 'Fve',
    'H. sapiens': 'Hsapiens',
    'R. chinensis': 'Rchinensis',
    # 'S. cerevisiae': 'Sce',
    'T. thermophil': 'Tthermophil',
    # 'Tolypocladiu': 'Tol',
    'Xoc': 'Xoc',
    # 'BLS25': 'BLS25'
}
