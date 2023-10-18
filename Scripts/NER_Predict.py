# %% 
# Load libraries
#
from operator import index
import os
from pathlib import Path
import pandas as pd
from collections import Counter

import spacy
from spacy import displacy
from spacy.tokens import DocBin
import json
from datetime import datetime
from tqdm import tqdm
import re

# specify the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cpuPath = './NER/'
gpuPath = './NER_gpu/'


# %% #################################################################
# Prediction with trained model
# ####################################################################

def EntityPredictDisplay(test_txt):
    # pass our test instance into the trained pipeline
    doc = nlp_output(test_txt)

    # customize the label colors
    colors = {'D-spi': "lightblue", 'D-rac':'lightblue', 'D-gen':'lightblue', 'D-sxo':'lightblue', 'D-soc':'lightblue',
            'D-age':'lightblue', 'B-ces':'#b9db57','B-tme':'#b9db57','B-use':'#b9db57','B-int':'#b9db57','B-pcp':'#b9db57',
            'B-hlt':'#b9db57','B-exp':'#b9db57','B-prv':'#b9db57','B-stm':'#b9db57',
            'M-mth':'#57db94','M-sts':'#57db94','M-dat':'#57db94',
            'C-con':'#5784db','C-chm':'#5784db','C-flv':'#5784db','C-dgn':'#5784db','C-oth':'#5784db',
            'T-etc':'#c957db','T-com':'#c957db','T-oth':'#c957db','T-mkt':'#c957db',
            'P-reg':'lightyellow','P-bod':'lightyellow','P-lic':'lightyellow','P-mkt':'lightyellow',
            'P-red':'lightyellow','P-trt':'lightyellow','P-lbl':'lightyellow',
            'R-rel':'#62d835','L-loc':'#ff2600'}

    options = {"ents": ['D-spi', 'D-rac', 'D-gen', 'D-sxo', 'D-soc',
                        'D-age', 'B-ces','B-tme','B-use','B-int','B-pcp',
                        'B-hlt','B-exp','B-prv','B-stm',
                        'M-mth','M-sts','M-dat',
                        'C-con','C-chm','C-flv','C-dgn','C-oth',
                        'T-etc','T-com','T-oth','T-mkt',
                        'P-reg','P-bod','P-lic','P-mkt',
                        'P-red','P-trt','P-lbl',
                        'R-rel','L-loc'], 
                "colors": colors}

    # visualize the identified entities
    ### svg = displacy.render(doc, style="dep", options=options, jupyter=True)
    
    svg = displacy.render(doc, style="ent", options=options)
    return doc

# load the CPU-trained model 
nlp_output = spacy.load("./NER/model_all_expand/model-best")
# !!!!!!! make sure use the right model!!!!!!!!


# %% ####################################################
# Read in test dataset
#
df_test = pd.read_csv(cpuPath+'testDataset.csv', index_col=0, encoding='utf8')
print(len(df_test))
# shuffle data
df_test = df_test.sample(frac = 1, random_state=6).reset_index()


# %% 
# predict an abstract
#
num = 19
test_txt = df_test.loc[num,'abstract']
doc = EntityPredictDisplay(test_txt)


# %%
# output NER predictions for testDataset
#
"""
with open(cpuPath + 'testDatasetNER.csv', 'w') as f:
    f.writelines('index,named_entities\n')
    for idx, row in df_test.iterrows():
        print(idx)
        doc = EntityPredictDisplay(row['abstract'])
        ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
        f.writelines(str(row['index']) + ',' + str(ents) + '\n') 
"""
