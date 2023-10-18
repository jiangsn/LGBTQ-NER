# %% 
# Load libraries
#
from operator import index
import os
import pandas as pd
from collections import Counter

import json
from datetime import datetime
from tqdm import tqdm
import re

# specify the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
inputPath = './NER/'
outPath = './SpacyProdigy/'


# %% #################################################################
# Read pre-defined NER tags
#    #################################################################
df_tags = pd.read_csv(inputPath + 'TobaccoNamedEntitiesV2.csv', encoding='utf8')

# global list to store the NER list
kw_list = []    # [0] tag, [1] entity name

for idx, row in df_tags.iterrows():
    # if idx != 25:
    #     continue
    tag = row['tag']
    entities = row['entities'].split(',')
    for entity in entities:
        entClean = entity.strip()
        if entClean != '':
            kw_list.append([tag, entClean])

tag_classes = Counter([x[0] for x in kw_list])
print('Total %d tags in %d classes' % (len(kw_list),len(tag_classes)))
tag_classes


# %%
# Function to label entities in text
#
def structure_training_data(text, source, kw_list):
    """
    text is a string with cases
    kw[0] is tag name, e.g., 'M-sts'
    kw[1] is entity name, e.g., '95% CI'
    """
    results = []
    entities = []

    # search for instances of keywords within the text (ignoring letter case)
    for kw in kw_list:
        # \b are word boundaries, and s is optional
        search = re.finditer(r'\b'+kw[1]+r's?\b', text, flags=re.IGNORECASE)
        
        # store the start/end character positions
        all_instances = [[m.start(),m.end()] for m in search] 
        
        # if the callable_iterator found matches, create an 'entities' list
        if len(all_instances)>0:
            for i in all_instances:
                start = i[0]
                end = i[1]
            ### case 1: if a single-word entity is within a phrase entity, ignore it
            ### case 2: if an entity phrase envelops a found entity, remove the found entity
                already = False
                for ent in entities:
                    if ent['start']<=start and ent['end']>=end:   # case 1
                        already = True          # break immediately
                        break
                    if start<=ent['start'] and ent['end']<=end:   # case 2
                        entities.remove(ent)    # remove as many as is enveloped
                if not already:
                    entities.append({"start":start, "end":end, "label":kw[0]}) 
                
    # add any found entities into a JSON format within collective_dict
    if len(entities)>0:
        # results = [text, {"entities": entities}]
        results = {"text":text, "meta":{"source":source},
                "spans":entities}
        # this is a global list containing all annotated examples
        collective_dict.append(results)
        return True
    else:
        print('No entities identified: ', end='')
        return False


# %%        
# Test the label-entity function

text = ('nicotine and nicotine replacement therapy Family Smoking '
        'Prevention and Tobacco Control Act is enacted.'
        ' smokeless tobacco are used more in China than in the United '
        'States. smoking cesation is the goal to pursue when the')
source = "Tobacco Control"

# this dictionary will contain all annotated examples
collective_dict = []
structure_training_data(text, source, kw_list)

for item in collective_dict:
    print(json.dumps(item))


# %% ####################################################
# Read in text dataset and split into train, dev, and test
#
df_text = pd.read_csv(inputPath+'testDataset.csv', index_col=0, encoding='utf8')
# shuffle the dataset
df_text = df_text.sample(frac = 1, random_state=6)

# split into train, val, and test
# train_num = int(len(df_text) * 0.8)
# dev_num = int(len(df_text) * 0.1)
# df_train = df_text[:train_num].reset_index()
# df_dev = df_text[train_num:train_num+dev_num].reset_index()
# df_test = df_text[train_num+dev_num:].reset_index()

df_train = df_text[:20].reset_index()
df_train['doc_id'] = ''
df_train['rank'] = ''
for idx, row in df_train.iterrows():
    df_train.loc[idx,'doc_id'] = 'd' + str(idx)
    df_train.loc[idx,'rank'] = str(idx)
    df_train.loc[idx,'index'] = str(idx)
    df_train.loc[idx,'abstract'] = row['abstract'].replace(', ', ' , ').replace('. ', ' . ').replace('? ', ' ? ')
df_train = df_train[['index','doc_id','abstract','rank']]
df_train.columns = ['id','doc_id','doc_content','rank']

# df_train.to_csv(outPath + 'documents.csv', index=False)
df_train.head()


# %% #################################################################
# Label the training data using predefined kw_list
# ####################################################################

collective_dict = []
for idx, row in tqdm(df_train.iterrows()):
    text = row['abstract'].replace('\u2005','') # replace the unicode space
    source = "paper index %d" % row['index']
    result = structure_training_data(text, source, kw_list)
    if not result:
        print(idx)

# {""type"":""annotation"",""text"":""association"",""start"":195,""end"":195,""category"":""T-com"",""date"":""5/23/2022, 13:59:23""}


# %%
with open(outPath+'prodigyNER01.jsonl','w') as f:
    for item in collective_dict:
        f.writelines(json.dumps(item) + '\n')

        
# %%
